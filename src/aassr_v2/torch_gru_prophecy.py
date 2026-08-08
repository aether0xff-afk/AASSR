from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Iterable, Sequence

from .gru_prophecy import GRUTrainingStats
from .prophecy import ProphecyStep
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class TorchGRUMemory:
    hidden: Any


class TorchGRUProphecy:
    """Torch/CUDA backend for the canonical OnlineGRUProphecy algorithm.

    It preserves the custom GRU equations, dimensions, confidence rule and
    elementwise-clipped online SGD update while adding vectorized prediction for
    holdout batches. Float32 CUDA is intended for the local high-throughput run;
    float64 CPU can be used for close numerical equivalence checks.
    """

    def __init__(
        self,
        state_size: int,
        *,
        action_feature_size: int = 16,
        hidden_size: int = 24,
        learning_rate: float = 0.02,
        seed: int = 7,
        replay_limit: int = 512,
        device: str = "auto",
        dtype: str = "float32",
        allow_cpu_fallback: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchGRUProphecy requires PyTorch") from exc
        if min(state_size, action_feature_size, hidden_size, replay_limit) <= 0:
            raise ValueError("sizes and replay_limit must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

        self.torch = torch
        requested = str(device).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(requested)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
            resolved = torch.device("cpu")
        self.device = resolved
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.requested_device = device
        self.state_size = int(state_size)
        self.action_feature_size = int(action_feature_size)
        self.hidden_size = int(hidden_size)
        self.input_size = self.state_size + self.action_feature_size
        self.learning_rate = float(learning_rate)
        self.replay_limit = int(replay_limit)
        self._name = f"torch-gru:{self.device.type}:{dtype}"
        self._updates = 0
        self._last_loss = 0.0
        self._mean_loss = 0.0
        self._action_counts: dict[str, int] = {}
        self._action_feature_cache: dict[str, Any] = {}
        self._template_order: deque[tuple[float, ...]] = deque()
        self._template_buckets: dict[tuple[float, ...], deque[StateSnapshot]] = defaultdict(deque)
        self._template_dirty = True
        self._template_matrix: Any | None = None
        self._template_states: tuple[StateSnapshot, ...] = ()

        rng = random.Random(seed)

        def matrix(rows: int, columns: int):
            scale = 1.0 / math.sqrt(max(1, columns))
            values = [[rng.uniform(-scale, scale) for _ in range(columns)] for _ in range(rows)]
            return torch.tensor(values, device=self.device, dtype=self.dtype, requires_grad=True)

        def zeros(size: int):
            return torch.zeros(size, device=self.device, dtype=self.dtype, requires_grad=True)

        self.Wz = matrix(self.hidden_size, self.input_size)
        self.Uz = matrix(self.hidden_size, self.hidden_size)
        self.bz = zeros(self.hidden_size)
        self.Wr = matrix(self.hidden_size, self.input_size)
        self.Ur = matrix(self.hidden_size, self.hidden_size)
        self.br = zeros(self.hidden_size)
        self.Wh = matrix(self.hidden_size, self.input_size)
        self.Uh = matrix(self.hidden_size, self.hidden_size)
        self.bh = zeros(self.hidden_size)
        self.Wo = matrix(self.state_size, self.hidden_size)
        self.bo = zeros(self.state_size)
        self._parameters = (
            self.Wz, self.Uz, self.bz,
            self.Wr, self.Ur, self.br,
            self.Wh, self.Uh, self.bh,
            self.Wo, self.bo,
        )
        self._train_memory = self.initial_memory()

    @property
    def name(self) -> str:
        return self._name

    @property
    def training_stats(self) -> GRUTrainingStats:
        return GRUTrainingStats(self._updates, self._last_loss, self._mean_loss)

    def initial_memory(self) -> TorchGRUMemory:
        return TorchGRUMemory(self.torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype))

    def reset_sequence(self) -> None:
        self._train_memory = self.initial_memory()

    def reset_context(self) -> None:
        self.reset_sequence()

    def _action_features_cpu(self, action: Action) -> tuple[float, ...]:
        vector = [0.0] * self.action_feature_size
        tokens = [action.verb_name, action.signature]
        tokens.extend(f"{key}={value}" for key, value in sorted(action.parameters.items()))
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.action_feature_size
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)

    def _action_features(self, action: Action):
        cached = self._action_feature_cache.get(action.signature)
        if cached is None:
            cached = self.torch.tensor(
                self._action_features_cpu(action), device=self.device, dtype=self.dtype
            )
            self._action_feature_cache[action.signature] = cached
        return cached

    def _inputs(self, states: Sequence[StateSnapshot], actions: Sequence[Action]):
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if not states:
            return self.torch.empty((0, self.input_size), device=self.device, dtype=self.dtype)
        for state in states:
            if len(state.vector) != self.state_size:
                raise ValueError(
                    f"expected state vector of length {self.state_size}, got {len(state.vector)}"
                )
        state_tensor = self.torch.tensor(
            [tuple(state.vector) for state in states], device=self.device, dtype=self.dtype
        )
        action_tensor = self.torch.stack([self._action_features(action) for action in actions], dim=0)
        return self.torch.cat((state_tensor, action_tensor), dim=1)

    def _forward_batch(self, x, hidden):
        z = self.torch.sigmoid(x @ self.Wz.T + hidden @ self.Uz.T + self.bz)
        r = self.torch.sigmoid(x @ self.Wr.T + hidden @ self.Ur.T + self.br)
        rh = r * hidden
        n = self.torch.tanh(x @ self.Wh.T + rh @ self.Uh.T + self.bh)
        next_hidden = (1.0 - z) * n + z * hidden
        output = next_hidden @ self.Wo.T + self.bo
        return output, next_hidden

    def predict_vector(
        self, state: StateSnapshot, action: Action, *, memory: TorchGRUMemory | None = None
    ) -> tuple[float, ...]:
        with self.torch.inference_mode():
            x = self._inputs((state,), (action,))
            hidden = (memory or self.initial_memory()).hidden.reshape(1, -1)
            output, _ = self._forward_batch(x, hidden)
            return tuple(float(value) for value in output[0].cpu().tolist())

    def _add_template(self, state: StateSnapshot) -> None:
        key = tuple(float(value) for value in state.vector)
        self._template_order.append(key)
        self._template_buckets[key].append(state)
        while len(self._template_order) > self.replay_limit:
            oldest = self._template_order.popleft()
            bucket = self._template_buckets[oldest]
            bucket.popleft()
            if not bucket:
                del self._template_buckets[oldest]
        self._template_dirty = True

    def _templates(self):
        if self._template_dirty:
            states = [bucket[-1] for bucket in self._template_buckets.values()]
            states.sort(key=lambda state: tuple(sorted(state.facts)))
            self._template_states = tuple(states)
            self._template_matrix = (
                self.torch.tensor(
                    [tuple(state.vector) for state in states], device=self.device, dtype=self.dtype
                )
                if states else None
            )
            self._template_dirty = False
        return self._template_matrix, self._template_states

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state
        seen = self._action_counts.get(action.signature, 0)
        experience = seen / (seen + 4.0)
        loss_quality = 1.0 / (1.0 + max(0.0, self._mean_loss))
        return max(0.0, min(1.0, experience * loss_quality))

    def coverage(self, state: StateSnapshot, actions: Iterable[Action]) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(self.confidence(state, action) for action in materialized) / len(materialized)

    def _decode_outputs(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        outputs,
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        matrix, templates = self._templates()
        if matrix is None:
            return tuple(
                (Prediction(state, 1.0, source=f"{self.name}:unseen"),)
                for state in states
            )
        out_norm = self.torch.linalg.vector_norm(outputs, dim=1, keepdim=True)
        template_norm = self.torch.linalg.vector_norm(matrix, dim=1, keepdim=True).T
        denom = out_norm * template_norm
        similarities = outputs @ matrix.T
        similarities = self.torch.where(
            denom > 0,
            similarities / self.torch.clamp_min(denom, 1e-12),
            self.torch.zeros_like(similarities),
        )
        k = min(samples, len(templates))
        order = self.torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :k]
        scores = self.torch.gather(similarities, 1, order)
        probabilities = self.torch.softmax(scores * 4.0, dim=1)
        order_cpu = order.cpu().tolist()
        probs_cpu = probabilities.cpu().tolist()
        rows: list[tuple[Prediction, ...]] = []
        for row, (state, action) in enumerate(zip(states, actions, strict=True)):
            confidence = self.confidence(state, action)
            if confidence <= 0.0:
                rows.append((Prediction(state, 1.0, source=f"{self.name}:unseen"),))
                continue
            suffix = "exact" if confidence >= 0.75 else "action-family"
            predictions = [
                Prediction(
                    templates[int(index)],
                    float(probability) * confidence,
                    source=f"{self.name}:{suffix}",
                )
                for index, probability in zip(order_cpu[row], probs_cpu[row], strict=True)
            ]
            if confidence < 1.0:
                predictions.append(
                    Prediction(state, 1.0 - confidence, source=f"{self.name}:uncertain")
                )
            rows.append(tuple(predictions))
        return tuple(rows)

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if not states:
            return ()
        with self.torch.inference_mode():
            x = self._inputs(states, actions)
            hidden = self.torch.zeros(
                (len(states), self.hidden_size), device=self.device, dtype=self.dtype
            )
            outputs, _ = self._forward_batch(x, hidden)
            return self._decode_outputs(states, actions, outputs, samples=samples)

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: TorchGRUMemory | None,
        samples: int,
    ) -> ProphecyStep:
        with self.torch.inference_mode():
            x = self._inputs((state,), (action,))
            hidden = (memory or self.initial_memory()).hidden.reshape(1, -1)
            output, next_hidden = self._forward_batch(x, hidden)
            predictions = self._decode_outputs((state,), (action,), output, samples=samples)[0]
            return ProphecyStep(predictions, TorchGRUMemory(next_hidden[0].detach().clone()))

    def predict(self, state: StateSnapshot, action: Action, *, samples: int) -> tuple[Prediction, ...]:
        return self.predict_batch((state,), (action,), samples=samples)[0]

    def advance_sequence(self, state: StateSnapshot, action: Action) -> None:
        with self.torch.inference_mode():
            x = self._inputs((state,), (action,))
            _, hidden = self._forward_batch(x, self._train_memory.hidden.reshape(1, -1))
            self._train_memory = TorchGRUMemory(hidden[0].detach().clone())

    def learn(self, state: StateSnapshot, action: Action, actual_next_state: StateSnapshot) -> None:
        if len(actual_next_state.vector) != self.state_size:
            raise ValueError("next state vector size changed")
        for parameter in self._parameters:
            parameter.grad = None
        x = self._inputs((state,), (action,))
        current = self._train_memory.hidden.reshape(1, -1).detach()
        output, next_hidden = self._forward_batch(x, current)
        target = self.torch.tensor(
            actual_next_state.vector, device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        error = output - target
        reported_loss = float(error.square().mean().detach().cpu())
        # Canonical implementation uses dL/doutput = output-target and clips each
        # parameter gradient element independently before the SGD update.
        (0.5 * error.square().sum()).backward()
        with self.torch.no_grad():
            for parameter in self._parameters:
                if parameter.grad is not None:
                    parameter.add_(
                        -self.learning_rate * parameter.grad.clamp(min=-1.0, max=1.0)
                    )
        self._train_memory = TorchGRUMemory(next_hidden[0].detach().clone())
        self._add_template(actual_next_state)
        self._action_counts[action.signature] = self._action_counts.get(action.signature, 0) + 1
        self._updates += 1
        self._last_loss = reported_loss
        self._mean_loss += (reported_loss - self._mean_loss) / self._updates

    def runtime_diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "backend": "torch",
            "requested_device": str(self.requested_device),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "action_feature_cache": len(self._action_feature_cache),
            "unique_templates": len(self._template_buckets),
        }
        if self.device.type == "cuda":
            result.update(
                {
                    "cuda_device_name": self.torch.cuda.get_device_name(self.device),
                    "cuda_memory_allocated": int(self.torch.cuda.memory_allocated(self.device)),
                    "cuda_memory_reserved": int(self.torch.cuda.memory_reserved(self.device)),
                }
            )
        return result
