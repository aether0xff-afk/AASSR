from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

from .metrics import cosine_similarity
from .prophecy import ProphecyStep
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class TorchGRUMemory:
    hidden: Any


@dataclass(frozen=True, slots=True)
class TorchGRUTrainingStats:
    updates: int
    last_loss: float
    mean_loss: float
    requested_device: str
    resolved_device: str


class TorchGRUProphecy:
    """Optional PyTorch GRU world model with CUDA acceleration.

    PyTorch is imported lazily so the core package remains dependency-free.
    The model uses a GRUCell for online one-step updates and keeps symbolic
    state templates on the Python side for compatibility with StateSnapshot.
    """

    def __init__(
        self,
        state_size: int,
        *,
        action_feature_size: int = 32,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 7,
        replay_limit: int = 2048,
        device: str = "auto",
        allow_cpu_fallback: bool = True,
    ) -> None:
        if min(state_size, action_feature_size, hidden_size, replay_limit) <= 0:
            raise ValueError("sizes and replay_limit must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "TorchGRUProphecy requires PyTorch. Install a CUDA-enabled "
                "PyTorch build and then install the project with .[gpu]."
            ) from error

        self.torch = torch
        self.state_size = state_size
        self.action_feature_size = action_feature_size
        self.hidden_size = hidden_size
        self.input_size = state_size + action_feature_size
        self.learning_rate = learning_rate
        self.replay_limit = replay_limit
        self.requested_device = device
        self.device = self._resolve_device(device, allow_cpu_fallback)
        self._name = "torch-gru"
        self._updates = 0
        self._last_loss = 0.0
        self._mean_loss = 0.0
        self._templates: list[StateSnapshot] = []
        self._action_counts: dict[str, int] = {}

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.gru = torch.nn.GRUCell(self.input_size, hidden_size).to(self.device)
        self.output = torch.nn.Linear(hidden_size, state_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            lr=learning_rate,
        )
        self.loss_fn = torch.nn.MSELoss()
        self._train_hidden = self._zero_hidden()

    def _resolve_device(self, requested: str, allow_cpu_fallback: bool):
        torch = self.torch
        normalized = requested.lower().strip()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(normalized)
        if device.type == "cuda" and not torch.cuda.is_available():
            if allow_cpu_fallback:
                return torch.device("cpu")
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return device

    @property
    def name(self) -> str:
        return self._name

    @property
    def training_stats(self) -> TorchGRUTrainingStats:
        return TorchGRUTrainingStats(
            self._updates,
            self._last_loss,
            self._mean_loss,
            self.requested_device,
            str(self.device),
        )

    def _zero_hidden(self):
        return self.torch.zeros((1, self.hidden_size), device=self.device)

    def initial_memory(self) -> TorchGRUMemory:
        return TorchGRUMemory(self._zero_hidden())

    def reset_sequence(self) -> None:
        self._train_hidden = self._zero_hidden()

    def _action_features(self, action: Action):
        vector = [0.0] * self.action_feature_size
        tokens = [action.verb_name, action.signature]
        tokens.extend(
            f"{key}={value}" for key, value in sorted(action.parameters.items())
        )
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.action_feature_size
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def _input_tensor(self, state: StateSnapshot, action: Action):
        if len(state.vector) != self.state_size:
            raise ValueError(
                f"expected state vector of length {self.state_size}, "
                f"got {len(state.vector)}"
            )
        values = [*state.vector, *self._action_features(action)]
        return self.torch.tensor(
            values, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)

    def _forward(self, state: StateSnapshot, action: Action, hidden):
        next_hidden = self.gru(self._input_tensor(state, action), hidden)
        output = self.output(next_hidden)
        return output, next_hidden

    def _nearest_templates(
        self, vector: tuple[float, ...], limit: int
    ) -> list[tuple[float, StateSnapshot]]:
        unique: dict[tuple[float, ...], StateSnapshot] = {}
        for state in self._templates:
            unique[state.vector] = state
        scored = [
            (cosine_similarity(vector, template.vector), template)
            for template in unique.values()
        ]
        scored.sort(
            key=lambda item: (-item[0], tuple(sorted(item[1].facts)))
        )
        return scored[:limit]

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state
        seen = self._action_counts.get(action.signature, 0)
        experience = seen / (seen + 4.0)
        loss_quality = 1.0 / (1.0 + max(0.0, self._mean_loss))
        return max(0.0, min(1.0, experience * loss_quality))

    def coverage(
        self, state: StateSnapshot, actions: Iterable[Action]
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(self.confidence(state, action) for action in materialized) / len(
            materialized
        )

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: TorchGRUMemory | None,
        samples: int,
    ) -> ProphecyStep:
        if samples <= 0:
            raise ValueError("samples must be positive")
        hidden = self._zero_hidden() if memory is None else memory.hidden
        with self.torch.no_grad():
            output, next_hidden = self._forward(state, action, hidden)
        vector = tuple(float(value) for value in output.squeeze(0).cpu().tolist())
        nearest = self._nearest_templates(vector, samples)
        confidence = self.confidence(state, action)
        if not nearest or confidence <= 0.0:
            return ProphecyStep(
                (Prediction(state, 1.0, source=f"{self.name}:unseen"),),
                TorchGRUMemory(next_hidden.detach()),
            )

        logits = self.torch.tensor(
            [score * 4.0 for score, _ in nearest], dtype=self.torch.float32
        )
        probabilities = self.torch.softmax(logits, dim=0).tolist()
        suffix = "exact" if confidence >= 0.75 else "action-family"
        predictions = [
            Prediction(
                template,
                float(probability) * confidence,
                source=f"{self.name}:{suffix}",
            )
            for probability, (_, template) in zip(
                probabilities, nearest, strict=True
            )
        ]
        if confidence < 1.0:
            predictions.append(
                Prediction(
                    state,
                    1.0 - confidence,
                    source=f"{self.name}:uncertain",
                )
            )
        return ProphecyStep(
            tuple(predictions), TorchGRUMemory(next_hidden.detach())
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.predict_step(
            state,
            action,
            memory=self.initial_memory(),
            samples=samples,
        ).predictions

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if len(actual_next_state.vector) != self.state_size:
            raise ValueError("next state vector size changed")
        self.optimizer.zero_grad(set_to_none=True)
        output, next_hidden = self._forward(state, action, self._train_hidden)
        target = self.torch.tensor(
            actual_next_state.vector,
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()), 1.0
        )
        self.optimizer.step()

        loss_value = float(loss.detach().cpu())
        self._train_hidden = next_hidden.detach()
        self._templates.append(actual_next_state)
        if len(self._templates) > self.replay_limit:
            self._templates.pop(0)
        self._action_counts[action.signature] = (
            self._action_counts.get(action.signature, 0) + 1
        )
        self._updates += 1
        self._last_loss = loss_value
        self._mean_loss += (loss_value - self._mean_loss) / self._updates
