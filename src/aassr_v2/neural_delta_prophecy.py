from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Iterable, Protocol, Sequence

from .types import Action, Prediction, StateSnapshot


class StateCodec(Protocol):
    """Translate explicit snapshots to and from a learnable numeric state."""

    @property
    def dimension(self) -> int: ...

    def encode(self, state: StateSnapshot) -> tuple[float, ...]: ...

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot: ...


@dataclass(frozen=True, slots=True)
class NeuralDeltaConfig:
    action_feature_size: int = 16
    hidden_units: int = 128
    ensemble_size: int = 3
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 128
    learning_rate: float = 1e-3
    gradient_steps_per_observation: int = 1
    gradient_clip: float = 10.0
    variance_scale: float = 8.0
    confidence_prior: float = 256.0

    def __post_init__(self) -> None:
        positive = (
            self.action_feature_size,
            self.hidden_units,
            self.ensemble_size,
            self.replay_capacity,
            self.batch_size,
            self.warmup_steps,
            self.gradient_steps_per_observation,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("neural delta sizes must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class NeuralDeltaStats:
    observations: int
    gradient_updates: int
    replay_size: int
    mean_training_loss: float
    last_ensemble_variance: float
    parameter_count: int


class NeuralDeltaProphecy:
    """Online action-conditioned ensemble predicting explicit state deltas.

    The model does not receive task rules, object names, or reward shaping. It
    learns ``encoded state + opaque action -> encoded next-state delta`` from
    real transitions stored in replay. An ensemble supplies an uncertainty
    estimate used by Imagination confidence and gating.

    A small domain codec is required because AASSR states contain both numeric
    vectors and symbolic facts. The codec describes representation only; it must
    not implement transition rules.
    """

    name = "neural-delta"

    def __init__(
        self,
        codec: StateCodec,
        *,
        config: NeuralDeltaConfig | None = None,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("NeuralDeltaProphecy requires torch") from exc

        self.torch = torch
        self.nn = nn
        self.codec = codec
        self.config = config or NeuralDeltaConfig()
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover
            torch.use_deterministic_algorithms(True)

        input_size = codec.dimension + self.config.action_feature_size
        output_size = codec.dimension + 3
        self.models = [
            nn.Sequential(
                nn.Linear(input_size, self.config.hidden_units),
                nn.SiLU(),
                nn.Linear(self.config.hidden_units, self.config.hidden_units),
                nn.SiLU(),
                nn.Linear(self.config.hidden_units, output_size),
            )
            for _ in range(self.config.ensemble_size)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            for model in self.models
        ]
        self.replay: deque[
            tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]
        ] = deque(maxlen=self.config.replay_capacity)
        self.observations = 0
        self.gradient_updates = 0
        self._losses: deque[float] = deque(maxlen=256)
        self._last_ensemble_variance = 1.0

    def _action_features(self, action: Action) -> tuple[float, ...]:
        """Stable signed hashing keeps action handling schema-free."""

        values = [0.0] * self.config.action_feature_size
        signature = action.signature.encode("utf-8")
        # FNV-1a style stable hash; Python's randomized hash is not reproducible.
        value = 2166136261
        for byte in signature:
            value ^= byte
            value = (value * 16777619) & 0xFFFFFFFF
        primary = value % len(values)
        secondary = ((value >> 16) ^ value) % len(values)
        values[primary] += 1.0
        values[secondary] += -1.0 if value & 1 else 1.0
        return tuple(values)

    def _input(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        return self.codec.encode(state) + self._action_features(action)

    @staticmethod
    def _terminal_class(state: StateSnapshot) -> int:
        if state.available_actions:
            return 0
        return 1 if state.goal_progress >= 1.0 or "success" in state.facts else 2

    def _tensor(self, values: Any, *, dtype: Any | None = None) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=dtype or self.torch.float32,
        )

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        before = self.codec.encode(state)
        after = self.codec.encode(actual_next_state)
        delta = tuple(right - left for left, right in zip(before, after, strict=True))
        self.replay.append(
            (
                self._input(state, action),
                before,
                delta,
                self._terminal_class(actual_next_state),
            )
        )
        self.observations += 1
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return
        for _ in range(self.config.gradient_steps_per_observation):
            self._train_step()

    def _train_step(self) -> None:
        torch = self.torch
        nn = self.nn
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers, strict=True)
        ):
            # Independent bootstrap batches create useful ensemble disagreement.
            local = random.Random(
                (self.observations + 1) * 1_000_003
                + (self.gradient_updates + 1) * 97
                + model_index
            )
            batch = [
                self.replay[local.randrange(len(self.replay))]
                for _ in range(self.config.batch_size)
            ]
            inputs = self._tensor([item[0] for item in batch])
            deltas = self._tensor([item[2] for item in batch])
            terminal = self._tensor(
                [item[3] for item in batch],
                dtype=torch.int64,
            )
            output = model(inputs)
            predicted_delta = output[:, : self.codec.dimension]
            terminal_logits = output[:, self.codec.dimension :]
            delta_loss = nn.functional.smooth_l1_loss(
                predicted_delta,
                deltas,
            )
            terminal_loss = nn.functional.cross_entropy(
                terminal_logits,
                terminal,
            )
            loss = delta_loss + 0.25 * terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            optimizer.step()
            self._losses.append(float(loss.detach().item()))
        self.gradient_updates += 1

    def _raw_predictions(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[list[list[float]], list[list[float]]]:
        encoded = self.codec.encode(state)
        input_tensor = self._tensor(self._input(state, action)).unsqueeze(0)
        deltas: list[list[float]] = []
        terminal_probabilities: list[list[float]] = []
        with self.torch.no_grad():
            for model in self.models:
                output = model(input_tensor)[0]
                deltas.append(
                    [
                        float(value)
                        for value in output[: self.codec.dimension].tolist()
                    ]
                )
                terminal_probabilities.append(
                    [
                        float(value)
                        for value in self.torch.softmax(
                            output[self.codec.dimension :],
                            dim=0,
                        ).tolist()
                    ]
                )
        next_states = [
            [left + delta for left, delta in zip(encoded, item, strict=True)]
            for item in deltas
        ]
        return next_states, terminal_probabilities

    def _confidence_from_ensemble(
        self,
        next_states: Sequence[Sequence[float]],
        terminal_probabilities: Sequence[Sequence[float]],
    ) -> float:
        if len(next_states) <= 1:
            variance = 0.0
        else:
            means = [
                fmean(item[index] for item in next_states)
                for index in range(self.codec.dimension)
            ]
            variance = fmean(
                (item[index] - means[index]) ** 2
                for item in next_states
                for index in range(self.codec.dimension)
            )
        terminal_means = [
            fmean(item[index] for item in terminal_probabilities)
            for index in range(3)
        ]
        terminal_variance = fmean(
            (item[index] - terminal_means[index]) ** 2
            for item in terminal_probabilities
            for index in range(3)
        )
        variance += terminal_variance
        self._last_ensemble_variance = variance
        sample_confidence = self.observations / (
            self.observations + self.config.confidence_prior
        )
        uncertainty_confidence = math.exp(
            -self.config.variance_scale * variance
        )
        return max(
            0.05,
            min(0.995, sample_confidence * uncertainty_confidence),
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        if self.observations < self.config.warmup_steps:
            return (
                Prediction(
                    next_state=state,
                    probability=0.0,
                    source="neural-delta:unseen",
                ),
            )

        next_states, terminal_probabilities = self._raw_predictions(state, action)
        confidence = self._confidence_from_ensemble(
            next_states,
            terminal_probabilities,
        )
        mean_state = [
            fmean(item[index] for item in next_states)
            for index in range(self.codec.dimension)
        ]
        mean_terminal = [
            fmean(item[index] for item in terminal_probabilities)
            for index in range(3)
        ]
        terminal_class = max(range(3), key=lambda index: mean_terminal[index])
        decoded = self.codec.decode(
            mean_state,
            scaffold=state,
            terminal_class=terminal_class,
            source="neural-delta:ensemble",
        )
        return (
            Prediction(
                next_state=decoded,
                probability=confidence,
                source="neural-delta:ensemble",
            ),
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        if self.observations < self.config.warmup_steps:
            return 0.0
        next_states, terminal_probabilities = self._raw_predictions(state, action)
        return self._confidence_from_ensemble(
            next_states,
            terminal_probabilities,
        )

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return fmean(self.confidence(state, action) for action in materialized)

    def stats(self) -> NeuralDeltaStats:
        return NeuralDeltaStats(
            observations=self.observations,
            gradient_updates=self.gradient_updates,
            replay_size=len(self.replay),
            mean_training_loss=fmean(self._losses) if self._losses else 0.0,
            last_ensemble_variance=self._last_ensemble_variance,
            parameter_count=sum(
                parameter.numel()
                for model in self.models
                for parameter in model.parameters()
            ),
        )
