from __future__ import annotations

import hashlib
import io
import math
import random
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Protocol, Sequence

from .types import Action, Prediction, StateSnapshot


class PrefixStateCodec(Protocol):
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
class TransitionPrefixConfig:
    action_feature_size: int = 16
    model_dim: int = 64
    attention_heads: int = 4
    layers: int = 2
    feedforward_dim: int = 128
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 128
    learning_rate: float = 1e-3
    gradient_steps_per_observation: int = 1
    confidence_prior: float = 256.0

    def __post_init__(self) -> None:
        values = (
            self.action_feature_size,
            self.model_dim,
            self.attention_heads,
            self.layers,
            self.feedforward_dim,
            self.replay_capacity,
            self.batch_size,
            self.warmup_steps,
            self.gradient_steps_per_observation,
        )
        if any(value <= 0 for value in values):
            raise ValueError("transition-prefix sizes must be positive")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class TransitionPrefixStats:
    observations: int
    gradient_updates: int
    replay_size: int
    mean_training_loss: float
    parameter_count: int
    model_bytes: int


class _PrefixNetwork:
    def __init__(self, torch: Any, nn: Any, state_size: int, config: TransitionPrefixConfig):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.state_projection = nn.Linear(state_size, config.model_dim)
                self.action_projection = nn.Linear(
                    config.action_feature_size, config.model_dim
                )
                self.position = nn.Parameter(torch.zeros(1, 2, config.model_dim))
                layer = nn.TransformerEncoderLayer(
                    d_model=config.model_dim,
                    nhead=config.attention_heads,
                    dim_feedforward=config.feedforward_dim,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                    dropout=0.0,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
                self.output = nn.Linear(config.model_dim, state_size + 3)

            def forward(self, states: Any, actions: Any) -> Any:
                tokens = torch.stack(
                    (self.state_projection(states), self.action_projection(actions)),
                    dim=1,
                )
                tokens = tokens + self.position
                mask = torch.tensor(
                    [[0.0, float("-inf")], [0.0, 0.0]],
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                encoded = self.encoder(tokens, mask=mask)
                return self.output(encoded[:, 1, :])

        self.model = Model()


class TransitionPrefixProphecy:
    """Predict the next state from the transition prefix [state, action].

    The training unit is conceptually [current state, action, next state], but
    only the prefix [current state, action] is visible to the model. The model
    receives no reward, goal distance, object importance, or transition rule.
    """

    name = "transition-prefix"

    def __init__(
        self,
        codec: PrefixStateCodec,
        *,
        config: TransitionPrefixConfig | None = None,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("TransitionPrefixProphecy requires torch") from exc
        self.torch = torch
        self.nn = nn
        self.codec = codec
        self.config = config or TransitionPrefixConfig()
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        network = _PrefixNetwork(torch, nn, codec.dimension, self.config)
        self.model = network.model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.replay: deque[
            tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]
        ] = deque(maxlen=self.config.replay_capacity)
        self.observations = 0
        self.gradient_updates = 0
        self._losses: deque[float] = deque(maxlen=512)

    def _action_features(self, action: Action) -> tuple[float, ...]:
        values = [0.0] * self.config.action_feature_size
        digest = hashlib.sha256(action.signature.encode("utf-8")).digest()
        for offset in range(0, min(len(digest), 16), 4):
            index = int.from_bytes(digest[offset : offset + 4], "big") % len(values)
            values[index] += -1.0 if digest[(offset + 1) % len(digest)] & 1 else 1.0
        norm = sum(value * value for value in values) ** 0.5
        return tuple(value / norm for value in values) if norm else tuple(values)

    @staticmethod
    def _terminal_class(state: StateSnapshot) -> int:
        if state.available_actions:
            return 0
        return 1 if state.goal_progress >= 1.0 or "success" in state.facts else 2

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.replay.append(
            (
                self.codec.encode(state),
                self._action_features(action),
                self.codec.encode(actual_next_state),
                self._terminal_class(actual_next_state),
            )
        )
        self.observations += 1
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return
        for _ in range(self.config.gradient_steps_per_observation):
            self._train_step()

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.config.batch_size)
        states = self.torch.tensor([item[0] for item in batch], dtype=self.torch.float32)
        actions = self.torch.tensor([item[1] for item in batch], dtype=self.torch.float32)
        targets = self.torch.tensor([item[2] for item in batch], dtype=self.torch.float32)
        terminal = self.torch.tensor(
            [item[3] for item in batch], dtype=self.torch.int64
        )
        output = self.model(states, actions)
        predicted_state = output[:, : self.codec.dimension]
        terminal_logits = output[:, self.codec.dimension :]
        state_loss = self.nn.functional.smooth_l1_loss(predicted_state, targets)
        terminal_loss = self.nn.functional.cross_entropy(terminal_logits, terminal)
        loss = state_loss + 0.25 * terminal_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1
        self._losses.append(float(loss.detach().item()))

    def _raw(self, state: StateSnapshot, action: Action) -> tuple[list[float], list[float]]:
        states = self.torch.tensor(
            self.codec.encode(state), dtype=self.torch.float32
        ).unsqueeze(0)
        actions = self.torch.tensor(
            self._action_features(action), dtype=self.torch.float32
        ).unsqueeze(0)
        with self.torch.no_grad():
            output = self.model(states, actions)[0]
            encoded = [float(value) for value in output[: self.codec.dimension].tolist()]
            terminal = [
                float(value)
                for value in self.torch.softmax(
                    output[self.codec.dimension :], dim=0
                ).tolist()
            ]
        return encoded, terminal

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
            return (Prediction(state, 0.0, source="transition-prefix:unseen"),)
        encoded, terminal = self._raw(state, action)
        terminal_class = max(range(3), key=lambda index: terminal[index])
        decoded = self.codec.decode(
            encoded,
            scaffold=state,
            terminal_class=terminal_class,
            source="transition-prefix",
        )
        confidence = self.observations / (
            self.observations + self.config.confidence_prior
        )
        entropy = -sum(value * math.log(max(value, 1e-8)) for value in terminal)
        confidence *= math.exp(-0.25 * entropy)
        return (
            Prediction(
                decoded,
                max(0.05, min(0.995, confidence)),
                source="transition-prefix",
            ),
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        return self.predict(state, action, samples=1)[0].probability

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return fmean(self.confidence(state, action) for action in materialized)

    def stats(self) -> TransitionPrefixStats:
        handle = io.BytesIO()
        self.torch.save(self.model.state_dict(), handle)
        return TransitionPrefixStats(
            observations=self.observations,
            gradient_updates=self.gradient_updates,
            replay_size=len(self.replay),
            mean_training_loss=fmean(self._losses) if self._losses else 0.0,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            model_bytes=len(handle.getvalue()),
        )
