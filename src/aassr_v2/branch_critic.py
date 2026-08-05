from __future__ import annotations

import hashlib
import io
import random
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Sequence

from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class CriticTransition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot
    prophecy_confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class BranchCriticStep:
    value: float
    memory: Any = None


@dataclass(frozen=True, slots=True)
class BranchCriticStats:
    episodes: int
    transitions: int
    gradient_updates: int
    mean_loss: float
    parameter_count: int
    model_bytes: int


class _TransitionEncoder:
    def __init__(
        self,
        state_encoder: Callable[[StateSnapshot], Sequence[float]],
        state_size: int,
        *,
        action_feature_size: int = 16,
    ) -> None:
        if state_size <= 0 or action_feature_size <= 0:
            raise ValueError("encoder sizes must be positive")
        self.state_encoder = state_encoder
        self.state_size = int(state_size)
        self.action_feature_size = int(action_feature_size)

    @property
    def feature_size(self) -> int:
        return self.state_size * 2 + self.action_feature_size + 1

    def _action_features(self, action: Action) -> tuple[float, ...]:
        values = [0.0] * self.action_feature_size
        tokens = [action.verb_name, action.signature]
        tokens.extend(
            f"{key}={value}" for key, value in sorted(action.parameters.items())
        )
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % len(values)
            values[index] += -1.0 if digest[4] & 1 else 1.0
        norm = sum(value * value for value in values) ** 0.5
        if norm:
            values = [value / norm for value in values]
        return tuple(values)

    def encode(self, transition: CriticTransition) -> tuple[float, ...]:
        before = tuple(float(value) for value in self.state_encoder(transition.before))
        after = tuple(float(value) for value in self.state_encoder(transition.after))
        if len(before) != self.state_size or len(after) != self.state_size:
            raise ValueError("state encoder returned an unexpected size")
        confidence = max(0.0, min(1.0, float(transition.prophecy_confidence)))
        return before + self._action_features(transition.action) + after + (confidence,)


class ParentTransitionCritic:
    """Judge one parent-action-child transition without branch history.

    The only target supplied by the experiment is the real episode's final
    success bit. No task distance, object importance, or shaped progress label
    is used.
    """

    name = "parent-transition-critic"

    def __init__(
        self,
        state_encoder: Callable[[StateSnapshot], Sequence[float]],
        state_size: int,
        *,
        hidden_units: int = 64,
        action_feature_size: int = 16,
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        gradient_steps_per_episode: int = 4,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ParentTransitionCritic requires torch") from exc
        if min(hidden_units, replay_capacity, batch_size, gradient_steps_per_episode) <= 0:
            raise ValueError("critic sizes must be positive")
        self.torch = torch
        self.nn = nn
        self.encoder = _TransitionEncoder(
            state_encoder,
            state_size,
            action_feature_size=action_feature_size,
        )
        self.batch_size = int(batch_size)
        self.gradient_steps_per_episode = int(gradient_steps_per_episode)
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.model = nn.Sequential(
            nn.Linear(self.encoder.feature_size, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay: deque[tuple[tuple[float, ...], float]] = deque(
            maxlen=int(replay_capacity)
        )
        self.episodes = 0
        self.transitions = 0
        self.gradient_updates = 0
        self._losses: deque[float] = deque(maxlen=512)

    def initial_memory(self) -> None:
        return None

    def observe_episode(
        self,
        trajectory: Sequence[CriticTransition],
        *,
        success: bool,
    ) -> None:
        target = float(bool(success))
        for item in trajectory:
            self.replay.append((self.encoder.encode(item), target))
        self.episodes += 1
        self.transitions += len(trajectory)
        if len(self.replay) < self.batch_size:
            return
        for _ in range(self.gradient_steps_per_episode):
            self._train_step()

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        inputs = self.torch.tensor(
            [item[0] for item in batch], dtype=self.torch.float32
        )
        targets = self.torch.tensor(
            [item[1] for item in batch], dtype=self.torch.float32
        )
        logits = self.model(inputs).squeeze(1)
        loss = self.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        self.gradient_updates += 1
        self._losses.append(float(loss.detach().item()))

    def score_step(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        memory: Any = None,
        prophecy_confidence: float = 1.0,
    ) -> BranchCriticStep:
        del memory
        encoded = self.encoder.encode(
            CriticTransition(before, action, after, prophecy_confidence)
        )
        with self.torch.no_grad():
            tensor = self.torch.tensor(encoded, dtype=self.torch.float32).unsqueeze(0)
            value = float(self.torch.sigmoid(self.model(tensor)[0, 0]).item())
        return BranchCriticStep(value, None)

    def stats(self) -> BranchCriticStats:
        handle = io.BytesIO()
        self.torch.save(self.model.state_dict(), handle)
        return BranchCriticStats(
            episodes=self.episodes,
            transitions=self.transitions,
            gradient_updates=self.gradient_updates,
            mean_loss=fmean(self._losses) if self._losses else 0.0,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            model_bytes=len(handle.getvalue()),
        )


class GRUBranchCritic:
    """Judge a branch from its ordered transition history.

    Model parameters are shared by every imagined branch. Only the compact GRU
    hidden state is copied when a branch splits, so branches remain independent
    without cloning the neural network.
    """

    name = "gru-branch-critic"

    def __init__(
        self,
        state_encoder: Callable[[StateSnapshot], Sequence[float]],
        state_size: int,
        *,
        hidden_units: int = 64,
        action_feature_size: int = 16,
        replay_capacity: int = 4_000,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        gradient_steps_per_episode: int = 2,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("GRUBranchCritic requires torch") from exc
        if min(hidden_units, replay_capacity, batch_size, gradient_steps_per_episode) <= 0:
            raise ValueError("critic sizes must be positive")
        self.torch = torch
        self.nn = nn
        self.encoder = _TransitionEncoder(
            state_encoder,
            state_size,
            action_feature_size=action_feature_size,
        )
        self.hidden_units = int(hidden_units)
        self.batch_size = int(batch_size)
        self.gradient_steps_per_episode = int(gradient_steps_per_episode)
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.gru = nn.GRUCell(self.encoder.feature_size, self.hidden_units)
        self.output = nn.Linear(self.hidden_units, 1)
        self.optimizer = torch.optim.Adam(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()),
            lr=learning_rate,
        )
        self.replay: deque[tuple[tuple[tuple[float, ...], ...], float]] = deque(
            maxlen=int(replay_capacity)
        )
        self.episodes = 0
        self.transitions = 0
        self.gradient_updates = 0
        self._losses: deque[float] = deque(maxlen=512)

    def initial_memory(self) -> Any:
        return self.torch.zeros((1, self.hidden_units), dtype=self.torch.float32)

    def observe_episode(
        self,
        trajectory: Sequence[CriticTransition],
        *,
        success: bool,
    ) -> None:
        encoded = tuple(self.encoder.encode(item) for item in trajectory)
        if encoded:
            self.replay.append((encoded, float(bool(success))))
        self.episodes += 1
        self.transitions += len(encoded)
        if len(self.replay) < self.batch_size:
            return
        for _ in range(self.gradient_steps_per_episode):
            self._train_step()

    def _episode_loss(
        self,
        encoded: Sequence[tuple[float, ...]],
        target: float,
    ) -> Any:
        hidden = self.initial_memory()
        logits = []
        for item in encoded:
            tensor = self.torch.tensor(item, dtype=self.torch.float32).unsqueeze(0)
            hidden = self.gru(tensor, hidden)
            logits.append(self.output(hidden)[0, 0])
        stacked = self.torch.stack(logits)
        targets = self.torch.full_like(stacked, target)
        return self.nn.functional.binary_cross_entropy_with_logits(stacked, targets)

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        losses = [self._episode_loss(encoded, target) for encoded, target in batch]
        loss = self.torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.nn.utils.clip_grad_norm_(
            tuple(self.gru.parameters()) + tuple(self.output.parameters()), 5.0
        )
        self.optimizer.step()
        self.gradient_updates += 1
        self._losses.append(float(loss.detach().item()))

    def score_step(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        memory: Any = None,
        prophecy_confidence: float = 1.0,
    ) -> BranchCriticStep:
        encoded = self.encoder.encode(
            CriticTransition(before, action, after, prophecy_confidence)
        )
        hidden = self.initial_memory() if memory is None else memory
        with self.torch.no_grad():
            tensor = self.torch.tensor(encoded, dtype=self.torch.float32).unsqueeze(0)
            next_hidden = self.gru(tensor, hidden)
            value = float(self.torch.sigmoid(self.output(next_hidden)[0, 0]).item())
        return BranchCriticStep(value, next_hidden.detach().clone())

    def stats(self) -> BranchCriticStats:
        handle = io.BytesIO()
        self.torch.save(
            {"gru": self.gru.state_dict(), "output": self.output.state_dict()},
            handle,
        )
        parameters = tuple(self.gru.parameters()) + tuple(self.output.parameters())
        return BranchCriticStats(
            episodes=self.episodes,
            transitions=self.transitions,
            gradient_updates=self.gradient_updates,
            mean_loss=fmean(self._losses) if self._losses else 0.0,
            parameter_count=sum(parameter.numel() for parameter in parameters),
            model_bytes=len(handle.getvalue()),
        )
