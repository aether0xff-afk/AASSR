from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .autonomous_agent import (
    ActionDecision,
    ObservationMetrics,
    StateKey,
    state_key,
)
from .types import Action, StateSnapshot


def _empty_metrics(*, error: bool = False, repeated: bool = False) -> ObservationMetrics:
    return ObservationMetrics(
        prediction_score=0.0,
        holdout_before=0.0,
        holdout_after=0.0,
        holdout_gain=0.0,
        intrinsic_value=0.0,
        repeated=repeated,
        error=error,
    )


class TabularQLearningAgent:
    """One-step tabular Q-learning baseline with the standard agent interface."""

    def __init__(
        self,
        *,
        seed: int,
        learning_rate: float = 0.2,
        gamma: float = 0.97,
        epsilon_start: float = 0.8,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 1000,
    ) -> None:
        self.randomizer = random.Random(seed)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.q: dict[tuple[StateKey, str], float] = defaultdict(float)
        self._recent: list[tuple[StateKey, str]] = []

    def epsilon(self, episode: int) -> float:
        fraction = min(1.0, max(0.0, episode / self.epsilon_decay_episodes))
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def _value(self, state: StateSnapshot, action: Action) -> float:
        return self.q.get((state_key(state), action.signature), 0.0)

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool = True,
    ) -> ActionDecision:
        actions = state.available_actions
        if not actions:
            raise ValueError("state has no available actions")
        if explore and self.randomizer.random() < self.epsilon(episode):
            return ActionDecision(self.randomizer.choice(actions), False)
        action = min(
            actions,
            key=lambda item: (-self._value(state, item), item.signature),
        )
        return ActionDecision(action, False)

    def observe(
        self, before: StateSnapshot, action: Action, outcome: object
    ) -> ObservationMetrics:
        after: StateSnapshot = getattr(outcome, "snapshot")
        reward = float(getattr(outcome, "reward", 0.0))
        next_value = max(
            (self._value(after, candidate) for candidate in after.available_actions),
            default=0.0,
        )
        key = (state_key(before), action.signature)
        self.q[key] += self.learning_rate * (
            reward + self.gamma * next_value - self.q[key]
        )
        repeated = key in self._recent[-16:]
        self._recent.append(key)
        return _empty_metrics(
            error=bool(getattr(outcome, "error", False)),
            repeated=repeated,
        )

    def finish_episode(self, *, final_return: float) -> None:
        del final_return
        self._recent.clear()

    def discard_episode(self) -> None:
        self._recent.clear()

    def learning_fingerprint(self) -> str:
        payload = sorted(
            (repr(key), round(value, 12)) for key, value in self.q.items()
        )
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class DQNAgent:
    """Small optional PyTorch DQN baseline over state/action feature pairs."""

    def __init__(
        self,
        state_size: int,
        *,
        seed: int,
        action_feature_size: int = 16,
        hidden_size: int = 32,
        learning_rate: float = 1e-3,
        gamma: float = 0.97,
        epsilon_start: float = 0.8,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 1000,
        device: str = "auto",
        allow_cpu_fallback: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "DQN baseline requires PyTorch; install the paper extra"
            ) from error
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError("CUDA requested for DQN but unavailable")
            device = "cpu"
        self.device = torch.device(device)
        self.state_size = state_size
        self.action_feature_size = action_feature_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_size + action_feature_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate
        )
        self.loss = torch.nn.SmoothL1Loss()
        self._recent: list[tuple[StateKey, str]] = []

    def epsilon(self, episode: int) -> float:
        fraction = min(1.0, max(0.0, episode / self.epsilon_decay_episodes))
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def _action_vector(self, action: Action) -> list[float]:
        result = [0.0] * self.action_feature_size
        digest = hashlib.sha256(action.signature.encode("utf-8")).digest()
        for offset in range(0, len(digest), 2):
            index = digest[offset] % self.action_feature_size
            result[index] += 1.0 if digest[offset + 1] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in result))
        return [value / norm for value in result] if norm else result

    def _tensor(self, state: StateSnapshot, action: Action):
        values = [*state.vector, *self._action_vector(action)]
        return self.torch.tensor(
            values, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)

    def _value(self, state: StateSnapshot, action: Action):
        return self.network(self._tensor(state, action)).squeeze()

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool = True,
    ) -> ActionDecision:
        actions = state.available_actions
        if not actions:
            raise ValueError("state has no available actions")
        if explore and self.randomizer.random() < self.epsilon(episode):
            return ActionDecision(self.randomizer.choice(actions), False)
        with self.torch.no_grad():
            values = [
                float(self._value(state, action).item()) for action in actions
            ]
        index = min(
            range(len(actions)),
            key=lambda item: (-values[item], actions[item].signature),
        )
        return ActionDecision(actions[index], False)

    def observe(
        self, before: StateSnapshot, action: Action, outcome: object
    ) -> ObservationMetrics:
        after: StateSnapshot = getattr(outcome, "snapshot")
        reward = float(getattr(outcome, "reward", 0.0))
        predicted = self._value(before, action)
        with self.torch.no_grad():
            next_value = max(
                (
                    float(self._value(after, candidate).item())
                    for candidate in after.available_actions
                ),
                default=0.0,
            )
            target = self.torch.tensor(
                reward + self.gamma * next_value,
                dtype=self.torch.float32,
                device=self.device,
            )
        loss = self.loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        key = (state_key(before), action.signature)
        repeated = key in self._recent[-16:]
        self._recent.append(key)
        return _empty_metrics(
            error=bool(getattr(outcome, "error", False)),
            repeated=repeated,
        )

    def finish_episode(self, *, final_return: float) -> None:
        del final_return
        self._recent.clear()

    def discard_episode(self) -> None:
        self._recent.clear()

    def learning_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.network.state_dict().items()):
            digest.update(name.encode("utf-8"))
            values = tensor.detach().cpu().contiguous().view(
                self.torch.uint8
            ).flatten().tolist()
            digest.update(bytes(values))
        return digest.hexdigest()


class OracleAgent:
    """Privileged upper bound. Never include this condition as a fair baseline."""

    def select_action_for_environment(
        self,
        environment: Any,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ) -> ActionDecision:
        del state, episode, explore
        return ActionDecision(environment.oracle_action(), False)

    def observe(
        self, before: StateSnapshot, action: Action, outcome: object
    ) -> ObservationMetrics:
        del before, action
        return _empty_metrics(error=bool(getattr(outcome, "error", False)))

    def finish_episode(self, *, final_return: float) -> None:
        del final_return

    def discard_episode(self) -> None:
        pass

    def learning_fingerprint(self) -> str:
        return "oracle-static"
