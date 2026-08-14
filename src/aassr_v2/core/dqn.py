from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence
import io
import random

from ..action_plugins import PluginOutcome
from ..autonomous_agent_core import RunningValue
from ..policy import PolicyMemory, ScoredAction
from ..skills import SKILL_VERB
from ..types import Action, StateSnapshot
from .representation import SchemaDrivenRepresentation


class CoreDynamicActionDQN:
    """Domain-independent DQN over Core-owned state/action representations."""

    name = "core-dynamic-action-dqn-v1"

    def __init__(
        self,
        seed: int,
        *,
        representation: SchemaDrivenRepresentation,
        train_transitions: int,
        hidden_units: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.98,
        replay_capacity: int = 50_000,
        batch_size: int = 64,
        warmup_steps: int = 128,
        target_update_interval: int = 250,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CoreDynamicActionDQN requires torch") from exc
        self.torch = torch
        self.nn = nn
        self.representation = representation
        self.randomizer = random.Random(int(seed))
        torch.manual_seed(int(seed))
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover
            torch.use_deterministic_algorithms(True)
        self.device = torch.device(device)
        self.train_transitions = int(train_transitions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        input_size = representation.state_size + representation.action_feature_size
        self.online = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, 1),
        ).to(self.device)
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, 1),
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.replay: deque[
            tuple[
                tuple[float, ...],
                tuple[float, ...],
                float,
                tuple[float, ...],
                tuple[tuple[float, ...], ...],
                bool,
            ]
        ] = deque(maxlen=int(replay_capacity))
        self.environment_steps = 0
        self.gradient_updates = 0
        self.raw_actions_scored = 0
        self.unique_features_scored = 0
        self.raw_next_actions_stored = 0
        self.unique_next_features_stored = 0
        self._episode_boundary_pending = False
        self.forced_episode_boundaries = 0

    def _tensor(self, values: Any) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=self.torch.float32,
            device=self.device,
        )

    def encode_state(self, state: StateSnapshot) -> tuple[float, ...]:
        return self.representation.state_vector(state)

    def action_features(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        return self.representation.action_features(state, action)

    def _deduplicate(
        self,
        state: StateSnapshot,
        actions: Sequence[Action],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
        unique: dict[tuple[float, ...], int] = {}
        indices: list[int] = []
        for action in actions:
            features = self.action_features(state, action)
            indices.append(unique.setdefault(features, len(unique)))
        return tuple(unique), tuple(indices)

    def score_actions(
        self,
        state: StateSnapshot,
        actions: Sequence[Action],
    ) -> tuple[float, ...]:
        if not actions:
            return ()
        unique_features, indices = self._deduplicate(state, actions)
        state_values = self.encode_state(state)
        inputs = [state_values + features for features in unique_features]
        with self.torch.no_grad():
            values = self.online(self._tensor(inputs)).squeeze(1).tolist()
        self.raw_actions_scored += len(actions)
        self.unique_features_scored += len(unique_features)
        return tuple(float(values[index]) for index in indices)

    def mark_episode_boundary(self) -> None:
        self._episode_boundary_pending = True

    def _consume_episode_boundary(self) -> bool:
        value = self._episode_boundary_pending
        self._episode_boundary_pending = False
        if value:
            self.forced_episode_boundaries += 1
        return value

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: PluginOutcome,
        *,
        reward: float,
        terminal: bool | None = None,
    ) -> None:
        boundary = self._consume_episode_boundary()
        if terminal is None:
            terminal = bool(outcome.snapshot.metadata.get("core_terminated", False))
            terminal = terminal or bool(
                outcome.snapshot.metadata.get("core_truncated", False)
            )
            terminal = terminal or not outcome.snapshot.available_actions
        terminal = boundary or bool(terminal)
        raw_next = tuple(outcome.snapshot.available_actions)
        next_features, _ = self._deduplicate(outcome.snapshot, raw_next)
        self.raw_next_actions_stored += len(raw_next)
        self.unique_next_features_stored += len(next_features)
        self.replay.append(
            (
                self.encode_state(before),
                self.action_features(before, action),
                float(reward),
                self.encode_state(outcome.snapshot),
                next_features,
                terminal,
            )
        )
        self.environment_steps += 1
        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            self._train_step()

    def _train_step(self) -> None:
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        inputs = self._tensor([item[0] + item[1] for item in batch])
        predicted = self.online(inputs).squeeze(1)
        next_values: list[float] = []
        with self.torch.no_grad():
            for _, _, _, next_state, next_actions, terminal in batch:
                if terminal or not next_actions:
                    next_values.append(0.0)
                    continue
                candidates = self._tensor(
                    [next_state + features for features in next_actions]
                )
                next_values.append(float(self.target(candidates).max().item()))
        rewards = self._tensor([item[2] for item in batch])
        maxima = self._tensor(next_values)
        terminals = self._tensor([float(item[5]) for item in batch])
        targets = rewards + self.gamma * (1.0 - terminals) * maxima
        loss = self.loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1
        if self.gradient_updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def model_stats(self) -> dict[str, int | float | str]:
        handle = io.BytesIO()
        self.torch.save(self.online.state_dict(), handle)
        return {
            "device": str(self.device),
            "model_units": sum(
                parameter.numel() for parameter in self.online.parameters()
            ),
            "model_bytes": len(handle.getvalue()),
            "gradient_updates": self.gradient_updates,
            "replay_size": len(self.replay),
            "environment_steps": self.environment_steps,
            "raw_actions_scored": self.raw_actions_scored,
            "unique_features_scored": self.unique_features_scored,
            "raw_next_actions_stored": self.raw_next_actions_stored,
            "unique_next_features_stored": self.unique_next_features_stored,
            "forced_episode_boundaries": self.forced_episode_boundaries,
        }


class CorePolicy:
    """External-reward DQN plus a separate internal information residual."""

    name = "core-policy-v1"

    def __init__(
        self,
        dqn: CoreDynamicActionDQN,
        *,
        information_learning_rate: float = 0.2,
    ) -> None:
        self.dqn = dqn
        self.representation = dqn.representation
        self.information_learning_rate = float(information_learning_rate)
        self._information: dict[
            tuple[tuple[float, ...], tuple[float, ...]], RunningValue
        ] = {}
        self._skill_values: dict[str, RunningValue] = {}

    def _key(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (
            self.representation.state_key(state),
            self.representation.action_structure(state, action),
        )

    def value(self, state: StateSnapshot, action: Action) -> float:
        if action.verb_name == SKILL_VERB:
            return self._skill_values.get(str(action.target), RunningValue()).mean
        external = self.dqn.score_actions(state, (action,))[0]
        return external + self._information.get(self._key(state, action), RunningValue()).mean

    def rank(
        self,
        state: StateSnapshot,
        *,
        limit: int,
        memory: PolicyMemory | None = None,
    ) -> tuple[ScoredAction, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        actions = tuple(state.available_actions)
        if not actions:
            return ()
        primitive = tuple(
            action for action in actions if action.verb_name != SKILL_VERB
        )
        external_values = (
            self.dqn.score_actions(state, primitive) if primitive else ()
        )
        external = {
            action.signature: value
            for action, value in zip(primitive, external_values, strict=True)
        }
        deltas: Mapping[str, float] = {} if memory is None else memory.deltas
        rows = []
        for action in actions:
            if action.verb_name == SKILL_VERB:
                base = self._skill_values.get(
                    str(action.target),
                    RunningValue(),
                ).mean
            else:
                information = self._information.get(
                    self._key(state, action),
                    RunningValue(),
                ).mean
                base = float(external[action.signature]) + information
            rows.append(
                ScoredAction(
                    action,
                    float(base) + float(deltas.get(action.signature, 0.0)),
                )
            )
        rows.sort(key=lambda item: (-item.score, item.action.signature))
        return tuple(rows[:limit])

    def select(
        self,
        state: StateSnapshot,
        *,
        randomizer: random.Random,
        epsilon: float,
        exploration_bonus: float,
    ) -> Action:
        del exploration_bonus
        if not state.available_actions:
            raise ValueError("cannot select from an empty action set")
        if epsilon > 0.0 and randomizer.random() < epsilon:
            return randomizer.choice(state.available_actions)
        ranked = self.rank(state, limit=len(state.available_actions))
        best = ranked[0].score
        tied = tuple(
            item.action
            for item in ranked
            if abs(float(item.score) - float(best)) <= 1e-12
        )
        return randomizer.choice(tied)

    def imagine_update(
        self,
        memory: PolicyMemory,
        action: Action,
        value: float,
    ) -> PolicyMemory:
        deltas = dict(memory.deltas)
        deltas[action.signature] = deltas.get(action.signature, 0.0) + 0.1 * float(value)
        return PolicyMemory(deltas)

    def reinforce(self, action: Action, advantage: float) -> None:
        del action, advantage

    def observe_return(
        self,
        state: StateSnapshot,
        action: Action,
        target: float,
    ) -> None:
        del state
        if action.verb_name != SKILL_VERB:
            return
        entry = self._skill_values.setdefault(
            str(action.target),
            RunningValue(),
        )
        entry.observe(
            float(target),
            learning_rate=self.information_learning_rate,
        )

    def observe_information_return(
        self,
        state: StateSnapshot,
        action: Action,
        value: float,
    ) -> None:
        entry = self._information.setdefault(self._key(state, action), RunningValue())
        entry.observe(float(value), learning_rate=self.information_learning_rate)

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            "information_entries": len(self._information),
            "skill_value_entries": len(self._skill_values),
            **{f"dqn:{key}": value for key, value in self.dqn.model_stats().items()},
        }
