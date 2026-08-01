from __future__ import annotations

import ast
import hashlib
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from .paper_v2_types import FullAgentCheckpoint, RawCausalObservation


@dataclass(frozen=True, slots=True)
class ObservableTransition:
    before: RawCausalObservation
    action: str
    after: RawCausalObservation
    action_succeeded: bool
    inventory_delta: Mapping[str, int]
    facts_added: int
    facts_removed: int
    unlocked_actions: int
    resource_cost: float
    damage: float
    spatial_changed: bool
    terminal_reward: float


@dataclass(slots=True)
class _EffectAccumulator:
    executions: int = 0
    successes: int = 0
    inventory_change: float = 0.0
    facts_added: float = 0.0
    facts_removed: float = 0.0
    unlocks: float = 0.0
    resource_cost: float = 0.0
    damage: float = 0.0
    spatial_changes: float = 0.0
    terminal_returns: float = 0.0

    def observe(self, transition: ObservableTransition) -> None:
        self.executions += 1
        self.successes += int(transition.action_succeeded)
        self.inventory_change += sum(abs(value) for value in transition.inventory_delta.values())
        self.facts_added += transition.facts_added
        self.facts_removed += transition.facts_removed
        self.unlocks += transition.unlocked_actions
        self.resource_cost += transition.resource_cost
        self.damage += transition.damage
        self.spatial_changes += int(transition.spatial_changed)
        self.terminal_returns += transition.terminal_reward

    def profile(self) -> tuple[float, ...]:
        count = max(1, self.executions)
        return (
            self.successes / count,
            self.inventory_change / count,
            self.facts_added / count,
            self.facts_removed / count,
            self.unlocks / count,
            self.resource_cost / count,
            self.damage / count,
            self.spatial_changes / count,
            self.terminal_returns / count,
        )


class LearnedEffectMemory:
    """Learns token-local effect profiles from visible transition deltas only."""

    def __init__(self) -> None:
        self._actions: dict[str, _EffectAccumulator] = {}
        self.update_count = 0

    def observe(self, transition: ObservableTransition) -> None:
        self._actions.setdefault(transition.action, _EffectAccumulator()).observe(
            transition
        )
        self.update_count += 1

    def profile(self, action: str) -> tuple[float, ...] | None:
        accumulator = self._actions.get(action)
        return None if accumulator is None else accumulator.profile()

    def export(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count,
            "actions": {
                action: asdict(accumulator)
                for action, accumulator in sorted(self._actions.items())
            },
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        self.update_count = int(payload.get("update_count", 0))
        self._actions = {
            str(action): _EffectAccumulator(**dict(values))
            for action, values in dict(payload.get("actions", {})).items()
        }


class CausalEncoder(Protocol):
    name: str

    def state_key(self, observation: RawCausalObservation) -> str: ...

    def action_key(self, observation: RawCausalObservation, action: str) -> str: ...

    def observe(self, transition: ObservableTransition) -> None: ...

    def export(self) -> Mapping[str, Any]: ...

    def restore(self, payload: Mapping[str, Any]) -> None: ...


class IdentityEncoder:
    name = "identity_representation"

    def state_key(self, observation: RawCausalObservation) -> str:
        return repr(
            (
                tuple(sorted(observation.inventory.items())),
                tuple(sorted(observation.observable_facts)),
                observation.available_actions,
                tuple(sorted(observation.spatial_observations.items())),
                observation.health,
                observation.last_action_succeeded,
            )
        )

    def action_key(self, observation: RawCausalObservation, action: str) -> str:
        del observation
        return action

    def observe(self, transition: ObservableTransition) -> None:
        del transition

    def export(self) -> Mapping[str, Any]:
        return {"name": self.name}

    def restore(self, payload: Mapping[str, Any]) -> None:
        del payload


def _bin(value: float, width: float = 0.25) -> float:
    return round(round(value / width) * width, 6)


class RelationalEffectEncoder:
    name = "relational_effect_representation"

    def __init__(self, *, use_affordances: bool = False) -> None:
        self.memory = LearnedEffectMemory()
        self.use_affordances = bool(use_affordances)

    def state_key(self, observation: RawCausalObservation) -> str:
        # Opaque fact/action/spatial tokens are deliberately excluded.  The
        # encoder only uses observable quantities and learned effect profiles.
        known_profiles = sorted(
            self.action_key(observation, action)
            for action in observation.available_actions
            if self.memory.profile(action) is not None
        )
        return repr(
            (
                tuple(sorted(observation.inventory.values())),
                _bin(observation.health),
                _bin(observation.damage),
                observation.last_action_succeeded,
                len(observation.available_actions),
                tuple(known_profiles),
                observation.terminal,
            )
        )

    def action_key(self, observation: RawCausalObservation, action: str) -> str:
        profile = self.memory.profile(action)
        if profile is not None:
            return "effect:" + repr(tuple(_bin(value) for value in profile))
        if self.use_affordances and action in observation.action_affordances:
            return "affordance:" + repr(
                tuple(sorted(observation.action_affordances[action]))
            )
        digest = hashlib.sha256(action.encode()).hexdigest()[:12]
        return f"unknown:{digest}"

    def observe(self, transition: ObservableTransition) -> None:
        self.memory.observe(transition)

    def export(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "use_affordances": self.use_affordances,
            "memory": self.memory.export(),
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        self.use_affordances = bool(payload.get("use_affordances", False))
        self.memory.restore(dict(payload.get("memory", {})))


class RepresentedReturnAgent:
    """Same policy/update schedule for identity and relational encoders."""

    def __init__(
        self,
        encoder: CausalEncoder,
        *,
        seed: int,
        learning_rate: float = 0.2,
        capacity: int = 50_000,
    ) -> None:
        self.encoder = encoder
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.values: dict[tuple[str, str], float] = defaultdict(float)
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.episode: list[tuple[str, str]] = []
        self.update_count = 0

    def select_action(self, observation: RawCausalObservation, *, epsilon: float) -> str:
        actions = observation.available_actions
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return self.rng.choice(actions)
        state = self.encoder.state_key(observation)
        return min(
            actions,
            key=lambda action: (
                -self.values.get(
                    (state, self.encoder.action_key(observation, action)), 0.0
                ),
                action,
            ),
        )

    def q_value(self, observation: RawCausalObservation, action: str) -> float:
        key = (
            self.encoder.state_key(observation),
            self.encoder.action_key(observation, action),
        )
        return max(0.0, min(1.0, self.values.get(key, 0.0)))

    def observe_transition(self, transition: ObservableTransition) -> None:
        state = self.encoder.state_key(transition.before)
        action = self.encoder.action_key(transition.before, transition.action)
        self.episode.append((state, action))
        self.encoder.observe(transition)

    def finish_episode(self, success: bool, *, gamma: float = 0.97) -> None:
        target = float(success)
        for key in reversed(self.episode):
            self.counts[key] += 1
            self.values[key] += self.learning_rate * (target - self.values[key])
            self.update_count += 1
            target *= gamma
        self.episode.clear()
        if len(self.values) > self.capacity:
            keep = sorted(self.counts, key=self.counts.get, reverse=True)[: self.capacity]
            self.values = defaultdict(float, {key: self.values[key] for key in keep})
            self.counts = defaultdict(int, {key: self.counts[key] for key in keep})

    def export_full_checkpoint(self) -> FullAgentCheckpoint:
        return FullAgentCheckpoint(
            policy={
                "values": {repr(key): value for key, value in self.values.items()},
                "counts": {repr(key): value for key, value in self.counts.items()},
                "update_count": self.update_count,
                "capacity": self.capacity,
            },
            rng=repr(self.rng.getstate()),
            relational_representation=dict(self.encoder.export()),
        )

    def import_full_checkpoint(self, checkpoint: FullAgentCheckpoint) -> None:
        self.values = defaultdict(
            float,
            {
                tuple(ast.literal_eval(key)): float(value)
                for key, value in checkpoint.policy.get("values", {}).items()
            },
        )
        self.counts = defaultdict(
            int,
            {
                tuple(ast.literal_eval(key)): int(value)
                for key, value in checkpoint.policy.get("counts", {}).items()
            },
        )
        self.update_count = int(checkpoint.policy.get("update_count", 0))
        self.capacity = int(checkpoint.policy.get("capacity", self.capacity))
        self.rng.setstate(ast.literal_eval(str(checkpoint.rng)))
        self.encoder.restore(checkpoint.relational_representation)
        self.episode.clear()
