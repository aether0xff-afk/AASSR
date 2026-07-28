from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ActionVerb(str, Enum):
    """Legacy primitive families retained for GridWorld compatibility."""

    MOVE = "move"
    OBSERVE = "observe"
    PICKUP = "pickup"
    USE = "use"
    BREAK = "break"
    PLACE = "place"
    COMBINE = "combine"


def action_verb_name(verb: ActionVerb | str) -> str:
    return verb.value if isinstance(verb, ActionVerb) else str(verb)


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class Action:
    """One structured action.

    The old target/tool/destination fields remain for the first GridWorld.
    Plugins may instead use any opaque verb string and arbitrary parameters.
    The core never assumes what those names mean.
    """

    verb: ActionVerb | str
    target: str | None = None
    tool: str | None = None
    destination: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def verb_name(self) -> str:
        return action_verb_name(self.verb)

    @property
    def signature(self) -> str:
        parts = [self.verb_name, self.target, self.tool, self.destination]
        legacy = "|".join(part or "_" for part in parts)
        if not self.parameters:
            return legacy
        encoded = ",".join(
            f"{key}={_stable_value(value)}"
            for key, value in sorted(self.parameters.items())
        )
        return f"{legacy}|{encoded}"


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Comparable explicit state used by Prophecy and Imagination."""

    vector: tuple[float, ...]
    facts: frozenset[str] = frozenset()
    available_actions: tuple[Action, ...] = ()
    goal_progress: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def with_actions(self, actions: tuple[Action, ...]) -> StateSnapshot:
        return replace(self, available_actions=actions)


@dataclass(frozen=True, slots=True)
class Prediction:
    next_state: StateSnapshot
    probability: float = 1.0
    source: str = "prophecy"

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TransitionTrace:
    trace_id: str
    before: StateSnapshot
    action: Action
    predictions: tuple[Prediction, ...]
    after: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    real_reward: float = 0.0
    goal_ids: tuple[str, ...] = ()
