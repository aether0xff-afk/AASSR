from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ActionVerb(str, Enum):
    """Environment-independent primitive action families."""

    MOVE = "move"
    OBSERVE = "observe"
    PICKUP = "pickup"
    USE = "use"
    BREAK = "break"
    PLACE = "place"
    COMBINE = "combine"


@dataclass(frozen=True, slots=True)
class Action:
    verb: ActionVerb
    target: str | None = None
    tool: str | None = None
    destination: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def signature(self) -> str:
        parts = [self.verb.value, self.target, self.tool, self.destination]
        return "|".join(part or "_" for part in parts)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """A comparable state representation used by Prophecy and Imagination.

    The numerical vector is for similarity metrics. Facts and available actions
    remain explicit so the research code does not hide causal structure inside
    an embedding.
    """

    vector: tuple[float, ...]
    facts: frozenset[str] = frozenset()
    available_actions: tuple[Action, ...] = ()
    goal_progress: float = 0.0


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
