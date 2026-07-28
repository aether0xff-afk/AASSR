from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class ScoredAction:
    action: Action
    score: float


@dataclass(frozen=True, slots=True)
class PolicyMemory:
    """Branch-local policy changes made only inside one imagined universe."""

    deltas: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "deltas", MappingProxyType(dict(self.deltas)))

    @classmethod
    def empty(cls) -> PolicyMemory:
        return cls({})


class WeightedPolicy:
    """Minimal transparent policy used by the first imagination-tree baseline.

    The global weights belong to the real policy. ``PolicyMemory`` stores the
    temporary changes of one imagined branch, so sibling universes never share
    policy updates.
    """

    def __init__(
        self,
        initial_weights: Mapping[str, float] | None = None,
        *,
        imagination_learning_rate: float = 0.1,
        real_learning_rate: float = 0.2,
    ) -> None:
        if imagination_learning_rate < 0.0 or real_learning_rate < 0.0:
            raise ValueError("learning rates must be non-negative")
        self._weights = dict(initial_weights or {})
        self.imagination_learning_rate = imagination_learning_rate
        self.real_learning_rate = real_learning_rate

    def weight(self, action: Action) -> float:
        return self._weights.get(action.signature, 0.0)

    def rank(
        self,
        state: StateSnapshot,
        *,
        limit: int,
        memory: PolicyMemory | None = None,
    ) -> tuple[ScoredAction, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        deltas: Mapping[str, float] = {} if memory is None else memory.deltas
        ranked = sorted(
            (
                ScoredAction(
                    action,
                    self.weight(action) + deltas.get(action.signature, 0.0),
                )
                for action in state.available_actions
            ),
            key=lambda item: (-item.score, item.action.signature),
        )
        return tuple(ranked[:limit])

    def imagine_update(
        self,
        memory: PolicyMemory,
        action: Action,
        value: float,
    ) -> PolicyMemory:
        deltas = dict(memory.deltas)
        deltas[action.signature] = (
            deltas.get(action.signature, 0.0)
            + self.imagination_learning_rate * value
        )
        return PolicyMemory(deltas)

    def reinforce(self, action: Action, advantage: float) -> None:
        self._weights[action.signature] = (
            self.weight(action) + self.real_learning_rate * advantage
        )

    def snapshot(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._weights))
