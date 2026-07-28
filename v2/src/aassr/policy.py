from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from .knowledge import KK


CandidateT = TypeVar("CandidateT")
DmpT = TypeVar("DmpT")


class CandidateScorer(Protocol[CandidateT, DmpT]):
    def choose(self, candidates: list[CandidateT], dmp: DmpT) -> CandidateT:
        ...


class RandomScorer:
    """C0-style selector: no learning, no prophecy, no imagination."""

    def __init__(self, *, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def choose(self, candidates: list[CandidateT], dmp: DmpT) -> CandidateT:
        return self.random.choice(candidates)


@dataclass
class PolicyABC:
    """Paper-aligned Policy A/B/C probability tables.

    A = WHAT action template to use.
    B = HOW binding/selection strategy.
    C = WHERE KK slot or candidate pool to target.
    """

    policy_a: dict[str, float] = field(default_factory=dict)
    policy_b: dict[str, float] = field(default_factory=dict)
    policy_c: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.05
    min_prob: float = 0.02
    seed: int | None = None

    def __post_init__(self) -> None:
        self.random = random.Random(self.seed)
        self._normalize_all()

    @classmethod
    def uniform_gridworld(
        cls,
        *,
        learning_rate: float = 0.05,
        min_prob: float = 0.02,
        seed: int | None = None,
    ) -> PolicyABC:
        return cls(
            policy_a={
                "MOVE_TOWARD": 0.25,
                "INSPECT_CELL": 0.25,
                "USE_OBJECT": 0.25,
                "FOLLOW_HINT": 0.25,
            },
            policy_b={
                "random": 0.25,
                "nearest": 0.25,
                "least_tried": 0.25,
                "high_uncertainty": 0.25,
                "normal": 0.25,
                "prophecy_best": 0.25,
            },
            policy_c={
                "KK_UNKNOWN_NEIGHBOR": 1 / 8,
                "KK_FRONTIER_CELL": 1 / 8,
                "KK_HINT_CELL": 1 / 8,
                "KK_HINT_VALUE": 1 / 8,
                "KK_KEY_CELL": 1 / 8,
                "KK_KEY_OBJECT": 1 / 8,
                "KK_DOOR_CELL": 1 / 8,
                "KK_FLAG_CELL": 1 / 8,
            },
            learning_rate=learning_rate,
            min_prob=min_prob,
            seed=seed,
        )

    def choose(self, candidates: list[CandidateT], dmp: DmpT) -> CandidateT:
        weights = [self.candidate_probability(candidate) for candidate in candidates]
        total = sum(weights)
        if total <= 0:
            return self.random.choice(candidates)
        threshold = self.random.random() * total
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights):
            cumulative += weight
            if cumulative >= threshold:
                return candidate
        return candidates[-1]

    def select_candidate(self, candidates: list[CandidateT]) -> CandidateT:
        return self.choose(candidates, None)

    def sample_axes(self) -> tuple[str, str, str]:
        """Sample WHAT/HOW/WHERE before candidate binding.

        This is the paper-aligned PolicyABC role: the policy tables provide a
        stochastic action-generation prior, while candidate binding and
        imagination remain separate stages.
        """
        return (
            self._sample_axis(self.policy_a),
            self._sample_axis(self.policy_b),
            self._sample_axis(self.policy_c),
        )

    def update(self, candidate: CandidateT, reward: float) -> None:
        what, how, where = candidate_axes(candidate)
        self._update_axis(self.policy_a, what, reward)
        self._update_axis(self.policy_b, how, reward)
        self._update_axis(self.policy_c, where, reward)

    def update_weighted(self, candidate: CandidateT, reward: float, weight: float) -> None:
        if weight <= 0.0:
            return
        self.update(candidate, reward * weight)

    def candidate_probability(self, candidate: CandidateT) -> float:
        what, how, where = candidate_axes(candidate)
        return (
            self.policy_a.get(what, 0.0)
            * self.policy_b.get(how, 0.0)
            * self.policy_c.get(where, 0.0)
        )

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "WHAT": dict(sorted(self.policy_a.items())),
            "HOW": dict(sorted(self.policy_b.items())),
            "WHERE": dict(sorted(self.policy_c.items())),
        }

    def _update_axis(self, table: dict[str, float], selected_key: str, reward: float) -> None:
        if selected_key not in table:
            table[selected_key] = 1e-9
            self._normalize(table)
        table[selected_key] *= math.exp(self.learning_rate * reward)
        self._normalize(table)

    def _normalize_all(self) -> None:
        self._normalize(self.policy_a)
        self._normalize(self.policy_b)
        self._normalize(self.policy_c)

    def _normalize(self, table: dict[str, float]) -> None:
        if not table:
            return
        total = sum(max(value, 0.0) for value in table.values())
        if total <= 0:
            uniform = 1.0 / len(table)
            for key in table:
                table[key] = uniform
            return
        for key in table:
            table[key] = max(table[key], 0.0) / total
        self._apply_floor(table)

    def _apply_floor(self, table: dict[str, float]) -> None:
        if not table or self.min_prob <= 0:
            return
        floor = min(self.min_prob, 1.0 / len(table))
        floored = {key: max(value, floor) for key, value in table.items()}
        total = sum(floored.values())
        for key in table:
            table[key] = floored[key] / total

    def _sample_axis(self, table: dict[str, float]) -> str:
        if not table:
            raise ValueError("PolicyABC cannot sample from an empty axis table")
        total = sum(max(value, 0.0) for value in table.values())
        if total <= 0:
            return self.random.choice(list(table))
        threshold = self.random.random() * total
        cumulative = 0.0
        for key, value in table.items():
            cumulative += max(value, 0.0)
            if cumulative >= threshold:
                return key
        return next(reversed(table))


def candidate_axes(candidate: Any) -> tuple[str, str, str]:
    what = getattr(candidate, "name").value
    how = getattr(candidate, "strategy", "normal")
    where = where_slot(candidate)
    return what, how, where


def where_slot(candidate: Any) -> str:
    bindings = getattr(candidate, "bindings")
    if KK.DOOR_CELL in bindings:
        return KK.DOOR_CELL.value
    for kk in getattr(candidate, "required_kk_slots"):
        if kk != KK.CURRENT_POS:
            return kk.value
    return KK.CURRENT_POS.value
