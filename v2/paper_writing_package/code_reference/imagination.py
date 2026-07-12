from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .knowledge import KK


@dataclass(frozen=True)
class ImaginationConfig:
    knowledge_weight: float = 2.0
    flag_weight: float = 10.0
    error_weight: float = 3.0
    repeat_weight: float = 0.5
    policy_prior_weight: float = 0.2
    rollout_depth: int = 2
    rollout_branching: int = 3
    rollout_discount: float = 0.65
    dependency_weight: float = 1.5


@dataclass(frozen=True)
class ImaginationScore:
    candidate: Any
    score: float
    expected_kk_gain: float
    predicted_flag_prob: float
    predicted_error_prob: float
    repeat_penalty: float
    policy_prior: float
    rollout_value: float = 0.0
    rollout_depth: int = 1


@dataclass(frozen=True)
class ImaginationTrace:
    selected: Any
    scores: tuple[ImaginationScore, ...]

    @property
    def selected_score(self) -> ImaginationScore:
        return max(self.scores, key=lambda score: score.score)


class ImaginationCycle:
    """Depth-limited candidate rollout using only prophecy predictions.

    The rollout does not execute actions and does not read hidden environment
    state. It estimates whether an action's predicted knowledge change would
    make later action-template slots more useful.
    """

    def __init__(self, prophecy: Any, config: ImaginationConfig | None = None) -> None:
        self.prophecy = prophecy
        self.config = config or ImaginationConfig()

    def score_candidate(
        self,
        state_signature: Any,
        candidate: Any,
        *,
        policy: Any = None,
        dmp: Any = None,
        candidates: list[Any] | None = None,
    ) -> ImaginationScore:
        return self._score_candidate(
            state_signature,
            candidate,
            policy=policy,
            dmp=dmp,
            candidates=tuple(candidates or ()),
            remaining_depth=max(1, self.config.rollout_depth),
            visited=frozenset(),
        )

    def _score_candidate(
        self,
        state_signature: Any,
        candidate: Any,
        *,
        policy: Any = None,
        dmp: Any = None,
        candidates: tuple[Any, ...] = (),
        remaining_depth: int = 1,
        visited: frozenset[Any] = frozenset(),
    ) -> ImaginationScore:
        prediction = self.prophecy.predict(state_signature, candidate)
        expected_kk_gain = prediction.expected_knowledge_gain()
        predicted_flag_prob = prediction.flag_prob
        predicted_error_prob = prediction.error_prob
        repeat_penalty = self._repeat_penalty(candidate, dmp)
        policy_prior = self._policy_prior(candidate, policy)
        rollout_value = self._rollout_value(
            state_signature,
            prediction,
            candidates=candidates,
            policy=policy,
            dmp=dmp,
            remaining_depth=remaining_depth - 1,
            visited=visited | {self._candidate_signature(candidate)},
        )
        score = (
            self.config.knowledge_weight * expected_kk_gain
            + self.config.flag_weight * predicted_flag_prob
            - self.config.error_weight * predicted_error_prob
            - self.config.repeat_weight * repeat_penalty
            + self.config.policy_prior_weight * policy_prior
            + self.config.rollout_discount * rollout_value
        )
        return ImaginationScore(
            candidate=candidate,
            score=score,
            expected_kk_gain=expected_kk_gain,
            predicted_flag_prob=predicted_flag_prob,
            predicted_error_prob=predicted_error_prob,
            repeat_penalty=repeat_penalty,
            policy_prior=policy_prior,
            rollout_value=rollout_value,
            rollout_depth=max(1, self.config.rollout_depth - remaining_depth + 1),
        )

    def choose(
        self,
        state_signature: Any,
        candidates: list[Any],
        *,
        policy: Any = None,
        dmp: Any = None,
    ) -> ImaginationTrace:
        scores = tuple(
            self.score_candidate(
                state_signature,
                candidate,
                policy=policy,
                dmp=dmp,
                candidates=candidates,
            )
            for candidate in candidates
        )
        selected = max(scores, key=lambda score: score.score)
        return ImaginationTrace(selected=selected.candidate, scores=scores)

    def _rollout_value(
        self,
        state_signature: Any,
        prediction: Any,
        *,
        candidates: tuple[Any, ...],
        policy: Any,
        dmp: Any,
        remaining_depth: int,
        visited: frozenset[Any],
    ) -> float:
        if remaining_depth <= 0 or not candidates:
            return 0.0

        future_scores = []
        for future in candidates:
            signature = self._candidate_signature(future)
            if signature in visited:
                continue
            dependency_bonus = self._dependency_bonus(prediction, future)
            future_score = self._score_candidate(
                state_signature,
                future,
                policy=policy,
                dmp=dmp,
                candidates=candidates,
                remaining_depth=remaining_depth,
                visited=visited,
            )
            future_scores.append(future_score.score + dependency_bonus)

        if not future_scores:
            return 0.0
        future_scores.sort(reverse=True)
        top = future_scores[: max(1, self.config.rollout_branching)]
        return sum(top) / len(top)

    def _dependency_bonus(self, prediction: Any, future: Any) -> float:
        usable_probability = 0.0
        for kk in future.required_kk_slots:
            if kk == KK.CURRENT_POS:
                continue
            usable_probability += prediction.kk_probs.get(kk, 0.0)
        return self.config.dependency_weight * usable_probability

    def _policy_prior(self, candidate: Any, policy: Any) -> float:
        probability = getattr(policy, "candidate_probability", None)
        if callable(probability):
            return probability(candidate)
        return 1.0

    def _repeat_penalty(self, candidate: Any, dmp: Any) -> float:
        if dmp is None:
            return 0.0
        used_counts = []
        for kk, value in candidate.bindings.items():
            if kk == KK.CURRENT_POS:
                continue
            for kv in dmp.store.values(kk, include_inactive=True):
                if kv.value == value:
                    used_counts.append(kv.used_count)
        return float(max(used_counts or [0]))

    def _candidate_signature(self, candidate: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            candidate.template,
            tuple(
                sorted(
                    (kk.value, repr(value))
                    for kk, value in candidate.bindings.items()
                    if kk != KK.CURRENT_POS
                )
            ),
        )
