from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .knowledge import KK, KV, KnowledgeSource, KnowledgeStatus, KnowledgeStore, ValueType


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
    predicted_delta_threshold: float = 0.5
    imagined_policy_update: bool = False
    imagined_weight: float = 0.05
    imagined_discount: float = 0.65
    minimum_confidence: float = 0.25
    tie_tolerance: float = 1e-9
    seed: int | None = None
    calibrated_imagination_enabled: bool = False
    candidate_dedup_enabled: bool = False
    placeholder_confidence_scale: float = 0.35
    mixed_grounding_confidence_scale: float = 0.65


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
    trajectories: tuple[Any, ...] = ()
    imagined_updates: tuple[tuple[Any, float, float], ...] = ()
    initial_candidate_signatures: frozenset[Any] = frozenset()

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
        self.random = random.Random(self.config.seed)

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
        selected = self._select_score(scores)
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

    def _select_score(self, scores: tuple[ImaginationScore, ...]) -> ImaginationScore:
        best = max(score.score for score in scores)
        ties = [score for score in scores if abs(score.score - best) <= self.config.tie_tolerance]
        return self.random.choice(ties)


@dataclass(frozen=True)
class PredictedKnowledgeDelta:
    added: tuple[tuple[KK, KV], ...]
    confidence: float

    @property
    def changed_kk(self) -> tuple[KK, ...]:
        return tuple(kk for kk, _ in self.added)


@dataclass(frozen=True)
class ImaginedState:
    knowledge: KnowledgeStore
    position_or_position_belief: Any
    opened_doors: frozenset[Any]
    step_depth: int
    last_action: Any = None
    last_semantic_delta: tuple[str, ...] = ()
    last_error: float = 0.0


@dataclass(frozen=True)
class ImaginedStep:
    depth: int
    state_before: ImaginedState
    state_signature_before: Any
    action: Any
    prediction: Any
    predicted_delta: PredictedKnowledgeDelta
    state_after: ImaginedState
    state_signature_after: Any
    immediate_value: float
    future_value: float = 0.0
    generated_candidate_count: int = 0
    unique_generated_candidate_count: int = 0
    duplicate_future_candidate_count: int = 0
    newly_unlocked_candidates: tuple[Any, ...] = ()
    unique_newly_unlocked_candidates: tuple[Any, ...] = ()
    unlocked_by_kk: tuple[KK, ...] = ()
    transition_confidence: float = 1.0
    path_confidence: float = 1.0
    grounding_factor: float = 1.0
    effective_confidence: float = 1.0
    placeholder_dependent: bool = False
    mixed_grounding: bool = False


@dataclass(frozen=True)
class ImaginedTrajectory:
    steps: tuple[ImaginedStep, ...]
    immediate_value: float
    future_value: float
    discounted_value: float
    uncalibrated_discounted_value: float
    calibrated_future_value: float
    uncalibrated_future_value: float
    selected_path_confidence: float
    max_depth: int
    newly_unlocked_action_count: int
    unique_newly_unlocked_action_count: int


@dataclass(frozen=True)
class ImaginationDiagnostics:
    rollout_count: int = 0
    imagined_state_transition_count: int = 0
    imagined_step_count: int = 0
    imagined_trajectory_count: int = 0
    selected_trajectory_depth: int = 0
    trajectory_depth_mean: float = 0.0
    trajectory_depth_max: int = 0
    future_candidate_generation_count: int = 0
    future_candidate_count: int = 0
    future_candidate_count_by_depth: tuple[tuple[int, int], ...] = ()
    newly_unlocked_action_count: int = 0
    raw_future_candidate_count: int = 0
    unique_future_candidate_count: int = 0
    duplicate_future_candidate_count: int = 0
    future_candidate_dedup_ratio: float = 0.0
    raw_newly_unlocked_action_count: int = 0
    unique_newly_unlocked_action_count: int = 0
    unique_unlock_ratio: float = 0.0
    selected_action_newly_unlocked_count: int = 0
    selected_action_has_future_dependency: bool = False
    selected_action_immediate_value: float = 0.0
    selected_action_future_value: float = 0.0
    selected_action_future_value_ratio: float = 0.0
    mean_transition_confidence: float = 0.0
    mean_selected_path_confidence: float = 0.0
    mean_placeholder_grounding_factor: float = 0.0
    mean_selected_effective_confidence: float = 0.0
    uncalibrated_selected_future_value: float = 0.0
    calibrated_selected_future_value: float = 0.0
    future_value_discount_ratio: float = 0.0
    placeholder_dependent_transition_count: int = 0
    concrete_transition_count: int = 0
    mixed_grounding_transition_count: int = 0
    setup_action_selected: bool = False
    predicted_placeholder_kv_count: int = 0
    placeholder_generated_candidate_count: int = 0
    placeholder_selected_candidate_count: int = 0
    unlocked_by_kk_counts: tuple[tuple[str, int], ...] = ()
    imagined_next_action: Any = None


def candidate_diagnostic_signature(candidate: Any) -> tuple[str, str, str, str, tuple[tuple[str, str], ...]]:
    from .policy import candidate_axes

    what, how, where = candidate_axes(candidate)
    return (
        getattr(candidate, "template", ""),
        what,
        how,
        where,
        tuple(
            sorted(
                (kk.value, repr(value))
                for kk, value in getattr(candidate, "bindings", {}).items()
                if kk != KK.CURRENT_POS
            )
        ),
    )


def candidate_canonical_signature(candidate: Any) -> tuple[str, str, str, str, tuple[tuple[str, str], ...]]:
    from .policy import candidate_axes

    what, how, where = candidate_axes(candidate)
    return (
        getattr(candidate, "template", ""),
        what,
        how,
        where,
        tuple(
            sorted(
                (kk.value, _canonical_binding_value(kk, value))
                for kk, value in getattr(candidate, "bindings", {}).items()
                if kk != KK.CURRENT_POS
            )
        ),
    )


def _canonical_binding_value(kk: KK, value: Any) -> str:
    if is_placeholder_value(value):
        return f"placeholder:{kk.value}"
    return repr(value)


def unique_candidates_by_signature(candidates: list[Any] | tuple[Any, ...]) -> list[Any]:
    seen: set[Any] = set()
    unique = []
    for candidate in candidates:
        signature = candidate_canonical_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
    return unique


def is_placeholder_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("imagined-") or value.startswith("imagined-key")
    if isinstance(value, tuple) and value:
        return isinstance(value[0], int) and value[0] >= 10_000
    return False


def candidate_has_placeholder(candidate: Any) -> bool:
    return any(is_placeholder_value(value) for value in getattr(candidate, "bindings", {}).values())


def trace_diagnostics(trace: ImaginationTrace | None) -> ImaginationDiagnostics:
    if trace is None or not trace.trajectories:
        return ImaginationDiagnostics()
    selected_trajectory = _selected_trajectory(trace)
    depths = [len(trajectory.steps) for trajectory in trace.trajectories]
    future_counts_by_depth: dict[int, int] = {}
    newly_unlocked = 0
    placeholder_kv = 0
    placeholder_candidates = 0
    unlocked_by: dict[str, int] = {}
    transitions = 0
    future_generation_count = 0
    future_candidate_count = 0
    unique_future_candidate_count = 0
    duplicate_future_candidate_count = 0
    unique_unlocked = 0
    confidence_values = []
    selected_path_values = []
    placeholder_grounding_values = []
    selected_effective_values = []
    placeholder_dependent_transitions = 0
    concrete_transitions = 0
    mixed_grounding_transitions = 0
    for trajectory in trace.trajectories:
        transitions += len(trajectory.steps)
        newly_unlocked += trajectory.newly_unlocked_action_count
        unique_unlocked += trajectory.unique_newly_unlocked_action_count
        for step in trajectory.steps:
            if step.depth > 0:
                future_generation_count += 1
                future_candidate_count += step.generated_candidate_count
                unique_future_candidate_count += step.unique_generated_candidate_count
                duplicate_future_candidate_count += step.duplicate_future_candidate_count
                future_counts_by_depth[step.depth] = future_counts_by_depth.get(step.depth, 0) + step.generated_candidate_count
            placeholder_kv += sum(1 for _, kv in step.predicted_delta.added if is_placeholder_value(kv.value))
            placeholder_candidates += sum(1 for candidate in step.newly_unlocked_candidates if candidate_has_placeholder(candidate))
            confidence_values.append(step.transition_confidence)
            if step.grounding_factor < 1.0:
                placeholder_grounding_values.append(step.grounding_factor)
            if step.placeholder_dependent:
                placeholder_dependent_transitions += 1
            elif step.mixed_grounding:
                mixed_grounding_transitions += 1
            else:
                concrete_transitions += 1
            for kk in step.unlocked_by_kk:
                unlocked_by[kk.value] = unlocked_by.get(kk.value, 0) + len(step.newly_unlocked_candidates)
    selected_depth = len(selected_trajectory.steps) if selected_trajectory is not None else 0
    selected_immediate = selected_trajectory.immediate_value if selected_trajectory is not None else 0.0
    selected_future = selected_trajectory.future_value if selected_trajectory is not None else 0.0
    selected_uncalibrated_future = selected_trajectory.uncalibrated_future_value if selected_trajectory is not None else 0.0
    selected_calibrated_future = selected_trajectory.calibrated_future_value if selected_trajectory is not None else 0.0
    selected_path_values = [step.path_confidence for step in selected_trajectory.steps] if selected_trajectory is not None else []
    selected_effective_values = [step.effective_confidence for step in selected_trajectory.steps] if selected_trajectory is not None else []
    future_ratio = selected_future / abs(selected_immediate) if selected_immediate else (1.0 if selected_future > 0 else 0.0)
    selected_unlocked = selected_trajectory.newly_unlocked_action_count if selected_trajectory is not None else 0
    selected_unique_unlocked = selected_trajectory.unique_newly_unlocked_action_count if selected_trajectory is not None else 0
    future_discount_ratio = _safe_ratio(selected_calibrated_future, selected_uncalibrated_future)
    immediate_values = [
        trajectory.immediate_value
        for trajectory in trace.trajectories
        if trajectory.steps
    ]
    max_immediate = max(immediate_values or [0.0])
    setup_selected = bool(
        selected_trajectory is not None
        and selected_unlocked > 0
        and selected_immediate < max_immediate
        and selected_trajectory.discounted_value >= max(trajectory.discounted_value for trajectory in trace.trajectories)
    )
    imagined_next = None
    if selected_trajectory is not None and len(selected_trajectory.steps) > 1:
        imagined_next = selected_trajectory.steps[1].action
    return ImaginationDiagnostics(
        rollout_count=len(trace.trajectories),
        imagined_state_transition_count=transitions,
        imagined_step_count=transitions,
        imagined_trajectory_count=len(trace.trajectories),
        selected_trajectory_depth=selected_depth,
        trajectory_depth_mean=sum(depths) / len(depths) if depths else 0.0,
        trajectory_depth_max=max(depths or [0]),
        future_candidate_generation_count=future_generation_count,
        future_candidate_count=future_candidate_count,
        future_candidate_count_by_depth=tuple(sorted(future_counts_by_depth.items())),
        newly_unlocked_action_count=newly_unlocked,
        raw_future_candidate_count=future_candidate_count,
        unique_future_candidate_count=unique_future_candidate_count,
        duplicate_future_candidate_count=duplicate_future_candidate_count,
        future_candidate_dedup_ratio=_safe_ratio(unique_future_candidate_count, future_candidate_count),
        raw_newly_unlocked_action_count=newly_unlocked,
        unique_newly_unlocked_action_count=unique_unlocked,
        unique_unlock_ratio=_safe_ratio(unique_unlocked, newly_unlocked),
        selected_action_newly_unlocked_count=selected_unique_unlocked if selected_unique_unlocked else selected_unlocked,
        selected_action_has_future_dependency=(selected_unique_unlocked if selected_unique_unlocked else selected_unlocked) > 0,
        selected_action_immediate_value=selected_immediate,
        selected_action_future_value=selected_future,
        selected_action_future_value_ratio=future_ratio,
        mean_transition_confidence=sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        mean_selected_path_confidence=sum(selected_path_values) / len(selected_path_values) if selected_path_values else 0.0,
        mean_placeholder_grounding_factor=sum(placeholder_grounding_values) / len(placeholder_grounding_values) if placeholder_grounding_values else 0.0,
        mean_selected_effective_confidence=sum(selected_effective_values) / len(selected_effective_values) if selected_effective_values else 0.0,
        uncalibrated_selected_future_value=selected_uncalibrated_future,
        calibrated_selected_future_value=selected_calibrated_future,
        future_value_discount_ratio=future_discount_ratio,
        placeholder_dependent_transition_count=placeholder_dependent_transitions,
        concrete_transition_count=concrete_transitions,
        mixed_grounding_transition_count=mixed_grounding_transitions,
        setup_action_selected=setup_selected,
        predicted_placeholder_kv_count=placeholder_kv,
        placeholder_generated_candidate_count=placeholder_candidates,
        placeholder_selected_candidate_count=1 if candidate_has_placeholder(trace.selected) else 0,
        unlocked_by_kk_counts=tuple(sorted(unlocked_by.items())),
        imagined_next_action=imagined_next,
    )


def _selected_trajectory(trace: ImaginationTrace) -> ImaginedTrajectory | None:
    for trajectory in trace.trajectories:
        if trajectory.steps and trajectory.steps[0].action == trace.selected:
            return trajectory
    return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


class PredictedStateImaginationCycle(ImaginationCycle):
    """Predicted-state rollout with virtual KnowledgeStore transitions."""

    def choose(
        self,
        state_signature: Any,
        candidates: list[Any],
        *,
        policy: Any = None,
        dmp: Any = None,
    ) -> ImaginationTrace:
        if dmp is None:
            return super().choose(state_signature, candidates, policy=policy, dmp=dmp)
        initial_state = ImaginedState(
            knowledge=dmp.store.clone(),
            position_or_position_belief=dmp.position,
            opened_doors=frozenset(dmp.world.opened_doors),
            step_depth=dmp.step_index,
        )
        scored: list[ImaginationScore] = []
        trajectories: list[ImaginedTrajectory] = []
        prediction_cache: dict[Any, Any] = {}
        candidate_cache: dict[Any, list[Any]] = {}
        immediate_cache: dict[Any, float] = {}
        signature_cache: dict[tuple[str, int], Any] = {}
        for candidate in candidates:
            trajectory = self._rollout(
                initial_state,
                candidate,
                policy=policy,
                dmp=dmp,
                prediction_cache=prediction_cache,
                candidate_cache=candidate_cache,
                immediate_cache=immediate_cache,
                signature_cache=signature_cache,
            )
            first = trajectory.steps[0]
            scored.append(
                ImaginationScore(
                    candidate=candidate,
                    score=trajectory.discounted_value,
                    expected_kk_gain=first.prediction.expected_knowledge_gain(),
                    predicted_flag_prob=first.prediction.flag_prob,
                    predicted_error_prob=first.prediction.error_prob,
                    repeat_penalty=self._repeat_penalty(candidate, dmp),
                    policy_prior=self._policy_prior(candidate, policy),
                    rollout_value=trajectory.discounted_value - first.immediate_value,
                    rollout_depth=len(trajectory.steps),
                )
            )
            trajectories.append(trajectory)
        scores = tuple(scored)
        selected = self._select_score(scores)
        selected_trajectory = trajectories[scored.index(selected)]
        return ImaginationTrace(
            selected=selected.candidate,
            scores=scores,
            trajectories=tuple(trajectories),
            imagined_updates=self._imagined_updates(selected_trajectory),
            initial_candidate_signatures=frozenset(candidate_diagnostic_signature(candidate) for candidate in candidates),
        )

    def _rollout(
        self,
        state: ImaginedState,
        first_action: Any,
        *,
        policy: Any,
        dmp: Any,
        prediction_cache: dict[Any, Any] | None = None,
        candidate_cache: dict[Any, list[Any]] | None = None,
        immediate_cache: dict[Any, float] | None = None,
        signature_cache: dict[tuple[str, int], Any] | None = None,
    ) -> ImaginedTrajectory:
        steps: list[ImaginedStep] = []
        current = state
        action = first_action
        total = 0.0
        uncalibrated_total = 0.0
        discount = 1.0
        path_confidence = 1.0
        visited: set[Any] = set()
        prediction_cache = prediction_cache if prediction_cache is not None else {}
        candidate_cache = candidate_cache if candidate_cache is not None else {}
        immediate_cache = immediate_cache if immediate_cache is not None else {}
        signature_cache = signature_cache if signature_cache is not None else {}
        for depth in range(max(1, self.config.rollout_depth)):
            state_signature = self._imagined_state_signature(current, dmp=dmp)
            prediction = self._cached_prediction(prediction_cache, state_signature, action)
            immediate = self._cached_immediate_value(
                immediate_cache,
                state_signature,
                action,
                prediction,
                policy=policy,
                dmp=dmp,
            )
            predicted_delta = self._predicted_delta(prediction, current.step_depth + 1)
            after = self._apply_prediction(current, action, prediction)
            after_signature = self._imagined_state_signature(after, dmp=dmp)
            before_candidates = self._cached_future_candidates(candidate_cache, current, dmp=dmp)
            future = self._cached_future_candidates(candidate_cache, after, dmp=dmp)
            before_signatures = {
                self._cached_candidate_signature(signature_cache, candidate, canonical=False)
                for candidate in before_candidates
            }
            raw_newly_unlocked_candidates = tuple(
                candidate
                for candidate in future
                if self._cached_candidate_signature(signature_cache, candidate, canonical=False) not in before_signatures
            )
            unique_future = (
                self._unique_candidates_by_cached_signature(signature_cache, future)
                if self.config.candidate_dedup_enabled
                else list(future)
            )
            unique_unlocked_candidates = tuple(
                self._unique_candidates_by_cached_signature(signature_cache, raw_newly_unlocked_candidates)
            )
            transition_confidence = self._transition_confidence(prediction, predicted_delta)
            grounding_factor = self._grounding_factor(action, predicted_delta, raw_newly_unlocked_candidates)
            effective_confidence = _clamp01(transition_confidence * grounding_factor)
            if self.config.calibrated_imagination_enabled:
                path_confidence *= effective_confidence
                step_confidence = 1.0 if depth == 0 else path_confidence
                step_value = discount * step_confidence * immediate
            else:
                step_value = discount * immediate
            steps.append(
                ImaginedStep(
                    depth=depth,
                    state_before=current,
                    state_signature_before=state_signature,
                    action=action,
                    prediction=prediction,
                    predicted_delta=predicted_delta,
                    state_after=after,
                    state_signature_after=after_signature,
                    immediate_value=immediate,
                    generated_candidate_count=len(future),
                    unique_generated_candidate_count=len(unique_future),
                    duplicate_future_candidate_count=max(0, len(future) - len(unique_future)),
                    newly_unlocked_candidates=raw_newly_unlocked_candidates,
                    unique_newly_unlocked_candidates=unique_unlocked_candidates,
                    unlocked_by_kk=predicted_delta.changed_kk,
                    transition_confidence=transition_confidence,
                    path_confidence=path_confidence,
                    grounding_factor=grounding_factor,
                    effective_confidence=effective_confidence,
                    placeholder_dependent=self._placeholder_dependent(action, predicted_delta, raw_newly_unlocked_candidates),
                    mixed_grounding=self._mixed_grounding(action, predicted_delta, raw_newly_unlocked_candidates),
                )
            )
            total += step_value
            uncalibrated_total += discount * immediate
            discount *= self.config.rollout_discount
            visited.add(self._candidate_signature(action))
            if depth + 1 >= self.config.rollout_depth:
                break
            if self.config.candidate_dedup_enabled:
                future = unique_future
            future = [candidate for candidate in future if self._candidate_signature(candidate) not in visited]
            if not future:
                break
            ranked = sorted(
                future,
                key=lambda candidate: self._cached_immediate_value(
                    immediate_cache,
                    self._imagined_state_signature(after, dmp=dmp),
                    candidate,
                    self._cached_prediction(
                        prediction_cache,
                        self._imagined_state_signature(after, dmp=dmp),
                        candidate,
                    ),
                    policy=policy,
                    dmp=dmp,
                ),
                reverse=True,
            )
            action = self.random.choice(ranked[: max(1, min(self.config.rollout_branching, len(ranked)))])
            current = after
        immediate_value = steps[0].immediate_value if steps else 0.0
        future_value = total - immediate_value
        uncalibrated_future_value = uncalibrated_total - immediate_value
        return ImaginedTrajectory(
            steps=tuple(steps),
            immediate_value=immediate_value,
            future_value=future_value,
            discounted_value=total,
            uncalibrated_discounted_value=uncalibrated_total,
            calibrated_future_value=future_value,
            uncalibrated_future_value=uncalibrated_future_value,
            selected_path_confidence=path_confidence,
            max_depth=len(steps),
            newly_unlocked_action_count=sum(len(step.newly_unlocked_candidates) for step in steps),
            unique_newly_unlocked_action_count=sum(len(step.unique_newly_unlocked_candidates) for step in steps),
        )

    def _cached_prediction(self, cache: dict[Any, Any], state_signature: Any, action: Any) -> Any:
        key = (state_signature, candidate_diagnostic_signature(action))
        if key not in cache:
            cache[key] = self.prophecy.predict(state_signature, action)
        return cache[key]

    def _cached_future_candidates(self, cache: dict[Any, list[Any]], state: ImaginedState, *, dmp: Any) -> list[Any]:
        key = self._imagined_state_signature(state, dmp=dmp)
        if key not in cache:
            cache[key] = self._future_candidates(state, dmp=dmp)
        return cache[key]

    def _cached_immediate_value(
        self,
        cache: dict[Any, float],
        state_signature: Any,
        action: Any,
        prediction: Any,
        *,
        policy: Any,
        dmp: Any,
    ) -> float:
        key = (state_signature, candidate_diagnostic_signature(action))
        if key not in cache:
            cache[key] = self._immediate_value(action, prediction, policy=policy, dmp=dmp)
        return cache[key]

    def _cached_candidate_signature(
        self,
        cache: dict[tuple[str, int], Any],
        candidate: Any,
        *,
        canonical: bool,
    ) -> Any:
        key = ("canonical" if canonical else "diagnostic", id(candidate))
        if key not in cache:
            cache[key] = (
                candidate_canonical_signature(candidate)
                if canonical
                else candidate_diagnostic_signature(candidate)
            )
        return cache[key]

    def _unique_candidates_by_cached_signature(
        self,
        cache: dict[tuple[str, int], Any],
        candidates: list[Any] | tuple[Any, ...],
    ) -> list[Any]:
        seen: set[Any] = set()
        unique = []
        for candidate in candidates:
            signature = self._cached_candidate_signature(cache, candidate, canonical=True)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(candidate)
        return unique

    def _transition_confidence(self, prediction: Any, predicted_delta: PredictedKnowledgeDelta) -> float:
        positive_probs = [
            prediction.kk_probs.get(kk, 0.0)
            for kk, _ in predicted_delta.added
        ]
        if positive_probs:
            kk_confidence = sum(positive_probs) / len(positive_probs)
            confidence = (kk_confidence + (1.0 - prediction.error_prob)) / 2.0
        else:
            confidence = 1.0 - prediction.error_prob
        return _clamp01(confidence)

    def _grounding_factor(
        self,
        action: Any,
        predicted_delta: PredictedKnowledgeDelta,
        newly_unlocked_candidates: tuple[Any, ...],
    ) -> float:
        mixed = self._mixed_grounding(action, predicted_delta, newly_unlocked_candidates)
        if mixed:
            return _clamp01(self.config.mixed_grounding_confidence_scale)
        placeholder = self._placeholder_dependent(action, predicted_delta, newly_unlocked_candidates)
        if placeholder:
            return _clamp01(self.config.placeholder_confidence_scale)
        return 1.0

    def _placeholder_dependent(
        self,
        action: Any,
        predicted_delta: PredictedKnowledgeDelta,
        newly_unlocked_candidates: tuple[Any, ...],
    ) -> bool:
        if self._mixed_grounding(action, predicted_delta, newly_unlocked_candidates):
            return False
        if candidate_has_placeholder(action):
            return True
        if any(is_placeholder_value(kv.value) for _, kv in predicted_delta.added):
            return True
        return any(candidate_has_placeholder(candidate) for candidate in newly_unlocked_candidates)

    def _mixed_grounding(
        self,
        action: Any,
        predicted_delta: PredictedKnowledgeDelta,
        newly_unlocked_candidates: tuple[Any, ...],
    ) -> bool:
        values = [
            value
            for kk, value in getattr(action, "bindings", {}).items()
            if kk != KK.CURRENT_POS
        ]
        for _, kv in predicted_delta.added:
            values.append(kv.value)
        for candidate in newly_unlocked_candidates:
            values.extend(
                value
                for kk, value in getattr(candidate, "bindings", {}).items()
                if kk != KK.CURRENT_POS
            )
        has_placeholder = any(is_placeholder_value(value) for value in values)
        has_concrete = any(not is_placeholder_value(value) for value in values)
        return has_placeholder and has_concrete

    def _immediate_value(self, action: Any, prediction: Any, *, policy: Any, dmp: Any) -> float:
        return (
            self.config.knowledge_weight * prediction.expected_knowledge_gain()
            + self.config.flag_weight * prediction.flag_prob
            - self.config.error_weight * prediction.error_prob
            - self.config.repeat_weight * self._repeat_penalty(action, dmp)
            + self.config.policy_prior_weight * self._policy_prior(action, policy)
        )

    def _apply_prediction(self, state: ImaginedState, action: Any, prediction: Any) -> ImaginedState:
        knowledge = state.knowledge.clone()
        delta = self._predicted_delta(prediction, state.step_depth + 1)
        for kk, kv in delta.added:
            if kk == KK.CURRENT_POS:
                knowledge.set_singleton(kk, kv.value, kv.type, source=kv.source, confidence=kv.confidence, step=kv.last_updated)
            else:
                knowledge.upsert(kk, kv)
        position = state.position_or_position_belief
        if getattr(action, "name", None) is not None and getattr(action.name, "value", "") == "MOVE_TOWARD":
            try:
                position = next(value for kk, value in action.bindings.items() if kk != KK.CURRENT_POS and isinstance(value, tuple))
                knowledge.set_singleton(KK.CURRENT_POS, position, ValueType.CELL_COORD, source=KnowledgeSource.IMAGINED, confidence=delta.confidence, step=state.step_depth + 1)
                knowledge.add(KK.VISITED_CELL, position, ValueType.CELL_COORD, source=KnowledgeSource.IMAGINED, confidence=delta.confidence, step=state.step_depth + 1)
                knowledge.add(KK.KNOWN_CELL, position, ValueType.CELL_COORD, source=KnowledgeSource.IMAGINED, confidence=delta.confidence, step=state.step_depth + 1)
            except StopIteration:
                pass
        opened_doors = state.opened_doors
        if getattr(action, "name", None) is not None and getattr(action.name, "value", "") == "USE_OBJECT":
            door = action.bindings.get(KK.DOOR_CELL)
            if door is not None and prediction.error_prob < 0.5:
                opened_doors = frozenset(set(opened_doors) | {door})
                knowledge.mark(KK.DOOR_CELL, door, KnowledgeStatus.CONSUMED, step=state.step_depth + 1)
        return ImaginedState(
            knowledge=knowledge,
            position_or_position_belief=position,
            opened_doors=opened_doors,
            step_depth=state.step_depth + 1,
            last_action=action,
            last_semantic_delta=tuple(sorted(kk.value for kk, _ in delta.added)),
            last_error=prediction.error_prob,
        )

    def _predicted_delta(self, prediction: Any, step: int) -> PredictedKnowledgeDelta:
        added = []
        confidences = []
        for kk, probability in prediction.kk_probs.items():
            if kk in {KK.CURRENT_POS, KK.DIRECTION, KK.SELF} or probability < self.config.predicted_delta_threshold:
                continue
            confidences.append(probability)
            added.append((kk, self._imagined_kv(kk, probability, step)))
        confidence = min(confidences or [0.0])
        return PredictedKnowledgeDelta(added=tuple(added), confidence=confidence)

    def _imagined_kv(self, kk: KK, confidence: float, step: int) -> KV:
        if kk.name.endswith("CELL") or kk in {KK.UNKNOWN_NEIGHBOR, KK.FRONTIER_CELL, KK.WALL_CELL, KK.VISITED_CELL, KK.KNOWN_CELL}:
            value: Any = (10_000 + step, len(kk.value))
            value_type = ValueType.CELL_COORD
        elif kk == KK.KEY_OBJECT:
            value = f"imagined-key#{step}"
            value_type = ValueType.OBJECT_INSTANCE
        elif kk == KK.HINT_VALUE:
            value = f"imagined-hint#{step}"
            value_type = ValueType.HINT_TEXT
        else:
            value = f"imagined-{kk.value}-{step}"
            value_type = ValueType.OBJECT_INSTANCE
        return KV(
            value=value,
            type=value_type,
            source=KnowledgeSource.IMAGINED,
            confidence=confidence,
            status=KnowledgeStatus.ACTIVE,
            last_updated=step,
        )

    def _future_candidates(self, state: ImaginedState, *, dmp: Any) -> list[Any]:
        from .gridworld import CandidateGenerator, GridKnowledgeState

        generator = CandidateGenerator(top_k=dmp.top_k, independent_how=True)
        return generator.generate(
            state.knowledge,
            GridKnowledgeState(
                position=state.position_or_position_belief,
                width=dmp.world.width,
                height=dmp.world.height,
                opened_doors=state.opened_doors,
                step_depth=state.step_depth,
            ),
        )

    def _imagined_state_signature(self, state: ImaginedState, *, dmp: Any) -> tuple[Any, ...]:
        from .prophecy import gridworld_knowledge_state_signature

        return gridworld_knowledge_state_signature(
            state.knowledge,
            position=state.position_or_position_belief,
            width=dmp.world.width,
            height=dmp.world.height,
            last_action=state.last_action,
            last_semantic_delta=state.last_semantic_delta,
            last_error=state.last_error,
            recent_transitions=getattr(dmp, "recent_transitions", ()),
        )

    def _imagined_updates(self, trajectory: ImaginedTrajectory) -> tuple[tuple[Any, float, float], ...]:
        if not self.config.imagined_policy_update:
            return ()
        updates = []
        for depth, step in enumerate(trajectory.steps):
            confidence = min(1.0, max(step.prediction.kk_probs.values() or [0.0]))
            if confidence < self.config.minimum_confidence:
                continue
            weight = self.config.imagined_weight * confidence * (self.config.imagined_discount ** depth)
            updates.append((step.action, step.immediate_value, weight))
        return tuple(updates)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
