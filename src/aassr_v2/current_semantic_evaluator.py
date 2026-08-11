from __future__ import annotations

from itertools import combinations
from statistics import fmean
from typing import Any, Iterable, Sequence

from .current_generation import relational_action_key
from .current_relational_codec import (
    descriptor,
    legal_action_mask,
    semantic_prediction_score,
    terminal_class,
)
from .current_relational_state import relational_state_key_v2
from .knowledge import KnowledgeEntry, KnowledgeStore
from .learning import (
    AdvancedEvaluation,
    AdvancedTransitionEvaluator,
    CreditedTrace,
    DelayedCreditAssigner,
    LearningEffectReport,
    RunningMean,
)
from .replay import ReplayBuffer, ReplayTransition
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot, TransitionTrace


def _mask_jaccard(left: StateSnapshot, right: StateSnapshot) -> float:
    a = legal_action_mask(left)
    b = legal_action_mask(right)
    left_set = {index for index, value in enumerate(a) if value >= 0.5}
    right_set = {index for index, value in enumerate(b) if value >= 0.5}
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _normalized_outcome_masses(
    predictions: Sequence[Prediction],
) -> tuple[float, ...]:
    raw = [
        max(0.0, float(getattr(item, "outcome_probability", 1.0)))
        for item in predictions
    ]
    total = sum(raw)
    if total <= 1e-12:
        return (1.0 / len(raw),) * len(raw) if raw else ()
    return tuple(value / total for value in raw)


def semantic_prediction_uncertainty(
    predictions: Sequence[Prediction],
) -> float:
    """Relational uncertainty with reliability and outcome mass kept separate.

    Epistemic unreliability is weighted by stochastic outcome mass. Legitimate
    multimodal diversity is aleatoric uncertainty and is measured with pairwise
    outcome-mass weights, so a 1% rare mode cannot count as much as a 50% mode.
    """
    materialized = tuple(predictions)
    if not materialized:
        return 1.0
    masses = _normalized_outcome_masses(materialized)
    reliability = sum(
        mass * max(0.0, min(1.0, float(item.probability)))
        for mass, item in zip(masses, materialized, strict=True)
    )
    confidence_uncertainty = 1.0 - reliability

    weighted_distance = 0.0
    total_pair_mass = 0.0
    for (left_index, left), (right_index, right) in combinations(
        tuple(enumerate(materialized)),
        2,
    ):
        left_state = left.next_state
        right_state = right.next_state
        state_distance = fmean(
            abs(a - b)
            for a, b in zip(
                descriptor(left_state),
                descriptor(right_state),
                strict=True,
            )
        )
        distance = (
            0.55 * state_distance
            + 0.30 * (1.0 - _mask_jaccard(left_state, right_state))
            + 0.15 * float(terminal_class(left_state) != terminal_class(right_state))
        )
        pair_mass = masses[left_index] * masses[right_index]
        weighted_distance += pair_mass * distance
        total_pair_mass += pair_mass
    diversity = (
        weighted_distance / total_pair_mass
        if total_pair_mass > 1e-12
        else 0.0
    )
    return max(
        0.0,
        min(1.0, 0.5 * confidence_uncertainty + 0.5 * diversity),
    )


def _action_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", *relational_action_key(state, action))


class RelationalActionUnlockValueEstimator:
    """Delayed unlock value keyed by structural action identity, never raw IDs."""

    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], RunningMean] = {}

    def estimate(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        return sum(
            self._values.get(_action_key(state, action), RunningMean()).mean
            for action in actions
        )

    def observe_future_return(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
        future_return: float,
    ) -> None:
        for action in actions:
            self._values.setdefault(
                _action_key(state, action),
                RunningMean(),
            ).observe(float(future_return))


class RelationalAdvancedTransitionEvaluator(AdvancedTransitionEvaluator):
    """Information-value evaluator aligned with the repaired relational model.

    Replay/holdout anti-hindsight behavior is unchanged, but prediction quality,
    uncertainty, repeat identity and unlocked-action value are all relational.
    The v2 state key is referenced directly rather than depending on import-time
    monkeypatching of the historical current-generation module.
    """

    name = "relational-advanced-transition-evaluator-v2"

    def __init__(
        self,
        prophecy: object,
        *,
        replay: ReplayBuffer,
        validator: object,
        predictor: object,
        logger: object | None = None,
        samples: int = 3,
        intrinsic_cap: float = 1.0,
    ) -> None:
        super().__init__(
            prophecy,
            replay=replay,
            validator=validator,
            predictor=predictor,
            logger=logger,
            samples=samples,
            intrinsic_cap=intrinsic_cap,
        )
        self.relational_unlock_estimator = RelationalActionUnlockValueEstimator()
        self.unlock_estimator = self.relational_unlock_estimator
        self._recent_pairs: list[tuple[Any, ...]] = []

    def execute(
        self,
        environment: object,
        action: Action,
        knowledge: KnowledgeStore,
    ) -> AdvancedEvaluation:
        before = environment.snapshot()
        knowledge_before = knowledge.clone()

        predictions_without_context = self._predict(before, action, None)
        context_predictions = self._predict(
            before,
            action,
            knowledge_before,
        )
        uncertainty_before = semantic_prediction_uncertainty(context_predictions)

        holdout_reference = self.replay.holdout()
        holdout_before = self.validator.evaluate(
            self.prophecy,
            holdout_reference,
        ).mean_similarity

        outcome = environment.step(action)
        actual = outcome.snapshot
        latest_before = semantic_prediction_score(
            predictions_without_context,
            actual,
        )
        context_score = semantic_prediction_score(
            context_predictions,
            actual,
        )

        self._index += 1
        trace_id = f"aseq-{self._index:06d}"
        enabled = tuple(
            candidate.signature for candidate in outcome.unlocked_actions
        )
        entries = tuple(
            KnowledgeEntry(
                fact,
                True,
                trace_id,
                enabled_action_signatures=enabled,
            )
            for fact in outcome.added_facts
        )

        sample = ReplayTransition(before, action, actual, trace_id)
        train_sample = self.replay.add(sample)
        if train_sample:
            self.prophecy.learn(before, action, actual)
        else:
            advance = getattr(self.prophecy, "advance_sequence", None)
            if callable(advance):
                advance(before, action)

        predictions_after = self._predict(
            before,
            action,
            knowledge_before,
        )
        uncertainty_after = semantic_prediction_uncertainty(predictions_after)
        latest_after = semantic_prediction_score(predictions_after, actual)
        holdout_after = self.validator.evaluate(
            self.prophecy,
            holdout_reference,
        ).mean_similarity

        knowledge.apply(entries, outcome.removed_facts)

        repeat_key = (
            relational_state_key_v2(before),
            _action_key(before, action),
        )
        repeat_penalty = 1.0 if repeat_key in self._recent_pairs[-16:] else 0.0
        self._recent_pairs.append(repeat_key)
        unlock_value = self.relational_unlock_estimator.estimate(
            actual,
            outcome.unlocked_actions,
        )
        goal_gain = actual.goal_progress - before.goal_progress
        features = {
            "uncertainty_reduction": uncertainty_before - uncertainty_after,
            "holdout_gain": holdout_after - holdout_before,
            "unlocked_action_value": unlock_value,
            "goal_progress_gain": goal_gain,
            "repeat_penalty": repeat_penalty,
            "error_penalty": 1.0 if outcome.error else 0.0,
        }
        immediate = (
            features["uncertainty_reduction"]
            + features["holdout_gain"]
            + features["unlocked_action_value"]
            + features["goal_progress_gain"]
            - features["repeat_penalty"]
            - features["error_penalty"]
        )
        immediate = max(-self.intrinsic_cap, min(self.intrinsic_cap, immediate))
        predicted = self.predictor.predict(features)

        trace = TransitionTrace(
            trace_id,
            before,
            action,
            tuple(context_predictions),
            actual,
            outcome.added_facts,
            outcome.removed_facts,
            outcome.unlocked_actions,
            outcome.error,
            real_reward=float(getattr(outcome, "reward", 0.0)),
        )
        effect = LearningEffectReport(
            latest_before,
            context_score,
            latest_after,
            context_score - latest_before,
            latest_after - context_score,
            holdout_before,
            holdout_after,
            holdout_after - holdout_before,
        )
        result = AdvancedEvaluation(
            trace,
            effect,
            uncertainty_before,
            uncertainty_after,
            features,
            predicted,
            immediate,
        )
        if self.logger is not None:
            self.logger.append_trace(
                trace,
                effect={
                    "latest_prediction_before": effect.latest_prediction_before,
                    "knowledge_context_score": effect.knowledge_context_score,
                    "latest_prediction_after": effect.latest_prediction_after,
                    "knowledge_only_gain": effect.knowledge_only_gain,
                    "model_parameter_gain": effect.model_parameter_gain,
                    "holdout_before": effect.holdout_before,
                    "holdout_after": effect.holdout_after,
                    "holdout_gain": effect.holdout_gain,
                    "semantic_evaluator": True,
                },
                features=dict(features),
                predicted_information_value=predicted,
                immediate_information_value=immediate,
            )
        return result

    def finish_episode(
        self,
        evaluations: Iterable[AdvancedEvaluation],
        *,
        final_return: float,
        policy: object | None = None,
    ) -> tuple[CreditedTrace, ...]:
        materialized = tuple(evaluations)
        credits = DelayedCreditAssigner().assign(
            (item.trace for item in materialized),
            final_return,
        )
        by_id = {item.trace.trace_id: item for item in materialized}
        for credited in credits:
            evaluation = by_id[credited.trace_id]
            self.predictor.learn(evaluation.features, credited.credit)
            self.relational_unlock_estimator.observe_future_return(
                evaluation.trace.after,
                evaluation.trace.unlocked_actions,
                credited.credit,
            )
            if policy is not None:
                reinforce = getattr(policy, "reinforce", None)
                if callable(reinforce):
                    reinforce(
                        evaluation.trace.action,
                        credited.credit + evaluation.immediate_information_value,
                    )
        return credits
