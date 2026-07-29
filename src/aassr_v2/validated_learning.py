from __future__ import annotations

from dataclasses import dataclass, field

from .imagination import ImaginationBatch
from .knowledge import KnowledgeEntry, KnowledgeStore
from .learning import (
    AdvancedEvaluation,
    AdvancedTransitionEvaluator,
    LearningEffectReport,
)
from .metrics import expected_prediction_vector, prediction_similarity
from .replay import ReplayTransition
from .types import Action, TransitionTrace


@dataclass(slots=True)
class FixedReplayBuffer:
    """Replay split whose held-out set is frozen across one before/after comparison.

    The next transition is assigned to train or holdout deterministically, but it is
    committed only after both validation scores are measured. A held-out sample is
    therefore never used to grade the same update that introduced it.
    """

    capacity: int = 2048
    holdout_stride: int = 5
    _train: list[ReplayTransition] = field(default_factory=list)
    _holdout: list[ReplayTransition] = field(default_factory=list)
    _seen: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.holdout_stride <= 1:
            raise ValueError(
                "capacity must be positive and holdout_stride must exceed one"
            )

    def next_is_train(self) -> bool:
        return (self._seen + 1) % self.holdout_stride != 0

    def commit(self, transition: ReplayTransition, *, train: bool) -> None:
        self._seen += 1
        target = self._train if train else self._holdout
        target.append(transition)
        if len(target) > self.capacity:
            target.pop(0)

    def train(self) -> tuple[ReplayTransition, ...]:
        return tuple(self._train)

    def holdout(self) -> tuple[ReplayTransition, ...]:
        return tuple(self._holdout)


class FixedHoldoutTransitionEvaluator(AdvancedTransitionEvaluator):
    """Advanced evaluator with uncontaminated, same-set holdout gain.

    Validation before and after learning uses the exact same frozen transitions.
    The current transition is added to its partition only after the comparison.
    Holdout gain remains zero until a configurable warm-up count is available.
    """

    def __init__(
        self,
        prophecy: object,
        *,
        replay: FixedReplayBuffer | None = None,
        minimum_holdout_count: int = 4,
        **kwargs,
    ) -> None:
        if minimum_holdout_count <= 0:
            raise ValueError("minimum_holdout_count must be positive")
        fixed_replay = replay or FixedReplayBuffer()
        super().__init__(prophecy, replay=fixed_replay, **kwargs)
        self.minimum_holdout_count = minimum_holdout_count

    def execute(
        self,
        environment: object,
        action: Action,
        knowledge: KnowledgeStore,
    ) -> AdvancedEvaluation:
        before = environment.snapshot()
        knowledge_before = knowledge.clone()
        predictions_before = self._predict(before, action, knowledge_before)
        uncertainty_before = ImaginationBatch(
            before,
            action,
            predictions_before,
        ).uncertainty

        frozen_holdout = self.replay.holdout()
        holdout_ready = len(frozen_holdout) >= self.minimum_holdout_count
        holdout_before = (
            self.validator.evaluate(self.prophecy, frozen_holdout).mean_similarity
            if holdout_ready
            else 0.0
        )

        outcome = environment.step(action)
        actual = outcome.snapshot
        latest_before = prediction_similarity(
            expected_prediction_vector(predictions_before),
            actual.vector,
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
        knowledge.apply(entries, outcome.removed_facts)

        context_predictions = self._predict(before, action, knowledge)
        context_score = prediction_similarity(
            expected_prediction_vector(context_predictions),
            actual.vector,
        )

        sample = ReplayTransition(before, action, actual, trace_id)
        train_sample = self.replay.next_is_train()
        if train_sample:
            self.prophecy.learn(before, action, actual)

        predictions_after = self._predict(before, action, knowledge)
        uncertainty_after = ImaginationBatch(
            before,
            action,
            predictions_after,
        ).uncertainty
        latest_after = prediction_similarity(
            expected_prediction_vector(predictions_after),
            actual.vector,
        )
        holdout_after = (
            self.validator.evaluate(self.prophecy, frozen_holdout).mean_similarity
            if holdout_ready
            else 0.0
        )
        holdout_gain = holdout_after - holdout_before if holdout_ready else 0.0

        self.replay.commit(sample, train=train_sample)

        repeat_key = (action.signature, before.vector)
        repeat_penalty = 1.0 if repeat_key in self._recent_pairs[-16:] else 0.0
        self._recent_pairs.append(repeat_key)
        unlock_value = self.unlock_estimator.estimate(outcome.unlocked_actions)
        goal_gain = actual.goal_progress - before.goal_progress
        features = {
            "uncertainty_reduction": uncertainty_before - uncertainty_after,
            "holdout_gain": holdout_gain,
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
            predictions_before,
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
            holdout_gain,
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
                    "holdout_count": len(frozen_holdout),
                    "train_sample": train_sample,
                },
                features=dict(features),
                predicted_information_value=predicted,
                immediate_information_value=immediate,
            )
        return result
