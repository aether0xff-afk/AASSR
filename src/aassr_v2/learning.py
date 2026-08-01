from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .imagination import ImaginationBatch
from .knowledge import KnowledgeEntry, KnowledgeStore
from .metrics import expected_prediction_vector, prediction_similarity
from .replay import PredictionValidator, ReplayBuffer, ReplayTransition
from .serialization import JsonlLedgerWriter
from .types import Action, StateSnapshot, TransitionTrace


@dataclass(frozen=True, slots=True)
class LearningEffectReport:
    latest_prediction_before: float
    knowledge_context_score: float
    latest_prediction_after: float
    knowledge_only_gain: float
    model_parameter_gain: float
    holdout_before: float
    holdout_after: float
    holdout_gain: float


@dataclass(slots=True)
class RunningMean:
    count: int = 0
    mean: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count


class ActionUnlockValueEstimator:
    """Learn the delayed usefulness of actions that a transition unlocked."""

    def __init__(self) -> None:
        self._values: dict[str, RunningMean] = {}

    def estimate(self, actions: Iterable[Action]) -> float:
        return sum(
            self._values.get(
                action.signature,
                RunningMean(),
            ).mean
            for action in actions
        )

    def observe_future_return(
        self,
        actions: Iterable[Action],
        future_return: float,
    ) -> None:
        for action in actions:
            self._values.setdefault(
                action.signature,
                RunningMean(),
            ).observe(future_return)


class InformationValuePredictor:
    """Online linear predictor for delayed realized information value."""

    FEATURE_NAMES = (
        "uncertainty_reduction",
        "holdout_gain",
        "unlocked_action_value",
        "goal_progress_gain",
        "repeat_penalty",
        "error_penalty",
    )

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
    ) -> None:
        self.learning_rate = learning_rate
        self.weights = {
            name: 0.0 for name in self.FEATURE_NAMES
        }
        self.bias = 0.0

    def predict(
        self,
        features: Mapping[str, float],
    ) -> float:
        return self.bias + sum(
            self.weights[name]
            * float(features.get(name, 0.0))
            for name in self.FEATURE_NAMES
        )

    def learn(
        self,
        features: Mapping[str, float],
        target: float,
    ) -> float:
        prediction = self.predict(features)
        error = prediction - target
        self.bias -= self.learning_rate * error
        for name in self.FEATURE_NAMES:
            self.weights[name] -= (
                self.learning_rate
                * error
                * float(features.get(name, 0.0))
            )
        return error * error


@dataclass(frozen=True, slots=True)
class CreditedTrace:
    trace_id: str
    credit: float


class DelayedCreditAssigner:
    def __init__(
        self,
        *,
        discount: float = 0.95,
    ) -> None:
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        self.discount = discount

    def assign(
        self,
        traces: Iterable[TransitionTrace],
        final_return: float,
    ) -> tuple[CreditedTrace, ...]:
        materialized = tuple(traces)
        return tuple(
            CreditedTrace(
                trace.trace_id,
                final_return
                * self.discount
                ** (len(materialized) - index - 1),
            )
            for index, trace in enumerate(materialized)
        )


@dataclass(frozen=True, slots=True)
class AdvancedEvaluation:
    trace: TransitionTrace
    effect: LearningEffectReport
    uncertainty_before: float
    uncertainty_after: float
    features: Mapping[str, float]
    predicted_information_value: float
    immediate_information_value: float


class AdvancedTransitionEvaluator:
    """Separate KK-context effects, model learning and held-out gain."""

    def __init__(
        self,
        prophecy: object,
        *,
        replay: ReplayBuffer | None = None,
        validator: PredictionValidator | None = None,
        predictor: InformationValuePredictor | None = None,
        unlock_estimator: ActionUnlockValueEstimator | None = None,
        credit_assigner: DelayedCreditAssigner | None = None,
        goal_progress_estimator: Callable[
            [StateSnapshot, StateSnapshot], float
        ]
        | None = None,
        logger: JsonlLedgerWriter | None = None,
        samples: int = 4,
        intrinsic_cap: float = 1.0,
    ) -> None:
        self.prophecy = prophecy
        self.replay = replay or ReplayBuffer()
        self.validator = validator or PredictionValidator(
            samples=samples
        )
        self.predictor = predictor or InformationValuePredictor()
        self.unlock_estimator = (
            unlock_estimator
            or ActionUnlockValueEstimator()
        )
        self.credit_assigner = credit_assigner or DelayedCreditAssigner()
        self.goal_progress_estimator = goal_progress_estimator
        self.logger = logger
        self.samples = samples
        self.intrinsic_cap = intrinsic_cap
        self._index = 0
        self._recent_pairs: list[
            tuple[str, tuple[float, ...]]
        ] = []

    def _predict(
        self,
        state: StateSnapshot,
        action: Action,
        knowledge: KnowledgeStore | None = None,
    ):
        contextual = getattr(
            self.prophecy,
            "predict_with_context",
            None,
        )
        if knowledge is not None and callable(contextual):
            return contextual(
                state,
                action,
                knowledge=knowledge,
                samples=self.samples,
            )
        return self.prophecy.predict(
            state,
            action,
            samples=self.samples,
        )

    def execute(
        self,
        environment: object,
        action: Action,
        knowledge: KnowledgeStore,
        *,
        learn: bool = True,
    ) -> AdvancedEvaluation:
        before = environment.snapshot()
        knowledge_before = knowledge.clone()
        predictions_before = self._predict(
            before,
            action,
            knowledge_before,
        )
        uncertainty_before = ImaginationBatch(
            before,
            action,
            predictions_before,
        ).uncertainty
        holdout_before = self.validator.evaluate(
            self.prophecy,
            self.replay.holdout(),
        ).mean_similarity

        outcome = environment.step(action)
        actual = outcome.snapshot
        latest_before = prediction_similarity(
            expected_prediction_vector(predictions_before),
            actual.vector,
        )

        if learn:
            self._index += 1
            trace_id = f"aseq-{self._index:06d}"
        else:
            trace_id = "frozen-evaluation"
        enabled = tuple(
            candidate.signature
            for candidate in outcome.unlocked_actions
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
        if learn:
            knowledge.apply(entries, outcome.removed_facts)

        context_predictions = self._predict(
            before,
            action,
            knowledge,
        )
        context_score = prediction_similarity(
            expected_prediction_vector(context_predictions),
            actual.vector,
        )

        sample = ReplayTransition(
            before,
            action,
            actual,
            trace_id,
        )
        train_sample = self.replay.add(sample) if learn else False
        if train_sample:
            self.prophecy.learn(
                before,
                action,
                actual,
            )
        predictions_after = self._predict(
            before,
            action,
            knowledge,
        )
        uncertainty_after = ImaginationBatch(
            before,
            action,
            predictions_after,
        ).uncertainty
        latest_after = prediction_similarity(
            expected_prediction_vector(predictions_after),
            actual.vector,
        )
        holdout_after = self.validator.evaluate(
            self.prophecy,
            self.replay.holdout(),
        ).mean_similarity

        repeat_key = (
            action.signature,
            before.vector,
        )
        repeat_penalty = (
            1.0
            if repeat_key in self._recent_pairs[-16:]
            else 0.0
        )
        if learn:
            self._recent_pairs.append(repeat_key)
        unlock_value = self.unlock_estimator.estimate(
            outcome.unlocked_actions
        )
        goal_gain = (
            self.goal_progress_estimator(before, actual)
            if self.goal_progress_estimator is not None
            else actual.goal_progress - before.goal_progress
        )
        features = {
            "uncertainty_reduction": (
                uncertainty_before - uncertainty_after
            ),
            "holdout_gain": holdout_after - holdout_before,
            "unlocked_action_value": unlock_value,
            "goal_progress_gain": goal_gain,
            "repeat_penalty": repeat_penalty,
            "error_penalty": (
                1.0 if outcome.error else 0.0
            ),
        }
        immediate = (
            features["uncertainty_reduction"]
            + features["holdout_gain"]
            + features["unlocked_action_value"]
            + features["goal_progress_gain"]
            - features["repeat_penalty"]
            - features["error_penalty"]
        )
        immediate = max(
            -self.intrinsic_cap,
            min(self.intrinsic_cap, immediate),
        )
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
            real_reward=float(
                getattr(outcome, "reward", 0.0)
            ),
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
        learn: bool = True,
    ) -> tuple[CreditedTrace, ...]:
        materialized = tuple(evaluations)
        credits = self.credit_assigner.assign(
            (
                item.trace
                for item in materialized
            ),
            final_return,
        )
        if not learn:
            return credits
        by_id = {
            item.trace.trace_id: item
            for item in materialized
        }
        for credited in credits:
            evaluation = by_id[credited.trace_id]
            self.predictor.learn(
                evaluation.features,
                credited.credit,
            )
            self.unlock_estimator.observe_future_return(
                evaluation.trace.unlocked_actions,
                credited.credit,
            )
            if policy is not None:
                reinforce = getattr(
                    policy,
                    "reinforce",
                    None,
                )
                if callable(reinforce):
                    reinforce(
                        evaluation.trace.action,
                        credited.credit
                        + evaluation.predicted_information_value
                        + evaluation.immediate_information_value,
                    )
        return credits
