from __future__ import annotations

from dataclasses import dataclass

from .confidence import AdaptiveDepthController
from .imagination import ImaginationBatch
from .imagination_tree import ImaginationResult, ImaginationTree
from .information_value import (
    InformationValueBreakdown,
    information_value_from_measurements,
)
from .knowledge import KnowledgeEntry, KnowledgeStore
from .metrics import expected_prediction_vector, prediction_similarity
from .trace import TraceLedger
from .types import Action, TransitionTrace


@dataclass(frozen=True, slots=True)
class EvaluatedTransition:
    trace: TransitionTrace
    information_value: InformationValueBreakdown
    uncertainty_before: float
    uncertainty_after: float
    prediction_score_before: float
    prediction_score_after: float


@dataclass(frozen=True, slots=True)
class PlannedTransition:
    plan: ImaginationResult
    evaluated: EvaluatedTransition
    depth_before: int
    depth_after: int


class TransitionEvaluator:
    """Execute one real action and measure what the experience improved."""

    def __init__(self, prophecy: object, *, samples: int = 8) -> None:
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.prophecy = prophecy
        self.samples = samples
        self._index = 0

    def execute(
        self,
        environment: object,
        action: Action,
        knowledge: KnowledgeStore,
        ledger: TraceLedger,
    ) -> EvaluatedTransition:
        before = environment.snapshot()
        predictions_before = self.prophecy.predict(
            before,
            action,
            samples=self.samples,
        )
        uncertainty_before = ImaginationBatch(
            before,
            action,
            predictions_before,
        ).uncertainty

        outcome = environment.step(action)
        actual = outcome.snapshot
        prediction_score_before = prediction_similarity(
            expected_prediction_vector(predictions_before),
            actual.vector,
        )

        self.prophecy.learn(before, action, actual)
        predictions_after = self.prophecy.predict(
            before,
            action,
            samples=self.samples,
        )
        uncertainty_after = ImaginationBatch(
            before,
            action,
            predictions_after,
        ).uncertainty
        prediction_score_after = prediction_similarity(
            expected_prediction_vector(predictions_after),
            actual.vector,
        )

        self._index += 1
        trace_id = f"aseq-{self._index:06d}"
        enabled_signatures = tuple(
            candidate.signature for candidate in outcome.unlocked_actions
        )
        entries = tuple(
            KnowledgeEntry(
                key=fact,
                value=True,
                source_trace_id=trace_id,
                enabled_action_signatures=enabled_signatures,
            )
            for fact in outcome.added_facts
        )
        knowledge.apply(entries, outcome.removed_facts)

        trace = TransitionTrace(
            trace_id=trace_id,
            before=before,
            action=action,
            predictions=predictions_before,
            after=actual,
            added_facts=outcome.added_facts,
            removed_facts=outcome.removed_facts,
            unlocked_actions=outcome.unlocked_actions,
            error=outcome.error,
        )
        ledger.append(trace)

        information_value = information_value_from_measurements(
            uncertainty_before=uncertainty_before,
            uncertainty_after=uncertainty_after,
            prediction_score_before=prediction_score_before,
            prediction_score_after=prediction_score_after,
            unlocked_action_value=float(len(outcome.unlocked_actions)),
            goal_progress_before=before.goal_progress,
            goal_progress_after=actual.goal_progress,
            error_penalty=1.0 if outcome.error else 0.0,
        )

        return EvaluatedTransition(
            trace=trace,
            information_value=information_value,
            uncertainty_before=uncertainty_before,
            uncertainty_after=uncertainty_after,
            prediction_score_before=prediction_score_before,
            prediction_score_after=prediction_score_after,
        )


class ImaginationTransitionEvaluator:
    """Plan in Prophecy universes, execute only the chosen root action in reality."""

    def __init__(
        self,
        planner: ImaginationTree,
        evaluator: TransitionEvaluator,
        *,
        depth_controller: AdaptiveDepthController | None = None,
    ) -> None:
        self.planner = planner
        self.evaluator = evaluator
        self.depth_controller = depth_controller

    def execute(
        self,
        environment: object,
        knowledge: KnowledgeStore,
        ledger: TraceLedger,
    ) -> PlannedTransition:
        depth_before = (
            self.depth_controller.depth
            if self.depth_controller is not None
            else self.planner.config.maximum_depth
        )
        plan = self.planner.plan(
            environment.snapshot(),
            maximum_depth=depth_before,
        )
        evaluated = self.evaluator.execute(
            environment,
            plan.chosen_action,
            knowledge,
            ledger,
        )

        depth_after = depth_before
        if self.depth_controller is not None:
            depth_after = self.depth_controller.observe(
                evaluated.prediction_score_before
            )

        return PlannedTransition(
            plan=plan,
            evaluated=evaluated,
            depth_before=depth_before,
            depth_after=depth_after,
        )
