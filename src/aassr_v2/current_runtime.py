from __future__ import annotations

from statistics import fmean
from typing import Any, Hashable

from .current_generation import (
    CURRENT_COMPONENTS,
    CURRENT_GENERATION_VERSION,
    LEGACY_COMPONENTS_ACTIVE,
    CurrentPentestAASSRAgent,
    KnowledgeBoundProphecy,
    RelationalContextualSkillAwareProphecy,
    ReplayRelationalCalibratedProphecy,
    relational_action_key,
)
from .integrated_agent import IntegratedProphecyView
from .types import Action, StateSnapshot


class FrozenReplayRelationalCalibratedProphecy(
    ReplayRelationalCalibratedProphecy
):
    """Use one exact pre-transition holdout view for all same-step calibration.

    The evaluator already freezes the holdout set used for its before/after score.
    The current Neural-Delta calibrator must obey the same boundary: if the real
    transition itself is assigned to holdout, it may calibrate *later* decisions
    but never a prediction of the transition that produced it.
    """

    name = "current-frozen-relational-holdout-calibrated-prophecy"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_holdout: tuple[Any, ...] | None = None
        self.freeze_count = 0

    def freeze_holdout(self, items: tuple[Any, ...]) -> None:
        self._frozen_holdout = tuple(items)
        self.freeze_count += 1

    def release_holdout(self) -> None:
        self._frozen_holdout = None

    def _calibration(self, state: StateSnapshot, action: Action) -> float:
        key = self._key(state, action)
        source = (
            self._frozen_holdout
            if self._frozen_holdout is not None
            else self.replay.holdout()
        )
        items = [
            item
            for item in source
            if self._key(item.state, item.action) == key
        ]
        revision = int(self.base.gradient_updates)
        cache_key: tuple[Hashable, int, int] = (
            key,
            len(items) // self.refresh_stride,
            revision // self.refresh_stride,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.refreshes += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            scores = []
            for item in items[-self.evaluation_limit :]:
                prediction = self.base.predict(
                    item.state,
                    item.action,
                    samples=1,
                )[0]
                predicted = prediction.next_state
                error = fmean(
                    abs(left - right)
                    for left, right in zip(
                        predicted.vector,
                        item.next_state.vector,
                        strict=True,
                    )
                )
                terminal_match = (
                    self._terminal_class(predicted)
                    == self._terminal_class(item.next_state)
                )
                action_ratio = min(
                    len(predicted.available_actions),
                    len(item.next_state.available_actions),
                ) / float(
                    max(
                        1,
                        len(predicted.available_actions),
                        len(item.next_state.available_actions),
                    )
                )
                scores.append(
                    max(0.0, 1.0 - error)
                    * float(terminal_match)
                    * action_ratio
                )
            value = max(
                0.0,
                min(1.0, fmean(scores) if scores else 0.0),
            )
        self._cache[cache_key] = value
        return value

    def diagnostics(self) -> dict[str, int | float]:
        return {
            **super().diagnostics(),
            "holdout_freezes": self.freeze_count,
            "holdout_currently_frozen": int(self._frozen_holdout is not None),
        }


class CurrentSkillProphecy(RelationalContextualSkillAwareProphecy):
    """Relational Skill wrapper with explicit current-model diagnostics."""

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = getattr(self.base, "diagnostics", None)
        return dict(diagnostics()) if callable(diagnostics) else {}


class CurrentPentestRuntimeAgent(CurrentPentestAASSRAgent):
    """Safe canonical current-generation runtime.

    This thin layer exists so methodology guardrails can advance without editing
    any historical v0.4/reproduction implementation. Legacy classes remain
    importable, but the public current builder below always returns this runtime.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        calibrated = FrozenReplayRelationalCalibratedProphecy(
            self.base_neural_prophecy,
            self.evaluator.replay,
        )
        knowledge = KnowledgeBoundProphecy(calibrated)
        skill = CurrentSkillProphecy(
            knowledge,
            self.skills,
            self.knowledge,
        )
        self.calibrated_prophecy = calibrated
        self.knowledge_prophecy = knowledge
        self.skill_prophecy = skill

        # Rewire every active prediction consumer. Historical objects constructed
        # by the base initializer become unreachable from the current runtime.
        self.core.base_prophecy = skill
        self.core.prophecy = skill
        self.effect_prophecy = skill
        self.prophecy = IntegratedProphecyView(skill, skill)
        self.core.planner.prophecy = self.prophecy
        self.evaluator.prophecy = self.prophecy

    def _execute_primitive(
        self,
        environment: object,
        action: Action,
        *,
        training: bool,
    ):
        if not training:
            return super()._execute_primitive(
                environment,
                action,
                training=False,
            )

        # Freeze before AdvancedTransitionEvaluator can add the current sample to
        # replay. Both pre-update and post-update model checks therefore use the
        # exact same calibration evidence.
        self.calibrated_prophecy.freeze_holdout(
            tuple(self.evaluator.replay.holdout())
        )
        try:
            return super()._execute_primitive(
                environment,
                action,
                training=True,
            )
        finally:
            self.calibrated_prophecy.release_holdout()

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        base.update(
            {
                "canonical_runtime": "current_runtime",
                "calibration_same_transition_frozen": True,
                "current_generation_version": CURRENT_GENERATION_VERSION,
                "current_components": dict(CURRENT_COMPONENTS),
                "legacy_components_active": list(LEGACY_COMPONENTS_ACTIVE),
            }
        )
        return base


def build_current_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
) -> CurrentPentestRuntimeAgent:
    """Public current-generation builder; legacy builders are reproduction-only."""

    return CurrentPentestRuntimeAgent(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
