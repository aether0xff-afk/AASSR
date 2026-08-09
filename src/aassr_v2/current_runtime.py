from __future__ import annotations

from statistics import fmean
from typing import Any, Hashable

from .current_generation import (
    CurrentNeuralDeltaProphecy,
    CurrentPentestAASSRAgent,
    KnowledgeBoundProphecy,
    RelationalContextualSkillAwareProphecy,
    ReplayRelationalCalibratedProphecy,
    relational_action_key,
    relational_state_vector,
)
from .current_manifest import (
    CURRENT_COMPONENTS,
    CURRENT_GENERATION_VERSION,
    LEGACY_COMPONENTS_ACTIVE,
)
from .integrated_agent import IntegratedProphecyView
from .neural_delta_prophecy import NeuralDeltaConfig
from .pentest_agent_main_test import ACTION_FEATURE_SIZE, HttpAgentCodec
from .types import Action, StateSnapshot


class FullyRelationalNeuralDeltaProphecy(CurrentNeuralDeltaProphecy):
    """Current Neural Delta with rename-invariant state *and* action input.

    The model predicts a concrete-scaffold delta: its learned input contains no
    raw route/profile/object indices, while the predicted numerical delta is
    applied to the caller's current concrete vector by the inherited decoder.
    This preserves the real current action surface without letting seed-specific
    identifier slots become a Policy/Prophecy lookup key.
    """

    name = "current-relational-state-action-neural-delta"

    def _input(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        return relational_state_vector(state) + relational_action_key(
            state,
            action,
        )

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            "state_input_relational": 1,
            "action_input_relational": 1,
            "prediction_output": "concrete-scaffold-delta",
        }


class FrozenReplayRelationalCalibratedProphecy(
    ReplayRelationalCalibratedProphecy
):
    """Frozen same-transition calibration with batched holdout refreshes.

    A calibration refresh evaluates exactly the same selected holdout transitions
    as the scalar implementation, but sends them through one Neural-Delta batch.
    This removes repeated GPU launch/synchronization overhead without changing the
    holdout set, refresh cadence, scoring equation, or anti-hindsight boundary.
    """

    name = "current-frozen-relational-holdout-calibrated-prophecy"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_holdout: tuple[Any, ...] | None = None
        self.freeze_count = 0
        self.calibration_batch_refreshes = 0
        self.calibration_batch_rows = 0

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
            selected = tuple(items[-self.evaluation_limit :])
            rows = self.base.predict_batch(
                tuple(item.state for item in selected),
                tuple(item.action for item in selected),
                samples=1,
            )
            self.calibration_batch_refreshes += 1
            self.calibration_batch_rows += len(selected)
            scores = []
            for item, row in zip(selected, rows, strict=True):
                prediction = row[0]
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
            "calibration_batch_refreshes": self.calibration_batch_refreshes,
            "calibration_batch_rows": self.calibration_batch_rows,
            "calibration_refresh_batching": 1,
        }


class CurrentSkillProphecy(RelationalContextualSkillAwareProphecy):
    """Relational Skill wrapper with explicit current-model diagnostics."""

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = getattr(self.base, "diagnostics", None)
        return dict(diagnostics()) if callable(diagnostics) else {}


class CurrentPentestRuntimeAgent(CurrentPentestAASSRAgent):
    """Transitional current wrapper retained for compatibility tests only.

    The package-level current builder now constructs the standalone current agent;
    this class remains importable so earlier current-generation branches/tests can
    reproduce their wiring without modifying frozen v0.4 code.
    """

    def __init__(
        self,
        *,
        seed: int,
        train_transitions: int,
        use_imagination: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            seed=int(seed),
            train_transitions=int(train_transitions),
            use_imagination=bool(use_imagination),
            device=device,
        )
        self.current_generation_version = CURRENT_GENERATION_VERSION
        self.current_components = dict(CURRENT_COMPONENTS)
        self.legacy_components_active = LEGACY_COMPONENTS_ACTIVE

        neural = FullyRelationalNeuralDeltaProphecy(
            HttpAgentCodec(),
            config=NeuralDeltaConfig(
                action_feature_size=ACTION_FEATURE_SIZE,
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=int(seed) ^ 0x4E455552,
            device=device,
        )
        self.base_neural_prophecy = neural

        calibrated = FrozenReplayRelationalCalibratedProphecy(
            neural,
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
                "prophecy_state_input_relational": True,
                "prophecy_action_input_relational": True,
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
    """Compatibility builder for the transitional runtime, not package default."""

    return CurrentPentestRuntimeAgent(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
