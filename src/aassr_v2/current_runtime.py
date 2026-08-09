from __future__ import annotations

from statistics import fmean
from typing import Any, Hashable, Sequence

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
from .types import Action, Prediction, StateSnapshot


class FullyRelationalNeuralDeltaProphecy(CurrentNeuralDeltaProphecy):
    """Current Neural Delta with relational input and bulk batch decoding.

    The model input contains no raw route/profile/object index. Batched outputs are
    copied from the accelerator in whole tensors before Python StateSnapshot
    decoding, avoiding one host synchronization per imagined branch.
    """

    name = "current-relational-state-action-neural-delta"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_host_transfer_groups = 0
        self.batch_host_transfer_rows = 0

    def _input(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        return relational_state_vector(state) + relational_action_key(
            state,
            action,
        )

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not states:
            return ()
        self.batch_prediction_calls += 1
        self.batch_prediction_rows += len(states)
        if self.observations < self.config.warmup_steps:
            return tuple(
                (Prediction(state, 0.0, source=f"{self.name}:unseen"),)
                for state in states
            )

        next_states, terminal, _ = self._batch_outputs(states, actions)
        confidence = self._batch_confidence(next_states, terminal)
        mean_states = next_states.mean(dim=0)
        terminal_classes = terminal.mean(dim=0).argmax(dim=1)

        # Three bulk transfers per Neural-Delta batch regardless of branch count.
        host_states = mean_states.detach().cpu().tolist()
        host_terminal_classes = terminal_classes.detach().cpu().tolist()
        host_confidences = confidence.detach().cpu().tolist()
        self.batch_host_transfer_groups += 1
        self.batch_host_transfer_rows += len(states)

        rows = []
        for state, mean_state, terminal_class, probability in zip(
            states,
            host_states,
            host_terminal_classes,
            host_confidences,
            strict=True,
        ):
            decoded = self.codec.decode(
                mean_state,
                scaffold=state,
                terminal_class=int(terminal_class),
                source=f"{self.name}:ensemble",
            )
            rows.append(
                (
                    Prediction(
                        decoded,
                        float(probability),
                        source=f"{self.name}:ensemble",
                    ),
                )
            )
        return tuple(rows)

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **super().diagnostics(),
            "state_input_relational": 1,
            "action_input_relational": 1,
            "prediction_output": "concrete-scaffold-delta",
            "batch_host_transfer_groups": self.batch_host_transfer_groups,
            "batch_host_transfer_rows": self.batch_host_transfer_rows,
            "per_row_batch_host_sync": 0,
        }


class FrozenReplayRelationalCalibratedProphecy(
    ReplayRelationalCalibratedProphecy
):
    """Frozen same-transition calibration with batched holdout refreshes."""

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
    """Transitional current wrapper retained for compatibility tests only."""

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
