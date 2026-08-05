from __future__ import annotations

import io
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any, Sequence

from .autonomous_agent_core import ActionDecision
from .baseline_efficiency_benchmark import (
    CHOICE_ACTIONS,
    AgentDecision,
    BenchmarkGridPushWorld,
    encode_gridpush_state,
)
from .bottleneck_sota_diagnostic import (
    BenchmarkOracleProphecy,
    HybridDQNImaginationAgent,
    _current_scorer,
    remaining_oracle_steps,
)
from .metrics import (
    structured_prediction_components,
    structured_prediction_similarity,
)
from .neural_delta_prophecy import (
    NeuralDeltaConfig,
    NeuralDeltaProphecy,
    StateCodec,
)
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class BenchmarkGridPushCodec(StateCodec):
    """Representation codec only; contains no transition or reward rules."""

    @property
    def dimension(self) -> int:
        return 25

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        return encode_gridpush_state(state)

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != self.dimension:
            raise ValueError("benchmark neural state must have 25 values")
        values = [self._bounded(value) for value in encoded]
        # This is only a representation schema. It does not contain a movement,
        # object-order, success-path, or reward rule.
        for index in range(12):
            values[index] = round(values[index] * 2.0) / 2.0
        phase = min(6, max(0, int(round(values[12] * 6.0))))
        values[12] = phase / 6.0
        for index in range(13, 16):
            values[index] = float(values[index] >= 0.5)
        for index in range(16, 25):
            values[index] = float(values[index] >= 0.5)

        facts = {f"phase:{phase}"}
        for index, occupied in enumerate(values[16:25]):
            if occupied >= 0.5:
                x = index % BenchmarkGridPushWorld.grid_size
                y = index // BenchmarkGridPushWorld.grid_size
                facts.add(f"used:{x}:{y}")
        if values[13] >= 0.5:
            facts.add("bridge_built")
        if values[14] >= 0.5:
            facts.add("key_held")
        if values[15] >= 0.5:
            facts.add("door_open")

        if terminal_class == 1:
            facts.add("success")
        elif terminal_class == 2:
            facts.add("failed")
        available_actions = () if terminal_class else CHOICE_ACTIONS
        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "imagined_neural_delta": True,
                "imagined_neural_delta_source": source,
            }
        )
        return StateSnapshot(
            vector=tuple(float(value) for value in values[:16]),
            facts=frozenset(facts),
            available_actions=available_actions,
            goal_progress=1.0 if terminal_class == 1 else 0.0,
            metadata=metadata,
        )


class EmpiricallyCalibratedProphecy:
    """Scale model confidence by action-specific frozen-holdout accuracy.

    Ensemble agreement only detects disagreement inside the ensemble. All models
    can agree on the same wrong transition. This wrapper therefore evaluates the
    base Prophecy on transitions that the agent deliberately did not train on and
    multiplies imagined path probability by that empirical score.
    """

    name = "empirically-calibrated-neural-delta"

    def __init__(
        self,
        base: NeuralDeltaProphecy,
        holdout: Any,
        *,
        minimum_count: int = 8,
        evaluation_limit: int = 64,
        refresh_stride: int = 32,
        calibration_power: float = 1.0,
    ) -> None:
        self.base = base
        self.holdout = holdout
        self.minimum_count = int(minimum_count)
        self.evaluation_limit = int(evaluation_limit)
        self.refresh_stride = int(refresh_stride)
        self.calibration_power = float(calibration_power)
        self._cache: dict[tuple[int, str], float] = {}
        self._calibration_calls = Counter()

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.base.learn(state, action, actual_next_state)

    def _calibration(self, action: Action) -> float:
        items = [
            item
            for item in getattr(self.holdout, "_items", ())
            if item.action.signature == action.signature
        ]
        bucket = len(items) // self.refresh_stride
        key = (bucket, action.signature)
        if key in self._cache:
            return self._cache[key]
        self._calibration_calls["refreshes"] += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            selected = items[-self.evaluation_limit :]
            scores = []
            for item in selected:
                predictions = self.base.predict(
                    item.before,
                    item.action,
                    samples=1,
                )
                scores.append(
                    structured_prediction_similarity(
                        predictions,
                        item.after,
                    )
                )
            value = fmean(scores) if scores else 0.0
            value = value ** self.calibration_power
        value = max(0.0, min(1.0, value))
        self._cache[key] = value
        return value

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        calibration = self._calibration(action)
        predictions = self.base.predict(state, action, samples=samples)
        return tuple(
            Prediction(
                next_state=item.next_state,
                probability=item.probability * calibration,
                source=f"{item.source}:holdout-calibrated",
            )
            for item in predictions
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        raw = self.base.confidence(state, action)
        return min(raw, self._calibration(action))

    def coverage(
        self,
        state: StateSnapshot,
        actions: Sequence[Action],
    ) -> float:
        del state
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return fmean(
            min(
                self.base.confidence(
                    StateSnapshot(
                        (),
                        frozenset(),
                        (),
                    ),
                    action,
                )
                if False
                else 1.0,
                self._calibration(action),
            )
            for action in materialized
        )

    def action_calibrations(self) -> dict[str, float]:
        return {
            action.signature: self._calibration(action)
            for action in CHOICE_ACTIONS
        }

    def diagnostics(self) -> dict[str, int | float]:
        values = self.action_calibrations()
        return {
            **dict(self._calibration_calls),
            "calibration_mean": fmean(values.values()) if values else 0.0,
            "calibration_min": min(values.values()) if values else 0.0,
            "calibration_max": max(values.values()) if values else 0.0,
            **{
                f"calibration:{signature}": value
                for signature, value in values.items()
            },
        }


class NeuralProphecyHybridAgent(HybridDQNImaginationAgent):
    """Identity-preserving DQN Policy + learned Prophecy + Imagination."""

    def __init__(
        self,
        seed: int,
        *,
        train_episodes: int,
        adaptive: bool = False,
        calibrated: bool = False,
        conservative: bool = False,
    ) -> None:
        prophecy = NeuralDeltaProphecy(
            BenchmarkGridPushCodec(),
            config=NeuralDeltaConfig(
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=seed ^ 0x4E455552,
        )
        if conservative:
            name = "hybrid_neural_prophecy_calibrated_conservative"
        elif calibrated:
            name = "hybrid_neural_prophecy_calibrated"
        elif adaptive:
            name = "hybrid_neural_prophecy_adaptive"
        else:
            name = "hybrid_neural_prophecy"
        super().__init__(
            seed,
            train_episodes=train_episodes,
            prophecy=prophecy,
            scorer=_current_scorer(),
            depth=5,
            branching_factor=2,
            beam_width=16,
            outcome_samples=1,
            imagination_interval=1,
            use_effect_composition=False,
            name=name,
        )
        self.calibrated = bool(calibrated or conservative)
        self.conservative = bool(conservative)
        if self.calibrated:
            calibrated_prophecy = EmpiricallyCalibratedProphecy(
                prophecy,
                self.agent.holdout,
                calibration_power=1.35 if conservative else 1.0,
            )
            self.agent.prophecy = calibrated_prophecy
            self.agent.planner.prophecy = calibrated_prophecy
            self.agent.config = replace(
                self.agent.config,
                imagination_minimum_coverage=(
                    0.70 if conservative else 0.45
                ),
                imagination_intervention_margin=(
                    0.15 if conservative else 0.05
                ),
                imagination_uncertainty_margin=(
                    1.25 if conservative else 0.75
                ),
            )
        else:
            self.agent.config = replace(
                self.agent.config,
                imagination_minimum_coverage=0.30,
            )
        self.adaptive = bool(adaptive)
        self._original_depth = self.agent.planner.config.maximum_depth
        self._audit_counts: Counter[str] = Counter()
        self._audit_sums: defaultdict[str, float] = defaultdict(float)
        self._audit_decisions = 0
        self._oracle = BenchmarkOracleProphecy()

    @staticmethod
    def _distance_rank(value: int | None) -> float:
        return float("inf") if value is None else float(value)

    def _audit_intervention(
        self,
        state: StateSnapshot,
        decision: ActionDecision,
        *,
        training: bool,
    ) -> None:
        phase = "train" if training else "eval"
        if decision.used_imagination:
            self._audit_counts[f"{phase}:imagination_runs"] += 1
        if not decision.imagination_changed_action:
            return
        self._audit_counts[f"{phase}:interventions"] += 1
        policy_action = next(
            (
                action
                for action in state.available_actions
                if action.signature == decision.policy_action_signature
            ),
            None,
        )
        if policy_action is None:
            self._audit_counts[f"{phase}:unresolved_policy_action"] += 1
            return
        policy_next = self._oracle.predict(
            state,
            policy_action,
            samples=1,
        )[0].next_state
        imagined_next = self._oracle.predict(
            state,
            decision.action,
            samples=1,
        )[0].next_state
        policy_rank = self._distance_rank(remaining_oracle_steps(policy_next))
        imagined_rank = self._distance_rank(
            remaining_oracle_steps(imagined_next)
        )
        if imagined_rank < policy_rank:
            outcome = "correction"
        elif imagined_rank > policy_rank:
            outcome = "harm"
        else:
            outcome = "neutral"
        self._audit_counts[f"{phase}:{outcome}"] += 1

    def _audit_prediction_accuracy(
        self,
        state: StateSnapshot,
        *,
        training: bool,
    ) -> None:
        self._audit_decisions += 1
        if training and self._audit_decisions % 32:
            return
        prophecy = self.agent.base_prophecy
        stats = prophecy.stats()
        if stats.observations < prophecy.config.warmup_steps:
            return
        phase = "train" if training else "eval"
        for action in state.available_actions:
            predictions = prophecy.predict(state, action, samples=1)
            actual = self._oracle.predict(state, action, samples=1)[0].next_state
            components = structured_prediction_components(
                predictions,
                actual,
            )
            overall = structured_prediction_similarity(predictions, actual)
            self._audit_counts[f"{phase}:prediction_count"] += 1
            self._audit_sums[f"{phase}:prediction_overall"] += overall
            for name, value in components.items():
                self._audit_sums[f"{phase}:prediction_{name}"] += value

    def _decide(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        decision: ActionDecision = self.agent.select_action(
            state,
            episode=episode,
            explore=training,
        )
        self._audit_intervention(state, decision, training=training)
        self._audit_prediction_accuracy(state, training=training)
        return AgentDecision(
            action=decision.action,
            imagined_nodes=decision.imagined_nodes,
            used_imagination=decision.used_imagination,
        )

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        if not self.adaptive:
            return self._decide(
                state,
                episode=episode,
                training=training,
            )

        # This first adaptive prototype is kept as an ablation. The expanded
        # pilot showed that confidence-only depth reduction loses too much task
        # performance, so it is not the recommended implementation.
        coverage = self.agent.model_coverage(state)
        if coverage < 0.30:
            depth = 1
        elif coverage < 0.55:
            depth = 2
        elif coverage < 0.75:
            depth = 3
        else:
            depth = self._original_depth
        original = self.agent.planner.config
        self.agent.planner.config = replace(original, maximum_depth=depth)
        try:
            return self._decide(
                state,
                episode=episode,
                training=training,
            )
        finally:
            self.agent.planner.config = original

    def diagnostics(self) -> dict[str, int | float | dict[str, float]]:
        result: dict[str, int | float | dict[str, float]] = {
            **dict(self._audit_counts),
            **{
                f"sum:{name}": value
                for name, value in self._audit_sums.items()
            },
            "imagination": self.agent.imagination_diagnostics(),
        }
        for phase in ("train", "eval"):
            count = self._audit_counts[f"{phase}:prediction_count"]
            if count:
                result[f"{phase}:prediction_overall_mean"] = (
                    self._audit_sums[f"{phase}:prediction_overall"] / count
                )
                for component in (
                    "vector",
                    "facts",
                    "actions",
                    "goal",
                    "terminal",
                ):
                    result[f"{phase}:prediction_{component}_mean"] = (
                        self._audit_sums[
                            f"{phase}:prediction_{component}"
                        ]
                        / count
                    )
            interventions = self._audit_counts[f"{phase}:interventions"]
            if interventions:
                result[f"{phase}:correction_rate"] = (
                    self._audit_counts[f"{phase}:correction"]
                    / interventions
                )
                result[f"{phase}:harm_rate"] = (
                    self._audit_counts[f"{phase}:harm"] / interventions
                )
        calibrated = self.agent.prophecy
        diagnostics = getattr(calibrated, "diagnostics", None)
        if callable(diagnostics):
            result["calibration"] = dict(diagnostics())
        return result

    def model_stats(self) -> dict[str, int | float]:
        stats = dict(super().model_stats())
        prophecy = self.agent.base_prophecy
        neural_stats = prophecy.stats()
        stats.update(
            {
                "prophecy_observations": neural_stats.observations,
                "prophecy_gradient_updates": neural_stats.gradient_updates,
                "prophecy_replay_size": neural_stats.replay_size,
                "prophecy_mean_training_loss": neural_stats.mean_training_loss,
                "prophecy_last_ensemble_variance": (
                    neural_stats.last_ensemble_variance
                ),
            }
        )
        stats["model_units"] = int(stats.get("model_units", 0)) + int(
            neural_stats.parameter_count
        )
        handle = io.BytesIO()
        self.dqn.torch.save(
            [model.state_dict() for model in prophecy.models],
            handle,
        )
        stats["model_bytes"] = int(stats.get("model_bytes", 0)) + len(
            handle.getvalue()
        )
        return stats