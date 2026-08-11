from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Any, Iterable, Sequence

from .current_generation import relational_action_key
from .current_relational_codec import semantic_prediction_score
from .current_relational_model import RelationalStochasticProphecy
from .prophecy import ProphecyStep
from .replay import ReplayBuffer, ReplayTransition, ValidationScore
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


class SemanticPredictionValidator:
    def __init__(self, *, samples: int = 3, recent_limit: int = 64) -> None:
        self.samples = int(samples)
        self.recent_limit = int(recent_limit)
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_calls = 0
        self.expected_vector_calls = 0

    def evaluate(
        self,
        prophecy: object,
        transitions: Iterable[ReplayTransition],
    ) -> ValidationScore:
        selected = tuple(transitions)[-self.recent_limit :]
        if not selected:
            return ValidationScore(0, 0.0)
        self.cache_misses += 1
        rows = prophecy.predict_batch(
            tuple(item.state for item in selected),
            tuple(item.action for item in selected),
            samples=self.samples,
        )
        self.batch_calls += 1
        scores = [
            semantic_prediction_score(predictions, item.next_state)
            for item, predictions in zip(selected, rows, strict=True)
        ]
        return ValidationScore(len(scores), fmean(scores))

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batch_calls": self.batch_calls,
            "expected_vector_calls": self.expected_vector_calls,
        }


class SemanticCalibratedProphecy:
    """Frozen holdout calibration on semantic state/action/terminal correctness."""

    name = "current-semantic-holdout-calibrated-prophecy-v1"

    def __init__(
        self,
        base: RelationalStochasticProphecy,
        replay: ReplayBuffer,
        *,
        minimum_count: int = 8,
        evaluation_limit: int = 48,
        refresh_stride: int = 32,
    ) -> None:
        self.base = base
        self.replay = replay
        self.minimum_count = int(minimum_count)
        self.evaluation_limit = int(evaluation_limit)
        self.refresh_stride = int(refresh_stride)
        self._frozen_holdout: tuple[ReplayTransition, ...] | None = None
        self._cache: dict[tuple[Any, int, int], float] = {}
        self.refreshes = 0
        self.freeze_count = 0

    @property
    def training_stats(self) -> Any:
        return self.base.training_stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def freeze_holdout(self, items: tuple[ReplayTransition, ...]) -> None:
        self._frozen_holdout = tuple(items)
        self.freeze_count += 1

    def release_holdout(self) -> None:
        self._frozen_holdout = None

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.base.learn(state, action, actual_next_state)

    def _calibration(self, state: StateSnapshot, action: Action) -> float:
        key = relational_action_key(state, action)
        source = (
            self._frozen_holdout
            if self._frozen_holdout is not None
            else self.replay.holdout()
        )
        items = tuple(
            item
            for item in source
            if relational_action_key(item.state, item.action) == key
        )
        revision = int(self.base.gradient_updates)
        cache_key = (
            key,
            len(items) // max(1, self.refresh_stride),
            revision // max(1, self.refresh_stride),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.refreshes += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            selected = items[-self.evaluation_limit :]
            rows = self.base.predict_batch(
                tuple(item.state for item in selected),
                tuple(item.action for item in selected),
                samples=self.base.config.ensemble_size,
            )
            value = fmean(
                semantic_prediction_score(predictions, item.next_state)
                for item, predictions in zip(selected, rows, strict=True)
            )
            value = max(0.0, min(1.0, value))
        self._cache[cache_key] = value
        return value

    def _calibrated(
        self,
        state: StateSnapshot,
        action: Action,
        predictions: Sequence[Prediction],
    ) -> tuple[Prediction, ...]:
        calibration = self._calibration(state, action)
        return tuple(
            replace(
                prediction,
                probability=max(
                    0.0,
                    min(1.0, float(prediction.probability) * calibration),
                ),
                source=f"{prediction.source}:semantic-calibrated",
            )
            for prediction in predictions
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self._calibrated(
            state, action, self.base.predict(state, action, samples=samples)
        )

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        rows = self.base.predict_batch(states, actions, samples=samples)
        return tuple(
            self._calibrated(state, action, predictions)
            for state, action, predictions in zip(states, actions, rows, strict=True)
        )

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: object,
        samples: int,
    ) -> tuple[Prediction, ...]:
        # Public response facts already live in state. Re-unioning concrete
        # Knowledge into canonical imagined IDs would double-count renamed entities.
        del knowledge
        return self.predict(state, action, samples=samples)

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        # Same reliability contract as the Prediction path: learned ensemble
        # confidence multiplied by semantic holdout calibration. Outcome mass is
        # deliberately separate and never enters this gate.
        return max(
            0.0,
            min(
                1.0,
                float(self.base.confidence(state, action))
                * self._calibration(state, action),
            ),
        )

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return fmean(self.confidence(state, action) for action in materialized)

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            **self.base.diagnostics(),
            "calibration": self.name,
            "calibration_refreshes": self.refreshes,
            "calibration_cache_entries": len(self._cache),
            "holdout_freezes": self.freeze_count,
            "semantic_calibration": 1,
            "calibrated_reliability_product": 1,
            "outcome_probability_preserved": 1,
        }


class RelationalDepthBatchedProphecyView:
    """Batch primitive outcomes without concrete Knowledge re-injection."""

    name = "current-relational-stochastic-depth-batched-v1"

    def __init__(self, agent: object) -> None:
        self.agent = agent
        self._batch_calls = 0
        self._batch_rows = 0
        self._skill_fallback_rows = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent.prophecy, name)

    def predict_step_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        memories: Sequence[Any],
        *,
        samples: int,
    ) -> tuple[ProphecyStep, ...]:
        if not (len(states) == len(actions) == len(memories)):
            raise ValueError("states/actions/memories batch length mismatch")
        results: list[ProphecyStep | None] = [None] * len(states)
        primitive = [
            i for i, action in enumerate(actions) if action.verb_name != SKILL_VERB
        ]
        if primitive:
            rows = self.agent.calibrated_prophecy.predict_batch(
                tuple(states[i] for i in primitive),
                tuple(actions[i] for i in primitive),
                samples=samples,
            )
            self._batch_calls += 1
            self._batch_rows += len(primitive)
            for output_index, predictions in zip(primitive, rows, strict=True):
                results[output_index] = ProphecyStep(
                    tuple(
                        replace(
                            prediction,
                            next_state=self.agent.skills.augment_state(
                                prediction.next_state
                            ),
                        )
                        for prediction in predictions
                    ),
                    memories[output_index],
                )
        for index, (state, action, memory) in enumerate(
            zip(states, actions, memories, strict=True)
        ):
            if results[index] is not None:
                continue
            self._skill_fallback_rows += 1
            results[index] = self.agent.skill_prophecy.predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )
        if any(item is None for item in results):
            raise RuntimeError("relational batched Prophecy left an unresolved row")
        return tuple(item for item in results if item is not None)

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "current_imagination_batch_calls": self._batch_calls,
            "current_imagination_batch_rows": self._batch_rows,
            "current_imagination_skill_fallback_rows": self._skill_fallback_rows,
        }
