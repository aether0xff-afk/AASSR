from __future__ import annotations

from dataclasses import replace
from math import exp
from statistics import fmean
from typing import Any, Iterable, Sequence

from .current_generation import relational_action_key
from .current_relational_codec import legal_action_mask, semantic_prediction_score
from .current_relational_model import RelationalStochasticProphecy
from .current_relational_state import (
    ACTION_COUNT_INDEX,
    CONTROL_SIZE,
    KNOWN_OBJECT_INDEX,
    KNOWN_PROFILE_INDEX,
    KNOWN_ROUTE_INDEX,
    OBJECT_REQUEST_FRACTION_INDEX,
    REQUEST_FRACTION_INDEX,
    ROLE_START_INDEX,
    UNIQUE_ACTION_COUNT_INDEX,
    WORKFLOW_PROGRESS_INDEX,
)
from .current_relational_state_v3 import (
    latest_status_code,
    relational_state_descriptor_v3,
    relational_state_key_v3,
)
from .pentest_curriculum_env import PROFILE_RELATIONS, ROUTE_RELATIONS
from .prophecy import ProphecyStep
from .replay import ReplayBuffer, ReplayTransition, ValidationScore
from .skills import SKILL_VERB
from .types import Action, Prediction, StateSnapshot


CALIBRATION_LOCALITY_SCALE = 4.0
CALIBRATION_CACHE_LIMIT = 20_000


def probability_weighted_semantic_score(
    predictions: Sequence[Prediction],
    actual: StateSnapshot,
    *,
    prediction_score: Any | None = None,
) -> float:
    """Expected semantic correctness under the model's stochastic outcome mass."""
    materialized = tuple(predictions)
    if not materialized:
        return 0.0
    if prediction_score is None:
        from .current_relational_state_v3 import semantic_prediction_score_v3

        prediction_score = semantic_prediction_score_v3
    raw = [
        max(0.0, float(getattr(item, "outcome_probability", 1.0)))
        for item in materialized
    ]
    total = sum(raw)
    if total <= 1e-12:
        weights = [1.0 / len(materialized)] * len(materialized)
    else:
        weights = [value / total for value in raw]
    return sum(
        weight * prediction_score((prediction,), actual)
        for weight, prediction in zip(weights, materialized, strict=True)
    )


def _mask_jaccard_distance(left: StateSnapshot, right: StateSnapshot) -> float:
    left_mask = legal_action_mask(left)
    right_mask = legal_action_mask(right)
    a = {index for index, value in enumerate(left_mask) if float(value) >= 0.5}
    b = {index for index, value in enumerate(right_mask) if float(value) >= 0.5}
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)


def _relative_count_distance(left: float, right: float, *, scale: float) -> float:
    left_count = int(round(max(0.0, float(left)) * scale))
    right_count = int(round(max(0.0, float(right)) * scale))
    denominator = max(1, left_count, right_count)
    return abs(left_count - right_count) / denominator


def _public_state_distance(left: StateSnapshot, right: StateSnapshot) -> float:
    """Public structural locality without averaging rare differences away.

    Calibration is a reliability gate, not a value model. A holdout row should
    therefore count as local only when it resembles the current state in every
    decision-relevant public regime: controls, workflow progress, entity scale,
    semantic role distributions, object evidence, action-surface scale/composition,
    latest status, and legal structural slots. No curriculum level, hidden limit,
    target identity, or success path is used.
    """
    a = relational_state_descriptor_v3(left)
    b = relational_state_descriptor_v3(right)

    control_distance = max(
        (abs(a[index] - b[index]) for index in range(CONTROL_SIZE)),
        default=0.0,
    )
    workflow_distance = abs(
        a[WORKFLOW_PROGRESS_INDEX] - b[WORKFLOW_PROGRESS_INDEX]
    )
    entity_count_distance = max(
        _relative_count_distance(a[index], b[index], scale=32.0)
        for index in (KNOWN_ROUTE_INDEX, KNOWN_PROFILE_INDEX, KNOWN_OBJECT_INDEX)
    )

    route_end = ROLE_START_INDEX + len(ROUTE_RELATIONS)
    profile_end = route_end + len(PROFILE_RELATIONS)
    route_role_distance = 0.5 * sum(
        abs(a[index] - b[index]) for index in range(ROLE_START_INDEX, route_end)
    )
    profile_role_distance = 0.5 * sum(
        abs(a[index] - b[index]) for index in range(route_end, profile_end)
    )

    own_index = profile_end
    target_index = own_index + 1
    tried_index = target_index + 1
    object_distance = max(
        abs(a[own_index] - b[own_index]),
        abs(a[target_index] - b[target_index]),
        _relative_count_distance(a[tried_index], b[tried_index], scale=32.0),
    )

    action_scale_distance = max(
        _relative_count_distance(
            a[ACTION_COUNT_INDEX],
            b[ACTION_COUNT_INDEX],
            scale=128.0,
        ),
        _relative_count_distance(
            a[UNIQUE_ACTION_COUNT_INDEX],
            b[UNIQUE_ACTION_COUNT_INDEX],
            scale=32.0,
        ),
        abs(a[REQUEST_FRACTION_INDEX] - b[REQUEST_FRACTION_INDEX]),
        abs(
            a[OBJECT_REQUEST_FRACTION_INDEX]
            - b[OBJECT_REQUEST_FRACTION_INDEX]
        ),
    )
    status_distance = float(latest_status_code(left) != latest_status_code(right))
    legal_surface_distance = _mask_jaccard_distance(left, right)

    return max(
        control_distance,
        workflow_distance,
        entity_count_distance,
        route_role_distance,
        profile_role_distance,
        object_distance,
        action_scale_distance,
        status_distance,
        legal_surface_distance,
    )


class SemanticPredictionValidator:
    def __init__(
        self,
        *,
        samples: int = 3,
        recent_limit: int = 64,
        prediction_score: Any | None = None,
    ) -> None:
        self.samples = int(samples)
        self.recent_limit = int(recent_limit)
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_calls = 0
        self.expected_vector_calls = 0
        if prediction_score is None:
            from .current_relational_state_v3 import semantic_prediction_score_v3

            prediction_score = semantic_prediction_score_v3
        self.prediction_score = prediction_score

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
            probability_weighted_semantic_score(
                predictions,
                item.next_state,
                prediction_score=self.prediction_score,
            )
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
    """Frozen holdout reliability localized to the current public state regime."""

    name = "current-semantic-state-local-holdout-calibrated-prophecy-v3"

    def __init__(
        self,
        base: RelationalStochasticProphecy,
        replay: ReplayBuffer,
        *,
        minimum_count: int = 8,
        evaluation_limit: int = 48,
        refresh_stride: int = 32,
        representation: object | None = None,
    ) -> None:
        self.base = base
        self.replay = replay
        self.minimum_count = int(minimum_count)
        self.evaluation_limit = int(evaluation_limit)
        self.refresh_stride = int(refresh_stride)
        self.representation = representation
        self._frozen_holdout: tuple[ReplayTransition, ...] | None = None
        self._cache: dict[tuple[Any, ...], float] = {}
        self.refreshes = 0
        self.freeze_count = 0
        self.batch_refreshes = 0
        self.batch_rows = 0

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

    def _cache_store(self, key: tuple[Any, ...], value: float) -> None:
        if len(self._cache) >= CALIBRATION_CACHE_LIMIT and key not in self._cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def _calibration(self, state: StateSnapshot, action: Action) -> float:
        action_structure = (
            self.representation.action_structure
            if self.representation is not None
            else relational_action_key
        )
        state_key = (
            self.representation.state_key
            if self.representation is not None
            else relational_state_key_v3
        )
        prediction_score = (
            self.representation.prediction_score
            if self.representation is not None
            else semantic_prediction_score
        )
        action_key = action_structure(state, action)
        source = (
            self._frozen_holdout
            if self._frozen_holdout is not None
            else self.replay.holdout()
        )
        items = tuple(
            item
            for item in source
            if action_structure(item.state, item.action) == action_key
        )
        revision = int(self.base.gradient_updates)
        cache_key = (
            action_key,
            state_key(state),
            len(items) // max(1, self.refresh_stride),
            revision // max(1, self.refresh_stride),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        self.refreshes += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            nearest = sorted(
                ((_public_state_distance(state, item.state), item) for item in items),
                key=lambda row: row[0],
            )[: self.evaluation_limit]
            selected = tuple(item for _, item in nearest)
            locality = tuple(
                exp(-CALIBRATION_LOCALITY_SCALE * max(0.0, distance))
                for distance, _ in nearest
            )
            rows = self.base.predict_batch(
                tuple(item.state for item in selected),
                tuple(item.action for item in selected),
                samples=self.base.config.ensemble_size,
            )
            self.batch_refreshes += 1
            self.batch_rows += len(selected)
            scores = tuple(
                probability_weighted_semantic_score(
                    predictions,
                    item.next_state,
                    prediction_score=prediction_score,
                )
                for item, predictions in zip(selected, rows, strict=True)
            )
            total = sum(locality)
            if total <= 1e-12:
                value = 0.0
            else:
                local_score = sum(
                    weight * score
                    for weight, score in zip(locality, scores, strict=True)
                ) / total
                # Even a perfectly accurate far-away holdout cannot claim local
                # reliability. Exact/nearby support approaches a multiplier of 1.
                value = local_score * max(locality)
            value = max(0.0, min(1.0, value))
        self._cache_store(cache_key, value)
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
        del knowledge
        return self.predict(state, action, samples=samples)

    def confidence(self, state: StateSnapshot, action: Action) -> float:
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
            "calibration_cache_limit": CALIBRATION_CACHE_LIMIT,
            "calibration_batch_refreshes": self.batch_refreshes,
            "calibration_batch_rows": self.batch_rows,
            "calibration_refresh_batching": 1,
            "holdout_freezes": self.freeze_count,
            "semantic_calibration": 1,
            "state_local_calibration": 1,
            "calibration_locality_metric": "groupwise-public-structural-max-v2",
            "calibration_locality_scale": CALIBRATION_LOCALITY_SCALE,
            "probability_weighted_calibration": 1,
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
