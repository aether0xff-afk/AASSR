from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Iterable

from .metrics import expected_prediction_vector, prediction_similarity
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    state: StateSnapshot
    action: Action
    next_state: StateSnapshot
    trace_id: str = ""


@dataclass(slots=True)
class ReplayBuffer:
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

    def add(self, transition: ReplayTransition) -> bool:
        """Return True when the sample belongs to the training partition."""

        self._seen += 1
        is_holdout = self._seen % self.holdout_stride == 0
        target = self._holdout if is_holdout else self._train
        target.append(transition)
        if len(target) > self.capacity:
            target.pop(0)
        return not is_holdout

    def train(self) -> tuple[ReplayTransition, ...]:
        return tuple(self._train)

    def holdout(self) -> tuple[ReplayTransition, ...]:
        return tuple(self._holdout)


@dataclass(frozen=True, slots=True)
class ValidationScore:
    count: int
    mean_similarity: float


def _prophecy_learning_revision(prophecy: object) -> tuple[tuple[str, int], ...]:
    """Cheap revision fingerprint containing learning state, not diagnostics."""

    seen: set[int] = set()
    values: list[tuple[str, int]] = []

    def visit(obj: Any, path: str) -> None:
        identity = id(obj)
        if identity in seen:
            return
        seen.add(identity)
        try:
            stats = getattr(obj, "training_stats", None)
        except Exception:
            stats = None
        if stats is not None and hasattr(stats, "updates"):
            values.append((f"{path}.updates", int(stats.updates)))
        try:
            effect_observations = getattr(obj, "effect_observations", None)
        except Exception:
            effect_observations = None
        if effect_observations is not None:
            values.append((f"{path}.effects", int(effect_observations)))
        for name in ("_prophecy", "_base", "base"):
            try:
                child = getattr(obj, name, None)
            except Exception:
                child = None
            if child is not None and child is not obj:
                visit(child, f"{path}.{name}")

    visit(prophecy, "root")
    return tuple(sorted(values))


class PredictionValidator:
    def __init__(
        self,
        *,
        samples: int = 4,
        recent_limit: int = 64,
    ) -> None:
        if samples <= 0 or recent_limit <= 0:
            raise ValueError(
                "samples and recent_limit must be positive"
            )
        self.samples = samples
        self.recent_limit = recent_limit
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_value: ValidationScore | None = None
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

        cache_key = (
            id(prophecy),
            self.samples,
            tuple(id(item) for item in selected),
            _prophecy_learning_revision(prophecy),
        )
        if cache_key == self._cache_key and self._cache_value is not None:
            self.cache_hits += 1
            return self._cache_value
        self.cache_misses += 1

        states = tuple(item.state for item in selected)
        actions = tuple(item.action for item in selected)
        expected_batch = getattr(prophecy, "expected_vector_batch", None)
        if callable(expected_batch):
            self.batch_calls += 1
            self.expected_vector_calls += 1
            expected_rows = expected_batch(
                states,
                actions,
                samples=self.samples,
            )
            if len(expected_rows) != len(selected):
                raise RuntimeError("Prophecy expected_vector_batch returned wrong row count")
            scores = [
                prediction_similarity(expected, transition.next_state.vector)
                for transition, expected in zip(
                    selected, expected_rows, strict=True
                )
            ]
        else:
            batch_predict = getattr(prophecy, "predict_batch", None)
            if callable(batch_predict):
                self.batch_calls += 1
                prediction_rows = batch_predict(
                    states,
                    actions,
                    samples=self.samples,
                )
                if len(prediction_rows) != len(selected):
                    raise RuntimeError("Prophecy predict_batch returned wrong row count")
                scores = [
                    prediction_similarity(
                        expected_prediction_vector(predictions),
                        transition.next_state.vector,
                    )
                    for transition, predictions in zip(
                        selected, prediction_rows, strict=True
                    )
                ]
            else:
                scores = []
                for transition in selected:
                    predictions = prophecy.predict(
                        transition.state,
                        transition.action,
                        samples=self.samples,
                    )
                    scores.append(
                        prediction_similarity(
                            expected_prediction_vector(predictions),
                            transition.next_state.vector,
                        )
                    )
        result = ValidationScore(len(scores), fmean(scores))
        self._cache_key = cache_key
        self._cache_value = result
        return result

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batch_calls": self.batch_calls,
            "expected_vector_calls": self.expected_vector_calls,
        }
