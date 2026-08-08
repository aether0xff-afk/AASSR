from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Iterable

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

    def evaluate(
        self,
        prophecy: object,
        transitions: Iterable[ReplayTransition],
    ) -> ValidationScore:
        selected = tuple(transitions)[-self.recent_limit :]
        if not selected:
            return ValidationScore(0, 0.0)

        batch_predict = getattr(prophecy, "predict_batch", None)
        if callable(batch_predict):
            prediction_rows = batch_predict(
                tuple(item.state for item in selected),
                tuple(item.action for item in selected),
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
        return ValidationScore(
            len(scores),
            fmean(scores),
        )
