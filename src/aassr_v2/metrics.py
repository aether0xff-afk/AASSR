from __future__ import annotations

from math import sqrt
from statistics import fmean
from typing import Iterable, Sequence

from .types import Prediction


Vector = Sequence[float]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    if not left:
        raise ValueError("vectors must not be empty")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    similarity = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def mean_pairwise_cosine(vectors: Iterable[Vector]) -> float:
    items = tuple(vectors)
    if not items:
        raise ValueError("at least one vector is required")
    if len(items) == 1:
        return 1.0

    similarities = [
        cosine_similarity(items[index], items[other])
        for index in range(len(items))
        for other in range(index + 1, len(items))
    ]
    return fmean(similarities)


def imagination_uncertainty(vectors: Iterable[Vector]) -> float:
    """Uncertainty among predictions of the same state-action pair.

    0 means the imagined next states agree. Values near 1 mean disagreement.
    Negative cosine similarities can produce values above 1, so the metric is
    bounded to [0, 2].
    """

    uncertainty = 1.0 - mean_pairwise_cosine(vectors)
    return max(0.0, min(2.0, uncertainty))


def uncertainty_reduction(before: float, after: float) -> float:
    """Positive when a knowledge update makes imagined futures agree more."""

    return before - after


def prediction_similarity(predicted: Vector, actual: Vector) -> float:
    """Similarity between Prophecy's next-state prediction and reality."""

    return cosine_similarity(predicted, actual)


def expected_prediction_vector(
    predictions: Iterable[Prediction],
) -> tuple[float, ...]:
    """Return the probability-weighted expected next-state vector."""

    items = tuple(predictions)
    if not items:
        raise ValueError("at least one prediction is required")

    vector_size = len(items[0].next_state.vector)
    if any(len(item.next_state.vector) != vector_size for item in items):
        raise ValueError("all prediction vectors must have the same length")

    probability_sum = sum(item.probability for item in items)
    if probability_sum > 0.0:
        weights = tuple(item.probability / probability_sum for item in items)
    else:
        weights = tuple(1.0 / len(items) for _ in items)

    return tuple(
        sum(
            weight * prediction.next_state.vector[index]
            for weight, prediction in zip(weights, items, strict=True)
        )
        for index in range(vector_size)
    )
