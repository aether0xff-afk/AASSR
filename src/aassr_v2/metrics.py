from __future__ import annotations

from math import sqrt
from statistics import fmean
from typing import Iterable, Sequence


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
