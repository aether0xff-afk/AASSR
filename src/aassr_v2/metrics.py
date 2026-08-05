from __future__ import annotations

from math import exp, log, sqrt
from statistics import fmean
from typing import Iterable, Sequence

from .types import Prediction, StateSnapshot


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
    """Cosine similarity retained for backward-compatible vector metrics."""

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


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _terminal_class(state: StateSnapshot) -> str:
    if state.available_actions:
        return "active"
    if state.goal_progress >= 1.0 or "success" in state.facts:
        return "success"
    return "failure"


def structured_prediction_components(
    predictions: Iterable[Prediction],
    actual: StateSnapshot,
) -> dict[str, float]:
    """Measure whether a Prophecy prediction preserves actionable structure.

    Cosine similarity alone can be near one even when a model predicts the
    wrong terminal status, used facts, or available actions. This metric keeps
    the numeric vector term but also scores the explicit state interface that
    Policy and Imagination actually consume.
    """

    items = tuple(predictions)
    if not items:
        raise ValueError("at least one prediction is required")
    expected = expected_prediction_vector(items)
    if len(expected) != len(actual.vector):
        raise ValueError("predicted and actual vectors must have equal length")
    mae = fmean(
        abs(left - right)
        for left, right in zip(expected, actual.vector, strict=True)
    )
    vector_score = exp(-4.0 * mae)
    representative = max(
        items,
        key=lambda item: (item.probability, item.source),
    ).next_state
    action_score = _jaccard(
        (action.signature for action in representative.available_actions),
        (action.signature for action in actual.available_actions),
    )
    facts_score = _jaccard(representative.facts, actual.facts)
    goal_score = exp(
        -4.0 * abs(representative.goal_progress - actual.goal_progress)
    )
    terminal_score = float(
        _terminal_class(representative) == _terminal_class(actual)
    )
    return {
        "vector": max(0.0, min(1.0, vector_score)),
        "facts": max(0.0, min(1.0, facts_score)),
        "actions": max(0.0, min(1.0, action_score)),
        "goal": max(0.0, min(1.0, goal_score)),
        "terminal": terminal_score,
    }


def structured_prediction_similarity(
    predictions: Iterable[Prediction],
    actual: StateSnapshot,
) -> float:
    """Weighted geometric mean of explicit transition fidelity.

    The geometric mean deliberately penalizes a prediction that is numerically
    close but structurally unusable. It is generic to ``StateSnapshot`` and does
    not contain benchmark-specific transition or reward knowledge.
    """

    components = structured_prediction_components(predictions, actual)
    weights = {
        "vector": 0.35,
        "facts": 0.25,
        "actions": 0.15,
        "goal": 0.10,
        "terminal": 0.15,
    }
    floor = 1e-6
    value = exp(
        sum(
            weights[name] * log(max(floor, components[name]))
            for name in weights
        )
    )
    return max(0.0, min(1.0, value))
