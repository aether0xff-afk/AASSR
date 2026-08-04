from __future__ import annotations

from math import log
from typing import Iterable


def outcome_consistency(counts: Iterable[int | float]) -> float:
    """Return 1 for deterministic outcomes and 0 for maximally split outcomes.

    The score is one minus normalized Shannon entropy over positive empirical
    outcome counts.  A transition seen many times is therefore not considered
    reliable when the same observable state-action pair leads to contradictory
    next states.
    """

    values = tuple(float(value) for value in counts if float(value) > 0.0)
    if not values:
        return 0.0
    if len(values) == 1:
        return 1.0
    total = sum(values)
    probabilities = tuple(value / total for value in values)
    entropy = -sum(probability * log(probability) for probability in probabilities)
    maximum_entropy = log(len(probabilities))
    if maximum_entropy <= 0.0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - entropy / maximum_entropy))


def empirical_confidence(
    counts: Iterable[int | float],
    *,
    prior_strength: float,
    tier: float = 1.0,
) -> float:
    """Combine sample amount, outcome consistency and representation tier."""

    if prior_strength < 0.0:
        raise ValueError("prior_strength must be non-negative")
    if not 0.0 <= tier <= 1.0:
        raise ValueError("tier must be in [0, 1]")
    values = tuple(float(value) for value in counts if float(value) > 0.0)
    total = sum(values)
    if total <= 0.0:
        return 0.0
    experience = total / (total + prior_strength)
    return min(1.0, max(0.0, tier * experience * outcome_consistency(values)))
