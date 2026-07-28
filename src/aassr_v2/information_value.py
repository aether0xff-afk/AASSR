from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InformationValueWeights:
    uncertainty_reduction: float = 1.0
    prediction_gain: float = 1.0
    unlocked_action_value: float = 1.0
    goal_progress_gain: float = 1.0
    repeat_penalty: float = 1.0
    error_penalty: float = 1.0


@dataclass(frozen=True, slots=True)
class InformationValueBreakdown:
    uncertainty_reduction: float
    prediction_gain: float
    unlocked_action_value: float
    goal_progress_gain: float
    repeat_penalty: float = 0.0
    error_penalty: float = 0.0

    def total(self, weights: InformationValueWeights | None = None) -> float:
        weights = weights or InformationValueWeights()
        return (
            weights.uncertainty_reduction * self.uncertainty_reduction
            + weights.prediction_gain * self.prediction_gain
            + weights.unlocked_action_value * self.unlocked_action_value
            + weights.goal_progress_gain * self.goal_progress_gain
            - weights.repeat_penalty * self.repeat_penalty
            - weights.error_penalty * self.error_penalty
        )


def information_value_from_measurements(
    *,
    uncertainty_before: float,
    uncertainty_after: float,
    prediction_score_before: float,
    prediction_score_after: float,
    unlocked_action_value: float,
    goal_progress_before: float,
    goal_progress_after: float,
    repeat_penalty: float = 0.0,
    error_penalty: float = 0.0,
) -> InformationValueBreakdown:
    """Build a transparent information-value score from measured changes.

    Prediction gain should normally be evaluated on a recent replay or holdout
    set, not only on the transition just used for learning. That prevents a
    model from receiving reward merely for memorizing the latest transition.
    """

    return InformationValueBreakdown(
        uncertainty_reduction=uncertainty_before - uncertainty_after,
        prediction_gain=prediction_score_after - prediction_score_before,
        unlocked_action_value=unlocked_action_value,
        goal_progress_gain=goal_progress_after - goal_progress_before,
        repeat_penalty=repeat_penalty,
        error_penalty=error_penalty,
    )
