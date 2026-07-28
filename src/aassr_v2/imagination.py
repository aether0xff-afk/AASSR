from __future__ import annotations

from dataclasses import dataclass

from .metrics import imagination_uncertainty
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class ImaginationBatch:
    """Predictions for exactly one state-action pair."""

    state: StateSnapshot
    action: Action
    predictions: tuple[Prediction, ...]

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("an imagination batch needs at least one prediction")

    @property
    def uncertainty(self) -> float:
        return imagination_uncertainty(
            prediction.next_state.vector for prediction in self.predictions
        )
