from __future__ import annotations

from typing import Protocol

from .types import Action, Prediction, StateSnapshot


class ProphecyModel(Protocol):
    """Predict multiple possible next states for one state-action pair."""

    @property
    def name(self) -> str: ...

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]: ...

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None: ...
