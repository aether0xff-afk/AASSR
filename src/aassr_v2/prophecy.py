from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class ProphecyStep:
    """One branch-local recurrent prediction step.

    ``memory`` is intentionally opaque. A GRU/LSTM implementation may place its
    hidden state here, while the tabular baseline simply keeps it as ``None``.
    """

    predictions: tuple[Prediction, ...]
    memory: Any = None

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("a prophecy step needs at least one prediction")


class ProphecyModel(Protocol):
    """Predict possible next states for one state-action pair."""

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


class RecurrentProphecyModel(Protocol):
    """Optional interface for RNN/GRU/LSTM Prophecy implementations."""

    def initial_memory(self) -> Any: ...

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep: ...
