from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class ProphecyStep:
    """One branch-local recurrent prediction step."""

    predictions: tuple[Prediction, ...]
    memory: Any = None

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("a prophecy step needs at least one prediction")


class ProphecyModel(Protocol):
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
    """Optional interface for RNN/GRU/LSTM implementations."""

    def initial_memory(self) -> Any: ...

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep: ...


class ContextualProphecyModel(Protocol):
    """Optional interface used to separate KK-context and model updates."""

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: object,
        samples: int,
    ) -> tuple[Prediction, ...]: ...
