from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Iterable
from statistics import fmean

from ..neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy
from ..types import Action, Prediction, StateSnapshot
from .representation import SchemaDrivenRepresentation, SchemaDrivenStateCodec


class SchemaDrivenNeuralProphecy(NeuralDeltaProphecy):
    """Neural world model whose representation is owned by the Core."""

    name = "core-schema-driven-neural-prophecy-v1"

    def __init__(
        self,
        representation: SchemaDrivenRepresentation,
        *,
        config: NeuralDeltaConfig | None = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.representation = representation
        super().__init__(
            SchemaDrivenStateCodec(representation),
            config=config
            or NeuralDeltaConfig(
                action_feature_size=representation.action_feature_size,
            ),
            seed=int(seed),
        )
        self.device = self.torch.device(device)
        for model in self.models:
            model.to(self.device)

    def _tensor(self, values: Any, *, dtype: Any | None = None) -> Any:
        return self.torch.as_tensor(
            values,
            dtype=dtype or self.torch.float32,
            device=self.device,
        )

    def _input(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        return (
            self.representation.state_vector(state)
            + self.representation.action_features(state, action)
        )

    @staticmethod
    def _terminal_class(state: StateSnapshot) -> int:
        """Generic lifecycle class: active / terminated / truncated."""

        if bool(state.metadata.get("core_truncated", False)):
            return 2
        if bool(state.metadata.get("core_terminated", False)):
            return 1
        return 0

    @property
    def training_stats(self) -> Any:
        return SimpleNamespace(updates=int(self.gradient_updates))


class CoreHoldoutCalibratedProphecy:
    """Generic real-transition calibration over the Core representation.

    The wrapper uses the AutonomousLearningAgent's frozen holdout partition. It
    never asks a plugin how to score predictions and never assigns meaning to a
    particular observation value.
    """

    name = "core-holdout-calibrated-prophecy-v1"

    def __init__(
        self,
        base: SchemaDrivenNeuralProphecy,
        holdout: object,
        representation: SchemaDrivenRepresentation,
        *,
        minimum_count: int = 4,
        evaluation_limit: int = 32,
        refresh_stride: int = 8,
    ) -> None:
        self.base = base
        self.holdout = holdout
        self.representation = representation
        self.minimum_count = int(minimum_count)
        self.evaluation_limit = int(evaluation_limit)
        self.refresh_stride = int(refresh_stride)
        self._cache: dict[tuple[tuple[float, ...], int, int], float] = {}
        self.refreshes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def training_stats(self) -> Any:
        return SimpleNamespace(updates=int(self.base.gradient_updates))

    def _key(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        return self.representation.action_structure(state, action)

    @staticmethod
    def _lifecycle(state: StateSnapshot) -> int:
        if bool(state.metadata.get("core_truncated", False)):
            return 2
        if bool(state.metadata.get("core_terminated", False)):
            return 1
        return 0

    def _items(self, state: StateSnapshot, action: Action) -> tuple[Any, ...]:
        key = self._key(state, action)
        rows = tuple(getattr(self.holdout, "_items", ()))
        return tuple(
            item
            for item in rows
            if self._key(item.before, item.action) == key
        )

    def _calibration(self, state: StateSnapshot, action: Action) -> float:
        items = self._items(state, action)
        revision = int(self.base.gradient_updates)
        cache_key = (
            self._key(state, action),
            len(items) // max(1, self.refresh_stride),
            revision // max(1, self.refresh_stride),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        self.refreshes += 1
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            scores = []
            for item in items[-self.evaluation_limit :]:
                prediction = self.base.predict(item.before, item.action, samples=1)[0]
                predicted = prediction.next_state
                vector_error = fmean(
                    abs(left - right)
                    for left, right in zip(
                        self.representation.state_vector(predicted),
                        self.representation.state_vector(item.after),
                        strict=True,
                    )
                )
                lifecycle_match = float(
                    self._lifecycle(predicted) == self._lifecycle(item.after)
                )
                scores.append(max(0.0, 1.0 - vector_error) * lifecycle_match)
            value = max(0.0, min(1.0, fmean(scores) if scores else 0.0))
        self._cache[cache_key] = value
        return value

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.base.learn(state, action, actual_next_state)

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        calibration = self._calibration(state, action)
        return tuple(
            replace(
                prediction,
                probability=float(prediction.probability) * calibration,
                source=f"{prediction.source}:core-calibrated",
            )
            for prediction in self.base.predict(state, action, samples=samples)
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        return min(
            float(self.base.confidence(state, action)),
            self._calibration(state, action),
        )

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        rows = tuple(actions)
        if not rows:
            return 1.0
        return fmean(self.confidence(state, action) for action in rows)

    def diagnostics(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "refreshes": self.refreshes,
            "cache_entries": len(self._cache),
            "minimum_count": self.minimum_count,
        }
