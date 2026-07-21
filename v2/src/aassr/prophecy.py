from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .knowledge import KK, KnowledgeDelta
from .policy import candidate_axes


@dataclass(frozen=True)
class ProphecyPrediction:
    kk_probs: dict[KK, float]
    error_prob: float
    flag_prob: float
    status_change_probs: dict[KK, float] = field(default_factory=dict)
    confidence: float = 1.0
    predicted_position_change: Any = None

    def expected_knowledge_gain(self) -> float:
        return sum(self.kk_probs.values())


@dataclass(frozen=True)
class ProphecyUpdate:
    prediction_error: float
    loss: float


@dataclass
class ProphecyStats:
    count: int = 0
    kk_counts: dict[KK, int] = field(default_factory=dict)
    error_count: int = 0
    flag_count: int = 0


class ProphecyModule(Protocol):
    """Common interface for Prophecy Module implementations."""

    def predict(self, state_signature: Any, candidate: Any) -> ProphecyPrediction:
        ...

    def update(
        self,
        state_signature: Any,
        candidate: Any,
        actual_delta: KnowledgeDelta,
        actual_error: bool,
        actual_flag: bool,
    ) -> ProphecyUpdate:
        ...


class TableProphecyModel:
    """Lightweight tabular implementation of the Prophecy Module."""

    def __init__(self, *, prior: float = 0.05) -> None:
        self.prior = prior
        self._stats: dict[tuple[Any, str, str, str, Any, Any], ProphecyStats] = {}

    def predict(self, state_signature: Any, candidate: Any) -> ProphecyPrediction:
        stats = self._stats.get(self._key(state_signature, candidate), ProphecyStats())
        denominator = stats.count + 2
        kk_probs = {
            kk: (stats.kk_counts.get(kk, 0) + self.prior) / denominator
            for kk in KK
        }
        return ProphecyPrediction(
            kk_probs=kk_probs,
            error_prob=(stats.error_count + self.prior) / denominator,
            flag_prob=(stats.flag_count + self.prior) / denominator,
        )

    def update(
        self,
        state_signature: Any,
        candidate: Any,
        actual_delta: KnowledgeDelta,
        actual_error: bool,
        actual_flag: bool,
    ) -> ProphecyUpdate:
        prediction = self.predict(state_signature, candidate)
        actual_kk = actual_delta.semantic_changed_kk()
        prediction_error = self._prediction_error(
            prediction,
            actual_kk=actual_kk,
            actual_error=actual_error,
            actual_flag=actual_flag,
        )

        key = self._key(state_signature, candidate)
        stats = self._stats.setdefault(key, ProphecyStats())
        stats.count += 1
        for kk in actual_kk:
            stats.kk_counts[kk] = stats.kk_counts.get(kk, 0) + 1
        stats.error_count += int(actual_error)
        stats.flag_count += int(actual_flag)

        return ProphecyUpdate(
            prediction_error=prediction_error,
            loss=prediction_error,
        )

    def _key(self, state_signature: Any, candidate: Any) -> tuple[Any, str, str, str, Any, Any]:
        what, how, where = candidate_axes(candidate)
        return (
            state_signature,
            what,
            how,
            where,
            binding_signature(candidate),
            recent_transition_summary(state_signature),
        )

    def _prediction_error(
        self,
        prediction: ProphecyPrediction,
        *,
        actual_kk: set[KK],
        actual_error: bool,
        actual_flag: bool,
    ) -> float:
        kk_error = 0.0
        for kk, probability in prediction.kk_probs.items():
            target = 1.0 if kk in actual_kk else 0.0
            kk_error += abs(target - probability)
        kk_error /= len(prediction.kk_probs)
        error_error = abs(float(actual_error) - prediction.error_prob)
        flag_error = abs(float(actual_flag) - prediction.flag_prob)
        return (kk_error + error_error + flag_error) / 3


class SequenceProphecyModel:
    """Optional recurrent implementation of the Prophecy Module.

    The model keeps a recurrent context per seed-level experiment component. It
    does not read the hidden world map; it encodes only the APASSR state
    signature and candidate axes, then updates from the actually observed
    semantic ΔK/error/flag target.
    """

    def __init__(
        self,
        *,
        input_dim: int = 96,
        hidden_dim: int = 32,
        learning_rate: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self._kk_order = tuple(KK)
        self._rng = np.random.default_rng(seed)
        self._wx = self._rng.normal(0.0, 0.12, size=(input_dim, hidden_dim))
        self._wh = self._rng.normal(0.0, 0.08, size=(hidden_dim, hidden_dim))
        self._bh = np.zeros(hidden_dim)
        self._wk = self._rng.normal(0.0, 0.10, size=(hidden_dim, len(self._kk_order)))
        self._bk = np.full(len(self._kk_order), -2.2)
        self._wo = self._rng.normal(0.0, 0.10, size=(hidden_dim, 2))
        self._bo = np.array([-2.2, -2.2])
        self._context = np.zeros(hidden_dim)

    def predict(self, state_signature: Any, candidate: Any) -> ProphecyPrediction:
        features = self._features(state_signature, candidate)
        hidden = self._candidate_hidden(features)
        kk_values = _sigmoid(hidden @ self._wk + self._bk)
        other = _sigmoid(hidden @ self._wo + self._bo)
        return ProphecyPrediction(
            kk_probs={kk: float(kk_values[index]) for index, kk in enumerate(self._kk_order)},
            error_prob=float(other[0]),
            flag_prob=float(other[1]),
        )

    def update(
        self,
        state_signature: Any,
        candidate: Any,
        actual_delta: KnowledgeDelta,
        actual_error: bool,
        actual_flag: bool,
    ) -> ProphecyUpdate:
        features = self._features(state_signature, candidate)
        hidden = self._candidate_hidden(features)
        kk_pred = _sigmoid(hidden @ self._wk + self._bk)
        other_pred = _sigmoid(hidden @ self._wo + self._bo)
        kk_target = np.array(
            [1.0 if kk in actual_delta.semantic_changed_kk() else 0.0 for kk in self._kk_order],
            dtype=float,
        )
        other_target = np.array([float(actual_error), float(actual_flag)], dtype=float)

        kk_error = kk_pred - kk_target
        other_error = other_pred - other_target
        self._wk -= self.learning_rate * np.outer(hidden, kk_error)
        self._bk -= self.learning_rate * kk_error
        self._wo -= self.learning_rate * np.outer(hidden, other_error)
        self._bo -= self.learning_rate * other_error

        hidden_error = kk_error @ self._wk.T + other_error @ self._wo.T
        hidden_grad = hidden_error * (1.0 - hidden * hidden)
        self._wx -= self.learning_rate * 0.2 * np.outer(features, hidden_grad)
        self._wh -= self.learning_rate * 0.2 * np.outer(self._context, hidden_grad)
        self._bh -= self.learning_rate * 0.2 * hidden_grad
        self._context = hidden

        prediction_error = self._prediction_error(
            kk_pred,
            other_pred,
            actual_kk=actual_delta.semantic_changed_kk(),
            actual_error=actual_error,
            actual_flag=actual_flag,
        )
        return ProphecyUpdate(
            prediction_error=prediction_error,
            loss=_binary_cross_entropy(kk_pred, kk_target) + _binary_cross_entropy(other_pred, other_target),
        )

    def reset_context(self) -> None:
        self._context = np.zeros(self.hidden_dim)

    def _candidate_hidden(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(features @ self._wx + self._context @ self._wh + self._bh)

    def _features(self, state_signature: Any, candidate: Any) -> np.ndarray:
        features = np.zeros(self.input_dim, dtype=float)
        for item in _flatten_feature_items(state_signature):
            features[_stable_index(("state", item), self.input_dim)] += 1.0
        what, how, where = candidate_axes(candidate)
        for item in (("what", what), ("how", how), ("where", where), ("template", candidate.template)):
            features[_stable_index(item, self.input_dim)] += 1.0
        for kk, value in candidate.bindings.items():
            if kk == KK.CURRENT_POS:
                continue
            features[_stable_index(("binding_kk", kk.value), self.input_dim)] += 1.0
            features[_stable_index(("binding_value", repr(value)), self.input_dim)] += 0.5
        norm = np.linalg.norm(features)
        if norm > 0.0:
            features /= norm
        return features

    def _prediction_error(
        self,
        kk_pred: np.ndarray,
        other_pred: np.ndarray,
        *,
        actual_kk: set[KK],
        actual_error: bool,
        actual_flag: bool,
    ) -> float:
        kk_target = np.array([1.0 if kk in actual_kk else 0.0 for kk in self._kk_order], dtype=float)
        other_target = np.array([float(actual_error), float(actual_flag)], dtype=float)
        kk_error = float(np.mean(np.abs(kk_pred - kk_target)))
        other_error = float(np.mean(np.abs(other_pred - other_target)))
        return (kk_error + other_error) / 2.0


def gridworld_state_signature(dmp: Any) -> tuple[Any, ...]:
    return gridworld_knowledge_state_signature(
        dmp.store,
        position=dmp.position,
        width=dmp.world.width,
        height=dmp.world.height,
        last_action=None,
        last_semantic_delta=(),
        last_error=0.0,
        recent_transitions=getattr(dmp, "recent_transitions", ()),
    )


def gridworld_knowledge_state_signature(
    store: Any,
    *,
    position: Any,
    width: int,
    height: int,
    last_action: Any = None,
    last_semantic_delta: tuple[str, ...] = (),
    last_error: float = 0.0,
    recent_transitions: Any = (),
) -> tuple[Any, ...]:
    frontier_count = len(store.values(KK.FRONTIER_CELL))
    unknown_neighbors = len(store.values(KK.UNKNOWN_NEIGHBOR))
    known_count = len(store.values(KK.KNOWN_CELL, include_inactive=True))
    total_cells = max(1, width * height)
    last_axes = candidate_axes(last_action) if last_action is not None else ("none", "none", "none")
    opened_door_count = sum(kv.status.value == "consumed" for kv in store.values(KK.DOOR_CELL, include_inactive=True))
    return (
        ("has_key", store.has_active(KK.KEY_OBJECT)),
        ("known_key_count", _count_bucket(len(store.values(KK.KEY_CELL, include_inactive=True)))),
        ("known_door_count", _count_bucket(len(store.values(KK.DOOR_CELL, include_inactive=True)))),
        ("opened_door_count", _count_bucket(opened_door_count)),
        ("known_hint_count", _count_bucket(len(store.values(KK.HINT_VALUE, include_inactive=True)))),
        ("known_flag_count", _count_bucket(len(store.values(KK.FLAG_CELL, include_inactive=True)))),
        ("frontier_count", _bucket(frontier_count)),
        ("unknown_neighbors", _bucket(unknown_neighbors)),
        ("visited_ratio", _ratio_bucket(len(store.values(KK.VISITED_CELL, include_inactive=True)) / total_cells)),
        ("position_region", _position_region(position, width, height)),
        ("last_axes", last_axes),
        ("last_semantic_delta", tuple(last_semantic_delta)),
        ("last_error", _ratio_bucket(last_error)),
        ("recent", recent_transition_summary(recent_transitions)),
    )


def binding_signature(candidate: Any) -> tuple[tuple[str, Any], ...]:
    features = []
    for kk, value in candidate.bindings.items():
        if kk == KK.CURRENT_POS:
            continue
        features.append((kk.value, _binding_value_features(value)))
    return tuple(sorted(features))


def recent_transition_summary(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple) and value and isinstance(value[-1], tuple) and value[-1][0] == "recent":
        return value[-1]
    records = list(value or ())[-3:]
    summary = []
    for record in records:
        if isinstance(record, dict):
            summary.append(
                (
                    tuple(record.get("action_axes", ()))[:2],
                    tuple(record.get("semantic_delta", ())),
                    bool(record.get("error", False)),
                    bool(record.get("flag_found", False)),
                )
            )
    return tuple(summary)


def _binding_value_features(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple) and len(value) == 2:
        x, y = value
        return ("cell", _count_bucket(abs(x) + abs(y)), _direction_bucket(x, y))
    text = repr(value)
    return ("object", text.split("#")[0].split("@")[0], _count_bucket(len(text)))


def _count_bucket(value: int) -> str:
    if value == 0:
        return "none"
    if value == 1:
        return "one"
    if value <= 3:
        return "few"
    return "many"


def _ratio_bucket(value: float) -> str:
    if value <= 0:
        return "none"
    if value < 0.25:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def _position_region(position: Any, width: int, height: int) -> str:
    if not isinstance(position, tuple):
        return "unknown"
    x, y = position
    horizontal = "left" if x < width / 3 else "right" if x >= 2 * width / 3 else "center"
    vertical = "top" if y < height / 3 else "bottom" if y >= 2 * height / 3 else "middle"
    return f"{vertical}-{horizontal}"


def _direction_bucket(x: int, y: int) -> str:
    if abs(x) >= abs(y):
        return "east" if x >= 0 else "west"
    return "south" if y >= 0 else "north"


def _legacy_gridworld_state_signature(dmp: Any) -> tuple[bool, bool, bool, bool, str, int]:
    frontier_count = len(dmp.store.values(KK.FRONTIER_CELL))
    unknown_neighbors = len(dmp.store.values(KK.UNKNOWN_NEIGHBOR))
    return (
        dmp.store.has_active(KK.KEY_OBJECT),
        dmp.store.has_active(KK.DOOR_CELL),
        dmp.store.has_active(KK.HINT_CELL),
        dmp.store.has_active(KK.FLAG_CELL),
        _bucket(frontier_count),
        unknown_neighbors,
    )


def _bucket(value: int) -> str:
    if value == 0:
        return "none"
    if value <= 2:
        return "low"
    if value <= 5:
        return "medium"
    return "high"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _binary_cross_entropy(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    return float(np.mean(-(target * np.log(prediction) + (1.0 - target) * np.log(1.0 - prediction))))


def _stable_index(value: Any, width: int) -> int:
    total = 0
    for character in repr(value):
        total = (total * 131 + ord(character)) % width
    return total


def _flatten_feature_items(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


class TransformerProphecyModel:
    """Optional self-attention implementation of the Prophecy Module.

    This is intentionally small and NumPy-only. It exists for prophecy
    implementation ablations, not as the main APASSR framework contribution.
    The model encodes the current knowledge-state signature and candidate
    action as tokens, applies one self-attention block, and trains only the
    prediction heads online from observed semantic ΔK/error/flag targets.
    """

    def __init__(
        self,
        *,
        input_dim: int = 128,
        model_dim: int = 32,
        learning_rate: float = 0.03,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.learning_rate = learning_rate
        self._kk_order = tuple(KK)
        self._rng = np.random.default_rng(seed)
        self._embedding = self._rng.normal(0.0, 0.10, size=(input_dim, model_dim))
        self._wq = self._rng.normal(0.0, 0.10, size=(model_dim, model_dim))
        self._wk_attn = self._rng.normal(0.0, 0.10, size=(model_dim, model_dim))
        self._wv = self._rng.normal(0.0, 0.10, size=(model_dim, model_dim))
        self._kk_head = self._rng.normal(0.0, 0.10, size=(model_dim, len(self._kk_order)))
        self._kk_bias = np.full(len(self._kk_order), -2.2)
        self._other_head = self._rng.normal(0.0, 0.10, size=(model_dim, 2))
        self._other_bias = np.array([-2.2, -2.2])

    def predict(self, state_signature: Any, candidate: Any) -> ProphecyPrediction:
        pooled = self._encode(state_signature, candidate)
        kk_values = _sigmoid(pooled @ self._kk_head + self._kk_bias)
        other = _sigmoid(pooled @ self._other_head + self._other_bias)
        return ProphecyPrediction(
            kk_probs={kk: float(kk_values[index]) for index, kk in enumerate(self._kk_order)},
            error_prob=float(other[0]),
            flag_prob=float(other[1]),
        )

    def update(
        self,
        state_signature: Any,
        candidate: Any,
        actual_delta: KnowledgeDelta,
        actual_error: bool,
        actual_flag: bool,
    ) -> ProphecyUpdate:
        pooled = self._encode(state_signature, candidate)
        kk_pred = _sigmoid(pooled @ self._kk_head + self._kk_bias)
        other_pred = _sigmoid(pooled @ self._other_head + self._other_bias)
        actual_kk = actual_delta.semantic_changed_kk()
        kk_target = np.array([1.0 if kk in actual_kk else 0.0 for kk in self._kk_order], dtype=float)
        other_target = np.array([float(actual_error), float(actual_flag)], dtype=float)

        kk_error = kk_pred - kk_target
        other_error = other_pred - other_target
        self._kk_head -= self.learning_rate * np.outer(pooled, kk_error)
        self._kk_bias -= self.learning_rate * kk_error
        self._other_head -= self.learning_rate * np.outer(pooled, other_error)
        self._other_bias -= self.learning_rate * other_error

        prediction_error = self._prediction_error(
            kk_pred,
            other_pred,
            actual_kk=actual_kk,
            actual_error=actual_error,
            actual_flag=actual_flag,
        )
        return ProphecyUpdate(
            prediction_error=prediction_error,
            loss=_binary_cross_entropy(kk_pred, kk_target) + _binary_cross_entropy(other_pred, other_target),
        )

    def _encode(self, state_signature: Any, candidate: Any) -> np.ndarray:
        token_indices = self._token_indices(state_signature, candidate)
        tokens = self._embedding[token_indices]
        q = tokens @ self._wq
        k = tokens @ self._wk_attn
        v = tokens @ self._wv
        scores = (q @ k.T) / max(1.0, self.model_dim ** 0.5)
        attention = _softmax(scores)
        attended = attention @ v
        return attended.mean(axis=0)

    def _token_indices(self, state_signature: Any, candidate: Any) -> list[int]:
        items: list[Any] = []
        items.extend(("state", item) for item in _flatten_feature_items(state_signature))
        what, how, where = candidate_axes(candidate)
        items.extend((("what", what), ("how", how), ("where", where), ("template", candidate.template)))
        for kk, value in candidate.bindings.items():
            if kk == KK.CURRENT_POS:
                continue
            items.append(("binding_kk", kk.value))
            items.append(("binding_value", repr(value)))
        return [_stable_index(item, self.input_dim) for item in items] or [0]

    def _prediction_error(
        self,
        kk_pred: np.ndarray,
        other_pred: np.ndarray,
        *,
        actual_kk: set[KK],
        actual_error: bool,
        actual_flag: bool,
    ) -> float:
        kk_target = np.array([1.0 if kk in actual_kk else 0.0 for kk in self._kk_order], dtype=float)
        other_target = np.array([float(actual_error), float(actual_flag)], dtype=float)
        kk_error = float(np.mean(np.abs(kk_pred - kk_target)))
        other_error = float(np.mean(np.abs(other_pred - other_target)))
        return (kk_error + other_error) / 2.0


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
