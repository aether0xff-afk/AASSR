from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

from .metrics import cosine_similarity
from .prophecy import ProphecyStep
from .types import Action, Prediction, StateSnapshot


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _matvec(
    matrix: list[list[float]], vector: tuple[float, ...]
) -> list[float]:
    return [
        sum(weight * value for weight, value in zip(row, vector, strict=True))
        for row in matrix
    ]


def _transpose_matvec(
    matrix: list[list[float]], vector: list[float]
) -> list[float]:
    if not matrix:
        return []
    return [
        sum(matrix[row][column] * vector[row] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    ]


def _outer(
    left: list[float], right: tuple[float, ...] | list[float]
) -> list[list[float]]:
    return [[a * b for b in right] for a in left]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _clip(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True, slots=True)
class GRUMemory:
    hidden: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GRUTrainingStats:
    updates: int
    last_loss: float
    mean_loss: float


class OnlineGRUProphecy:
    """Dependency-free online GRU world model.

    Confidence is represented as probability mass, not multiplied and then
    normalized away. The remaining mass is assigned to an uncertain self-state
    prediction, allowing Imagination to prune low-evidence branches.
    """

    def __init__(
        self,
        state_size: int,
        *,
        action_feature_size: int = 16,
        hidden_size: int = 24,
        learning_rate: float = 0.02,
        seed: int = 7,
        replay_limit: int = 512,
    ) -> None:
        if min(state_size, action_feature_size, hidden_size, replay_limit) <= 0:
            raise ValueError("sizes and replay_limit must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.state_size = state_size
        self.action_feature_size = action_feature_size
        self.hidden_size = hidden_size
        self.input_size = state_size + action_feature_size
        self.learning_rate = learning_rate
        self.replay_limit = replay_limit
        self._random = random.Random(seed)
        self._name = "online-gru"
        self._updates = 0
        self._last_loss = 0.0
        self._mean_loss = 0.0
        self._train_memory = self.initial_memory()
        self._templates: list[StateSnapshot] = []
        self._action_counts: dict[str, int] = {}

        def matrix(rows: int, columns: int) -> list[list[float]]:
            scale = 1.0 / math.sqrt(max(1, columns))
            return [
                [self._random.uniform(-scale, scale) for _ in range(columns)]
                for _ in range(rows)
            ]

        self.Wz = matrix(hidden_size, self.input_size)
        self.Uz = matrix(hidden_size, hidden_size)
        self.bz = [0.0] * hidden_size
        self.Wr = matrix(hidden_size, self.input_size)
        self.Ur = matrix(hidden_size, hidden_size)
        self.br = [0.0] * hidden_size
        self.Wh = matrix(hidden_size, self.input_size)
        self.Uh = matrix(hidden_size, hidden_size)
        self.bh = [0.0] * hidden_size
        self.Wo = matrix(state_size, hidden_size)
        self.bo = [0.0] * state_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def training_stats(self) -> GRUTrainingStats:
        return GRUTrainingStats(self._updates, self._last_loss, self._mean_loss)

    def initial_memory(self) -> GRUMemory:
        return GRUMemory((0.0,) * self.hidden_size)

    def reset_sequence(self) -> None:
        self._train_memory = self.initial_memory()

    def _action_features(self, action: Action) -> tuple[float, ...]:
        vector = [0.0] * self.action_feature_size
        tokens = [action.verb_name, action.signature]
        tokens.extend(
            f"{key}={value}" for key, value in sorted(action.parameters.items())
        )
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.action_feature_size
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)

    def _input(self, state: StateSnapshot, action: Action) -> tuple[float, ...]:
        if len(state.vector) != self.state_size:
            raise ValueError(
                f"expected state vector of length {self.state_size}, "
                f"got {len(state.vector)}"
            )
        return tuple(state.vector) + self._action_features(action)

    def _forward(self, x: tuple[float, ...], memory: GRUMemory):
        h = memory.hidden
        z = [
            _sigmoid(a + b + c)
            for a, b, c in zip(
                _matvec(self.Wz, x), _matvec(self.Uz, h), self.bz, strict=True
            )
        ]
        r = [
            _sigmoid(a + b + c)
            for a, b, c in zip(
                _matvec(self.Wr, x), _matvec(self.Ur, h), self.br, strict=True
            )
        ]
        rh = tuple(r[index] * h[index] for index in range(self.hidden_size))
        n = [
            math.tanh(a + b + c)
            for a, b, c in zip(
                _matvec(self.Wh, x), _matvec(self.Uh, rh), self.bh, strict=True
            )
        ]
        next_h = tuple(
            (1.0 - z[index]) * n[index] + z[index] * h[index]
            for index in range(self.hidden_size)
        )
        y = tuple(
            value + bias
            for value, bias in zip(
                _matvec(self.Wo, next_h), self.bo, strict=True
            )
        )
        return y, GRUMemory(next_h), (x, h, z, r, rh, n, next_h)

    def predict_vector(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: GRUMemory | None = None,
    ) -> tuple[float, ...]:
        output, _, _ = self._forward(
            self._input(state, action), memory or self.initial_memory()
        )
        return output

    def _nearest_templates(
        self, vector: tuple[float, ...], limit: int
    ) -> list[tuple[float, StateSnapshot]]:
        unique: dict[tuple[float, ...], StateSnapshot] = {}
        for state in self._templates:
            unique[state.vector] = state
        scored = [
            (cosine_similarity(vector, template.vector), template)
            for template in unique.values()
        ]
        scored.sort(key=lambda item: (-item[0], tuple(sorted(item[1].facts))))
        return scored[:limit]

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state
        seen = self._action_counts.get(action.signature, 0)
        experience = seen / (seen + 4.0)
        loss_quality = 1.0 / (1.0 + max(0.0, self._mean_loss))
        return max(0.0, min(1.0, experience * loss_quality))

    def coverage(
        self, state: StateSnapshot, actions: Iterable[Action]
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(self.confidence(state, action) for action in materialized) / len(
            materialized
        )

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: GRUMemory | None,
        samples: int,
    ) -> ProphecyStep:
        if samples <= 0:
            raise ValueError("samples must be positive")
        output, next_memory, _ = self._forward(
            self._input(state, action), memory or self.initial_memory()
        )
        nearest = self._nearest_templates(output, samples)
        confidence = self.confidence(state, action)
        if not nearest or confidence <= 0.0:
            return ProphecyStep(
                (Prediction(state, 1.0, source=f"{self.name}:unseen"),),
                next_memory,
            )
        probabilities = _softmax([score * 4.0 for score, _ in nearest])
        suffix = "exact" if confidence >= 0.75 else "action-family"
        predictions = [
            Prediction(
                template,
                probability * confidence,
                source=f"{self.name}:{suffix}",
            )
            for probability, (_, template) in zip(
                probabilities, nearest, strict=True
            )
        ]
        if confidence < 1.0:
            predictions.append(
                Prediction(
                    state,
                    1.0 - confidence,
                    source=f"{self.name}:uncertain",
                )
            )
        return ProphecyStep(tuple(predictions), next_memory)

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.predict_step(
            state,
            action,
            memory=self.initial_memory(),
            samples=samples,
        ).predictions

    def _apply_matrix_gradient(
        self, matrix: list[list[float]], gradient: list[list[float]]
    ) -> None:
        for row in range(len(matrix)):
            for column in range(len(matrix[row])):
                matrix[row][column] -= self.learning_rate * _clip(
                    gradient[row][column]
                )

    def _apply_vector_gradient(
        self, vector: list[float], gradient: list[float]
    ) -> None:
        for index in range(len(vector)):
            vector[index] -= self.learning_rate * _clip(gradient[index])

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if len(actual_next_state.vector) != self.state_size:
            raise ValueError("next state vector size changed")
        x = self._input(state, action)
        output, next_memory, cache = self._forward(x, self._train_memory)
        target = actual_next_state.vector
        error = [output[index] - target[index] for index in range(self.state_size)]
        loss = sum(value * value for value in error) / self.state_size
        x, h, z, r, rh, n, next_h = cache

        dWo = _outer(error, next_h)
        dbo = error[:]
        dh = _transpose_matvec(self.Wo, error)
        dn = [dh[index] * (1.0 - z[index]) for index in range(self.hidden_size)]
        dz = [
            dh[index] * (h[index] - n[index])
            for index in range(self.hidden_size)
        ]
        dan = [dn[index] * (1.0 - n[index] ** 2) for index in range(self.hidden_size)]
        dWh = _outer(dan, x)
        dUh = _outer(dan, rh)
        dbh = dan[:]
        drh = _transpose_matvec(self.Uh, dan)
        dr = [drh[index] * h[index] for index in range(self.hidden_size)]
        dar = [
            dr[index] * r[index] * (1.0 - r[index])
            for index in range(self.hidden_size)
        ]
        dWr = _outer(dar, x)
        dUr = _outer(dar, h)
        dbr = dar[:]
        daz = [
            dz[index] * z[index] * (1.0 - z[index])
            for index in range(self.hidden_size)
        ]
        dWz = _outer(daz, x)
        dUz = _outer(daz, h)
        dbz = daz[:]

        self._apply_matrix_gradient(self.Wo, dWo)
        self._apply_vector_gradient(self.bo, dbo)
        self._apply_matrix_gradient(self.Wh, dWh)
        self._apply_matrix_gradient(self.Uh, dUh)
        self._apply_vector_gradient(self.bh, dbh)
        self._apply_matrix_gradient(self.Wr, dWr)
        self._apply_matrix_gradient(self.Ur, dUr)
        self._apply_vector_gradient(self.br, dbr)
        self._apply_matrix_gradient(self.Wz, dWz)
        self._apply_matrix_gradient(self.Uz, dUz)
        self._apply_vector_gradient(self.bz, dbz)

        self._train_memory = next_memory
        self._templates.append(actual_next_state)
        if len(self._templates) > self.replay_limit:
            self._templates.pop(0)
        self._action_counts[action.signature] = (
            self._action_counts.get(action.signature, 0) + 1
        )
        self._updates += 1
        self._last_loss = loss
        self._mean_loss += (loss - self._mean_loss) / self._updates
