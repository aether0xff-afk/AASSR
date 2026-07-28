from __future__ import annotations

from collections import Counter, defaultdict

from .types import Action, Prediction, StateSnapshot


StateFingerprint = tuple[
    tuple[float, ...],
    tuple[str, ...],
    tuple[str, ...],
]


def state_fingerprint(state: StateSnapshot) -> StateFingerprint:
    return (
        tuple(round(value, 8) for value in state.vector),
        tuple(sorted(state.facts)),
        tuple(action.signature for action in state.available_actions),
    )


class TabularProphecy:
    """Transparent empirical transition model used as the first baseline.

    Exact state-action observations are preferred. Before an exact transition
    is known, the model falls back to transitions observed for the same action
    family during the curriculum.
    """

    def __init__(self, name: str = "tabular") -> None:
        self._name = name
        self._exact: dict[
            tuple[StateFingerprint, str],
            Counter[StateFingerprint],
        ] = defaultdict(Counter)
        self._global: dict[str, Counter[StateFingerprint]] = defaultdict(Counter)
        self._states: dict[StateFingerprint, StateSnapshot] = {}

    @property
    def name(self) -> str:
        return self._name

    def _key(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[StateFingerprint, str]:
        return state_fingerprint(state), action.signature

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        next_fingerprint = state_fingerprint(actual_next_state)
        self._states[next_fingerprint] = actual_next_state
        self._exact[self._key(state, action)][next_fingerprint] += 1
        self._global[action.verb.value][next_fingerprint] += 1

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if samples <= 0:
            raise ValueError("samples must be positive")

        counts = self._exact.get(self._key(state, action))
        source = f"{self.name}:exact"
        if not counts:
            counts = self._global.get(action.verb.value)
            source = f"{self.name}:action-family"

        if not counts:
            return (
                Prediction(
                    next_state=state,
                    probability=1.0,
                    source=f"{self.name}:unseen",
                ),
            )

        total = sum(counts.values())
        return tuple(
            Prediction(
                next_state=self._states[fingerprint],
                probability=count / total,
                source=source,
            )
            for fingerprint, count in counts.most_common(samples)
        )
