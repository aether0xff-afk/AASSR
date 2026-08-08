from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from dataclasses import replace
from math import log, sqrt
import random
from typing import Any

from .autonomous_agent_core import ContextualPolicy, RunningValue
from .policy import PolicyMemory, ScoredAction
from .types import Action, StateSnapshot


SemanticStateKey = Hashable
SemanticStateKeyFn = Callable[[StateSnapshot], SemanticStateKey]


def raw_semantic_state_key(state: StateSnapshot) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Backward-compatible state identity used when no domain contract is supplied."""

    return (
        tuple(round(value, 8) for value in state.vector),
        tuple(sorted(state.facts)),
    )


class SemanticContextualPolicy(ContextualPolicy):
    """Contextual Policy whose state identity is supplied by one shared contract.

    AASSR 0.4 uses the same semantic equivalence relation for Policy lookup,
    ASEQ self-loop memory, and Imagination cycle detection. Domains that need to
    ignore administrative counters (for example the audited pentest world) pass
    an observation-derived semantic key function instead of changing Policy
    internals independently.
    """

    def __init__(
        self,
        state_key_fn: SemanticStateKeyFn = raw_semantic_state_key,
        *,
        learning_rate: float = 0.2,
    ) -> None:
        super().__init__(learning_rate=learning_rate)
        self.state_key_fn = state_key_fn

    def _semantic_key(self, state: StateSnapshot) -> SemanticStateKey:
        return self.state_key_fn(state)

    def _entry(self, state: StateSnapshot, action: Action) -> RunningValue:
        return self._local.get(
            (self._semantic_key(state), action.signature),
            RunningValue(),
        )

    def select(
        self,
        state: StateSnapshot,
        *,
        randomizer: random.Random,
        epsilon: float,
        exploration_bonus: float,
    ) -> Action:
        if not state.available_actions:
            raise ValueError("state has no available actions")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon > 0.0 and randomizer.random() < epsilon:
            return randomizer.choice(state.available_actions)

        key = self._semantic_key(state)
        total = self._state_visits.get(key, 0)
        scored: list[ScoredAction] = []
        for action in state.available_actions:
            entry = self._entry(state, action)
            bonus = exploration_bonus * sqrt(
                log(total + 2.0) / (entry.count + 1.0)
            )
            scored.append(
                ScoredAction(action, self.value(state, action) + bonus)
            )
        scored.sort(key=lambda item: (-item.score, item.action.signature))
        return scored[0].action

    def observe_return(
        self,
        state: StateSnapshot,
        action: Action,
        target: float,
    ) -> None:
        key = self._semantic_key(state)
        self._state_visits[key] = self._state_visits.get(key, 0) + 1
        local = self._local.setdefault(
            (key, action.signature),
            RunningValue(),
        )
        local.observe(target, learning_rate=self.learning_rate)
        global_entry = self._global.setdefault(
            action.signature,
            RunningValue(),
        )
        global_entry.observe(target, learning_rate=self.learning_rate)


class SemanticSelfLoopASEQ:
    """Empirical ASEQ guard for repeated semantic ``S -> A -> S`` self-loops.

    The memory stores observed ``(S, A, S')`` outcomes. An action is guarded only
    when one exact ``(S, A)`` pair has produced *only* the same semantic state S
    at least ``repeat_threshold`` times. State-changing repeats remain legal. If
    every available action would be guarded, the raw action set is restored so
    ASEQ never removes all controller freedom.
    """

    def __init__(self, *, repeat_threshold: int = 2) -> None:
        if repeat_threshold <= 0:
            raise ValueError("repeat_threshold must be positive")
        self.repeat_threshold = int(repeat_threshold)
        self._outcomes: dict[
            tuple[SemanticStateKey, str],
            Counter[SemanticStateKey],
        ] = defaultdict(Counter)
        self.guard_events = 0
        self.all_guarded_fallbacks = 0
        self.guarded_candidates = 0

    def reset_episode(self) -> None:
        self._outcomes.clear()

    def observe(
        self,
        before: SemanticStateKey,
        action: Action,
        after: SemanticStateKey,
    ) -> None:
        self._outcomes[(before, action.signature)][after] += 1

    def guarded_signatures(self, semantic_state: SemanticStateKey) -> frozenset[str]:
        guarded: set[str] = set()
        for (state, signature), history in self._outcomes.items():
            if state != semantic_state or len(history) != 1:
                continue
            only_next_state, count = next(iter(history.items()))
            if count >= self.repeat_threshold and only_next_state == semantic_state:
                guarded.add(signature)
        return frozenset(guarded)

    def filter_state(
        self,
        state: StateSnapshot,
        semantic_state: SemanticStateKey,
    ) -> tuple[StateSnapshot, int, bool]:
        guarded = self.guarded_signatures(semantic_state)
        if not guarded:
            return state, 0, False

        kept = tuple(
            action
            for action in state.available_actions
            if action.signature not in guarded
        )
        self.guarded_candidates += len(guarded)
        if not kept:
            self.all_guarded_fallbacks += 1
            return state, len(guarded), True

        self.guard_events += 1
        return replace(state, available_actions=kept), len(guarded), False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "repeat_threshold": self.repeat_threshold,
            "observed_state_action_pairs": len(self._outcomes),
            "guard_events": self.guard_events,
            "guarded_candidates": self.guarded_candidates,
            "all_guarded_fallbacks": self.all_guarded_fallbacks,
        }
