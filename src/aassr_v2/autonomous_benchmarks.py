from __future__ import annotations

from dataclasses import dataclass
import random

from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class AutonomousStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    reward: float = 0.0


class OpaqueDependencyWorld:
    """Sparse-reward irreversible dependency world with opaque symbols.

    At every stage two opaque actions are available. One preserves the viable
    trajectory and one irreversibly corrupts it, but both advance to the next
    stage. Only terminal success gives reward 1.
    """

    def __init__(self, length: int = 4, *, seed: int = 0) -> None:
        if length < 2:
            raise ValueError("length must be at least two")
        self.length = length
        self.seed = seed
        randomizer = random.Random(seed)
        self._actions: list[tuple[Action, Action]] = []
        self._viable_signature: list[str] = []
        for _stage in range(length):
            tokens = [
                f"op_{randomizer.getrandbits(48):012x}",
                f"op_{randomizer.getrandbits(48):012x}",
            ]
            randomizer.shuffle(tokens)
            actions = (Action(tokens[0]), Action(tokens[1]))
            viable_index = randomizer.randrange(2)
            self._actions.append(actions)
            self._viable_signature.append(actions[viable_index].signature)
        self._labels = tuple(
            f"obs_{randomizer.getrandbits(48):012x}"
            for _ in range(length + 1)
        )
        slots = list(range(length + 1))
        randomizer.shuffle(slots)
        self._state_slots = tuple(slots)
        self.stage = 0
        self.corrupted = False
        self.terminal = False

    def _vector(self) -> tuple[float, ...]:
        one_hot = [0.0] * (self.length + 1)
        one_hot[self._state_slots[min(self.stage, self.length)]] = 1.0
        # Viability is deliberately absent from the agent-visible state.  A
        # wrong choice and a viable choice at the same stage must be
        # observationally indistinguishable until terminal reward is emitted.
        return (*one_hot, 0.0, float(self.terminal))

    def snapshot(self) -> StateSnapshot:
        success = self.terminal and not self.corrupted
        actions = () if self.terminal else self._actions[self.stage]
        facts = {self._labels[min(self.stage, self.length)]}
        return StateSnapshot(
            self._vector(),
            frozenset(facts),
            actions,
            1.0 if success else 0.0,
        )

    def step(self, action: Action) -> AutonomousStep:
        if self.terminal:
            raise RuntimeError("cannot step a terminal world")
        before = self.snapshot()
        available = {candidate.signature for candidate in before.available_actions}
        error = action.signature not in available
        reward = 0.0
        if not error:
            if action.signature != self._viable_signature[self.stage]:
                self.corrupted = True
            self.stage += 1
            if self.stage >= self.length:
                self.terminal = True
                reward = 1.0 if not self.corrupted else 0.0
        after = self.snapshot()
        before_actions = {candidate.signature for candidate in before.available_actions}
        unlocked = tuple(
            candidate
            for candidate in after.available_actions
            if candidate.signature not in before_actions
        )
        return AutonomousStep(
            after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=unlocked,
            error=error,
            reward=reward,
        )

    def oracle_action(self) -> Action:
        """Return the viable action for the explicitly privileged upper bound."""
        if self.terminal:
            raise RuntimeError("terminal world has no oracle action")
        signature = self._viable_signature[self.stage]
        return next(
            action
            for action in self._actions[self.stage]
            if action.signature == signature
        )
