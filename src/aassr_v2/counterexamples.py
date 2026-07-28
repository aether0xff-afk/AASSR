from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .action_plugins import PluginOutcome
from .types import Action, StateSnapshot


class NoisyInformationWrapper:
    """Add goal-irrelevant facts without changing wrapped dynamics."""

    def __init__(
        self,
        environment: object,
        *,
        facts_per_step: int = 8,
        seed: int = 0,
    ) -> None:
        self.environment = environment
        self.facts_per_step = facts_per_step
        self.random = random.Random(seed)
        self.index = 0

    def _decorate(
        self,
        state: StateSnapshot,
    ) -> StateSnapshot:
        noise = {
            f"noise:{self.index}:{self.random.randrange(1_000_000)}"
            for _ in range(self.facts_per_step)
        }
        return replace(
            state,
            facts=state.facts | noise,
        )

    def snapshot(self) -> StateSnapshot:
        return self._decorate(
            self.environment.snapshot()
        )

    def step(self, action: Action):
        outcome = self.environment.step(action)
        self.index += 1
        snapshot = self._decorate(outcome.snapshot)
        if isinstance(outcome, PluginOutcome):
            return replace(
                outcome,
                snapshot=snapshot,
                added_facts=(
                    outcome.added_facts
                    | (snapshot.facts - outcome.snapshot.facts)
                ),
            )
        return replace(outcome, snapshot=snapshot)


@dataclass(frozen=True, slots=True)
class UncertaintyStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    reward: float = 0.0


class LearnableVsRandomWorld:
    """Separate stable causal change from irreducible randomness."""

    def __init__(self, *, seed: int = 0) -> None:
        self.random = random.Random(seed)
        self.stable_value = 0
        self.random_value = 0
        self.steps = 0

    def snapshot(self) -> StateSnapshot:
        actions = (
            Action("probe_stable"),
            Action("probe_random"),
            Action("finish"),
        )
        progress = (
            1.0 if self.stable_value >= 2 else 0.0
        )
        facts = frozenset(
            {
                f"stable:{self.stable_value}",
                f"random:{self.random_value}",
            }
        )
        return StateSnapshot(
            (
                float(self.stable_value),
                float(self.random_value),
                progress,
            ),
            facts,
            actions,
            progress,
        )

    def step(self, action: Action) -> UncertaintyStep:
        before = self.snapshot()
        error = False
        if action.verb_name == "probe_stable":
            self.stable_value = min(
                2,
                self.stable_value + 1,
            )
        elif action.verb_name == "probe_random":
            self.random_value = self.random.randrange(3)
        elif action.verb_name == "finish":
            pass
        else:
            error = True
        self.steps += 1
        after = self.snapshot()
        return UncertaintyStep(
            after,
            after.facts - before.facts,
            before.facts - after.facts,
            (),
            error,
            after.goal_progress - before.goal_progress,
        )


class LongDependencyWorld:
    """Each observed token enables the next action in a longer chain."""

    def __init__(self, length: int = 6) -> None:
        if length < 2:
            raise ValueError("length must be at least two")
        self.length = length
        self.stage = 0
        self.known: set[str] = set()

    def snapshot(self) -> StateSnapshot:
        actions = [
            Action(
                "inspect",
                parameters={"stage": self.stage},
            )
        ]
        if f"token:{self.stage}" in self.known:
            actions.append(
                Action(
                    "advance",
                    parameters={"stage": self.stage},
                )
            )
        progress = self.stage / self.length
        vector = tuple(
            [progress]
            + [
                1.0
                if f"token:{index}" in self.known
                else 0.0
                for index in range(self.length)
            ]
        )
        return StateSnapshot(
            vector,
            frozenset(self.known),
            tuple(actions),
            (
                1.0
                if self.stage >= self.length
                else progress
            ),
        )

    def step(self, action: Action) -> UncertaintyStep:
        before = self.snapshot()
        error = False
        if (
            action.verb_name == "inspect"
            and int(action.parameters.get("stage", -1))
            == self.stage
        ):
            self.known.add(f"token:{self.stage}")
        elif (
            action.verb_name == "advance"
            and f"token:{self.stage}" in self.known
        ):
            self.stage = min(
                self.length,
                self.stage + 1,
            )
        else:
            error = True
        after = self.snapshot()
        before_actions = {
            item.signature
            for item in before.available_actions
        }
        unlocked = tuple(
            item
            for item in after.available_actions
            if item.signature not in before_actions
        )
        return UncertaintyStep(
            after,
            after.facts - before.facts,
            before.facts - after.facts,
            unlocked,
            error,
            after.goal_progress - before.goal_progress,
        )


def opaque_name_map(
    values: tuple[str, ...],
    *,
    seed: int = 0,
) -> dict[str, str]:
    """Rename semantic labels to opaque identifiers for leakage tests."""

    randomizer = random.Random(seed)
    identifiers = [
        f"object-{index:04d}"
        for index in range(len(values))
    ]
    randomizer.shuffle(identifiers)
    return dict(
        zip(
            sorted(values),
            identifiers,
            strict=True,
        )
    )


def permuted_positions(
    count: int,
    *,
    width: int,
    height: int,
    seed: int = 0,
) -> tuple[tuple[int, int], ...]:
    if count < 0 or count > width * height:
        raise ValueError("count must fit inside the grid")
    positions = [
        (x, y)
        for y in range(height)
        for x in range(width)
    ]
    random.Random(seed).shuffle(positions)
    return tuple(positions[:count])
