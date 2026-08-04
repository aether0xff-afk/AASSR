from __future__ import annotations

import random

from .goal_gridpush_experiment import GridPushStep
from .long_horizon_goal_experiment import (
    CorridorRoom,
    DIRECTION_DELTAS,
    DIRECTION_PAIRS,
)
from .types import Action, ActionVerb, StateSnapshot


class CompositionalLongHorizonWorld:
    """Long dependency chain with one reusable local state representation.

    The stage number remains an observable fact for Policy and logging, but it
    is deliberately excluded from the numerical vector used by the generic
    effect-composition lookup. Therefore a movement effect learned in an early
    room can be composed in a later room with the same local geometry. This is
    not a solution hint: the target coordinate and available actions were
    already observable, and the agent still has to learn every transition from
    experience.
    """

    def __init__(
        self,
        seed: int,
        *,
        stage_count: int = 10,
        room_length: int = 6,
    ) -> None:
        if stage_count <= 1:
            raise ValueError("stage_count must exceed one")
        if room_length <= 4:
            raise ValueError("room_length must exceed the short imagination depth")
        self.seed = int(seed)
        self.stage_count = int(stage_count)
        self.room_length = int(room_length)
        randomizer = random.Random(seed)
        self.rooms = tuple(
            CorridorRoom(
                choices := randomizer.choice(DIRECTION_PAIRS),
                randomizer.choice(choices),
            )
            for _ in range(stage_count)
        )
        self.stage = 0
        self.path_step = 0
        self.chosen_direction: str | None = None
        self.agent = (0, 0)
        self.just_opened_checkpoint = False
        self.success = False
        self.failed = False
        self.optimal_steps = self.stage_count * self.room_length

    @property
    def room(self) -> CorridorRoom:
        return self.rooms[min(self.stage, self.stage_count - 1)]

    @property
    def target(self) -> tuple[int, int]:
        dx, dy = DIRECTION_DELTAS[self.room.target_direction]
        return dx * self.room_length, dy * self.room_length

    def _facts(self) -> frozenset[str]:
        facts = {f"stage:{self.stage}"}
        if self.just_opened_checkpoint:
            facts.add("checkpoint_transition")
        if self.success:
            facts.add("success")
        if self.failed:
            facts.add("failed")
        return frozenset(facts)

    @staticmethod
    def _action(direction: str) -> Action:
        return Action(ActionVerb.MOVE, parameters={"direction": direction})

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        if self.chosen_direction is None:
            return tuple(self._action(direction) for direction in self.room.choices)
        return (self._action(self.chosen_direction),)

    def _normalize(self, point: tuple[int, int]) -> tuple[float, float]:
        scale = float(self.room_length)
        return point[0] / scale, point[1] / scale

    def snapshot(self) -> StateSnapshot:
        # Only local geometry is numerical. Stage identity remains observable
        # in facts but is intentionally absent here so Prophecy can reuse the
        # same learned effect at any depth in the dependency chain.
        vector = (
            *self._normalize(self.agent),
            *self._normalize(self.target),
            self.path_step / float(self.room_length),
        )
        return StateSnapshot(
            vector,
            self._facts(),
            self._available_actions(),
            1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "stage": self.stage,
                "stage_count": self.stage_count,
                "room_length": self.room_length,
                "optimal_steps": self.optimal_steps,
                "termination": "dependency_dead_end_or_final_success",
                "prophecy_scope": "reusable_local_room",
            },
        )

    def step(self, action: Action) -> GridPushStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        available = {item.signature for item in before.available_actions}
        if action.signature not in available:
            self.failed = True
            error = True
        else:
            direction = str(action.parameters.get("direction", ""))
            if self.chosen_direction is None:
                self.chosen_direction = direction
                self.just_opened_checkpoint = False
            if direction != self.chosen_direction:
                self.failed = True
                error = True
            else:
                dx, dy = DIRECTION_DELTAS[direction]
                self.agent = self.agent[0] + dx, self.agent[1] + dy
                self.path_step += 1
                if self.path_step >= self.room_length:
                    if self.chosen_direction != self.room.target_direction:
                        self.failed = True
                    else:
                        self.stage += 1
                        if self.stage >= self.stage_count:
                            self.success = True
                            reward = 1.0
                        else:
                            self.agent = (0, 0)
                            self.path_step = 0
                            self.chosen_direction = None
                            self.just_opened_checkpoint = True

        after = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        unlocked = tuple(
            item
            for item in after.available_actions
            if item.signature not in before_actions
        )
        return GridPushStep(
            after,
            after.facts - before.facts,
            before.facts - after.facts,
            unlocked,
            error,
            reward,
        )


def install_compositional_long_horizon_world() -> None:
    """Install the reusable local-room representation in the experiment runner."""

    from . import long_horizon_goal_experiment

    long_horizon_goal_experiment.LongHorizonDependencyWorld = (
        CompositionalLongHorizonWorld
    )
