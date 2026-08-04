from __future__ import annotations

import random
from typing import Mapping

from .goal_gridpush_experiment import DIRECTIONS, GridPushStep
from .types import Action, ActionVerb, StateSnapshot


class CollapsingGridPushWorld:
    """Finite GridPush world with no tick or energy limit.

    During each movement stage, a grid cell collapses after it is used and
    cannot be entered again. Reaching the stage target opens a fresh room for
    the next dependency. A wrong route therefore fails only when it naturally
    leaves no legal movement, rather than when an artificial step counter
    expires. The same movement, push, pickup and use rules are reused across
    procedural maps.
    """

    grid_size = 3

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        randomizer = random.Random(seed)
        points = randomizer.sample(
            [
                (x, y)
                for y in range(self.grid_size)
                for x in range(self.grid_size)
            ],
            6,
        )
        (
            self.agent,
            self.crate,
            self.pit,
            self.key,
            self.door,
            self.exit,
        ) = points
        self.phase = 0
        self.bridge_built = False
        self.key_held = False
        self.door_open = False
        self.success = False
        self.failed = False
        self.used_cells: set[tuple[int, int]] = {self.agent}
        self.optimal_steps = (
            self._distance(self.agent, self.crate)
            + self._distance(self.crate, self.pit)
            + self._distance(self.pit, self.key)
            + 1
            + self._distance(self.key, self.door)
            + 1
            + self._distance(self.door, self.exit)
        )

    @staticmethod
    def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _normalize(self, point: tuple[int, int]) -> tuple[float, float]:
        scale = float(self.grid_size - 1)
        return point[0] / scale, point[1] / scale

    def _facts(self) -> frozenset[str]:
        facts = {
            f"phase:{self.phase}",
            *(f"used:{x}:{y}" for x, y in sorted(self.used_cells)),
        }
        if self.bridge_built:
            facts.add("bridge_built")
        if self.key_held:
            facts.add("key_held")
        if self.door_open:
            facts.add("door_open")
        if self.success:
            facts.add("success")
        if self.failed:
            facts.add("failed")
        return frozenset(facts)

    def _next_point(
        self,
        point: tuple[int, int],
        direction: str,
    ) -> tuple[int, int] | None:
        delta = DIRECTIONS.get(direction)
        if delta is None:
            return None
        candidate = point[0] + delta[0], point[1] + delta[1]
        if not (
            0 <= candidate[0] < self.grid_size
            and 0 <= candidate[1] < self.grid_size
        ):
            return None
        if candidate in self.used_cells:
            return None
        return candidate

    def _movement_actions(
        self,
        verb: ActionVerb | str,
        point: tuple[int, int],
    ) -> tuple[Action, ...]:
        return tuple(
            Action(verb, parameters={"direction": direction})
            for direction in DIRECTIONS
            if self._next_point(point, direction) is not None
        )

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        if self.phase in {0, 2, 4, 6}:
            return self._movement_actions(ActionVerb.MOVE, self.agent)
        if self.phase == 1:
            return self._movement_actions("push", self.crate)
        if self.phase == 3:
            return (Action(ActionVerb.PICKUP),)
        if self.phase == 5:
            return (Action(ActionVerb.USE),)
        return ()

    def snapshot(self) -> StateSnapshot:
        vector = (
            *self._normalize(self.agent),
            *self._normalize(self.crate),
            *self._normalize(self.pit),
            *self._normalize(self.key),
            *self._normalize(self.door),
            *self._normalize(self.exit),
            self.phase / 6.0,
            float(self.bridge_built),
            float(self.key_held),
            float(self.door_open),
        )
        return StateSnapshot(
            vector,
            self._facts(),
            self._available_actions(),
            1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "optimal_steps": self.optimal_steps,
                "termination": "irreversible_path_only",
            },
        )

    def _enter_new_room(self) -> None:
        self.used_cells = {self.agent}

    def _finish_or_fail_movement(self) -> None:
        if not self.success and not self._available_actions():
            self.failed = True

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
            if self.phase in {0, 2, 4, 6}:
                candidate = self._next_point(self.agent, direction)
                if candidate is None:
                    self.failed = True
                    error = True
                else:
                    self.agent = candidate
                    self.used_cells.add(candidate)
                    if self.phase == 0 and self.agent == self.crate:
                        self.phase = 1
                        self._enter_new_room()
                    elif self.phase == 2 and self.agent == self.key:
                        self.phase = 3
                        self._enter_new_room()
                    elif self.phase == 4 and self.agent == self.door:
                        self.phase = 5
                        self._enter_new_room()
                    elif self.phase == 6 and self.agent == self.exit:
                        self.success = True
                    else:
                        self._finish_or_fail_movement()
            elif self.phase == 1:
                candidate = self._next_point(self.crate, direction)
                if candidate is None:
                    self.failed = True
                    error = True
                else:
                    self.crate = candidate
                    self.agent = candidate
                    self.used_cells.add(candidate)
                    if self.crate == self.pit:
                        self.bridge_built = True
                        self.phase = 2
                        self._enter_new_room()
                    else:
                        self._finish_or_fail_movement()
            elif self.phase == 3:
                if self.agent == self.key:
                    self.key_held = True
                    self.phase = 4
                    self._enter_new_room()
                else:
                    self.failed = True
                    error = True
            elif self.phase == 5:
                if self.agent == self.door and self.key_held:
                    self.door_open = True
                    self.phase = 6
                    self._enter_new_room()
                else:
                    self.failed = True
                    error = True
            else:
                self.failed = True
                error = True

        if self.success:
            reward = 1.0

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
