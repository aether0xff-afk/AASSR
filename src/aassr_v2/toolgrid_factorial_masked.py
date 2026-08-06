from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import toolgrid_factorial as base
from .types import Action, StateSnapshot


STAGE_COUNT = 2
TOOLGRID_STATE_SIZE = 2 + 1 + 1 + STAGE_COUNT * 3 + base.MAX_GRID_SIZE**2


class ToolGridWorld(base.ToolGridWorld):
    """ToolGrid with context-valid action masking.

    Navigation states expose only unvisited in-bounds moves. At a station the
    movement actions disappear and the tool choices become available. This keeps
    the semantic branching manipulation while avoiding a floor effect caused by
    asking a random explorer to choose among movement and tools at every tick.
    Episodes still have no artificial step limit: a self-avoiding walk either
    reaches the station or exhausts all valid moves and fails naturally.
    """

    def _valid_move_actions(self) -> tuple[Action, ...]:
        actions: list[Action] = []
        for index, name in enumerate(base.MOVE_NAMES):
            dx, dy = base.MOVE_DELTAS[name]
            candidate = self.agent[0] + dx, self.agent[1] + dy
            if (
                0 <= candidate[0] < self.grid_size
                and 0 <= candidate[1] < self.grid_size
                and candidate not in self.used_cells
            ):
                actions.append(self.actions[index])
        return tuple(actions)

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        if self.agent == self.current_station:
            return self.actions[4:]
        return self._valid_move_actions()

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            vector=self._vector(),
            facts=self._facts(),
            available_actions=self._available_actions(),
            goal_progress=1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "grid_size": self.grid_size,
                "action_count": self.action_count,
                "tool_count": self.tool_count,
                "stage_count": STAGE_COUNT,
                "stations": self.stations,
                "required_tools": self.required_tools,
                "optimal_steps": self.optimal_steps,
                "termination": "toolgrid_masked_irreversible",
            },
        )

    def step(self, action: Action) -> base.GridPushStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        self.steps += 1
        allowed = {item.signature for item in before.available_actions}
        if action.signature not in allowed:
            self.failed = True
            error = True
        else:
            index = base.action_index(action, self.actions)
            if index < 4:
                if not self._move(base.MOVE_NAMES[index]):
                    self.failed = True
                    error = True
                elif self.agent != self.current_station and not self._valid_move_actions():
                    self.failed = True
            else:
                tool_index = index - 4
                if self.agent != self.current_station or tool_index != self.current_tool:
                    self.failed = True
                    error = True
                else:
                    self.phase += 1
                    if self.phase >= STAGE_COUNT:
                        self.success = True
                        reward = 1.0
                    else:
                        self.used_cells = {self.agent}
        after = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        unlocked = tuple(
            item for item in after.available_actions if item.signature not in before_actions
        )
        return base.GridPushStep(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=unlocked,
            error=error,
            reward=reward,
        )


def encode_toolgrid_state(state: StateSnapshot) -> tuple[float, ...]:
    values = tuple(float(value) for value in state.vector)
    if len(values) != TOOLGRID_STATE_SIZE:
        raise ValueError(
            f"ToolGrid state must contain {TOOLGRID_STATE_SIZE} values, got {len(values)}"
        )
    return values


@dataclass(frozen=True, slots=True)
class ToolGridCodec(base.StateCodec):
    action_count: int

    def __post_init__(self) -> None:
        base.build_actions(self.action_count)

    @property
    def dimension(self) -> int:
        return TOOLGRID_STATE_SIZE

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        return encode_toolgrid_state(state)

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != self.dimension:
            raise ValueError("ToolGrid neural state has an unexpected size")
        values = [self._bounded(value) for value in encoded]
        grid_size = int(scaffold.metadata.get("grid_size", base.MAX_GRID_SIZE))
        scale = float(grid_size - 1)
        values[0] = round(values[0] * scale) / scale
        values[1] = round(values[1] * scale) / scale
        phase = min(STAGE_COUNT, max(0, int(round(values[2] * STAGE_COUNT))))
        values[2] = phase / float(STAGE_COUNT)
        tool_count = self.action_count - 4
        values[3] = tool_count / float(base.MAX_TOOL_COUNT)
        offset = 4
        stations: list[tuple[int, int]] = []
        tools: list[int] = []
        for _ in range(STAGE_COUNT):
            values[offset] = round(values[offset] * scale) / scale
            values[offset + 1] = round(values[offset + 1] * scale) / scale
            tool = min(
                tool_count - 1,
                max(0, int(round(values[offset + 2] * (base.MAX_TOOL_COUNT - 1)))),
            )
            values[offset + 2] = tool / float(base.MAX_TOOL_COUNT - 1)
            stations.append(
                (int(round(values[offset] * scale)), int(round(values[offset + 1] * scale)))
            )
            tools.append(tool)
            offset += 3
        for index in range(offset, len(values)):
            values[index] = float(values[index] >= 0.5)

        agent = int(round(values[0] * scale)), int(round(values[1] * scale))
        used: set[tuple[int, int]] = set()
        facts = {
            f"phase:{phase}",
            f"grid_size:{grid_size}",
            f"action_count:{self.action_count}",
        }
        for index, occupied in enumerate(values[offset:]):
            if occupied < 0.5:
                continue
            x = index % base.MAX_GRID_SIZE
            y = index // base.MAX_GRID_SIZE
            if x < grid_size and y < grid_size:
                used.add((x, y))
                facts.add(f"used:{x}:{y}")
        if phase < STAGE_COUNT:
            facts.add(f"required_tool:{tools[phase]}")
        if terminal_class == 1:
            facts.add("success")
        elif terminal_class == 2:
            facts.add("failed")

        actions = base.build_actions(self.action_count)
        available: tuple[Action, ...] = ()
        if terminal_class == 0 and phase < STAGE_COUNT:
            if agent == stations[phase]:
                available = actions[4:]
            else:
                candidates: list[Action] = []
                for index, name in enumerate(base.MOVE_NAMES):
                    dx, dy = base.MOVE_DELTAS[name]
                    point = agent[0] + dx, agent[1] + dy
                    if (
                        0 <= point[0] < grid_size
                        and 0 <= point[1] < grid_size
                        and point not in used
                    ):
                        candidates.append(actions[index])
                available = tuple(candidates)

        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "imagined_neural_delta": True,
                "imagined_neural_delta_source": source,
                "action_count": self.action_count,
                "stage_count": STAGE_COUNT,
            }
        )
        return StateSnapshot(
            vector=tuple(values),
            facts=frozenset(facts),
            available_actions=available,
            goal_progress=1.0 if terminal_class == 1 else 0.0,
            metadata=metadata,
        )


base.STAGE_COUNT = STAGE_COUNT
base.TOOLGRID_STATE_SIZE = TOOLGRID_STATE_SIZE
base.ToolGridWorld = ToolGridWorld
base.ToolGridCodec = ToolGridCodec
base.encode_toolgrid_state = encode_toolgrid_state

GRID_SIZES = base.GRID_SIZES
ACTION_COUNTS = base.ACTION_COUNTS
TOOLGRID_CONDITIONS = base.TOOLGRID_CONDITIONS
build_actions = base.build_actions
run_toolgrid_factorial = base.run_toolgrid_factorial
