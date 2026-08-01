from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .paper_v2_protocol import sha256_json
from .paper_v2_types import RawCausalObservation


Position = tuple[int, int]

MOVE_DELTAS: Mapping[str, Position] = {
    "MOVE_NORTH": (0, -1),
    "MOVE_SOUTH": (0, 1),
    "MOVE_WEST": (-1, 0),
    "MOVE_EAST": (1, 0),
}

GRID_PUSH_LAW_VERSION = "grid-push-physics-v2.0"
GRID_PUSH_LAW = {
    "movement": "an agent moves one cardinal cell when the destination is passable",
    "push": "entering a block cell pushes exactly one block one cardinal cell when its destination is passable",
    "pit": "a block entering a pit is consumed and irreversibly makes that pit passable",
    "pressure": "occupancy of a plate opens every privately linked door",
    "door": "an unpowered door blocks agents and blocks",
    "reward": "reward one is emitted only when the agent reaches the goal cell",
}
GRID_PUSH_LAW_SHA256 = sha256_json(GRID_PUSH_LAW)


def _positions(values: Iterable[Sequence[int]]) -> frozenset[Position]:
    return frozenset((int(value[0]), int(value[1])) for value in values)


@dataclass(frozen=True, slots=True)
class GridPushSpec:
    width: int
    height: int
    walls: frozenset[Position]
    start: Position
    goal: Position
    blocks: frozenset[Position]
    pits: frozenset[Position] = frozenset()
    plates: frozenset[Position] = frozenset()
    doors: frozenset[Position] = frozenset()
    plate_links: Mapping[Position, tuple[Position, ...]] = field(default_factory=dict)
    observation_mode: str = "full_map"
    view_radius: int = 2
    generator_seed: int | None = None

    def __post_init__(self) -> None:
        if self.width < 4 or self.height < 4:
            raise ValueError("grid must be at least 4 by 4")
        if self.observation_mode not in {"full_map", "limited_view"}:
            raise ValueError("unknown observation mode")
        all_positions = (
            set(self.walls)
            | {self.start, self.goal}
            | set(self.blocks)
            | set(self.pits)
            | set(self.plates)
            | set(self.doors)
        )
        if any(not (0 <= x < self.width and 0 <= y < self.height) for x, y in all_positions):
            raise ValueError("object outside grid")
        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("start and goal must be floor cells")
        if self.start == self.goal:
            raise ValueError("start and goal must differ")
        if self.blocks & self.walls or self.blocks & self.pits or self.blocks & self.doors:
            raise ValueError("blocks cannot start in walls, pits, or doors")
        if self.pits & self.walls or self.doors & self.walls:
            raise ValueError("pits and doors cannot be walls")
        if set(self.plate_links) - set(self.plates):
            raise ValueError("plate link source is not a plate")
        linked = {door for doors in self.plate_links.values() for door in doors}
        if linked - set(self.doors):
            raise ValueError("plate link target is not a door")

    def private_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "walls": sorted(self.walls),
            "start": self.start,
            "goal": self.goal,
            "blocks": sorted(self.blocks),
            "pits": sorted(self.pits),
            "plates": sorted(self.plates),
            "doors": sorted(self.doors),
            "plate_links": {
                f"{plate[0]},{plate[1]}": sorted(doors)
                for plate, doors in sorted(self.plate_links.items())
            },
            "observation_mode": self.observation_mode,
            "view_radius": self.view_radius,
            "generator_seed": self.generator_seed,
        }


@dataclass(frozen=True, slots=True)
class GridPushPrivateState:
    player: Position
    blocks: frozenset[Position]
    filled_pits: frozenset[Position] = frozenset()
    terminal: bool = False
    success: bool = False
    step_count: int = 0


@dataclass(frozen=True, slots=True)
class GridEvent:
    kind: str
    source: Position | None = None
    target: Position | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GridPushStep:
    observation: RawCausalObservation
    reward: float
    action_succeeded: bool


def _add(position: Position, delta: Position) -> Position:
    return position[0] + delta[0], position[1] + delta[1]


class GridPushWorld:
    """Sparse-reward grid physics. Solver data is never part of observation."""

    def __init__(self, spec: GridPushSpec) -> None:
        self._spec = spec
        self._state = GridPushPrivateState(spec.start, spec.blocks)
        self._last_succeeded: bool | None = None
        self._last_events: tuple[GridEvent, ...] = ()
        self._event_history: list[tuple[GridEvent, ...]] = []

    @property
    def causal_law_sha256(self) -> str:
        return GRID_PUSH_LAW_SHA256

    @property
    def world_sha256(self) -> str:
        return sha256_json(self._spec.private_dict())

    @property
    def analysis_spec(self) -> GridPushSpec:
        return self._spec

    @property
    def analysis_private_state(self) -> GridPushPrivateState:
        return self._state

    @property
    def analysis_last_events(self) -> tuple[GridEvent, ...]:
        return self._last_events

    @property
    def analysis_event_history(self) -> tuple[tuple[GridEvent, ...], ...]:
        return tuple(self._event_history)

    @property
    def terminal(self) -> bool:
        return self._state.terminal

    def clone(self) -> "GridPushWorld":
        return copy.deepcopy(self)

    def state_key(self) -> tuple[Any, ...]:
        return self._state.player, self._state.blocks, self._state.filled_pits

    def pressed_plates(self) -> frozenset[Position]:
        occupants = set(self._state.blocks) | {self._state.player}
        return frozenset(set(self._spec.plates) & occupants)

    def open_doors(self) -> frozenset[Position]:
        opened: set[Position] = set()
        for plate in self.pressed_plates():
            opened.update(self._spec.plate_links.get(plate, ()))
        return frozenset(opened)

    def _inside(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self._spec.width and 0 <= y < self._spec.height

    def _passable_for_player(self, position: Position) -> bool:
        return bool(
            self._inside(position)
            and position not in self._spec.walls
            and position not in self._state.blocks
            and (
                position not in self._spec.pits
                or position in self._state.filled_pits
            )
            and (
                position not in self._spec.doors
                or position in self.open_doors()
            )
        )

    def _passable_for_block(self, position: Position) -> bool:
        return bool(
            self._inside(position)
            and position not in self._spec.walls
            and position not in self._state.blocks
            and (
                position not in self._spec.doors
                or position in self.open_doors()
            )
        )

    def _base_tile(self, position: Position) -> str:
        if position in self._spec.walls:
            return "wall"
        if position in self._spec.doors:
            return "door_open" if position in self.open_doors() else "door_closed"
        if position in self._spec.pits:
            return "pit_filled" if position in self._state.filled_pits else "pit"
        if position in self._spec.plates:
            return "pressure_plate"
        if position == self._spec.goal:
            return "goal"
        return "floor"

    def _visible(self, position: Position) -> bool:
        if self._spec.observation_mode == "full_map":
            return True
        return (
            abs(position[0] - self._state.player[0])
            + abs(position[1] - self._state.player[1])
            <= self._spec.view_radius
        )

    def observe(self) -> RawCausalObservation:
        spatial: dict[str, str | float | int] = {
            "width": self._spec.width,
            "height": self._spec.height,
            "observation_mode": self._spec.observation_mode,
        }
        for y in range(self._spec.height):
            for x in range(self._spec.width):
                position = (x, y)
                if not self._visible(position):
                    continue
                layers = [self._base_tile(position)]
                if position in self._state.blocks:
                    layers.append("block")
                if position == self._state.player:
                    layers.append("player")
                spatial[f"cell:{x},{y}"] = "+".join(layers)
        actions = () if self.terminal else tuple(MOVE_DELTAS)
        return RawCausalObservation(
            available_actions=actions,
            action_affordances={
                action: ("move", action.removeprefix("MOVE_").lower())
                for action in actions
            },
            spatial_observations=spatial,
            last_action_succeeded=self._last_succeeded,
            terminal_reward=1.0 if self._state.success else 0.0,
            terminal=self._state.terminal,
        )

    def step(self, action: str) -> GridPushStep:
        if self.terminal:
            raise RuntimeError("cannot step a terminal world")
        if action not in MOVE_DELTAS:
            raise ValueError(f"unknown grid action: {action}")
        before_plates = self.pressed_plates()
        before_doors = self.open_doors()
        target = _add(self._state.player, MOVE_DELTAS[action])
        blocks = set(self._state.blocks)
        filled = set(self._state.filled_pits)
        events: list[GridEvent] = []
        succeeded = False
        if target in blocks:
            block_target = _add(target, MOVE_DELTAS[action])
            if self._passable_for_block(block_target):
                blocks.remove(target)
                if block_target in self._spec.pits and block_target not in filled:
                    filled.add(block_target)
                    events.append(GridEvent("block_moved", target, block_target))
                    events.append(GridEvent("pit_filled", block_target, block_target))
                else:
                    blocks.add(block_target)
                    events.append(GridEvent("block_moved", target, block_target))
                events.append(GridEvent("player_moved", self._state.player, target))
                succeeded = True
        elif self._passable_for_player(target):
            events.append(GridEvent("player_moved", self._state.player, target))
            succeeded = True
        if succeeded:
            player = target
        else:
            player = self._state.player
            events.append(GridEvent("action_failed", target, target))
        success = succeeded and player == self._spec.goal
        self._state = GridPushPrivateState(
            player=player,
            blocks=frozenset(blocks),
            filled_pits=frozenset(filled),
            terminal=success,
            success=success,
            step_count=self._state.step_count + 1,
        )
        after_plates = self.pressed_plates()
        after_doors = self.open_doors()
        events.extend(
            GridEvent("plate_pressed", plate, plate)
            for plate in sorted(after_plates - before_plates)
        )
        events.extend(
            GridEvent("plate_released", plate, plate)
            for plate in sorted(before_plates - after_plates)
        )
        events.extend(
            GridEvent("door_opened", door, door)
            for door in sorted(after_doors - before_doors)
        )
        events.extend(
            GridEvent("door_closed", door, door)
            for door in sorted(before_doors - after_doors)
        )
        if success:
            events.append(GridEvent("goal_reached", player, player))
        self._last_succeeded = succeeded
        self._last_events = tuple(events)
        self._event_history.append(self._last_events)
        return GridPushStep(
            observation=self.observe(),
            reward=1.0 if success else 0.0,
            action_succeeded=succeeded,
        )

    def render_ascii(self) -> str:
        glyphs = {
            "wall": "#", "floor": ".", "goal": "G", "pit": "O",
            "pit_filled": "=", "pressure_plate": "P",
            "door_closed": "D", "door_open": "d",
        }
        rows: list[str] = []
        for y in range(self._spec.height):
            row = []
            for x in range(self._spec.width):
                position = (x, y)
                glyph = glyphs[self._base_tile(position)]
                if position in self._state.blocks:
                    glyph = "B" if position not in self._spec.plates else "b"
                if position == self._state.player:
                    glyph = "A"
                row.append(glyph)
            rows.append("".join(row))
        return "\n".join(rows)



_SOLVER_EXPORTS = {
    "GridSolution",
    "SolverResult",
    "GridWorldCertification",
    "ProceduralGridPushGenerator",
    "solve_grid_world",
    "certify_grid_world",
}


def __getattr__(name: str):
    """Backward-compatible lazy access without loading solver in runtime use."""

    if name in _SOLVER_EXPORTS:
        from . import grid_push_solver

        return getattr(grid_push_solver, name)
    raise AttributeError(name)
