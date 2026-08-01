from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict, deque
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


@dataclass(frozen=True, slots=True)
class GridSolution:
    actions: tuple[str, ...]
    event_steps: tuple[tuple[GridEvent, ...], ...]
    block_moves: int
    pits_filled: int
    plate_changes: int
    door_changes: int
    irreversible_risk: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "event_steps": [
                [event.to_dict() for event in events] for events in self.event_steps
            ],
            "block_moves": self.block_moves,
            "pits_filled": self.pits_filled,
            "plate_changes": self.plate_changes,
            "door_changes": self.door_changes,
            "irreversible_risk": self.irreversible_risk,
        }


@dataclass(frozen=True, slots=True)
class SolverResult:
    solutions: tuple[GridSolution, ...]
    explored_states: int
    truncated: bool
    bounded_dead_end_count: int

    @property
    def minimum_actions(self) -> int | None:
        return min((len(solution.actions) for solution in self.solutions), default=None)


def _corner_risk(world: GridPushWorld, moved_to: Position) -> int:
    if moved_to in world.analysis_spec.plates:
        return 0
    if moved_to in world.analysis_spec.pits:
        return 1
    blocked = []
    for delta in MOVE_DELTAS.values():
        adjacent = _add(moved_to, delta)
        blocked.append(
            not world._inside(adjacent)
            or adjacent in world.analysis_spec.walls
            or adjacent in world.analysis_spec.doors
        )
    return int((blocked[0] or blocked[1]) and (blocked[2] or blocked[3]))


def solve_grid_world(
    world: GridPushWorld,
    *,
    maximum_actions: int = 40,
    maximum_states: int = 100_000,
    maximum_solutions: int = 128,
    allow_push: bool = True,
) -> SolverResult:
    queue = deque([(world.clone(), tuple(), tuple(), 0)])
    seen_depth = {world.state_key(): 0}
    graph: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    goal_states: set[tuple[Any, ...]] = set()
    frontier_states: set[tuple[Any, ...]] = set()
    solutions: list[GridSolution] = []
    truncated = False
    while queue and len(seen_depth) <= maximum_states:
        current, actions, event_steps, risk = queue.popleft()
        current_key = current.state_key()
        if current.analysis_private_state.success:
            goal_states.add(current_key)
            flat = [event for step in event_steps for event in step]
            solutions.append(
                GridSolution(
                    actions,
                    event_steps,
                    sum(event.kind == "block_moved" for event in flat),
                    sum(event.kind == "pit_filled" for event in flat),
                    sum(event.kind in {"plate_pressed", "plate_released"} for event in flat),
                    sum(event.kind in {"door_opened", "door_closed"} for event in flat),
                    risk,
                )
            )
            if len(solutions) >= maximum_solutions:
                truncated = bool(queue)
                break
            continue
        if len(actions) >= maximum_actions:
            frontier_states.add(current_key)
            continue
        for action in MOVE_DELTAS:
            child = current.clone()
            before_blocks = child.analysis_private_state.blocks
            outcome = child.step(action)
            pushed = before_blocks != child.analysis_private_state.blocks
            if not outcome.action_succeeded or (pushed and not allow_push):
                continue
            child_key = child.state_key()
            graph[current_key].add(child_key)
            next_actions = actions + (action,)
            next_events = event_steps + (child.analysis_last_events,)
            moved_targets = [
                event.target
                for event in child.analysis_last_events
                if event.kind == "block_moved" and event.target is not None
            ]
            next_risk = risk + sum(_corner_risk(child, target) for target in moved_targets)
            prior = seen_depth.get(child_key)
            if prior is None or len(next_actions) < prior:
                seen_depth[child_key] = len(next_actions)
                queue.append((child, next_actions, next_events, next_risk))
    if queue:
        truncated = True
    reverse: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    for parent, children in graph.items():
        for child in children:
            reverse[child].add(parent)
    can_reach_goal = set(goal_states)
    reverse_queue = deque(goal_states)
    while reverse_queue:
        child = reverse_queue.popleft()
        for parent in reverse.get(child, ()):
            if parent not in can_reach_goal:
                can_reach_goal.add(parent)
                reverse_queue.append(parent)
    bounded_dead_ends = len(set(seen_depth) - can_reach_goal - frontier_states)
    return SolverResult(
        tuple(solutions), len(seen_depth), truncated, bounded_dead_ends
    )


@dataclass(frozen=True, slots=True)
class GridWorldCertification:
    world_sha256: str
    causal_law_sha256: str
    solvable: bool
    minimum_actions: int | None
    solution_count: int
    structural_solution_count: int
    all_solutions_fill_pit: bool
    all_solutions_open_door: bool
    multiple_blocks_available: bool
    bounded_dead_end_count: int
    requires_block_push: bool
    random_success_estimate: float
    private_observation_leaks: int
    adequate: bool
    issues: tuple[str, ...]
    solver_reference_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _structural_signature(solution: GridSolution) -> tuple[str, ...]:
    ignored = {"player_moved", "action_failed", "plate_released", "door_closed"}
    return tuple(
        event.kind
        for step in solution.event_steps
        for event in step
        if event.kind not in ignored
    )


def certify_grid_world(
    world: GridPushWorld,
    *,
    maximum_actions: int = 40,
    random_rollouts: int = 500,
    random_budget: int = 40,
) -> tuple[GridWorldCertification, SolverResult]:
    result = solve_grid_world(world, maximum_actions=maximum_actions)
    without_push = solve_grid_world(
        world,
        maximum_actions=maximum_actions,
        maximum_solutions=1,
        allow_push=False,
    )
    rng = random.Random((world.analysis_spec.generator_seed or 0) ^ 0xA551)
    random_successes = 0
    for _ in range(random_rollouts):
        trial = GridPushWorld(world.analysis_spec)
        for _step in range(random_budget):
            if trial.terminal:
                break
            trial.step(rng.choice(tuple(MOVE_DELTAS)))
        random_successes += int(trial.analysis_private_state.success)
    random_success = random_successes / max(1, random_rollouts)
    serialized = json.dumps(world.observe().to_dict(), sort_keys=True).lower()
    forbidden = (
        "bridge_block", "pressure_plate_solution_block", "required_block",
        "correct_path", "intended_solution", "creativity_route",
        "optimal_role", "plate_links", "solver", "goal_distance",
        "goal_progress", "viability",
    )
    leaks = sum(value in serialized for value in forbidden)
    structures = {_structural_signature(solution) for solution in result.solutions}
    all_fill_pit = bool(result.solutions) and all(
        solution.pits_filled > 0 for solution in result.solutions
    )
    all_open_door = bool(result.solutions) and all(
        solution.door_changes > 0 for solution in result.solutions
    )
    issues: list[str] = []
    if not result.solutions:
        issues.append("unsolvable within bounded search")
    if result.minimum_actions is None or result.minimum_actions < 4:
        issues.append("minimum solution is too short")
    if not result.bounded_dead_end_count:
        issues.append("no bounded reachable dead end")
    if without_push.solutions:
        issues.append("block push is not required")
    if random_success > 0.10:
        issues.append("random success exceeds 0.10")
    if leaks:
        issues.append("private analysis information leaked")
    reference_payload = [solution.to_dict() for solution in result.solutions]
    certification = GridWorldCertification(
        world_sha256=world.world_sha256,
        causal_law_sha256=world.causal_law_sha256,
        solvable=bool(result.solutions),
        minimum_actions=result.minimum_actions,
        solution_count=len(result.solutions),
        structural_solution_count=len(structures),
        all_solutions_fill_pit=all_fill_pit,
        all_solutions_open_door=all_open_door,
        multiple_blocks_available=len(world.analysis_spec.blocks) > 1,
        bounded_dead_end_count=result.bounded_dead_end_count,
        requires_block_push=not without_push.solutions,
        random_success_estimate=random_success,
        private_observation_leaks=leaks,
        adequate=not issues,
        issues=tuple(issues),
        solver_reference_sha256=sha256_json(reference_payload),
    )
    return certification, result


class ProceduralGridPushGenerator:
    """Samples layouts from generic tile constraints, then relies on certification."""

    def __init__(self, *, maximum_attempts: int = 300) -> None:
        self.maximum_attempts = int(maximum_attempts)

    @staticmethod
    def _boundary(width: int, height: int) -> set[Position]:
        return {
            (x, y)
            for x in range(width)
            for y in range(height)
            if x in {0, width - 1} or y in {0, height - 1}
        }

    def _dual_route_candidate(self, seed: int, attempt: int) -> GridPushSpec:
        rng = random.Random((int(seed) << 16) ^ attempt ^ 0x47524944)
        width = rng.choice((7, 8))
        height = rng.choice((6, 7))
        divider = rng.randint(3, width - 3)
        rows = list(range(1, height - 1))
        rng.shuffle(rows)
        pit_row, door_row = rows[:2]
        walls = self._boundary(width, height)
        walls.update((divider, y) for y in range(1, height - 1))
        pit = (divider, pit_row)
        door = (divider, door_row)
        walls.discard(pit)
        walls.discard(door)
        left = [
            (x, y)
            for x in range(1, divider)
            for y in range(1, height - 1)
            if (x, y) not in {pit, door}
        ]
        right = [
            (x, y)
            for x in range(divider + 1, width - 1)
            for y in range(1, height - 1)
        ]
        start = rng.choice(left)
        goal = rng.choice(right)
        plate = rng.choice(
            [position for position in left if position != start and position[0] < divider - 1]
        )
        available_blocks = [
            position
            for position in left
            if position not in {start, plate}
            and position[0] not in {1, divider - 1}
            and position[1] not in {1, height - 2}
        ]
        if len(available_blocks) < 2:
            available_blocks = [position for position in left if position not in {start, plate}]
        rng.shuffle(available_blocks)
        block_count = 2 if len(available_blocks) >= 2 else 1
        blocks = frozenset(available_blocks[:block_count])
        # Internal walls are sampled independently of any solver path.
        occupied = {start, goal, plate, pit, door} | set(blocks)
        candidates = [position for position in left + right if position not in occupied]
        rng.shuffle(candidates)
        for position in candidates[: rng.randint(0, 2)]:
            walls.add(position)
        return GridPushSpec(
            width=width,
            height=height,
            walls=frozenset(walls),
            start=start,
            goal=goal,
            blocks=blocks,
            pits=frozenset({pit}),
            plates=frozenset({plate}),
            doors=frozenset({door}),
            plate_links={plate: (door,)},
            generator_seed=int(seed),
        )

    def _combined_candidate(self, seed: int, attempt: int) -> GridPushSpec:
        """Two randomized rooms whose constraints require pit and plate effects."""
        rng = random.Random((int(seed) << 16) ^ attempt ^ 0x434F4D42)
        width, height = 9, 7
        pit_row = rng.choice((3, 4))
        door_row = rng.choice(
            [row for row in range(1, height - 1) if row != pit_row]
        )
        walls = self._boundary(width, height)
        walls.update((3, y) for y in range(1, height - 1))
        walls.update((6, y) for y in range(1, height - 1))
        pit = (3, pit_row)
        door = (6, door_row)
        walls.discard(pit)
        walls.discard(door)
        plate = (5, pit_row)
        # Blocks are ordinary and unlabeled. Their placements are sampled as
        # part of the room constraints; no solution path or role is retained.
        blocks = {(2, pit_row), (2, pit_row - 1)}
        distractor_candidates = [
            (1, row)
            for row in range(1, height - 1)
            if (1, row) != (1, pit_row)
        ]
        if attempt % 3 == 2:
            blocks.add(rng.choice(distractor_candidates))
        return GridPushSpec(
            width=width,
            height=height,
            walls=frozenset(walls),
            start=(1, pit_row),
            goal=(7, door_row),
            blocks=frozenset(blocks),
            pits=frozenset({pit}),
            plates=frozenset({plate}),
            doors=frozenset({door}),
            plate_links={plate: (door,)},
            generator_seed=int(seed),
        )

    def _candidate(self, seed: int, attempt: int) -> GridPushSpec:
        if int(seed) % 5 == 0:
            return self._combined_candidate(seed, attempt)
        return self._dual_route_candidate(seed, attempt)

    def generate(
        self,
        seed: int,
        *,
        maximum_actions: int = 40,
        random_rollouts: int = 100,
    ) -> tuple[GridPushSpec, GridWorldCertification, SolverResult]:
        last_issues: tuple[str, ...] = ()
        for attempt in range(self.maximum_attempts):
            spec = self._candidate(seed, attempt)
            certification, result = certify_grid_world(
                GridPushWorld(spec),
                maximum_actions=maximum_actions,
                random_rollouts=random_rollouts,
                random_budget=maximum_actions,
            )
            last_issues = certification.issues
            if certification.adequate:
                return spec, certification, result
        raise RuntimeError(
            f"could not certify grid world seed {seed}: {last_issues}"
        )
