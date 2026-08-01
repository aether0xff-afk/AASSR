from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

from .grid_push_world import (
    MOVE_DELTAS,
    GridEvent,
    GridPushSpec,
    GridPushWorld,
    Position,
    _add,
)
from .paper_v2_protocol import sha256_json


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
    """Analysis-only generator: candidates are accepted only after certification."""

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
