from __future__ import annotations

import random
from enum import StrEnum

from .gridworld import Cell, CellKind, GridWorld


class WorldKind(StrEnum):
    FIXED = "fixed"
    RANDOM_FLAG = "random_flag"
    RANDOM_WALL_FLAG = "random_wall_flag"
    RANDOM_KEY_DOOR = "random_key_door"
    V2_COMPLEX = "v2_complex"
    LOCKED_BOTTLENECK = "locked_bottleneck"


def make_default_world() -> GridWorld:
    return GridWorld(
        width=6,
        height=4,
        start=(1, 1),
        cells={
            (2, 0): CellKind.WALL,
            (3, 1): CellKind.HINT,
            (1, 2): CellKind.KEY,
            (4, 2): CellKind.DOOR,
            (5, 2): CellKind.FLAG,
        },
        hints={(3, 1): (5, 2)},
    )


def make_world(kind: str | WorldKind, *, seed: int) -> GridWorld:
    kind = WorldKind(kind)
    if kind == WorldKind.FIXED:
        return make_default_world()
    if kind == WorldKind.LOCKED_BOTTLENECK:
        return make_locked_bottleneck_world(seed=seed)
    if kind == WorldKind.V2_COMPLEX:
        return make_v2_complex_world(seed=seed)
    return make_random_world(seed=seed, kind=kind)


def make_random_world(*, seed: int, kind: str | WorldKind) -> GridWorld:
    kind = WorldKind(kind)
    rng = random.Random(seed)
    width = 6
    height = 4
    start = (1, 1)
    cells: dict[Cell, CellKind] = {}

    flag = _choice(rng, _candidate_cells(width, height, start))
    if kind == WorldKind.RANDOM_FLAG:
        cells[flag] = CellKind.FLAG
        return GridWorld(width=width, height=height, start=start, cells=cells)

    wall_candidates = [
        cell for cell in _candidate_cells(width, height, start) if cell != flag
    ]
    wall_count = 3 if kind == WorldKind.RANDOM_WALL_FLAG else 2
    for wall in rng.sample(wall_candidates, k=wall_count):
        cells[wall] = CellKind.WALL

    if kind == WorldKind.RANDOM_WALL_FLAG:
        cells[flag] = CellKind.FLAG
        return GridWorld(width=width, height=height, start=start, cells=cells)

    empty_cells = [
        cell
        for cell in _candidate_cells(width, height, start)
        if cell != flag and cell not in cells
    ]
    key = _choice(rng, empty_cells)
    empty_cells.remove(key)
    door = _choice(rng, empty_cells)
    empty_cells.remove(door)
    hint = _choice(rng, empty_cells)

    cells.update(
        {
            key: CellKind.KEY,
            door: CellKind.DOOR,
            hint: CellKind.HINT,
            flag: CellKind.FLAG,
        }
    )
    return GridWorld(
        width=width,
        height=height,
        start=start,
        cells=cells,
        hints={hint: flag},
    )


def _candidate_cells(width: int, height: int, start: Cell) -> list[Cell]:
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) != start and abs(x - start[0]) + abs(y - start[1]) >= 2
    ]


def _choice(rng: random.Random, cells: list[Cell]) -> Cell:
    return cells[rng.randrange(len(cells))]


def make_v2_complex_world(*, seed: int) -> GridWorld:
    rng = random.Random(seed)
    width = 9
    height = 6
    start = (1, 1)
    candidates = _candidate_cells(width, height, start)

    for _ in range(100):
        cells: dict[Cell, CellKind] = {}
        flag = _choice(rng, _far_cells(candidates, start, minimum_distance=7))
        cells[flag] = CellKind.FLAG

        wall_candidates = [cell for cell in candidates if cell != flag]
        for wall in rng.sample(wall_candidates, k=10):
            cells[wall] = CellKind.WALL

        blocked = {cell for cell, kind in cells.items() if kind == CellKind.WALL}
        if not _reachable(width, height, start, flag, blocked=blocked):
            continue

        empty_cells = [cell for cell in candidates if cell not in cells]
        key_cells = rng.sample(empty_cells, k=2)
        for cell in key_cells:
            cells[cell] = CellKind.KEY
        empty_cells = [cell for cell in empty_cells if cell not in key_cells]

        door_cells = rng.sample(empty_cells, k=2)
        for cell in door_cells:
            cells[cell] = CellKind.DOOR
        empty_cells = [cell for cell in empty_cells if cell not in door_cells]

        hint_cells = rng.sample(empty_cells, k=2)
        hints = {}
        for cell in hint_cells:
            cells[cell] = CellKind.HINT
            hints[cell] = flag

        return GridWorld(
            width=width,
            height=height,
            start=start,
            cells=cells,
            hints=hints,
        )

    raise RuntimeError("Could not generate reachable v2 complex world")


def make_locked_bottleneck_world(*, seed: int) -> GridWorld:
    rng = random.Random(seed)
    width = 10
    height = 6
    start = (1, 1)
    cells: dict[Cell, CellKind] = {}

    for x in range(width):
        cells[(x, 3)] = CellKind.WALL
    door_one = (4, 3)
    door_two = (7, 3)
    cells[door_one] = CellKind.DOOR
    cells[door_two] = CellKind.DOOR

    for wall in ((3, 1), (3, 2), (6, 4), (8, 2), (1, 4), (5, 1)):
        cells[wall] = CellKind.WALL

    key_candidates = [(1, 2), (2, 2), (4, 1), (5, 2)]
    hint_candidates = [(2, 1), (5, 4), (8, 4)]
    key = key_candidates[rng.randrange(len(key_candidates))]
    flag = (8, 5)
    cells[key] = CellKind.KEY
    cells[flag] = CellKind.FLAG
    for hint in rng.sample(hint_candidates, k=2):
        if hint not in cells:
            cells[hint] = CellKind.HINT

    hints = {cell: flag for cell, kind in cells.items() if kind == CellKind.HINT}
    return GridWorld(
        width=width,
        height=height,
        start=start,
        cells=cells,
        hints=hints,
    )


def _far_cells(cells: list[Cell], start: Cell, *, minimum_distance: int) -> list[Cell]:
    far = [
        cell
        for cell in cells
        if abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= minimum_distance
    ]
    return far or cells


def _reachable(width: int, height: int, start: Cell, goal: Cell, *, blocked: set[Cell]) -> bool:
    frontier = [start]
    seen = {start}
    while frontier:
        x, y = frontier.pop(0)
        if (x, y) == goal:
            return True
        for neighbor in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            nx, ny = neighbor
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if neighbor in blocked or neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append(neighbor)
    return False
