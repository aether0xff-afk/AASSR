from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

from .gridworld import DIRECTIONS, Position, format_position, parse_position
from .types import Action, ActionVerb, StateSnapshot


DEFAULT_COLORS = ("red", "blue", "green")


@dataclass(frozen=True, slots=True)
class EscapeBox:
    box_id: str
    position: Position
    key_color: str | None


@dataclass(frozen=True, slots=True)
class EscapeGridSpec:
    width: int
    height: int
    start: Position
    goal: Position
    walls: frozenset[Position]
    doors: tuple[tuple[Position, str], ...]
    boxes: tuple[EscapeBox, ...]
    colors: tuple[str, ...]
    seed: int
    max_steps: int = 180

    def __post_init__(self) -> None:
        if self.width < 7 or self.height < 7:
            raise ValueError("escape grid must be at least 7x7")
        if not self.colors:
            raise ValueError("at least one key color is required")
        occupied = [
            self.start,
            self.goal,
            *self.walls,
            *(position for position, _ in self.doors),
            *(box.position for box in self.boxes),
        ]
        if any(
            not (0 <= x < self.width and 0 <= y < self.height)
            for x, y in occupied
        ):
            raise ValueError("all positions must be inside the grid")
        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("start and goal cannot be walls")
        if len({box.box_id for box in self.boxes}) != len(self.boxes):
            raise ValueError("box ids must be unique")


@dataclass(frozen=True, slots=True)
class EscapeGridStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    goal_reached: bool = False
    reward: float = 0.0
    event: str = ""


class EscapeGridWorld:
    """Deterministic escape room with neutral boxes and colored key-door pairs.

    The world exposes only primitive movement and interaction. Boxes are visible,
    but their contents are unknown until opened. Doors reveal their color. The
    only external reward is delivered when the agent reaches the exit.
    """

    def __init__(self, spec: EscapeGridSpec) -> None:
        self.spec = spec
        self.position = spec.start
        self.inventory: set[str] = set()
        self.open_boxes: set[str] = set()
        self.open_doors: set[Position] = set()
        self.steps = 0
        self.doors = dict(spec.doors)
        self.boxes_by_id = {box.box_id: box for box in spec.boxes}
        self.boxes_by_position = {box.position: box for box in spec.boxes}

    def clone(self) -> EscapeGridWorld:
        clone = EscapeGridWorld(self.spec)
        clone.position = self.position
        clone.inventory = set(self.inventory)
        clone.open_boxes = set(self.open_boxes)
        clone.open_doors = set(self.open_doors)
        clone.steps = self.steps
        return clone

    @property
    def done(self) -> bool:
        return self.position == self.spec.goal or self.steps >= self.spec.max_steps

    @property
    def success(self) -> bool:
        return self.position == self.spec.goal

    def _in_bounds(self, position: Position) -> bool:
        return (
            0 <= position[0] < self.spec.width
            and 0 <= position[1] < self.spec.height
        )

    def _adjacent(self, position: Position) -> bool:
        return (
            abs(position[0] - self.position[0])
            + abs(position[1] - self.position[1])
            <= 1
        )

    def _blocked(self, position: Position) -> bool:
        return (
            not self._in_bounds(position)
            or position in self.spec.walls
            or (position in self.doors and position not in self.open_doors)
        )

    def _available_actions(self) -> tuple[Action, ...]:
        if self.done:
            return ()
        actions: list[Action] = [
            Action(ActionVerb.MOVE, destination=direction)
            for direction in DIRECTIONS
        ]
        box = self.boxes_by_position.get(self.position)
        if box is not None and box.box_id not in self.open_boxes:
            actions.append(Action("interact", target=f"box:{box.box_id}"))
        for position in self.doors:
            if self._adjacent(position) and position not in self.open_doors:
                actions.append(
                    Action("interact", target=f"door:{format_position(position)}")
                )
        return tuple(sorted(actions, key=lambda item: item.signature))

    def _vector(self) -> tuple[float, ...]:
        positions = [
            (x, y)
            for y in range(self.spec.height)
            for x in range(self.spec.width)
        ]
        vector: list[float] = [
            1.0 if position == self.position else 0.0 for position in positions
        ]
        vector.extend(
            1.0 if color in self.inventory else 0.0
            for color in self.spec.colors
        )
        vector.extend(
            1.0 if box.box_id in self.open_boxes else 0.0
            for box in self.spec.boxes
        )
        vector.extend(
            1.0 if position in self.open_doors else 0.0
            for position, _ in self.spec.doors
        )
        vector.append(1.0 if self.success else 0.0)
        return tuple(vector)

    def state_key(self) -> tuple[object, ...]:
        return (
            self.position,
            tuple(sorted(self.inventory)),
            tuple(sorted(self.open_boxes)),
            tuple(sorted(self.open_doors)),
        )

    def snapshot(self) -> StateSnapshot:
        facts = {
            f"agent.position={format_position(self.position)}",
            *(f"inventory:key:{color}" for color in self.inventory),
            *(f"visible:wall:{format_position(position)}" for position in self.spec.walls),
            *(
                f"visible:door:{color}:{format_position(position)}:"
                f"{'open' if position in self.open_doors else 'closed'}"
                for position, color in self.spec.doors
            ),
            *(
                f"visible:box:{box.box_id}:{format_position(box.position)}:"
                f"{'open' if box.box_id in self.open_boxes else 'closed'}"
                for box in self.spec.boxes
            ),
            f"visible:goal:{format_position(self.spec.goal)}",
        }
        return StateSnapshot(
            vector=self._vector(),
            facts=frozenset(facts),
            available_actions=self._available_actions(),
            goal_progress=1.0 if self.success else 0.0,
            metadata={
                "position": self.position,
                "inventory": tuple(sorted(self.inventory)),
                "open_boxes": tuple(sorted(self.open_boxes)),
                "open_doors": tuple(sorted(self.open_doors)),
                "steps": self.steps,
                "max_steps": self.spec.max_steps,
            },
        )

    def step(self, action: Action) -> EscapeGridStep:
        if self.done:
            return EscapeGridStep(
                self.snapshot(),
                error=True,
                goal_reached=self.success,
                event="episode_finished",
            )
        before = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        before_facts = set(before.facts)
        error = False
        event = ""

        if action.verb_name == ActionVerb.MOVE.value and action.destination in DIRECTIONS:
            dx, dy = DIRECTIONS[action.destination]
            target = (self.position[0] + dx, self.position[1] + dy)
            if self._blocked(target):
                error = True
                event = "blocked"
            else:
                self.position = target
                event = "moved"
        elif action.verb_name == "interact" and action.target:
            if action.target.startswith("box:"):
                box_id = action.target.split(":", 1)[1]
                box = self.boxes_by_id.get(box_id)
                if (
                    box is None
                    or box.position != self.position
                    or box_id in self.open_boxes
                ):
                    error = True
                    event = "invalid_box"
                else:
                    self.open_boxes.add(box_id)
                    if box.key_color is not None:
                        self.inventory.add(box.key_color)
                        event = f"found_key:{box.key_color}"
                    else:
                        event = "empty_box"
            elif action.target.startswith("door:"):
                position = parse_position(action.target.split(":", 1)[1])
                color = self.doors.get(position)
                if (
                    color is None
                    or not self._adjacent(position)
                    or position in self.open_doors
                ):
                    error = True
                    event = "invalid_door"
                elif color not in self.inventory:
                    error = True
                    event = f"missing_key:{color}"
                else:
                    self.open_doors.add(position)
                    event = f"opened_door:{color}"
            else:
                error = True
                event = "invalid_target"
        else:
            error = True
            event = "invalid_action"

        self.steps += 1
        after = self.snapshot()
        after_actions = {
            item.signature: item for item in after.available_actions
        }
        reward = 1.0 if self.success else 0.0
        return EscapeGridStep(
            snapshot=after,
            added_facts=frozenset(set(after.facts) - before_facts),
            removed_facts=frozenset(before_facts - set(after.facts)),
            unlocked_actions=tuple(
                after_actions[signature]
                for signature in sorted(after_actions.keys() - before_actions)
            ),
            error=error,
            goal_reached=self.success,
            reward=reward,
            event=event,
        )


def _choose_free(
    randomizer: random.Random,
    candidates: Iterable[Position],
    occupied: set[Position],
) -> Position:
    options = [position for position in candidates if position not in occupied]
    if not options:
        raise RuntimeError("no free position remains while generating the world")
    return randomizer.choice(options)


def generate_escape_grid(
    seed: int,
    *,
    color_count: int = 3,
    distractor_boxes: int = 2,
    height: int = 9,
    max_steps: int = 180,
) -> EscapeGridSpec:
    if not 1 <= color_count <= len(DEFAULT_COLORS):
        raise ValueError(f"color_count must be between 1 and {len(DEFAULT_COLORS)}")
    if distractor_boxes < 0:
        raise ValueError("distractor_boxes must be non-negative")
    colors = DEFAULT_COLORS[:color_count]
    width = 4 * color_count + 5
    randomizer = random.Random(seed)

    walls: set[Position] = {
        *( (x, 0) for x in range(width) ),
        *( (x, height - 1) for x in range(width) ),
        *( (0, y) for y in range(height) ),
        *( (width - 1, y) for y in range(height) ),
    }
    doors: list[tuple[Position, str]] = []
    barrier_xs = [4 + 4 * index for index in range(color_count)]
    for index, barrier_x in enumerate(barrier_xs):
        door_y = randomizer.randint(2, height - 3)
        for y in range(1, height - 1):
            if y != door_y:
                walls.add((barrier_x, y))
        doors.append(((barrier_x, door_y), colors[index]))

    start = (1, height // 2)
    goal = (width - 2, height // 2)
    occupied = set(walls) | {start, goal} | {position for position, _ in doors}
    boxes: list[EscapeBox] = []

    for index, color in enumerate(colors):
        left = 1 if index == 0 else barrier_xs[index - 1] + 1
        right = barrier_xs[index] - 1
        candidates = (
            (x, y)
            for x in range(left, right + 1)
            for y in range(1, height - 1)
        )
        position = _choose_free(randomizer, candidates, occupied)
        occupied.add(position)
        boxes.append(EscapeBox(f"key_box_{index}", position, color))

    region_bounds = [(1, barrier_xs[0] - 1)] + [
        (barrier_xs[index - 1] + 1, barrier_xs[index] - 1)
        for index in range(1, color_count)
    ]
    for index in range(distractor_boxes):
        left, right = randomizer.choice(region_bounds)
        candidates = (
            (x, y)
            for x in range(left, right + 1)
            for y in range(1, height - 1)
        )
        position = _choose_free(randomizer, candidates, occupied)
        occupied.add(position)
        boxes.append(EscapeBox(f"distractor_{index}", position, None))

    return EscapeGridSpec(
        width=width,
        height=height,
        start=start,
        goal=goal,
        walls=frozenset(walls),
        doors=tuple(doors),
        boxes=tuple(boxes),
        colors=tuple(colors),
        seed=seed,
        max_steps=max_steps,
    )


def oracle_plan(spec: EscapeGridSpec) -> tuple[Action, ...]:
    """Return a shortest valid solution, or raise if the generated world is invalid."""

    start = EscapeGridWorld(spec)
    queue: deque[tuple[EscapeGridWorld, tuple[Action, ...]]] = deque([(start, ())])
    visited = {start.state_key()}
    while queue:
        environment, path = queue.popleft()
        if environment.success:
            return path
        for action in environment.snapshot().available_actions:
            candidate = environment.clone()
            outcome = candidate.step(action)
            if outcome.error:
                continue
            key = candidate.state_key()
            if key in visited:
                continue
            visited.add(key)
            queue.append((candidate, path + (action,)))
    raise RuntimeError(f"generated escape grid seed={spec.seed} is not solvable")
