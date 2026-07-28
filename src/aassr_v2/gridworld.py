from __future__ import annotations

from dataclasses import dataclass

from .types import Action, ActionVerb, StateSnapshot


Position = tuple[int, int]
DIRECTIONS: dict[str, Position] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}


def format_position(position: Position) -> str:
    return f"{position[0]},{position[1]}"


def parse_position(value: str) -> Position:
    x_value, y_value = value.split(",", 1)
    return int(x_value), int(y_value)


@dataclass(frozen=True, slots=True)
class GridWorldSpec:
    width: int
    height: int
    start: Position
    goal: Position
    walls: frozenset[Position] = frozenset()
    keys: tuple[tuple[Position, str], ...] = ()
    doors: tuple[tuple[Position, str], ...] = ()
    required_inventory_at_goal: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

        occupied = [
            self.start,
            self.goal,
            *self.walls,
            *(position for position, _ in self.keys),
            *(position for position, _ in self.doors),
        ]
        if any(
            not (0 <= x < self.width and 0 <= y < self.height)
            for x, y in occupied
        ):
            raise ValueError("all positions must be inside the grid")

    @property
    def colors(self) -> tuple[str, ...]:
        colors = (
            {color for _, color in self.keys}
            | {color for _, color in self.doors}
            | set(self.required_inventory_at_goal)
        )
        return tuple(sorted(colors))


@dataclass(frozen=True, slots=True)
class GridWorldStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    goal_reached: bool = False


class GridWorldEnv:
    """Minimal deterministic environment for validating the AASSR v2 loop.

    The environment exposes primitive controls but does not reveal object
    semantics until OBSERVE produces an explicit fact.
    """

    def __init__(self, spec: GridWorldSpec) -> None:
        self.spec = spec
        self.position = spec.start
        self.inventory: set[str] = set()
        self.keys = dict(spec.keys)
        self.doors = dict(spec.doors)
        self.open_doors: set[Position] = set()
        self.known_cells: dict[Position, str] = {}

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

    def _actual_cell_fact(self, position: Position) -> str:
        tag = format_position(position)
        if position in self.spec.walls:
            return f"cell:{tag}=wall"
        if position in self.doors:
            state = "open" if position in self.open_doors else "closed"
            return f"cell:{tag}=door:{self.doors[position]}:{state}"
        if position in self.keys:
            return f"cell:{tag}=key:{self.keys[position]}"
        if position == self.spec.goal:
            return f"cell:{tag}=goal"
        return f"cell:{tag}=empty"

    def _goal_reached(self) -> bool:
        return (
            self.position == self.spec.goal
            and self.spec.required_inventory_at_goal <= self.inventory
        )

    def _available_actions(self) -> tuple[Action, ...]:
        actions = [
            Action(ActionVerb.MOVE, destination=direction)
            for direction in DIRECTIONS
        ]

        for y_value in range(self.spec.height):
            for x_value in range(self.spec.width):
                position = (x_value, y_value)
                if self._adjacent(position):
                    actions.append(
                        Action(ActionVerb.OBSERVE, target=format_position(position))
                    )

        current_fact = self.known_cells.get(self.position, "")
        if "=key:" in current_fact:
            actions.append(
                Action(ActionVerb.PICKUP, target=format_position(self.position))
            )

        for position, fact in self.known_cells.items():
            if not self._adjacent(position):
                continue
            if "=door:" not in fact or ":closed" not in fact:
                continue

            color = fact.split("=door:", 1)[1].split(":", 1)[0]
            if color in self.inventory:
                actions.append(
                    Action(
                        ActionVerb.USE,
                        target=format_position(position),
                        tool=color,
                    )
                )

        return tuple(sorted(actions, key=lambda action: action.signature))

    def _vector(self) -> tuple[float, ...]:
        positions = [
            (x_value, y_value)
            for y_value in range(self.spec.height)
            for x_value in range(self.spec.width)
        ]

        vector: list[float] = [
            1.0 if position == self.position else 0.0
            for position in positions
        ]

        categories = ("unknown", "empty", "wall", "goal", "key", "door")
        for position in positions:
            fact = self.known_cells.get(position)
            category = "unknown"
            if fact:
                value = fact.split("=", 1)[1]
                if value.startswith("key:"):
                    category = "key"
                elif value.startswith("door:"):
                    category = "door"
                else:
                    category = value
            vector.extend(
                1.0 if category == candidate else 0.0
                for candidate in categories
            )

        vector.extend(
            1.0 if color in self.inventory else 0.0
            for color in self.spec.colors
        )
        vector.extend(
            1.0 if position in self.open_doors else 0.0
            for position, _ in sorted(self.spec.doors)
        )
        vector.append(1.0 if self._goal_reached() else 0.0)
        return tuple(vector)

    def snapshot(self) -> StateSnapshot:
        facts = (
            set(self.known_cells.values())
            | {f"inventory:key:{color}" for color in self.inventory}
            | {f"agent.position={format_position(self.position)}"}
        )
        goal_progress = 1.0 if self._goal_reached() else 0.0
        return StateSnapshot(
            vector=self._vector(),
            facts=frozenset(facts),
            available_actions=self._available_actions(),
            goal_progress=goal_progress,
        )

    def step(self, action: Action) -> GridWorldStep:
        before = self.snapshot()
        before_signatures = {
            candidate.signature for candidate in before.available_actions
        }
        added_facts: set[str] = set()
        removed_facts: set[str] = set()
        error = False

        if action.verb is ActionVerb.OBSERVE and action.target:
            position = parse_position(action.target)
            if self._in_bounds(position) and self._adjacent(position):
                old_fact = self.known_cells.get(position)
                new_fact = self._actual_cell_fact(position)
                self.known_cells[position] = new_fact
                if old_fact and old_fact != new_fact:
                    removed_facts.add(old_fact)
                if old_fact != new_fact:
                    added_facts.add(new_fact)
            else:
                error = True

        elif action.verb is ActionVerb.MOVE and action.destination in DIRECTIONS:
            delta_x, delta_y = DIRECTIONS[action.destination]
            target = (
                self.position[0] + delta_x,
                self.position[1] + delta_y,
            )
            blocked = (
                not self._in_bounds(target)
                or target in self.spec.walls
                or (target in self.doors and target not in self.open_doors)
            )
            if blocked:
                error = True
            else:
                self.position = target

        elif action.verb is ActionVerb.PICKUP:
            if self.position in self.keys:
                color = self.keys.pop(self.position)
                old_fact = self.known_cells.get(self.position)
                new_fact = self._actual_cell_fact(self.position)
                self.known_cells[self.position] = new_fact
                if old_fact:
                    removed_facts.add(old_fact)
                added_facts.update({new_fact, f"inventory:key:{color}"})
                self.inventory.add(color)
            else:
                error = True

        elif action.verb is ActionVerb.USE and action.target and action.tool:
            position = parse_position(action.target)
            required_color = self.doors.get(position)
            valid = (
                self._adjacent(position)
                and required_color == action.tool
                and action.tool in self.inventory
            )
            if valid:
                old_fact = self.known_cells.get(position)
                self.open_doors.add(position)
                new_fact = self._actual_cell_fact(position)
                self.known_cells[position] = new_fact
                if old_fact:
                    removed_facts.add(old_fact)
                added_facts.add(new_fact)
            else:
                error = True

        else:
            error = True

        after = self.snapshot()
        after_actions = {
            candidate.signature: candidate
            for candidate in after.available_actions
        }
        unlocked_actions = tuple(
            after_actions[signature]
            for signature in sorted(after_actions.keys() - before_signatures)
        )

        return GridWorldStep(
            snapshot=after,
            added_facts=frozenset(added_facts),
            removed_facts=frozenset(removed_facts),
            unlocked_actions=unlocked_actions,
            error=error,
            goal_reached=self._goal_reached(),
        )
