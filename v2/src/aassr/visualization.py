from __future__ import annotations

from .gridworld import ActionCandidate, Cell, CellKind, GridWorldDMP
from .knowledge import KK, KnowledgeStatus


def render_grid(dmp: GridWorldDMP) -> str:
    """Render observed GridWorld knowledge as an ASCII map."""

    rows: list[str] = []
    known = _values(dmp, KK.KNOWN_CELL, include_inactive=True)
    walls = _values(dmp, KK.WALL_CELL, include_inactive=True)
    frontiers = _values(dmp, KK.FRONTIER_CELL)
    hints = _values(dmp, KK.HINT_CELL)
    keys = _values(dmp, KK.KEY_CELL, include_inactive=True)
    doors = _values(dmp, KK.DOOR_CELL, include_inactive=True)
    flags = _values(dmp, KK.FLAG_CELL)

    for y in range(dmp.world.height):
        cells: list[str] = []
        for x in range(dmp.world.width):
            cell = (x, y)
            if cell == dmp.position:
                cells.append("A")
            elif cell in walls:
                cells.append("#")
            elif cell in flags:
                cells.append("F")
            elif cell in doors:
                cells.append("D")
            elif cell in keys:
                cells.append("K")
            elif cell in hints:
                cells.append("H")
            elif cell in frontiers:
                cells.append("?")
            elif cell in known:
                cells.append(".")
            else:
                cells.append(" ")
        rows.append("".join(cells))
    return "\n".join(rows)


def render_knowledge_summary(dmp: GridWorldDMP) -> str:
    """Render active and inactive KV pools grouped by KK."""

    lines = ["Knowledge Storage"]
    snapshot = dmp.store.snapshot()
    if not snapshot:
        return "Knowledge Storage\n  <empty>"

    for kk in sorted(snapshot):
        values = snapshot[kk]
        rendered_values = []
        for item in values:
            marker = "" if item["status"] == KnowledgeStatus.ACTIVE.value else f" [{item['status']}]"
            rendered_values.append(f"{item['value']}{marker}")
        lines.append(f"  {kk}: {', '.join(rendered_values)}")
    return "\n".join(lines)


def render_candidates(candidates: list[ActionCandidate]) -> str:
    """Render action candidates as KK-to-KV bindings."""

    if not candidates:
        return "Action Candidates\n  <none>"

    lines = ["Action Candidates"]
    for index, candidate in enumerate(candidates, start=1):
        bindings = ", ".join(f"{kk.value}={value}" for kk, value in candidate.bindings.items())
        lines.append(f"  {index}. {candidate.template} -> {candidate.name.value}({bindings})")
    return "\n".join(lines)


def render_dmp_state(dmp: GridWorldDMP) -> str:
    return "\n\n".join(
        [
            "GridWorld",
            render_grid(dmp),
            render_knowledge_summary(dmp),
            render_candidates(dmp.generate_candidates()),
        ]
    )


def render_dmp_mermaid() -> str:
    return """```mermaid
flowchart LR
    A["Action"] --> B["Observation"]
    B --> C["KV Extraction"]
    C --> D["Knowledge Storage"]
    D --> E["KK Slot Binding"]
    E --> F["Candidate Actions"]
    F --> G["Policy / Prophecy / Imagination"]
    G --> A
```"""


def _values(dmp: GridWorldDMP, kk: KK, *, include_inactive: bool = False) -> set[Cell]:
    values: set[Cell] = set()
    for kv in dmp.store.values(kk, include_inactive=include_inactive):
        if isinstance(kv.value, tuple) and len(kv.value) == 2:
            values.add(kv.value)
    return values


def demo_state() -> GridWorldDMP:
    from .gridworld import GridWorld

    world = GridWorld(
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
    return GridWorldDMP(world)


if __name__ == "__main__":
    print(render_dmp_state(demo_state()))
