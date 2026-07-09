from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from .gridworld import CellKind, GridWorld
from .worlds import WorldKind, make_world


CELL_STYLES = {
    CellKind.EMPTY: ("#f8fafc", "#cbd5e1", ""),
    CellKind.WALL: ("#334155", "#0f172a", "W"),
    CellKind.HINT: ("#fde68a", "#b45309", "H"),
    CellKind.KEY: ("#bfdbfe", "#1d4ed8", "K"),
    CellKind.DOOR: ("#fed7aa", "#c2410c", "D"),
    CellKind.FLAG: ("#bbf7d0", "#15803d", "F"),
}


def render_world_png(
    world: GridWorld,
    output_path: str | Path,
    *,
    title: str = "GridWorld",
    show_coordinates: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(6.0, world.width * 0.75)
    fig_height = max(4.5, world.height * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    ax.set_aspect("equal")
    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_title(title, fontsize=16, pad=14)

    for y in range(world.height):
        for x in range(world.width):
            cell = (x, y)
            kind = world.kind_at(cell)
            fill, edge, label = CELL_STYLES[kind]
            rect = patches.Rectangle((x, y), 1, 1, facecolor=fill, edgecolor=edge, linewidth=1.2)
            ax.add_patch(rect)
            if label:
                ax.text(x + 0.5, y + 0.52, label, ha="center", va="center", fontsize=14, weight="bold", color=edge)

    start_x, start_y = world.start
    start = patches.Circle((start_x + 0.5, start_y + 0.5), 0.32, facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=2.0)
    ax.add_patch(start)
    ax.text(start_x + 0.5, start_y + 0.52, "S", ha="center", va="center", fontsize=14, weight="bold", color="#0369a1")

    for hint_cell, target in world.hints.items():
        if isinstance(target, tuple) and len(target) == 2:
            hx, hy = hint_cell
            tx, ty = target
            ax.annotate(
                "",
                xy=(tx + 0.5, ty + 0.5),
                xytext=(hx + 0.5, hy + 0.5),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.6,
                    "color": "#92400e",
                    "alpha": 0.75,
                    "shrinkA": 14,
                    "shrinkB": 14,
                },
            )

    if show_coordinates:
        ax.set_xticks([x + 0.5 for x in range(world.width)], labels=[str(x) for x in range(world.width)])
        ax.set_yticks([y + 0.5 for y in range(world.height)], labels=[str(y) for y in range(world.height)])
        ax.tick_params(length=0, labelsize=9)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    _add_legend(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_worlds(
    *,
    worlds: tuple[WorldKind, ...],
    seed: int,
    output_dir: str | Path,
    show_coordinates: bool = True,
) -> list[Path]:
    output_path = Path(output_dir)
    paths = []
    for world_kind in worlds:
        world = make_world(world_kind, seed=seed)
        paths.append(
            render_world_png(
                world,
                output_path / f"{world_kind.value}_seed_{seed}.png",
                title=f"{world_kind.value} (seed={seed})",
                show_coordinates=show_coordinates,
            )
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GridWorld environments as presentation-ready PNG images.")
    parser.add_argument("--world", default="v2_complex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/world_renders")
    parser.add_argument("--hide-coordinates", action="store_true")
    return parser.parse_args()


def parse_worlds(value: str) -> tuple[WorldKind, ...]:
    if value == "all":
        return tuple(WorldKind)
    return tuple(WorldKind(item.strip()) for item in value.split(",") if item.strip())


def _add_legend(ax: plt.Axes) -> None:
    legend_items = [
        patches.Patch(facecolor="#e0f2fe", edgecolor="#0284c7", label="Start"),
        patches.Patch(facecolor=CELL_STYLES[CellKind.WALL][0], edgecolor=CELL_STYLES[CellKind.WALL][1], label="Wall"),
        patches.Patch(facecolor=CELL_STYLES[CellKind.KEY][0], edgecolor=CELL_STYLES[CellKind.KEY][1], label="Key"),
        patches.Patch(facecolor=CELL_STYLES[CellKind.DOOR][0], edgecolor=CELL_STYLES[CellKind.DOOR][1], label="Door"),
        patches.Patch(facecolor=CELL_STYLES[CellKind.HINT][0], edgecolor=CELL_STYLES[CellKind.HINT][1], label="Hint"),
        patches.Patch(facecolor=CELL_STYLES[CellKind.FLAG][0], edgecolor=CELL_STYLES[CellKind.FLAG][1], label="Flag"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=6,
        frameon=False,
        fontsize=9,
    )


def main() -> None:
    args = parse_args()
    paths = render_worlds(
        worlds=parse_worlds(args.world),
        seed=args.seed,
        output_dir=args.output_dir,
        show_coordinates=not args.hide_coordinates,
    )
    for path in paths:
        print(f"wrote {path.resolve()}")


if __name__ == "__main__":
    main()
