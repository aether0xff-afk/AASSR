from __future__ import annotations

import argparse
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Any

from .experiment import _run_seed_jobs, _write_csv, _world_seed
from .gridworld import Cell, CellKind, GridWorld
from .metrics import EpisodeMetric, StepMetric, SummaryMetric, summary_metric
from .worlds import WorldKind, make_world


State = tuple[Cell, bool, frozenset[Cell]]


def run_mdp_baseline(
    *,
    episodes: int,
    seeds: int,
    step_limit: int,
    world: str | WorldKind = WorldKind.RANDOM_KEY_DOOR,
    output_dir: str | Path | None = None,
    condition: str = "ORACLE_MDP",
    workers: int = 1,
    progress: bool = False,
    progress_label: str | None = None,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[SummaryMetric]]:
    world = WorldKind(world)

    args = [(seed, episodes, step_limit, world.value, condition) for seed in range(seeds)]
    chunks = _run_seed_jobs(
        _run_mdp_seed,
        args,
        workers=workers,
        progress=progress,
        label=progress_label or condition,
    )

    all_steps = [row for steps, _ in chunks for row in steps]
    all_episodes = [row for _, episode_rows in chunks for row in episode_rows]

    summaries = [summary_metric(condition, all_episodes)]
    if output_dir is not None:
        write_mdp_outputs(output_dir, all_steps, all_episodes, summaries)
    return all_steps, all_episodes, summaries


def _run_mdp_seed(args: tuple[int, int, int, str, str]) -> tuple[list[StepMetric], list[EpisodeMetric]]:
    seed, episodes, step_limit, world_value, condition = args
    world = WorldKind(world_value)
    seed_steps: list[StepMetric] = []
    seed_episodes: list[EpisodeMetric] = []
    for episode in range(episodes):
        grid = make_world(world, seed=_world_seed(seed, episode))
        path = shortest_mdp_plan(grid, step_limit=step_limit)
        success = bool(path) and len(path) <= step_limit
        step_count = len(path) if success else step_limit
        step_rows = [
            StepMetric(
                condition=condition,
                seed=seed,
                episode=episode,
                step=index,
                action=action,
                template="MDP_PRIMITIVE_ACTION",
                action_signature=f"MDP|{action}",
                total_reward=1.0 if success and index == step_count else 0.0,
                external_reward=1.0 if success and index == step_count else 0.0,
                intrinsic_reward=0.0,
                semantic_gain=0,
                prophecy_error=0.0,
                error=False,
                flag_found=success and index == step_count,
                done=success and index == step_count,
                imagination_score=0.0,
                imagination_candidate_count=0,
            )
            for index, action in enumerate(path if success else [], start=1)
        ]
        seed_steps.extend(step_rows)
        seed_episodes.append(
            EpisodeMetric(
                condition=condition,
                seed=seed,
                episode=episode,
                success=success,
                steps_to_flag=step_count,
                total_reward=1.0 if success else 0.0,
                external_reward=1.0 if success else 0.0,
                semantic_gain_total=0,
                prophecy_error_mean=0.0,
                repeat_count=0,
                error_count=0,
                knowledge_reuse_count=0,
                unique_action_count=len(set(path)) if success else 0,
            )
        )
    return seed_steps, seed_episodes


def shortest_mdp_plan(world: GridWorld, *, step_limit: int) -> list[str]:
    flags = {cell for cell, kind in world.cells.items() if kind == CellKind.FLAG}
    if not flags:
        return []
    initial: State = (world.start, False, frozenset())
    queue: deque[tuple[State, list[str]]] = deque([(initial, [])])
    seen = {initial}

    while queue:
        state, path = queue.popleft()
        position, has_key, opened = state
        if position in flags:
            return path
        if len(path) >= step_limit:
            continue
        for action, next_state in transitions(world, state):
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_state, path + [action]))
    return []


def transitions(world: GridWorld, state: State) -> list[tuple[str, State]]:
    position, has_key, opened = state
    results = []
    for action, delta in {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }.items():
        target = (position[0] + delta[0], position[1] + delta[1])
        if not _can_enter(world, target, has_key=has_key, opened=opened):
            continue
        next_has_key = has_key or world.kind_at(target) == CellKind.KEY
        results.append((action, (target, next_has_key, opened)))
    if has_key:
        for door in _adjacent_doors(world, position):
            if door not in opened:
                results.append((f"OPEN {door}", (position, has_key, frozenset(set(opened) | {door}))))
    return results


def _can_enter(world: GridWorld, cell: Cell, *, has_key: bool, opened: frozenset[Cell]) -> bool:
    if not world.in_bounds(cell):
        return False
    kind = world.kind_at(cell)
    if kind == CellKind.WALL:
        return False
    if kind == CellKind.DOOR and cell not in opened:
        return False
    return True


def _adjacent_doors(world: GridWorld, position: Cell) -> list[Cell]:
    x, y = position
    neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    return [cell for cell in neighbors if world.in_bounds(cell) and world.kind_at(cell) == CellKind.DOOR]


def write_mdp_outputs(
    output_dir: str | Path,
    steps: list[StepMetric],
    episodes: list[EpisodeMetric],
    summaries: list[SummaryMetric],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "gridworld_steps.csv", steps, StepMetric)
    _write_csv(output_path / "gridworld_episodes.csv", episodes, EpisodeMetric)
    _write_csv(output_path / "gridworld_summary.csv", summaries, SummaryMetric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full-map shortest-path oracle MDP upper bound for GridWorld.")
    parser.add_argument("--world", choices=[world.value for world in WorldKind], default=WorldKind.RANDOM_KEY_DOOR.value)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--output-dir", default="runs/mdp_baseline/ORACLE_MDP")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, summaries = run_mdp_baseline(
        episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        world=args.world,
        output_dir=args.output_dir,
        workers=args.workers,
        progress=True,
    )
    summary = summaries[0]
    print(
        "condition={condition} episodes={episodes} seeds={seeds} "
        "success_rate={success_rate:.3f} steps_to_flag_mean={steps_to_flag_mean:.3f}".format(
            **summary.to_dict()
        )
    )
    print(f"wrote {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
