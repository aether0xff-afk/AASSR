from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from .experiment import _run_seed_jobs, _world_seed
from .gridworld import ActionCandidate, DMPConfig, GridWorldDMP
from .knowledge import KK
from .mdp_baseline import write_mdp_outputs
from .metrics import (
    EpisodeMetric,
    StepMetric,
    action_signature,
    episode_metric,
    step_metric,
    summary_metric,
)
from .prophecy import gridworld_state_signature
from .worlds import WorldKind, make_world


def run_q_learning_baseline(
    *,
    episodes: int,
    seeds: int,
    step_limit: int,
    world: str | WorldKind = WorldKind.RANDOM_KEY_DOOR,
    output_dir: str | Path | None = None,
    condition: str = "QLEARN",
    alpha: float = 0.25,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    workers: int = 1,
    progress: bool = False,
    progress_label: str | None = None,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[Any]]:
    world = WorldKind(world)

    args = [
        (seed, episodes, step_limit, world.value, condition, alpha, gamma, epsilon)
        for seed in range(seeds)
    ]
    chunks = _run_seed_jobs(
        _run_q_learning_seed,
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


def _run_q_learning_seed(
    args: tuple[int, int, int, str, str, float, float, float],
) -> tuple[list[StepMetric], list[EpisodeMetric]]:
    seed, episodes, step_limit, world_value, condition, alpha, gamma, epsilon = args
    world = WorldKind(world_value)
    rng = random.Random(seed)
    q: dict[tuple[Any, str], float] = {}
    seed_steps: list[StepMetric] = []
    seed_episodes: list[EpisodeMetric] = []

    for episode in range(episodes):
        dmp = GridWorldDMP(
            make_world(world, seed=_world_seed(seed, episode)),
            config=DMPConfig(),
            step_limit=step_limit,
        )
        step_rows: list[StepMetric] = []
        while not dmp.done:
            state = q_state(dmp)
            candidates = dmp.generate_candidates()
            if not candidates:
                break
            candidate = choose_q_candidate(candidates, state, q, rng, epsilon=epsilon)
            action_key = q_action(candidate)
            result = dmp.execute(candidate)
            next_state = q_state(dmp)
            next_candidates = dmp.generate_candidates()
            next_best = max(
                (q.get((next_state, q_action(next_candidate)), 0.0) for next_candidate in next_candidates),
                default=0.0,
            )
            current = q.get((state, action_key), 0.0)
            q[(state, action_key)] = current + alpha * (result.total_reward + gamma * next_best - current)
            row = step_metric(
                condition=condition,
                seed=seed,
                episode=episode,
                result=result,
            )
            step_rows.append(row)
            seed_steps.append(row)
        seed_episodes.append(
            episode_metric(
                condition=condition,
                seed=seed,
                episode=episode,
                steps=step_rows,
                knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                step_limit=step_limit,
            )
        )
    return seed_steps, seed_episodes


def choose_q_candidate(
    candidates: list[ActionCandidate],
    state: Any,
    q: dict[tuple[Any, str], float],
    rng: random.Random,
    *,
    epsilon: float,
) -> ActionCandidate:
    if rng.random() < epsilon:
        return rng.choice(candidates)
    return max(
        candidates,
        key=lambda candidate: (
            q.get((state, q_action(candidate)), 0.0),
            -len(candidate.bindings),
        ),
    )


def q_state(dmp: GridWorldDMP) -> tuple[Any, tuple[int, int]]:
    return gridworld_state_signature(dmp), dmp.position


def q_action(candidate: ActionCandidate) -> str:
    return action_signature(candidate)


def used_count(dmp: GridWorldDMP, candidate: ActionCandidate) -> int:
    counts = []
    for kk, value in candidate.bindings.items():
        if kk == KK.CURRENT_POS:
            continue
        for kv in dmp.store.values(kk, include_inactive=True):
            if kv.value == value:
                counts.append(kv.used_count)
    return min(counts or [0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional GridWorld baselines.")
    parser.add_argument("--baseline", choices=["qlearn"], required=True)
    parser.add_argument("--world", choices=[world.value for world in WorldKind], default=WorldKind.RANDOM_KEY_DOOR.value)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition = "QLEARN"
    output_dir = args.output_dir or "runs/traditional/QLEARN"
    _, _, summaries = run_q_learning_baseline(
        episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        world=args.world,
        output_dir=output_dir,
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
    print(f"wrote {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
