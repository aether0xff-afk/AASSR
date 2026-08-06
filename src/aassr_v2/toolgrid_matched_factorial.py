from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

from . import toolgrid_factorial as base
from . import toolgrid_factorial_masked as env


MATCHED_HYBRID_CONDITION = "matched_hybrid"
MATCHED_EVALUATION_CONDITIONS = (
    "neural_policy_only",
    "imagination_v2",
)


def _evaluate_mode(
    agent: env.ProductionToolGridHybridAgent,
    *,
    use_imagination: bool,
    condition: str,
    research_seed: int,
    phase: str,
    grid_size: int,
    action_count: int,
    checkpoint_transition_target: int,
    map_seeds: Sequence[int],
    environment_steps_total: int,
    schedule_horizon: int,
) -> list[base.EpisodeRow]:
    """Evaluate one checkpoint mode without retraining or cloning the model."""

    original = agent.use_imagination
    agent.use_imagination = bool(use_imagination)
    try:
        return base._evaluate(
            agent,
            condition=condition,
            research_seed=research_seed,
            phase=phase,
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_transition_target,
            map_seeds=map_seeds,
            environment_steps_total=environment_steps_total,
            schedule_horizon=schedule_horizon,
        )
    finally:
        agent.use_imagination = original


def _write_manifest(
    output: Path,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    training_maps: Sequence[int],
    unseen_maps: Sequence[int],
) -> None:
    rows: list[dict[str, Any]] = []
    for split, seeds in (("training", training_maps), ("unseen", unseen_maps)):
        for map_seed in seeds:
            profile = env.ToolGridWorld(
                int(map_seed),
                grid_size=grid_size,
                action_count=action_count,
            ).map_profile()
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "grid_size": grid_size,
                    "action_count": action_count,
                    "tool_count": action_count - 4,
                    "effective_branching_factor": action_count - 4,
                    "map_seed": profile.map_seed,
                    "oracle_shortest_steps": profile.oracle_shortest_steps,
                    "start": json.dumps(profile.start),
                    "stations": json.dumps(profile.stations),
                    "required_tools": json.dumps(profile.required_tools),
                    "global_action_vocabulary": action_count,
                    "semantic_branching_factor": action_count - 4,
                }
            )
    with (output / "map_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_row(
    *,
    condition: str,
    seed: int,
    grid_size: int,
    action_count: int,
    checkpoint_target: int,
    environment_steps_total: int,
    training_episode_count: int,
    seen_rows: Sequence[base.EpisodeRow],
    unseen_rows: Sequence[base.EpisodeRow],
    stats: dict[str, int | float],
    training_wall_seconds: float,
) -> base.CheckpointRow:
    return base.CheckpointRow(
        condition=condition,
        seed=seed,
        grid_size=grid_size,
        action_count=action_count,
        checkpoint_transition_target=checkpoint_target,
        actual_training_transitions=environment_steps_total,
        training_episode_count=training_episode_count,
        seen_success_rate=base._mean_success(seen_rows),
        unseen_success_rate=base._mean_success(unseen_rows),
        seen_mean_steps_on_success=base._success_mean(seen_rows, "steps"),
        unseen_mean_steps_on_success=base._success_mean(unseen_rows, "steps"),
        imagined_nodes_mean=(
            sum(row.imagined_nodes for row in unseen_rows) / len(unseen_rows)
            if unseen_rows
            else 0.0
        ),
        imagination_use_rate=(
            sum(float(row.imagination_runs > 0) for row in unseen_rows)
            / len(unseen_rows)
            if unseen_rows
            else 0.0
        ),
        training_wall_seconds=training_wall_seconds,
        model_units=int(stats.get("model_units", 0)),
        model_bytes=int(stats.get("model_bytes", 0)),
        gradient_updates=int(stats.get("gradient_updates", 0)),
    )


def run_toolgrid_matched_hybrid(
    output_dir: str | Path,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    transition_budget: int = 5_000,
    train_map_count: int = 48,
    evaluation_map_count: int = 100,
    checkpoints: Sequence[int] = (0, 5_000),
) -> dict[str, Any]:
    """Train one hybrid stream and evaluate the exact checkpoint both ways.

    Policy-only and Imagination are not independently initialized or trained.
    They are two evaluation views of the same learned DQN, Prophecy, and branch
    critic checkpoint, eliminating cross-run numerical divergence.
    """

    if grid_size not in env.GRID_SIZES or action_count not in env.ACTION_COUNTS:
        raise ValueError("unsupported factorial cell")
    if min(transition_budget, train_map_count, evaluation_map_count) <= 0:
        raise ValueError("experiment sizes must be positive")
    normalized_checkpoints = tuple(
        sorted(
            {
                int(value)
                for value in checkpoints
                if 0 <= int(value) <= transition_budget
            }
            | {0, int(transition_budget)}
        )
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cell_base = (
        int(seed) * 10_000_000
        + int(grid_size) * 100_000
        + int(action_count) * 1_000
    )
    training_maps = tuple(cell_base + index for index in range(train_map_count))
    unseen_maps = tuple(
        cell_base + 500_000 + index for index in range(evaluation_map_count)
    )
    seen_maps = tuple(
        training_maps[index % len(training_maps)]
        for index in range(evaluation_map_count)
    )
    _write_manifest(
        output,
        seed=seed,
        grid_size=grid_size,
        action_count=action_count,
        training_maps=training_maps,
        unseen_maps=unseen_maps,
    )

    agent = env.ProductionToolGridHybridAgent(
        int(seed),
        action_count=action_count,
        train_transitions=transition_budget,
        use_imagination=True,
    )
    training_rows: list[base.EpisodeRow] = []
    evaluation_rows: list[base.EpisodeRow] = []
    checkpoint_rows: list[base.CheckpointRow] = []
    environment_steps_total = 0
    training_episode_count = 0
    training_started = time.perf_counter()
    env._TRAINING_SEGMENTS.clear()

    for checkpoint_target in normalized_checkpoints:
        while environment_steps_total < checkpoint_target:
            map_seed = training_maps[
                training_episode_count % len(training_maps)
            ]
            row, environment_steps_total = env._run_episode(
                agent,
                condition=MATCHED_HYBRID_CONDITION,
                research_seed=seed,
                phase="training",
                grid_size=grid_size,
                action_count=action_count,
                checkpoint_transition_target=checkpoint_target,
                episode_index=training_episode_count,
                map_seed=map_seed,
                training=True,
                environment_steps_total=environment_steps_total,
                schedule_horizon=transition_budget,
            )
            training_rows.append(row)
            training_episode_count += 1

        policy_seen = _evaluate_mode(
            agent,
            use_imagination=False,
            condition="neural_policy_only",
            research_seed=seed,
            phase="evaluation_seen",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_target,
            map_seeds=seen_maps,
            environment_steps_total=environment_steps_total,
            schedule_horizon=transition_budget,
        )
        policy_unseen = _evaluate_mode(
            agent,
            use_imagination=False,
            condition="neural_policy_only",
            research_seed=seed,
            phase="evaluation_unseen",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_target,
            map_seeds=unseen_maps,
            environment_steps_total=environment_steps_total,
            schedule_horizon=transition_budget,
        )
        imagination_seen = _evaluate_mode(
            agent,
            use_imagination=True,
            condition="imagination_v2",
            research_seed=seed,
            phase="evaluation_seen",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_target,
            map_seeds=seen_maps,
            environment_steps_total=environment_steps_total,
            schedule_horizon=transition_budget,
        )
        imagination_unseen = _evaluate_mode(
            agent,
            use_imagination=True,
            condition="imagination_v2",
            research_seed=seed,
            phase="evaluation_unseen",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_target,
            map_seeds=unseen_maps,
            environment_steps_total=environment_steps_total,
            schedule_horizon=transition_budget,
        )
        evaluation_rows.extend(policy_seen)
        evaluation_rows.extend(policy_unseen)
        evaluation_rows.extend(imagination_seen)
        evaluation_rows.extend(imagination_unseen)

        stats = agent.model_stats()
        elapsed = time.perf_counter() - training_started
        checkpoint_rows.append(
            _checkpoint_row(
                condition="neural_policy_only",
                seed=seed,
                grid_size=grid_size,
                action_count=action_count,
                checkpoint_target=checkpoint_target,
                environment_steps_total=environment_steps_total,
                training_episode_count=training_episode_count,
                seen_rows=policy_seen,
                unseen_rows=policy_unseen,
                stats=stats,
                training_wall_seconds=elapsed,
            )
        )
        checkpoint_rows.append(
            _checkpoint_row(
                condition="imagination_v2",
                seed=seed,
                grid_size=grid_size,
                action_count=action_count,
                checkpoint_target=checkpoint_target,
                environment_steps_total=environment_steps_total,
                training_episode_count=training_episode_count,
                seen_rows=imagination_seen,
                unseen_rows=imagination_unseen,
                stats=stats,
                training_wall_seconds=elapsed,
            )
        )

    base._write_rows(output / "training_episodes.csv", training_rows)
    base._write_rows(output / "evaluation_episodes.csv", evaluation_rows)
    base._write_rows(output / "checkpoints.csv", checkpoint_rows)
    final = checkpoint_rows[-1]
    payload: dict[str, Any] = {
        "config": {
            "condition": MATCHED_HYBRID_CONDITION,
            "evaluation_conditions": list(MATCHED_EVALUATION_CONDITIONS),
            "shared_checkpoint": True,
            "seed": seed,
            "grid_size": grid_size,
            "action_count": action_count,
            "tool_count": action_count - 4,
            "transition_budget": transition_budget,
            "train_map_count": train_map_count,
            "evaluation_map_count": evaluation_map_count,
            "checkpoints": list(normalized_checkpoints),
            "external_reward": "final_success_only",
            "artificial_tick_limit": False,
        },
        "final": final.row(),
        "model_stats": agent.model_stats(),
    }
    env._rewrite_experiment_metrics(output, payload)
    return payload
