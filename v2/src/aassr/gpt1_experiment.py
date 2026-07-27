from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import log
from pathlib import Path
from typing import Any

from .actionable import ActionableGridWorldDMP
from .curriculum import LearningProgressScheduler
from .experiment import ExperimentComponents, ExperimentCondition, write_experiment_outputs
from .gridworld import DMPConfig, GridWorld
from .imagination import ImaginationCycle, PredictedStateImaginationCycle
from .metrics import EpisodeMetric, StepMetric, SummaryMetric, episode_metric, step_metric, summary_metric
from .policy import PolicyABC
from .worlds import WorldKind, make_world


C6_REWARD = "C6_REWARD"
C7_ACADEMY_MODEL = "C7_ACADEMY_MODEL"
C7_ACADEMY_FULL = "C7_ACADEMY_FULL"


class AcademyTransferMode(StrEnum):
    MODEL_ONLY = "model_only"
    FULL_PRIOR = "full_prior"


@dataclass(frozen=True)
class AcademyRunReport:
    transfer_mode: str
    pretrain_episodes: int
    evaluation_episodes: int
    seeds: int
    target_world: str
    curriculum: dict[str, dict[str, float]]
    evaluation_summary: dict[str, Any]
    creativity: dict[str, float]


def make_actionable_dmp(
    components: ExperimentComponents,
    world: GridWorld,
    *,
    step_limit: int,
) -> ActionableGridWorldDMP:
    if components.use_imagination and components.prophecy is not None:
        imagination_type = (
            PredictedStateImaginationCycle
            if components.full_imagination
            else ImaginationCycle
        )
        imagination = imagination_type(components.prophecy, components.imagination_config)
    else:
        imagination = None

    reset_context = getattr(components.prophecy, "reset_context", None)
    if callable(reset_context):
        reset_context()

    return ActionableGridWorldDMP(
        world,
        scorer=components.scorer,
        prophecy=components.prophecy,
        imagination=imagination,
        config=DMPConfig(
            use_prophecy=components.use_prophecy,
            use_imagination=components.use_imagination,
            prophecy_beta=components.prophecy_beta,
            prediction_error_mode="disabled",
            independent_policy_axes=components.independent_policy_axes,
        ),
        step_limit=step_limit,
    )


def run_reward_experiment(
    *,
    episodes: int,
    seeds: int,
    step_limit: int,
    world: str | WorldKind,
    output_dir: str | Path | None = None,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[SummaryMetric]]:
    world = WorldKind(world)
    all_steps: list[StepMetric] = []
    all_episodes: list[EpisodeMetric] = []

    for seed in range(seeds):
        components = ExperimentComponents.for_condition(ExperimentCondition.C5, seed=seed)
        for episode in range(episodes):
            dmp = make_actionable_dmp(
                components,
                make_world(world, seed=_world_seed(seed, episode)),
                step_limit=step_limit,
            )
            step_rows = _run_episode(
                dmp,
                condition=C6_REWARD,
                seed=seed,
                episode=episode,
            )
            all_steps.extend(step_rows)
            all_episodes.append(
                episode_metric(
                    condition=C6_REWARD,
                    seed=seed,
                    episode=episode,
                    steps=step_rows,
                    knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                    step_limit=step_limit,
                )
            )

    summaries = [summary_metric(C6_REWARD, all_episodes)]
    if output_dir is not None:
        write_experiment_outputs(output_dir, all_steps, all_episodes, summaries)
    return all_steps, all_episodes, summaries


def run_academy_experiment(
    *,
    pretrain_episodes: int,
    evaluation_episodes: int,
    seeds: int,
    step_limit: int,
    target_world: str | WorldKind,
    transfer_mode: str | AcademyTransferMode = AcademyTransferMode.MODEL_ONLY,
    output_dir: str | Path | None = None,
) -> AcademyRunReport:
    target_world = WorldKind(target_world)
    transfer_mode = AcademyTransferMode(transfer_mode)
    condition = (
        C7_ACADEMY_MODEL
        if transfer_mode == AcademyTransferMode.MODEL_ONLY
        else C7_ACADEMY_FULL
    )

    all_steps: list[StepMetric] = []
    all_episodes: list[EpisodeMetric] = []
    aggregate_scheduler = LearningProgressScheduler(seed=0)
    academy_success_signatures: set[tuple[str, ...]] = set()
    evaluation_success_signatures: list[tuple[str, ...]] = []

    for seed in range(seeds):
        components = ExperimentComponents.for_condition(ExperimentCondition.C5, seed=seed)
        scheduler = LearningProgressScheduler(seed=seed)

        for academy_episode in range(pretrain_episodes):
            task = scheduler.next_task()
            dmp = make_actionable_dmp(
                components,
                make_world(task.world_kind, seed=task.seed),
                step_limit=step_limit,
            )
            rows = _run_episode(
                dmp,
                condition="ACADEMY_PRETRAIN",
                seed=seed,
                episode=academy_episode,
            )
            success = bool(rows and rows[-1].flag_found)
            if success:
                academy_success_signatures.add(tuple(row.template for row in rows))
            scheduler.observe(
                task,
                success=success,
                steps=len(rows),
                step_limit=step_limit,
            )
            aggregate_scheduler.observe(
                task,
                success=success,
                steps=len(rows),
                step_limit=step_limit,
            )

        if transfer_mode == AcademyTransferMode.MODEL_ONLY:
            # Keep learned Prophecy/transition knowledge, but remove the academy's
            # preferred action distribution so the target task can form a new strategy.
            components.scorer = PolicyABC.uniform_gridworld(seed=seed + 10_000)

        for episode in range(evaluation_episodes):
            dmp = make_actionable_dmp(
                components,
                make_world(target_world, seed=_world_seed(seed, episode)),
                step_limit=step_limit,
            )
            rows = _run_episode(
                dmp,
                condition=condition,
                seed=seed,
                episode=episode,
            )
            all_steps.extend(rows)
            if rows and rows[-1].flag_found:
                evaluation_success_signatures.append(tuple(row.template for row in rows))
            all_episodes.append(
                episode_metric(
                    condition=condition,
                    seed=seed,
                    episode=episode,
                    steps=rows,
                    knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                    step_limit=step_limit,
                )
            )

    summaries = [summary_metric(condition, all_episodes)]
    report = AcademyRunReport(
        transfer_mode=transfer_mode.value,
        pretrain_episodes=pretrain_episodes,
        evaluation_episodes=evaluation_episodes,
        seeds=seeds,
        target_world=target_world.value,
        curriculum=aggregate_scheduler.snapshot(),
        evaluation_summary=summaries[0].to_dict(),
        creativity=_creativity_summary(
            evaluation_success_signatures,
            academy_success_signatures,
        ),
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        write_experiment_outputs(output_path, all_steps, all_episodes, summaries)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "academy_report.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _run_episode(
    dmp: ActionableGridWorldDMP,
    *,
    condition: str,
    seed: int,
    episode: int,
) -> list[StepMetric]:
    rows: list[StepMetric] = []
    while not dmp.done:
        candidate = dmp.choose_candidate("scorer")
        if candidate is None:
            break
        result = dmp.execute(candidate)
        rows.append(
            step_metric(
                condition=condition,
                seed=seed,
                episode=episode,
                result=result,
            )
        )
    return rows


def _creativity_summary(
    evaluation_signatures: list[tuple[str, ...]],
    academy_signatures: set[tuple[str, ...]],
) -> dict[str, float]:
    total = len(evaluation_signatures)
    if total == 0:
        return {
            "successful_trajectory_count": 0.0,
            "unique_successful_trajectory_count": 0.0,
            "successful_trajectory_diversity": 0.0,
            "novel_strategy_rate": 0.0,
            "trajectory_entropy": 0.0,
        }

    counts: dict[tuple[str, ...], int] = {}
    for signature in evaluation_signatures:
        counts[signature] = counts.get(signature, 0) + 1
    entropy = -sum(
        (count / total) * log(count / total)
        for count in counts.values()
    )
    normalized_entropy = entropy / log(total) if total > 1 else 0.0
    novel = sum(signature not in academy_signatures for signature in evaluation_signatures)
    return {
        "successful_trajectory_count": float(total),
        "unique_successful_trajectory_count": float(len(counts)),
        "successful_trajectory_diversity": len(counts) / total,
        "novel_strategy_rate": novel / total,
        "trajectory_entropy": normalized_entropy,
    }


def _world_seed(seed: int, episode: int) -> int:
    return seed * 100_000 + episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-1 reward and autonomous curriculum experiments."
    )
    parser.add_argument("--mode", choices=("reward", "academy"), required=True)
    parser.add_argument(
        "--world",
        choices=[item.value for item in WorldKind],
        default=WorldKind.V2_COMPLEX.value,
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--pretrain-episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--step-limit", type=int, default=120)
    parser.add_argument(
        "--transfer-mode",
        choices=[item.value for item in AcademyTransferMode],
        default=AcademyTransferMode.MODEL_ONLY.value,
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"runs/gpt1/{args.mode}"
    if args.mode == "reward":
        _, _, summaries = run_reward_experiment(
            episodes=args.episodes,
            seeds=args.seeds,
            step_limit=args.step_limit,
            world=args.world,
            output_dir=output_dir,
        )
        print(json.dumps(summaries[0].to_dict(), indent=2))
        return

    report = run_academy_experiment(
        pretrain_episodes=args.pretrain_episodes,
        evaluation_episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        target_world=args.world,
        transfer_mode=args.transfer_mode,
        output_dir=output_dir,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
