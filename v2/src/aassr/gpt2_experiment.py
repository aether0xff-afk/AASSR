from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import log
from pathlib import Path
from typing import Any

from .experiment import (
    ExperimentComponents,
    ExperimentCondition,
    ProphecyKind,
    c5_imagination_config,
    make_prophecy,
    write_experiment_outputs,
)
from .gpt2_curriculum import CurriculumOutcome, LearningProgressScheduler
from .gpt2_reward import ActionableGridWorldDMP
from .gridworld import DMPConfig, GridWorld
from .imagination import ImaginationCycle
from .metrics import EpisodeMetric, StepMetric, SummaryMetric, episode_metric, step_metric, summary_metric
from .policy import PolicyABC, candidate_axes
from .worlds import WorldKind, make_world


GPT2_REWARD = "GPT2_REWARD"
GPT2_ACADEMY_MODEL = "GPT2_ACADEMY_MODEL"
GPT2_ACADEMY_FULL = "GPT2_ACADEMY_FULL"
GPT2_ACADEMY_BASELINE = "GPT2_ACADEMY_BASELINE"
GPT2_ACADEMY_PRETRAIN = "GPT2_ACADEMY_PRETRAIN"


class AcademyTransferMode(StrEnum):
    MODEL_ONLY = "model_only"
    FULL_PRIOR = "full_prior"


@dataclass(frozen=True)
class CreativityGuardrailConfig:
    min_success_delta: float = 0.0
    max_diversity_drop: float = 0.10
    max_entropy_drop: float = 0.10
    min_novel_strategy_rate: float = 0.20


@dataclass(frozen=True)
class AcademyRunReport:
    transfer_mode: str
    prophecy_kind: str
    pretrain_episodes: int
    evaluation_episodes: int
    seeds: int
    target_world: str
    curriculum: dict[str, dict[str, float]]
    baseline_summary: dict[str, Any]
    evaluation_summary: dict[str, Any]
    baseline_creativity: dict[str, float]
    evaluation_creativity: dict[str, float]
    policy_override_rate: float
    guardrail: dict[str, Any]


@dataclass
class EpisodeExecution:
    steps: list[StepMetric]
    diagnostics: list[dict[str, Any]]
    structural_signature: tuple[str, ...]
    policy_override_count: int = 0
    policy_decision_count: int = 0

    @property
    def success(self) -> bool:
        return any(step.flag_found for step in self.steps)


def make_actionable_dmp(
    components: ExperimentComponents,
    world: GridWorld,
    *,
    step_limit: int,
) -> ActionableGridWorldDMP:
    imagination = None
    if components.use_imagination and components.prophecy is not None:
        imagination = ImaginationCycle(components.prophecy, components.imagination_config)

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
            prophecy_beta=0.0,
            prediction_error_mode="disabled",
            independent_policy_axes=components.independent_policy_axes,
        ),
        step_limit=step_limit,
    )


def make_academy_components(
    *,
    model_seed: int,
    policy_seed: int,
    prophecy_kind: str | ProphecyKind,
) -> ExperimentComponents:
    kind = ProphecyKind(prophecy_kind)
    return ExperimentComponents(
        scorer=PolicyABC.uniform_gridworld(seed=policy_seed),
        prophecy=make_prophecy(kind, seed=model_seed),
        use_prophecy=True,
        use_imagination=True,
        prophecy_beta=0.0,
        imagination_config=c5_imagination_config(),
        full_imagination=False,
        independent_policy_axes=False,
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
    diagnostics: list[dict[str, Any]] = []

    for seed in range(seeds):
        components = ExperimentComponents.for_condition(ExperimentCondition.C5, seed=seed)
        components.prophecy_beta = 0.0
        for episode in range(episodes):
            dmp = make_actionable_dmp(
                components,
                make_world(world, seed=_world_seed(seed, episode)),
                step_limit=step_limit,
            )
            execution = _run_episode(
                dmp,
                condition=GPT2_REWARD,
                seed=seed,
                episode=episode,
            )
            all_steps.extend(execution.steps)
            diagnostics.extend(execution.diagnostics)
            all_episodes.append(
                episode_metric(
                    condition=GPT2_REWARD,
                    seed=seed,
                    episode=episode,
                    steps=execution.steps,
                    knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                    step_limit=step_limit,
                )
            )

    summaries = [summary_metric(GPT2_REWARD, all_episodes)]
    if output_dir is not None:
        output_path = Path(output_dir)
        write_experiment_outputs(output_path, all_steps, all_episodes, summaries)
        _write_dict_csv(output_path / "gpt2_reward_diagnostics.csv", diagnostics)
    return all_steps, all_episodes, summaries


def run_academy_experiment(
    *,
    pretrain_episodes: int,
    evaluation_episodes: int,
    seeds: int,
    step_limit: int,
    target_world: str | WorldKind,
    transfer_mode: str | AcademyTransferMode = AcademyTransferMode.MODEL_ONLY,
    prophecy_kind: str | ProphecyKind = ProphecyKind.SEQUENCE,
    guardrail_config: CreativityGuardrailConfig | None = None,
    output_dir: str | Path | None = None,
) -> AcademyRunReport:
    target_world = WorldKind(target_world)
    transfer_mode = AcademyTransferMode(transfer_mode)
    prophecy_kind = ProphecyKind(prophecy_kind)
    guardrail_config = guardrail_config or CreativityGuardrailConfig()
    condition = (
        GPT2_ACADEMY_MODEL
        if transfer_mode == AcademyTransferMode.MODEL_ONLY
        else GPT2_ACADEMY_FULL
    )

    evaluation_steps: list[StepMetric] = []
    evaluation_episodes_rows: list[EpisodeMetric] = []
    evaluation_diagnostics: list[dict[str, Any]] = []
    baseline_steps: list[StepMetric] = []
    baseline_episodes_rows: list[EpisodeMetric] = []
    baseline_diagnostics: list[dict[str, Any]] = []
    curriculum_history: list[dict[str, Any]] = []

    aggregate_scheduler = LearningProgressScheduler(seed=0)
    academy_success_signatures: set[tuple[str, ...]] = set()
    evaluation_success_signatures: list[tuple[str, ...]] = []
    baseline_success_signatures: list[tuple[str, ...]] = []
    override_count = 0
    decision_count = 0

    for seed in range(seeds):
        pretrain_components = make_academy_components(
            model_seed=seed,
            policy_seed=seed,
            prophecy_kind=prophecy_kind,
        )
        scheduler = LearningProgressScheduler(seed=seed)

        for academy_episode in range(pretrain_episodes):
            task = scheduler.next_task()
            dmp = make_actionable_dmp(
                pretrain_components,
                make_world(task.world_kind, seed=task.seed),
                step_limit=step_limit,
            )
            execution = _run_episode(
                dmp,
                condition=GPT2_ACADEMY_PRETRAIN,
                seed=seed,
                episode=academy_episode,
            )
            repeats = _repeat_count(execution.steps)
            errors = sum(1 for row in execution.steps if row.error)
            outcome = CurriculumOutcome(
                success=execution.success,
                steps=len(execution.steps),
                step_limit=step_limit,
                error_count=errors,
                repeat_count=repeats,
            )
            scheduler.observe(task, outcome)
            aggregate_scheduler.observe(task, outcome)
            curriculum_history.append(
                {
                    "seed": seed,
                    "episode": academy_episode,
                    **task.to_dict(),
                    "success": execution.success,
                    "steps": len(execution.steps),
                    "error_count": errors,
                    "repeat_count": repeats,
                    "teacher_score": outcome.score,
                }
            )
            if execution.success:
                academy_success_signatures.add(execution.structural_signature)

        prior_top_axes = _top_policy_axes(pretrain_components.scorer)
        evaluation_policy_seed = seed + 10_000
        if transfer_mode == AcademyTransferMode.MODEL_ONLY:
            pretrain_components.scorer = PolicyABC.uniform_gridworld(seed=evaluation_policy_seed)

        baseline_components = make_academy_components(
            model_seed=seed,
            policy_seed=evaluation_policy_seed,
            prophecy_kind=prophecy_kind,
        )

        for episode in range(evaluation_episodes):
            world_seed = _world_seed(seed, episode)
            transferred_dmp = make_actionable_dmp(
                pretrain_components,
                make_world(target_world, seed=world_seed),
                step_limit=step_limit,
            )
            transferred = _run_episode(
                transferred_dmp,
                condition=condition,
                seed=seed,
                episode=episode,
                prior_top_axes=prior_top_axes,
            )
            evaluation_steps.extend(transferred.steps)
            evaluation_diagnostics.extend(transferred.diagnostics)
            override_count += transferred.policy_override_count
            decision_count += transferred.policy_decision_count
            if transferred.success:
                evaluation_success_signatures.append(transferred.structural_signature)
            evaluation_episodes_rows.append(
                episode_metric(
                    condition=condition,
                    seed=seed,
                    episode=episode,
                    steps=transferred.steps,
                    knowledge_reuse_count=int(transferred_dmp.metrics()["knowledge_reuse_count"]),
                    step_limit=step_limit,
                )
            )

            baseline_dmp = make_actionable_dmp(
                baseline_components,
                make_world(target_world, seed=world_seed),
                step_limit=step_limit,
            )
            baseline = _run_episode(
                baseline_dmp,
                condition=GPT2_ACADEMY_BASELINE,
                seed=seed,
                episode=episode,
            )
            baseline_steps.extend(baseline.steps)
            baseline_diagnostics.extend(baseline.diagnostics)
            if baseline.success:
                baseline_success_signatures.append(baseline.structural_signature)
            baseline_episodes_rows.append(
                episode_metric(
                    condition=GPT2_ACADEMY_BASELINE,
                    seed=seed,
                    episode=episode,
                    steps=baseline.steps,
                    knowledge_reuse_count=int(baseline_dmp.metrics()["knowledge_reuse_count"]),
                    step_limit=step_limit,
                )
            )

    evaluation_summary = summary_metric(condition, evaluation_episodes_rows)
    baseline_summary = summary_metric(GPT2_ACADEMY_BASELINE, baseline_episodes_rows)
    evaluation_creativity = _creativity_summary(
        evaluation_success_signatures,
        academy_success_signatures,
    )
    baseline_creativity = _trajectory_summary(baseline_success_signatures)
    policy_override_rate = override_count / decision_count if decision_count else 0.0
    guardrail = _guardrail_summary(
        evaluation_summary=evaluation_summary,
        baseline_summary=baseline_summary,
        evaluation_creativity=evaluation_creativity,
        baseline_creativity=baseline_creativity,
        config=guardrail_config,
    )

    report = AcademyRunReport(
        transfer_mode=transfer_mode.value,
        prophecy_kind=prophecy_kind.value,
        pretrain_episodes=pretrain_episodes,
        evaluation_episodes=evaluation_episodes,
        seeds=seeds,
        target_world=target_world.value,
        curriculum=aggregate_scheduler.snapshot(),
        baseline_summary=baseline_summary.to_dict(),
        evaluation_summary=evaluation_summary.to_dict(),
        baseline_creativity=baseline_creativity,
        evaluation_creativity=evaluation_creativity,
        policy_override_rate=policy_override_rate,
        guardrail=guardrail,
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        write_experiment_outputs(
            output_path,
            evaluation_steps,
            evaluation_episodes_rows,
            [evaluation_summary],
        )
        write_experiment_outputs(
            output_path / "baseline",
            baseline_steps,
            baseline_episodes_rows,
            [baseline_summary],
        )
        _write_dict_csv(output_path / "gpt2_reward_diagnostics.csv", evaluation_diagnostics)
        _write_dict_csv(output_path / "baseline_reward_diagnostics.csv", baseline_diagnostics)
        _write_dict_csv(output_path / "curriculum_history.csv", curriculum_history)
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
    prior_top_axes: tuple[str, str, str] | None = None,
) -> EpisodeExecution:
    rows: list[StepMetric] = []
    diagnostics: list[dict[str, Any]] = []
    override_count = 0
    decision_count = 0

    while not dmp.done:
        candidate = dmp.choose_candidate("scorer")
        if candidate is None:
            break
        if prior_top_axes is not None:
            what, _, where = candidate_axes(candidate)
            prior_what, _, prior_where = prior_top_axes
            decision_count += 1
            override_count += int((what, where) != (prior_what, prior_where))
        result = dmp.execute(candidate)
        row = step_metric(
            condition=condition,
            seed=seed,
            episode=episode,
            result=result,
        )
        rows.append(row)
        diagnostics.append(
            {
                "condition": condition,
                "seed": seed,
                "episode": episode,
                "step": result.step,
                "action": result.action.name.value,
                "template": result.action.template,
                "total_reward": result.total_reward,
                **dmp.actionable_reward.last_diagnostics.to_dict(),
            }
        )

    return EpisodeExecution(
        steps=rows,
        diagnostics=diagnostics,
        structural_signature=_structural_signature(rows),
        policy_override_count=override_count,
        policy_decision_count=decision_count,
    )


def _structural_signature(rows: list[StepMetric]) -> tuple[str, ...]:
    tokens: list[str] = []
    for row in rows:
        if not tokens or tokens[-1] != row.template:
            tokens.append(row.template)
    return tuple(tokens)


def _trajectory_summary(signatures: list[tuple[str, ...]]) -> dict[str, float]:
    total = len(signatures)
    if total == 0:
        return {
            "successful_trajectory_count": 0.0,
            "unique_successful_trajectory_count": 0.0,
            "successful_trajectory_diversity": 0.0,
            "trajectory_entropy": 0.0,
        }
    counts: dict[tuple[str, ...], int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    entropy = -sum((count / total) * log(count / total) for count in counts.values())
    normalized_entropy = entropy / log(total) if total > 1 else 0.0
    return {
        "successful_trajectory_count": float(total),
        "unique_successful_trajectory_count": float(len(counts)),
        "successful_trajectory_diversity": len(counts) / total,
        "trajectory_entropy": normalized_entropy,
    }


def _creativity_summary(
    evaluation_signatures: list[tuple[str, ...]],
    academy_signatures: set[tuple[str, ...]],
) -> dict[str, float]:
    summary = _trajectory_summary(evaluation_signatures)
    total = len(evaluation_signatures)
    novel = sum(signature not in academy_signatures for signature in evaluation_signatures)
    return {
        **summary,
        "novel_strategy_rate": novel / total if total else 0.0,
    }


def _guardrail_summary(
    *,
    evaluation_summary: SummaryMetric,
    baseline_summary: SummaryMetric,
    evaluation_creativity: dict[str, float],
    baseline_creativity: dict[str, float],
    config: CreativityGuardrailConfig,
) -> dict[str, Any]:
    success_delta = evaluation_summary.success_rate - baseline_summary.success_rate
    diversity_delta = (
        evaluation_creativity["successful_trajectory_diversity"]
        - baseline_creativity["successful_trajectory_diversity"]
    )
    entropy_delta = (
        evaluation_creativity["trajectory_entropy"]
        - baseline_creativity["trajectory_entropy"]
    )
    novel_rate = evaluation_creativity["novel_strategy_rate"]
    passed = (
        evaluation_creativity["successful_trajectory_count"] > 0
        and success_delta >= config.min_success_delta
        and diversity_delta >= -config.max_diversity_drop
        and entropy_delta >= -config.max_entropy_drop
        and novel_rate >= config.min_novel_strategy_rate
    )
    return {
        "passed": passed,
        "success_delta": success_delta,
        "diversity_delta": diversity_delta,
        "entropy_delta": entropy_delta,
        "novel_strategy_rate": novel_rate,
        "criteria": asdict(config),
    }


def _top_policy_axes(policy: Any) -> tuple[str, str, str] | None:
    snapshot = getattr(policy, "snapshot", None)
    if not callable(snapshot):
        return None
    tables = snapshot()
    try:
        return tuple(
            max(tables[axis], key=tables[axis].get)
            for axis in ("WHAT", "HOW", "WHERE")
        )
    except (KeyError, ValueError):
        return None


def _repeat_count(rows: list[StepMetric]) -> int:
    signatures = [row.action_signature for row in rows]
    return len(signatures) - len(set(signatures))


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _world_seed(seed: int, episode: int) -> int:
    return seed * 100_000 + episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-2 actionable-reward and autonomous-academy experiments."
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
    parser.add_argument(
        "--prophecy-kind",
        choices=[item.value for item in ProphecyKind],
        default=ProphecyKind.SEQUENCE.value,
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or f"runs/gpt2/{args.mode}"
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
        prophecy_kind=args.prophecy_kind,
        output_dir=output_dir,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
