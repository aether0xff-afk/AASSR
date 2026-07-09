from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

from .gridworld import DMPConfig, GridWorld, GridWorldDMP
from .imagination import ImaginationConfig, ImaginationCycle
from .metrics import (
    EpisodeMetric,
    StepMetric,
    SummaryMetric,
    episode_metric,
    step_metric,
    summary_metric,
)
from .policy import PolicyABC, RandomScorer
from .progress import ProgressTracker
from .prophecy import SequenceProphecyModel, TableProphecyModel, TransformerProphecyModel
from .worlds import WorldKind, make_world


class ExperimentCondition(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"


class ProphecyKind(StrEnum):
    TABLE = "table"
    SEQUENCE = "sequence"
    TRANSFORMER = "transformer"


@dataclass(frozen=True)
class ExperimentSpec:
    condition: str
    scorer: str = "policy"
    use_prophecy: bool = True
    prophecy_kind: ProphecyKind = ProphecyKind.TABLE
    prophecy_beta: float = 0.3
    use_imagination: bool = True
    imagination_config: ImaginationConfig = field(default_factory=ImaginationConfig)


ALL_CONDITIONS = (
    ExperimentCondition.C0,
    ExperimentCondition.C1,
    ExperimentCondition.C2,
    ExperimentCondition.C3,
    ExperimentCondition.C4,
)


def run_experiment(
    *,
    condition: str | ExperimentCondition,
    episodes: int,
    seeds: int,
    step_limit: int = 200,
    world: str | WorldKind = WorldKind.FIXED,
    output_dir: str | Path | None = None,
    workers: int = 1,
    progress: bool = False,
    progress_label: str | None = None,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[SummaryMetric]]:
    condition = ExperimentCondition(condition)
    world = WorldKind(world)
    args = [
        (condition.value, seed, episodes, step_limit, world.value)
        for seed in range(seeds)
    ]
    chunks = _run_seed_jobs(
        _run_condition_seed,
        args,
        workers=workers,
        progress=progress,
        label=progress_label or condition.value,
    )

    all_steps = [row for steps, _ in chunks for row in steps]
    all_episodes = [row for _, episode_rows in chunks for row in episode_rows]

    summaries = [summary_metric(condition.value, all_episodes)]
    if output_dir is not None:
        write_experiment_outputs(output_dir, all_steps, all_episodes, summaries)
    return all_steps, all_episodes, summaries


def _run_seed_jobs(function: Any, args: list[Any], *, workers: int, progress: bool, label: str) -> list[Any]:
    tracker = ProgressTracker(label=label, total=len(args), enabled=progress)
    chunks: list[Any] = []
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(function, item): index
                for index, item in enumerate(args)
            }
            ordered: list[Any | None] = [None] * len(args)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ordered[index] = future.result()
                tracker.advance()
            return [item for item in ordered if item is not None]
    else:
        for item in args:
            chunks.append(function(item))
            tracker.advance()
    return chunks


def run_all_conditions(
    *,
    episodes: int,
    seeds: int,
    step_limit: int = 200,
    world: str | WorldKind = WorldKind.FIXED,
    output_dir: str | Path = "runs/gridworld",
    workers: int = 1,
    progress: bool = False,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[SummaryMetric]]:
    output_path = Path(output_dir)
    all_steps: list[StepMetric] = []
    all_episodes: list[EpisodeMetric] = []
    all_summaries: list[SummaryMetric] = []
    for condition in ALL_CONDITIONS:
        steps, episodes_rows, summaries = run_experiment(
            condition=condition,
            episodes=episodes,
            seeds=seeds,
            step_limit=step_limit,
            world=world,
            output_dir=output_path / condition.value,
            workers=workers,
            progress=progress,
            progress_label=f"{WorldKind(world).value}/{condition.value}",
        )
        all_steps.extend(steps)
        all_episodes.extend(episodes_rows)
        all_summaries.extend(summaries)
    _write_csv(output_path / "combined_summary.csv", all_summaries, SummaryMetric)
    return all_steps, all_episodes, all_summaries


def run_experiment_spec(
    *,
    spec: ExperimentSpec,
    episodes: int,
    seeds: int,
    step_limit: int = 200,
    world: str | WorldKind = WorldKind.FIXED,
    output_dir: str | Path | None = None,
    workers: int = 1,
    progress: bool = False,
    progress_label: str | None = None,
) -> tuple[list[StepMetric], list[EpisodeMetric], list[SummaryMetric]]:
    world = WorldKind(world)
    chunks = _run_seed_jobs(
        _run_spec_seed,
        [
            (spec, seed, episodes, step_limit, world.value)
            for seed in range(seeds)
        ],
        workers=workers,
        progress=progress,
        label=progress_label or spec.condition,
    )

    all_steps = [row for steps, _ in chunks for row in steps]
    all_episodes = [row for _, episode_rows in chunks for row in episode_rows]
    summaries = [summary_metric(spec.condition, all_episodes)]
    if output_dir is not None:
        write_experiment_outputs(output_dir, all_steps, all_episodes, summaries)
    return all_steps, all_episodes, summaries


class ExperimentComponents:
    def __init__(
        self,
        *,
        scorer: Any,
        prophecy: Any | None,
        use_prophecy: bool,
        use_imagination: bool,
        prophecy_beta: float = 0.3,
        imagination_config: ImaginationConfig | None = None,
    ) -> None:
        self.scorer = scorer
        self.prophecy = prophecy
        self.use_prophecy = use_prophecy
        self.use_imagination = use_imagination
        self.prophecy_beta = prophecy_beta
        self.imagination_config = imagination_config or ImaginationConfig()

    @classmethod
    def for_condition(cls, condition: ExperimentCondition, *, seed: int) -> ExperimentComponents:
        if condition == ExperimentCondition.C0:
            return cls(
                scorer=RandomScorer(seed=seed),
                prophecy=None,
                use_prophecy=False,
                use_imagination=False,
            )
        if condition == ExperimentCondition.C1:
            return cls(
                scorer=PolicyABC.uniform_gridworld(seed=seed),
                prophecy=None,
                use_prophecy=False,
                use_imagination=False,
            )
        if condition == ExperimentCondition.C4:
            prophecy = SequenceProphecyModel(seed=seed)
        else:
            prophecy = TableProphecyModel()
        return cls(
            scorer=PolicyABC.uniform_gridworld(seed=seed),
            prophecy=prophecy,
            use_prophecy=True,
            use_imagination=condition in {
                ExperimentCondition.C3,
                ExperimentCondition.C4,
            },
        )

    @classmethod
    def for_spec(cls, spec: ExperimentSpec, *, seed: int) -> ExperimentComponents:
        if spec.scorer == "random":
            scorer = RandomScorer(seed=seed)
        elif spec.scorer == "policy":
            scorer = PolicyABC.uniform_gridworld(seed=seed)
        else:
            raise ValueError(f"unknown scorer kind: {spec.scorer}")

        prophecy = None
        if spec.use_prophecy:
            prophecy = make_prophecy(spec.prophecy_kind, seed=seed)

        return cls(
            scorer=scorer,
            prophecy=prophecy,
            use_prophecy=spec.use_prophecy,
            use_imagination=spec.use_imagination,
            prophecy_beta=spec.prophecy_beta,
            imagination_config=spec.imagination_config,
        )

    def make_dmp(self, world: GridWorld, *, step_limit: int) -> GridWorldDMP:
        imagination = (
            ImaginationCycle(self.prophecy, self.imagination_config)
            if self.use_imagination and self.prophecy is not None
            else None
        )
        return GridWorldDMP(
            world,
            scorer=self.scorer,
            prophecy=self.prophecy,
            imagination=imagination,
            config=DMPConfig(
                use_prophecy=self.use_prophecy,
                use_imagination=self.use_imagination,
                prophecy_beta=self.prophecy_beta,
            ),
            step_limit=step_limit,
        )


def make_prophecy(kind: str | ProphecyKind, *, seed: int) -> Any:
    kind = ProphecyKind(kind)
    if kind == ProphecyKind.TABLE:
        return TableProphecyModel()
    if kind == ProphecyKind.SEQUENCE:
        return SequenceProphecyModel(seed=seed)
    if kind == ProphecyKind.TRANSFORMER:
        return TransformerProphecyModel(seed=seed)
    raise ValueError(f"unknown prophecy kind: {kind}")


def _run_condition_seed(args: tuple[str, int, int, int, str]) -> tuple[list[StepMetric], list[EpisodeMetric]]:
    condition_value, seed, episodes, step_limit, world_value = args
    condition = ExperimentCondition(condition_value)
    world = WorldKind(world_value)
    components = ExperimentComponents.for_condition(condition, seed=seed)
    seed_steps: list[StepMetric] = []
    seed_episodes: list[EpisodeMetric] = []

    for episode in range(episodes):
        dmp = components.make_dmp(
            make_world(world, seed=_world_seed(seed, episode)),
            step_limit=step_limit,
        )
        step_rows: list[StepMetric] = []

        while not dmp.done:
            candidate = dmp.choose_candidate("scorer")
            if candidate is None:
                break
            result = dmp.execute(candidate)
            row = step_metric(
                condition=condition.value,
                seed=seed,
                episode=episode,
                result=result,
            )
            step_rows.append(row)
            seed_steps.append(row)

        seed_episodes.append(
            episode_metric(
                condition=condition.value,
                seed=seed,
                episode=episode,
                steps=step_rows,
                knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                step_limit=step_limit,
            )
        )

    return seed_steps, seed_episodes


def _run_spec_seed(args: tuple[ExperimentSpec, int, int, int, str]) -> tuple[list[StepMetric], list[EpisodeMetric]]:
    spec, seed, episodes, step_limit, world_value = args
    world = WorldKind(world_value)
    components = ExperimentComponents.for_spec(spec, seed=seed)
    seed_steps: list[StepMetric] = []
    seed_episodes: list[EpisodeMetric] = []

    for episode in range(episodes):
        dmp = components.make_dmp(
            make_world(world, seed=_world_seed(seed, episode)),
            step_limit=step_limit,
        )
        step_rows: list[StepMetric] = []

        while not dmp.done:
            candidate = dmp.choose_candidate("scorer")
            if candidate is None:
                break
            result = dmp.execute(candidate)
            row = step_metric(
                condition=spec.condition,
                seed=seed,
                episode=episode,
                result=result,
            )
            step_rows.append(row)
            seed_steps.append(row)

        seed_episodes.append(
            episode_metric(
                condition=spec.condition,
                seed=seed,
                episode=episode,
                steps=step_rows,
                knowledge_reuse_count=int(dmp.metrics()["knowledge_reuse_count"]),
                step_limit=step_limit,
            )
        )

    return seed_steps, seed_episodes


def write_experiment_outputs(
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


def _write_csv(path: Path, rows: list[Any], row_type: type[Any]) -> None:
    fieldnames = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AASSR GridWorld C0/C1/C2/C3/C4 experiments.")
    parser.add_argument("--condition", choices=[condition.value for condition in ExperimentCondition] + ["all"], required=True)
    parser.add_argument("--world", choices=[world.value for world in WorldKind], default=WorldKind.FIXED.value)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.condition)
    if args.condition == "all":
        _, _, summaries = run_all_conditions(
            episodes=args.episodes,
            seeds=args.seeds,
            step_limit=args.step_limit,
            world=args.world,
            output_dir=output_dir,
            workers=args.workers,
            progress=True,
        )
        for summary in summaries:
            _print_summary(summary)
        print(f"wrote {Path(output_dir).resolve()}")
        return
    _, _, summaries = run_experiment(
        condition=args.condition,
        episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        world=args.world,
        output_dir=output_dir,
        workers=args.workers,
        progress=True,
    )
    _print_summary(summaries[0])
    print(f"wrote {Path(output_dir).resolve()}")


def _print_summary(summary: SummaryMetric) -> None:
    print(
        "condition={condition} episodes={episodes} seeds={seeds} "
        "success_rate={success_rate:.3f} steps_to_flag_mean={steps_to_flag_mean:.3f}".format(
            **summary.to_dict()
        )
    )


def default_output_dir(condition: str) -> str:
    return f"runs/gridworld/{condition}"


def _world_seed(seed: int, episode: int) -> int:
    return seed * 100_000 + episode


if __name__ == "__main__":
    main()
