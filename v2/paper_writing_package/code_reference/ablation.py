from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .analysis import analyze_results
from .experiment import ExperimentSpec, ProphecyKind, run_experiment_spec
from .imagination import ImaginationConfig
from .worlds import WorldKind


DEFAULT_ENVIRONMENTS = (
    WorldKind.RANDOM_KEY_DOOR,
    WorldKind.V2_COMPLEX,
    WorldKind.LOCKED_BOTTLENECK,
)


@dataclass(frozen=True)
class AblationSuite:
    name: str
    description: str
    specs: tuple[ExperimentSpec, ...]


def prophecy_implementation_suite() -> AblationSuite:
    """Ablation 1: compare Prophecy Module implementations."""
    return AblationSuite(
        name="ablation_1_prophecy_model",
        description="TableProphecyModel vs TransformerProphecyModel inside the same C3 loop.",
        specs=(
            ExperimentSpec(
                condition="A1_TABLE_C3",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
            ),
            ExperimentSpec(
                condition="A1_TRANSFORMER_C3",
                prophecy_kind=ProphecyKind.TRANSFORMER,
                prophecy_beta=0.3,
                use_imagination=True,
            ),
        ),
    )


def prophecy_reward_suite() -> AblationSuite:
    """Ablation 2: compare prediction-error reward on/off."""
    return AblationSuite(
        name="ablation_2_prophecy_reward",
        description="C3 with prediction-error reward enabled versus disabled.",
        specs=(
            ExperimentSpec(
                condition="A2_REWARD_ON",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
            ),
            ExperimentSpec(
                condition="A2_REWARD_OFF",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.0,
                use_imagination=True,
            ),
        ),
    )


def imagination_depth_branch_suite(
    *,
    depths: tuple[int, ...] = (1, 2, 3),
    branches: tuple[int, ...] = (1, 3, 5),
) -> AblationSuite:
    """Ablation 3: compare one-step/deeper candidate rollout settings."""
    specs = []
    for depth in depths:
        for branch in branches:
            specs.append(
                ExperimentSpec(
                    condition=f"A3_D{depth}_B{branch}",
                    prophecy_kind=ProphecyKind.TABLE,
                    prophecy_beta=0.3,
                    use_imagination=True,
                    imagination_config=ImaginationConfig(
                        rollout_depth=depth,
                        rollout_branching=branch,
                    ),
                )
            )
    return AblationSuite(
        name="ablation_3_imagination_depth_branch",
        description="C3 with controlled Imagination rollout depth and branching factor.",
        specs=tuple(specs),
    )


def imagination_mechanism_suite() -> AblationSuite:
    """Ablation 4: remove Imagination mechanism terms one at a time."""
    return AblationSuite(
        name="ablation_4_imagination_mechanisms",
        description=(
            "C3 mechanism ablation for dependency-aware rollout, repeat penalty, "
            "policy prior, and rollout contribution."
        ),
        specs=(
            ExperimentSpec(
                condition="A4_FULL_C3",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(),
            ),
            ExperimentSpec(
                condition="A4_NO_DEPENDENCY",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(dependency_weight=0.0),
            ),
            ExperimentSpec(
                condition="A4_NO_REPEAT_PENALTY",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(repeat_weight=0.0),
            ),
            ExperimentSpec(
                condition="A4_NO_POLICY_PRIOR",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(policy_prior_weight=0.0),
            ),
            ExperimentSpec(
                condition="A4_NO_ROLLOUT_VALUE",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(rollout_discount=0.0),
            ),
            ExperimentSpec(
                condition="A4_ONE_STEP_NO_DEP",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(
                    rollout_depth=1,
                    rollout_branching=1,
                    rollout_discount=0.0,
                    dependency_weight=0.0,
                ),
            ),
        ),
    )


def prophecy_score_component_suite() -> AblationSuite:
    """Ablation 5: remove Prophecy prediction score components one at a time."""
    return AblationSuite(
        name="ablation_5_prophecy_score_components",
        description="C3 score-component ablation for predicted knowledge gain, flag relevance, and error likelihood.",
        specs=(
            ExperimentSpec(
                condition="A5_FULL_C3",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(),
            ),
            ExperimentSpec(
                condition="A5_NO_KNOWLEDGE_GAIN",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(knowledge_weight=0.0),
            ),
            ExperimentSpec(
                condition="A5_NO_FLAG_PROB",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(flag_weight=0.0),
            ),
            ExperimentSpec(
                condition="A5_NO_ERROR_AVOIDANCE",
                prophecy_kind=ProphecyKind.TABLE,
                prophecy_beta=0.3,
                use_imagination=True,
                imagination_config=ImaginationConfig(error_weight=0.0),
            ),
        ),
    )


def suites_for_name(name: str) -> tuple[AblationSuite, ...]:
    suites = {
        "ablation_1": prophecy_implementation_suite,
        "ablation_2": prophecy_reward_suite,
        "ablation_3": imagination_depth_branch_suite,
        "ablation_4": imagination_mechanism_suite,
        "ablation_5": prophecy_score_component_suite,
    }
    if name == "all":
        return tuple(factory() for factory in suites.values())
    if name not in suites:
        raise ValueError(f"unknown ablation suite: {name}")
    return (suites[name](),)


def run_ablation_suite(
    *,
    suite: AblationSuite,
    world: str | WorldKind,
    episodes: int,
    seeds: int,
    step_limit: int,
    output_dir: str | Path,
    workers: int = 1,
    analyze: bool = True,
    progress: bool = False,
) -> Path:
    world = WorldKind(world)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for spec in suite.specs:
        run_experiment_spec(
            spec=spec,
            episodes=episodes,
            seeds=seeds,
            step_limit=step_limit,
            world=world,
            output_dir=output_path / spec.condition,
            workers=workers,
            progress=progress,
            progress_label=f"{world.value}/{suite.name}/{spec.condition}",
        )
    _write_suite_readme(output_path, suite=suite, world=world, episodes=episodes, seeds=seeds, step_limit=step_limit)
    if analyze:
        analyze_results(
            input_dir=output_path,
            output_dir=output_path / "analysis",
            learning_window=max(1, min(20, episodes)),
        )
    return output_path


def run_ablation_matrix(
    *,
    suite_name: str,
    worlds: tuple[WorldKind, ...],
    episodes: int,
    seeds: int,
    step_limit: int,
    output_dir: str | Path,
    workers: int = 1,
    analyze: bool = True,
    progress: bool = False,
) -> list[Path]:
    outputs = []
    for world in worlds:
        for suite in suites_for_name(suite_name):
            outputs.append(
                run_ablation_suite(
                    suite=suite,
                    world=world,
                    episodes=episodes,
                    seeds=seeds,
                    step_limit=step_limit,
                    output_dir=Path(output_dir) / world.value / suite.name,
                    workers=workers,
                    analyze=analyze,
                    progress=progress,
                )
            )
    return outputs


def parse_worlds(value: str) -> tuple[WorldKind, ...]:
    if value == "all":
        return DEFAULT_ENVIRONMENTS
    return tuple(WorldKind(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APASSR ablation suites.")
    parser.add_argument(
        "--suite",
        choices=["ablation_1", "ablation_2", "ablation_3", "ablation_4", "ablation_5", "all"],
        default="all",
    )
    parser.add_argument("--world", default=WorldKind.V2_COMPLEX.value)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--step-limit", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/ablations")
    parser.add_argument("--no-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_ablation_matrix(
        suite_name=args.suite,
        worlds=parse_worlds(args.world),
        episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        output_dir=args.output_dir,
        workers=args.workers,
        analyze=not args.no_analysis,
        progress=True,
    )
    for output in outputs:
        print(f"wrote {output.resolve()}")


def _write_suite_readme(
    output_path: Path,
    *,
    suite: AblationSuite,
    world: WorldKind,
    episodes: int,
    seeds: int,
    step_limit: int,
) -> None:
    lines = [
        f"# {suite.name}",
        "",
        suite.description,
        "",
        f"- World: `{world.value}`",
        f"- Episodes per seed: `{episodes}`",
        f"- Seeds: `{seeds}`",
        f"- Step limit: `{step_limit}`",
        "",
        "These ablations preserve C3 as the main paper-aligned APASSR loop and vary only the stated experimental factor.",
        "",
        "Conditions:",
    ]
    for spec in suite.specs:
        config = spec.imagination_config
        lines.append(
            f"- `{spec.condition}`: prophecy={spec.prophecy_kind.value}, "
            f"prophecy_beta={spec.prophecy_beta}, imagination={spec.use_imagination}, "
            f"depth={config.rollout_depth}, branch={config.rollout_branching}, "
            f"knowledge_w={config.knowledge_weight}, flag_w={config.flag_weight}, "
            f"error_w={config.error_weight}, repeat_w={config.repeat_weight}, "
            f"policy_prior_w={config.policy_prior_weight}, rollout_discount={config.rollout_discount}, "
            f"dependency_w={config.dependency_weight}"
        )
    output_path.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
