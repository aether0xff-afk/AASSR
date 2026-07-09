from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_results
from .dqn_baseline import run_dqn_partial_baseline
from .experiment import run_all_conditions
from .mdp_baseline import run_mdp_baseline
from .traditional_baselines import run_q_learning_baseline
from .worlds import WorldKind


def run_v2_comparison(
    *,
    episodes: int,
    seeds: int,
    step_limit: int,
    output_dir: str | Path,
    world: str | WorldKind = WorldKind.V2_COMPLEX,
    analyze: bool = True,
    workers: int = 1,
    progress: bool = False,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_all_conditions(
        episodes=episodes,
        seeds=seeds,
        step_limit=step_limit,
        world=world,
        output_dir=output_path,
        workers=workers,
        progress=progress,
    )
    run_q_learning_baseline(
        episodes=episodes,
        seeds=seeds,
        step_limit=step_limit,
        world=world,
        output_dir=output_path / "QLEARN",
        workers=workers,
        progress=progress,
        progress_label=f"{WorldKind(world).value}/QLEARN",
    )
    run_dqn_partial_baseline(
        episodes=episodes,
        seeds=seeds,
        step_limit=step_limit,
        world=world,
        output_dir=output_path / "DQN_PARTIAL",
        workers=workers,
        progress=progress,
        progress_label=f"{WorldKind(world).value}/DQN_PARTIAL",
    )
    run_mdp_baseline(
        episodes=episodes,
        seeds=seeds,
        step_limit=step_limit,
        world=world,
        output_dir=output_path / "ORACLE_MDP",
        workers=workers,
        progress=progress,
        progress_label=f"{WorldKind(world).value}/ORACLE_MDP",
    )
    if analyze:
        analyze_results(
            input_dir=output_path,
            output_dir=output_path / "analysis",
            learning_window=max(1, min(20, episodes)),
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 complex GridWorld comparison experiments.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--step-limit", type=int, default=120)
    parser.add_argument("--world", choices=[world.value for world in WorldKind], default=WorldKind.V2_COMPLEX.value)
    parser.add_argument("--output-dir", default="runs/v2_complex_compare")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_v2_comparison(
        episodes=args.episodes,
        seeds=args.seeds,
        step_limit=args.step_limit,
        world=args.world,
        output_dir=args.output_dir,
        analyze=not args.no_analysis,
        workers=args.workers,
        progress=True,
    )
    print(f"wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
