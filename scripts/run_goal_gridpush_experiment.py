from __future__ import annotations

import argparse
from pathlib import Path

from aassr_v2.goal_gridpush_diagnostic_setup import (
    install_goal_gridpush_diagnostic_agents,
)
from aassr_v2.goal_gridpush_experiment import run_goal_gridpush_experiment


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Policy, Prophecy, Imagination and separated GOAL Maker/Executor on GridPush.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(7, 13, 21, 42, 100))
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--train-maps", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--training-tail", type=int, default=80)
    args = parser.parse_args()

    install_goal_gridpush_diagnostic_agents()
    payload = run_goal_gridpush_experiment(
        args.output,
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        train_map_count=args.train_maps,
        evaluation_episodes=args.eval_episodes,
        training_tail=args.training_tail,
    )
    print(f"wrote {args.output / 'episodes.csv'}")
    print(f"wrote {args.output / 'summary.json'}")
    for row in payload["summary"]:
        if row["phase"] in {"evaluation_seen", "evaluation_unseen"}:
            print(
                f"{row['condition']:>24} {row['phase']:>18} "
                f"success={row['success_rate']:.3f} "
                f"steps={row['mean_steps_on_success']:.2f} "
                f"interventions={row['mean_interventions']:.2f} "
                f"goals={row['mean_goal_proposals']:.2f}"
            )


if __name__ == "__main__":
    main()
