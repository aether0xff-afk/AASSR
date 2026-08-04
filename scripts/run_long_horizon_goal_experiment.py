from __future__ import annotations

import argparse
from pathlib import Path

from aassr_v2.compositional_long_horizon_world import (
    install_compositional_long_horizon_world,
)
from aassr_v2.long_horizon_goal_experiment import run_long_horizon_goal_experiment


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare direct Imagination and hierarchical GOAL planning on a long sparse-reward dependency chain.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(7, 13, 21, 42, 100))
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--train-maps", type=int, default=48)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--training-tail", type=int, default=100)
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--room-length", type=int, default=6)
    args = parser.parse_args()

    install_compositional_long_horizon_world()
    payload = run_long_horizon_goal_experiment(
        args.output,
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        train_map_count=args.train_maps,
        evaluation_episodes=args.eval_episodes,
        training_tail=args.training_tail,
        stage_count=args.stages,
        room_length=args.room_length,
    )
    print(f"wrote {args.output / 'episodes.csv'}")
    print(f"wrote {args.output / 'summary.json'}")
    for row in payload["summary"]:  # type: ignore[index]
        if row["phase"] in {"evaluation_seen", "evaluation_unseen"}:
            print(
                f"{row['condition']:>24} {row['phase']:>18} "
                f"success={row['success_rate']:.3f} "
                f"nodes={row['mean_imagined_nodes']:.1f} "
                f"goals={row['mean_goal_proposals']:.2f}"
            )


if __name__ == "__main__":
    main()
