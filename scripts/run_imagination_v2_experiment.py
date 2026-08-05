from __future__ import annotations

import argparse

from aassr_v2.imagination_v2 import IMAGINATION_V2_CONDITIONS
from aassr_v2.imagination_v2_experiment import run_imagination_v2_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=IMAGINATION_V2_CONDITIONS,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--train-map-count", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=(0, 100, 250, 500, 1000),
    )
    args = parser.parse_args()
    run_imagination_v2_experiment(
        args.output,
        condition=args.condition,
        seed=args.seed,
        train_episodes=args.train_episodes,
        train_map_count=args.train_map_count,
        evaluation_episodes=args.evaluation_episodes,
        checkpoints=tuple(args.checkpoints),
    )


if __name__ == "__main__":
    main()
