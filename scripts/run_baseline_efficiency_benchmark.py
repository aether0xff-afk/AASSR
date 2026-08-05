from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.baseline_efficiency_benchmark import (
    run_gridpush_baseline_benchmark,
)


def _parse_checkpoints(value: str) -> tuple[int, ...]:
    items = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )
    if not items:
        raise argparse.ArgumentTypeError("at least one checkpoint is required")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Random, Q-learning, DQN, Policy-only and AASSR on the "
            "same solvable sparse-reward GridPush maps."
        )
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--train-map-count", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument(
        "--checkpoints",
        type=_parse_checkpoints,
        default=(0, 50, 100, 250, 500, 1000),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_gridpush_baseline_benchmark(
        args.output,
        condition=args.condition,
        seed=args.seed,
        train_episodes=args.train_episodes,
        train_map_count=args.train_map_count,
        evaluation_episodes=args.evaluation_episodes,
        checkpoints=args.checkpoints,
    )
    print(json.dumps(payload["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
