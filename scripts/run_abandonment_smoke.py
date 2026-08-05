from __future__ import annotations

import argparse
import json

from aassr_v2.abandonment_smoke_runner import run_abandonment_smoke


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train frozen Imagination v2 and audit voluntary abandonment."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--train-map-count", type=int, default=32)
    parser.add_argument("--evaluation-episodes", type=int, default=30)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.15, 0.30),
    )
    parser.add_argument("--minimum-steps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--safety-cap", type=int, default=128)
    args = parser.parse_args()
    result = run_abandonment_smoke(
        args.output,
        seed=args.seed,
        train_episodes=args.train_episodes,
        train_map_count=args.train_map_count,
        evaluation_episodes=args.evaluation_episodes,
        thresholds=args.thresholds,
        minimum_steps=args.minimum_steps,
        patience=args.patience,
        safety_cap=args.safety_cap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
