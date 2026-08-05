from __future__ import annotations

import argparse
import json

from aassr_v2.final_complexity_scaling import (
    FINAL_COMPLEXITY_CONDITIONS,
    run_final_complexity_scaling,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one condition/seed cell of the frozen final complexity scaling experiment."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=FINAL_COMPLEXITY_CONDITIONS,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--transition-budget", type=int, default=20_000)
    parser.add_argument("--train-maps-per-level", type=int, default=32)
    parser.add_argument("--evaluation-maps-per-level", type=int, default=100)
    parser.add_argument("--pool-oversample", type=int, default=3)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=(0, 2_500, 5_000, 10_000, 20_000),
    )
    args = parser.parse_args()
    payload = run_final_complexity_scaling(
        args.output,
        condition=args.condition,
        seed=args.seed,
        transition_budget=args.transition_budget,
        train_maps_per_level=args.train_maps_per_level,
        evaluation_maps_per_level=args.evaluation_maps_per_level,
        checkpoints=tuple(args.checkpoints),
        pool_oversample=args.pool_oversample,
    )
    print(json.dumps(payload["final"], indent=2, sort_keys=True, default=repr))


if __name__ == "__main__":
    main()
