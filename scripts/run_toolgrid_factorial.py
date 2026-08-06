from __future__ import annotations

import argparse
import json

from aassr_v2.toolgrid_factorial_masked import (
    ACTION_COUNTS,
    GRID_SIZES,
    run_toolgrid_factorial,
)
from aassr_v2.toolgrid_matched_factorial import (
    MATCHED_HYBRID_CONDITION,
    run_toolgrid_matched_hybrid,
)


RUN_CONDITIONS = ("dqn", MATCHED_HYBRID_CONDITION)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one ToolGrid map-size × semantic-branching factorial cell."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", required=True, choices=RUN_CONDITIONS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--grid-size", required=True, type=int, choices=GRID_SIZES)
    parser.add_argument("--action-count", required=True, type=int, choices=ACTION_COUNTS)
    parser.add_argument("--transition-budget", type=int, default=5_000)
    parser.add_argument("--train-map-count", type=int, default=48)
    parser.add_argument("--evaluation-map-count", type=int, default=100)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Evaluation checkpoints. By default only 0 and the final transition "
            "budget are used so an intermediate checkpoint cannot reset a "
            "partially completed training episode."
        ),
    )
    args = parser.parse_args()
    checkpoints = (
        (0, args.transition_budget)
        if args.checkpoints is None
        else tuple(args.checkpoints)
    )
    common = {
        "seed": args.seed,
        "grid_size": args.grid_size,
        "action_count": args.action_count,
        "transition_budget": args.transition_budget,
        "train_map_count": args.train_map_count,
        "evaluation_map_count": args.evaluation_map_count,
        "checkpoints": checkpoints,
    }
    if args.condition == MATCHED_HYBRID_CONDITION:
        payload = run_toolgrid_matched_hybrid(args.output, **common)
    else:
        payload = run_toolgrid_factorial(
            args.output,
            condition="dqn",
            **common,
        )
    print(json.dumps(payload["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
