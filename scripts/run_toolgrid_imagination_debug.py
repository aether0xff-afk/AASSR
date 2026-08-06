from __future__ import annotations

import argparse
import json

from aassr_v2 import toolgrid_debug_clone as _toolgrid_debug_clone  # noqa: F401
from aassr_v2.toolgrid_factorial_masked import ACTION_COUNTS, GRID_SIZES
from aassr_v2.toolgrid_imagination_fix_patch import install_toolgrid_imagination_fix


STRATEGIES = (
    "baseline",
    "calibration_fix",
    "balanced_tool_replay",
    "tool_decision_gate",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one ToolGrid hybrid without imagination interventions, then "
            "evaluate the identical checkpoint with imagination off and on."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--grid-size", required=True, type=int, choices=GRID_SIZES)
    parser.add_argument("--action-count", required=True, type=int, choices=ACTION_COUNTS)
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--transition-budget", type=int, default=5_000)
    parser.add_argument("--train-map-count", type=int, default=48)
    parser.add_argument("--evaluation-map-count", type=int, default=100)
    args = parser.parse_args()

    install_toolgrid_imagination_fix(args.strategy)
    from aassr_v2.toolgrid_imagination_debug import run_toolgrid_imagination_debug

    summary = run_toolgrid_imagination_debug(
        args.output,
        seed=args.seed,
        grid_size=args.grid_size,
        action_count=args.action_count,
        transition_budget=args.transition_budget,
        train_map_count=args.train_map_count,
        evaluation_map_count=args.evaluation_map_count,
    )
    summary["config"]["debug_strategy"] = args.strategy
    with open(f"{args.output}/summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=repr)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
