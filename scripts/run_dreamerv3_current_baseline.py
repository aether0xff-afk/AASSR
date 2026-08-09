from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.dreamerv3_external import (
    run_official_dreamerv3_current_baseline,
)
from aassr_v2.dreamerv3_hardware import stamp_dreamer_summary_hardware


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned official danijar/DreamerV3 implementation on the "
            "current AASSR pentest protocol through the relational dynamic-action adapter."
        )
    )
    parser.add_argument("--dreamer-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="runs/aassr_current_generation_main/seed-7/dreamerv3",
    )
    parser.add_argument("--research-seed", type=int, default=7)
    parser.add_argument("--transition-budget", type=int, default=10_000)
    parser.add_argument("--block-target", type=int, default=512)
    parser.add_argument("--jax-platform", default="cuda")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help=(
            "Override the official dmc_proprio preset train ratio. The canonical "
            "baseline leaves this unset and records the upstream value."
        ),
    )
    parser.add_argument(
        "--no-prealloc",
        action="store_true",
        help="Disable JAX GPU preallocation without changing Dreamer learning equations.",
    )
    parser.add_argument(
        "--allow-upstream-mismatch",
        action="store_true",
        help="Allow a non-pinned DreamerV3 checkout for diagnostic runs only.",
    )
    args = parser.parse_args()

    result = run_official_dreamerv3_current_baseline(
        args.output_dir,
        dreamer_root=args.dreamer_root,
        research_seed=args.research_seed,
        transition_budget=args.transition_budget,
        block_target=args.block_target,
        jax_platform=args.jax_platform,
        train_ratio=args.train_ratio,
        prealloc=not args.no_prealloc,
        allow_upstream_mismatch=args.allow_upstream_mismatch,
    )
    hardware = stamp_dreamer_summary_hardware(
        result,
        requested_platform=args.jax_platform,
    )
    artifact = Path(args.output_dir) / "summary_dreamerv3_relational.json"
    artifact.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "DreamerV3 current baseline:",
        f"seed={result['research_seed']}",
        f"transitions={result['transitions_used']}",
        f"driver_steps={result['dreamer_driver_steps']}",
        f"updates={result['gradient_updates']}",
        f"diag={result['diagnostic_successes']}",
        f"focus={result['final_focus_level']}",
        f"jax={hardware['actual_platforms']}",
        f"upstream={result['official_upstream']['actual_commit'][:12]}",
    )
    print("artifact:", artifact)


if __name__ == "__main__":
    main()
