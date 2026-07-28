from __future__ import annotations

import argparse
from pathlib import Path

from aassr_v2.experiment_runner import (
    load_config,
    planned_run_count,
    run_experiment,
)
from aassr_v2.experiment_statistics import regenerate_seed_level_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible AASSR v2 experiment suites from JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment JSON config.",
    )
    parser.add_argument(
        "--output",
        help="Override output directory from the config.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=(
            "prophecy",
            "imagination",
            "dependency",
            "skills",
            "information_value",
        ),
        help="Run only selected suite kinds. Repeat this option for several suites.",
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated integer seeds overriding the config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory instead of adding a timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the planned result-row count only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seeds = None
    if args.seeds:
        seeds = tuple(
            int(value.strip())
            for value in args.seeds.split(",")
            if value.strip()
        )
        if not seeds:
            raise SystemExit("--seeds must contain at least one integer")
        config["seeds"] = list(seeds)

    planned = planned_run_count(
        config,
        set(args.suite or ()) or None,
    )
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Experiment: {config['name']}")
    print(f"Seeds: {config['seeds']}")
    print(f"Planned result rows: {planned}")
    if args.dry_run:
        return 0

    artifacts = run_experiment(
        config,
        output_dir=args.output,
        overwrite=args.overwrite,
        suite_filter=args.suite,
        seed_override=seeds,
    )
    seed_summary, summary, report = regenerate_seed_level_summary(
        artifacts.output_dir
    )
    print(f"Completed rows: {artifacts.row_count}")
    print(f"Output: {artifacts.output_dir.resolve()}")
    print(f"Episodes: {artifacts.episodes_csv.resolve()}")
    print(f"Seed summary: {seed_summary.resolve()}")
    print(f"Summary: {summary.resolve()}")
    print(f"Report: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
