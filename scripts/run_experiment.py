from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.experiment_statistics import regenerate_seed_level_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible AASSR v2 experiment suites from JSON config."
    )
    parser.add_argument("--config", required=True, help="Path to an experiment JSON config.")
    parser.add_argument("--output", help="Override output directory from the config.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=(
            "prophecy",
            "imagination",
            "dependency",
            "skills",
            "information_value",
            "autonomous_discovery",
        ),
        help="Run only selected suite kinds. Repeat this option for several suites.",
    )
    parser.add_argument(
        "--seeds", help="Comma-separated integer seeds overriding the config."
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


def _seed_override(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    return seeds


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    runner = str(raw.get("runner", "legacy"))
    if runner == "final_pilot":
        from aassr_v2.final_pilot import (
            load_final_config as load_config,
            planned_final_run_count as planned_run_count,
            run_final_pilot as run_experiment,
        )
    elif runner == "autonomous_main":
        from aassr_v2.autonomous_experiment import (
            load_autonomous_config as load_config,
            planned_autonomous_run_count as planned_run_count,
            run_autonomous_experiment as run_experiment,
        )
    else:
        from aassr_v2.experiment_runner import (
            load_config,
            planned_run_count,
            run_experiment,
        )

    config = load_config(config_path)
    seeds = _seed_override(args.seeds)
    if seeds is not None:
        config["seeds"] = list(seeds)
    selected = set(args.suite or ()) or None
    planned = planned_run_count(config, selected)
    print(f"Config: {config_path.resolve()}")
    print(f"Experiment: {config['name']}")
    print(f"Runner: {runner}")
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
