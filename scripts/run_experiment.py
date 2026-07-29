from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aassr_v2.experiment_statistics import regenerate_seed_level_summary
from aassr_v2.progress import format_duration


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
    parser.add_argument(
        "--progress-every",
        type=int,
        help="Print and persist progress after this many completed episodes.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        help="Print and persist progress at least this often in seconds.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable console progress lines while still writing progress files.",
    )
    args = parser.parse_args()
    if args.progress_every is not None and args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.progress_seconds is not None and args.progress_seconds <= 0.0:
        parser.error("--progress-seconds must be positive")
    return args


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
    print(f"Config: {config_path.resolve()}", flush=True)
    print(f"Experiment: {config['name']}", flush=True)
    print(f"Runner: {runner}", flush=True)
    print(f"Seeds: {config['seeds']}", flush=True)
    print(f"Planned result rows: {planned:,}", flush=True)
    if runner == "autonomous_main":
        progress = config.get("progress", {})
        every_items = args.progress_every or progress.get("every_episodes", 100)
        every_seconds = args.progress_seconds or progress.get("every_seconds", 10.0)
        print(
            "Progress: "
            f"every {int(every_items):,} episodes or {float(every_seconds):g} seconds",
            flush=True,
        )
        print(
            "Progress files: progress.log, progress.jsonl, progress.json",
            flush=True,
        )
    if args.dry_run:
        return 0

    run_kwargs = {
        "output_dir": args.output,
        "overwrite": args.overwrite,
        "suite_filter": args.suite,
        "seed_override": seeds,
    }
    if runner == "autonomous_main":
        run_kwargs.update(
            {
                "progress_every": args.progress_every,
                "progress_seconds": args.progress_seconds,
                "progress_console": not args.quiet_progress,
            }
        )

    started = time.perf_counter()
    artifacts = run_experiment(config, **run_kwargs)
    experiment_elapsed = time.perf_counter() - started
    print(
        f"[AASSR:postprocess] episode execution complete in "
        f"{format_duration(experiment_elapsed)}; generating seed summaries...",
        flush=True,
    )
    postprocess_started = time.perf_counter()
    seed_summary, summary, report = regenerate_seed_level_summary(
        artifacts.output_dir
    )
    postprocess_elapsed = time.perf_counter() - postprocess_started
    print(
        f"[AASSR:postprocess] summaries complete in "
        f"{format_duration(postprocess_elapsed)}",
        flush=True,
    )
    print(f"Completed rows: {artifacts.row_count:,}", flush=True)
    print(f"Total elapsed: {format_duration(time.perf_counter() - started)}", flush=True)
    print(f"Output: {artifacts.output_dir.resolve()}", flush=True)
    print(f"Episodes: {artifacts.episodes_csv.resolve()}", flush=True)
    print(f"Seed summary: {seed_summary.resolve()}", flush=True)
    print(f"Summary: {summary.resolve()}", flush=True)
    print(f"Report: {report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
