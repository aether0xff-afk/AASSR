from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aassr_v2.paper_protocol import (
    load_paper_config,
    planned_paper_run_count,
)
from aassr_v2.paper_runner import run_paper_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the versioned AASSR paper experiment protocol."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--pilot", action="store_true")
    stage.add_argument("--final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    config = load_paper_config(args.config)
    requested_stage = "pilot" if args.pilot else "final" if args.final else ""
    if requested_stage and config["study_stage"] != requested_stage:
        raise SystemExit(
            f"config stage is {config['study_stage']}, not {requested_stage}"
        )
    planned = planned_paper_run_count(config)
    print(f"Protocol: {config['protocol_version']}")
    print(f"Stage: {config['study_stage']}")
    print(f"Research seeds: {len(config['research_seeds'])}")
    print(f"Planned episode rows (upper estimate): {planned:,}")
    if args.dry_run:
        return 0
    started = time.perf_counter()
    artifacts = run_paper_suite(
        config,
        output_dir=args.output,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"Completed rows: {artifacts.row_count:,}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")
    print(f"Output: {artifacts.output_dir.resolve()}")
    print(f"Report: {artifacts.report_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
