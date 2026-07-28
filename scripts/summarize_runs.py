from __future__ import annotations

import argparse
from pathlib import Path

from aassr_v2.experiment_runner import read_rows
from aassr_v2.experiment_statistics import regenerate_seed_level_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate seed_summary.csv, summary.csv and report.md "
            "from an AASSR run directory."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Directory containing episodes.csv and resolved_config.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = Path(args.run_dir)
    rows = read_rows(directory / "episodes.csv")
    seed_summary, summary, report = regenerate_seed_level_summary(directory)
    print(f"Rows: {len(rows)}")
    print(f"Seed summary: {seed_summary.resolve()}")
    print(f"Summary: {summary.resolve()}")
    print(f"Report: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
