from __future__ import annotations

import argparse
from pathlib import Path

from aassr_v2.experiment_runner import regenerate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate summary.csv and report.md from an AASSR run directory."
    )
    parser.add_argument(
        "run_dir",
        help="Directory containing episodes.csv and resolved_config.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = regenerate_summary(Path(args.run_dir))
    print(f"Rows: {artifacts.row_count}")
    print(f"Summary: {artifacts.summary_csv.resolve()}")
    print(f"Report: {artifacts.report_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
