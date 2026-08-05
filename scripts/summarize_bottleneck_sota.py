from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.input.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        final = payload["final"]
        spec = payload.get("condition", {})
        rows.append(
            {
                "condition": final["condition"],
                "seed": int(final["seed"]),
                "identity_role": spec.get("identity_role", ""),
                "diagnostic_question": spec.get("diagnostic_question", ""),
                "seen_success_rate": float(final["seen_success_rate"]),
                "unseen_success_rate": float(final["unseen_success_rate"]),
                "environment_steps_total": int(final["environment_steps_total"]),
                "training_wall_seconds": float(final["training_wall_seconds"]),
                "selection_seconds_total": float(final["selection_seconds_total"]),
                "update_seconds_total": float(final["update_seconds_total"]),
                "peak_rss_mb": float(final["peak_rss_mb"]),
                "model_bytes": int(final["model_bytes"]),
                "seen_path_efficiency": float(final["seen_path_efficiency"]),
                "unseen_path_efficiency": float(final["unseen_path_efficiency"]),
            }
        )
    if not rows:
        raise SystemExit("no summary.json files found")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["condition"], row["seed"])))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    aggregate: list[dict[str, Any]] = []
    for condition, items in sorted(grouped.items()):
        def mean(field: str) -> float:
            return fmean(float(item[field]) for item in items)

        def sd(field: str) -> float:
            values = [float(item[field]) for item in items]
            return pstdev(values) if len(values) > 1 else 0.0

        aggregate.append(
            {
                "condition": condition,
                "seeds": len(items),
                "identity_role": items[0]["identity_role"],
                "seen_success_rate_mean": mean("seen_success_rate"),
                "seen_success_rate_sd": sd("seen_success_rate"),
                "unseen_success_rate_mean": mean("unseen_success_rate"),
                "unseen_success_rate_sd": sd("unseen_success_rate"),
                "environment_steps_mean": mean("environment_steps_total"),
                "training_wall_seconds_mean": mean("training_wall_seconds"),
                "selection_seconds_mean": mean("selection_seconds_total"),
                "update_seconds_mean": mean("update_seconds_total"),
                "peak_rss_mb_mean": mean("peak_rss_mb"),
                "model_mb_mean": mean("model_bytes") / (1024.0 ** 2),
                "seen_path_efficiency_mean": mean("seen_path_efficiency"),
                "unseen_path_efficiency_mean": mean("unseen_path_efficiency"),
            }
        )

    with (args.output / "aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    lines = [
        "# Bottleneck/SOTA diagnostic summary",
        "",
        "| condition | role | seen | unseen | train s | select s | RSS MB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {condition} | {identity_role} | {seen_success_rate_mean:.1%} | "
            "{unseen_success_rate_mean:.1%} | {training_wall_seconds_mean:.2f} | "
            "{selection_seconds_mean:.2f} | {peak_rss_mb_mean:.1f} |".format(**row)
        )
    (args.output / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
