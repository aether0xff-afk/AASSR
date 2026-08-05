from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


_ELAPSED_PATTERN = re.compile(
    r"Elapsed \(wall clock\) time.*\):\s*([0-9:.]+)\s*$"
)
_RSS_PATTERN = re.compile(
    r"Maximum resident set size \(kbytes\):\s*(\d+)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return 60.0 * float(minutes) + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return 3600.0 * float(hours) + 60.0 * float(minutes) + float(seconds)
    return float(value)


def parse_time_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"wall_seconds": 0.0, "peak_rss_mb": 0.0}
    wall_seconds = 0.0
    peak_rss_mb = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        elapsed = _ELAPSED_PATTERN.search(line)
        if elapsed:
            wall_seconds = _elapsed_seconds(elapsed.group(1))
        rss = _RSS_PATTERN.search(line)
        if rss:
            peak_rss_mb = int(rss.group(1)) / 1024.0
    return {
        "wall_seconds": wall_seconds,
        "peak_rss_mb": peak_rss_mb,
    }


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    items = list(rows)
    successful = [item for item in items if int(item["success"])]
    return {
        "episodes": len(items),
        "success_rate": (
            fmean(float(item["success"]) for item in items)
            if items
            else 0.0
        ),
        "mean_return": (
            fmean(float(item["reward"]) for item in items)
            if items
            else 0.0
        ),
        "mean_steps": (
            fmean(float(item["steps"]) for item in items)
            if items
            else 0.0
        ),
        "mean_steps_on_success": (
            fmean(float(item["steps"]) for item in successful)
            if successful
            else 0.0
        ),
        "mean_path_efficiency": (
            fmean(float(item["path_efficiency"]) for item in items)
            if items
            else 0.0
        ),
        "environment_transitions": sum(
            int(item["steps"]) for item in items
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--seen-log", type=Path, required=True)
    parser.add_argument("--unseen-log", type=Path, required=True)
    parser.add_argument("--train-time", type=Path, required=True)
    parser.add_argument("--seen-time", type=Path, required=True)
    parser.add_argument("--unseen-time", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-tail", type=int, default=100)
    parser.add_argument("--upstream-commit", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_rows = read_jsonl(args.train_log)
    seen_rows = read_jsonl(args.seen_log)
    unseen_rows = read_jsonl(args.unseen_log)
    training_tail = train_rows[-min(args.training_tail, len(train_rows)) :]
    train_usage = parse_time_file(args.train_time)
    seen_usage = parse_time_file(args.seen_time)
    unseen_usage = parse_time_file(args.unseen_time)

    payload = {
        "algorithm": "dreamerv3_official_size1m",
        "seed": args.seed,
        "upstream_commit": args.upstream_commit,
        "observation": "same_25d_vector_as_native_baselines",
        "action_count": 4,
        "external_reward": "final_success_only",
        "training": summarize_rows(train_rows),
        "training_tail": summarize_rows(training_tail),
        "evaluation_seen": summarize_rows(seen_rows),
        "evaluation_unseen": summarize_rows(unseen_rows),
        "efficiency": {
            "training_wall_seconds": train_usage["wall_seconds"],
            "training_peak_rss_mb": train_usage["peak_rss_mb"],
            "seen_eval_wall_seconds": seen_usage["wall_seconds"],
            "unseen_eval_wall_seconds": unseen_usage["wall_seconds"],
            "seen_eval_peak_rss_mb": seen_usage["peak_rss_mb"],
            "unseen_eval_peak_rss_mb": unseen_usage["peak_rss_mb"],
            "checkpoint_bytes": directory_bytes(args.checkpoint),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
