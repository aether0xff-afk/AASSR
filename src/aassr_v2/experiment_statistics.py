from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from .experiment_runner import SUMMARY_METRICS, read_rows

GROUP_FIELDS = (
    "suite",
    "condition",
    "environment",
    "model",
    "action_family",
)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_level_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average episodes inside each seed before cross-seed statistics."""

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in GROUP_FIELDS) + (
            str(row.get("seed", "")),
        )
        groups.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        record: dict[str, Any] = {
            field: key[index]
            for index, field in enumerate((*GROUP_FIELDS, "seed"))
        }
        record["episode_rows"] = len(items)
        for metric in SUMMARY_METRICS:
            values = [
                numeric
                for item in items
                if (numeric := _to_float(item.get(metric))) is not None
            ]
            record[metric] = fmean(values) if values else ""
        result.append(record)
    return result


def cross_seed_summary(
    seed_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        key = tuple(str(row.get(field, "")) for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        record: dict[str, Any] = {
            field: key[index]
            for index, field in enumerate(GROUP_FIELDS)
        }
        record["seed_count"] = len(items)
        record["episode_rows"] = sum(int(item["episode_rows"]) for item in items)
        for metric in SUMMARY_METRICS:
            values = [
                numeric
                for item in items
                if (numeric := _to_float(item.get(metric))) is not None
            ]
            if not values:
                record[f"{metric}_mean"] = ""
                record[f"{metric}_sd"] = ""
                record[f"{metric}_ci95"] = ""
                continue
            mean = fmean(values)
            deviation = stdev(values) if len(values) > 1 else 0.0
            ci95 = (
                1.96 * deviation / (len(values) ** 0.5)
                if len(values) > 1
                else 0.0
            )
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = deviation
            record[f"{metric}_ci95"] = ci95
        result.append(record)
    return result


def seed_fields() -> tuple[str, ...]:
    return (
        *GROUP_FIELDS,
        "seed",
        "episode_rows",
        *SUMMARY_METRICS,
    )


def summary_fields() -> tuple[str, ...]:
    fields = [*GROUP_FIELDS, "seed_count", "episode_rows"]
    for metric in SUMMARY_METRICS:
        fields.extend(
            (
                f"{metric}_mean",
                f"{metric}_sd",
                f"{metric}_ci95",
            )
        )
    return tuple(fields)


def _format(value: Any) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric:.4f}"


def write_seed_report(
    path: Path,
    config: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        f"# {config['name']} seed-level report",
        "",
        "Episode results were averaged inside each seed first. The mean, standard deviation, and 95% interval below are computed across seed means, not across individual episodes.",
        "",
        "| Suite | Condition | Environment | Model | Action | Seeds | Success | Steps | Prediction | Actual return |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["suite"]),
                    str(row["condition"]),
                    str(row["environment"]),
                    str(row["model"]),
                    str(row["action_family"] or "-"),
                    str(row["seed_count"]),
                    _format(row.get("success_mean")),
                    _format(row.get("steps_mean")),
                    _format(row.get("prediction_score_mean")),
                    _format(row.get("actual_return_mean")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## 주의",
            "",
            "파일럿은 실행 배선과 지표 방향을 확인하는 용도다. 성능 주장은 본 실험 설정과 충분한 seed 수를 사용한 뒤 내려야 한다.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def regenerate_seed_level_summary(
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    directory = Path(output_dir)
    rows = read_rows(directory / "episodes.csv")
    config = json.loads(
        (directory / "resolved_config.json").read_text(encoding="utf-8")
    )
    per_seed = seed_level_rows(rows)
    summary = cross_seed_summary(per_seed)
    seed_csv = directory / "seed_summary.csv"
    summary_csv = directory / "summary.csv"
    report = directory / "report.md"
    _write_csv(seed_csv, per_seed, seed_fields())
    _write_csv(summary_csv, summary, summary_fields())
    write_seed_report(report, config, summary)
    return seed_csv, summary_csv, report
