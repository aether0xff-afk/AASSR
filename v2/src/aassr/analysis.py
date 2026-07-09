from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .labels import condition_label
from .plotting import write_analysis_plots


CONDITIONS = ("C0", "C1", "C2", "C3", "C4")


@dataclass(frozen=True)
class SummaryRow:
    condition: str
    success_rate_mean: float
    success_rate_ci95_low: float
    success_rate_ci95_high: float
    steps_to_flag_mean: float
    steps_to_flag_ci95_low: float
    steps_to_flag_ci95_high: float
    semantic_gain_mean: float
    repeat_rate_mean: float
    error_rate_mean: float
    prophecy_error_mean: float
    total_reward_mean: float


@dataclass(frozen=True)
class ConditionStatRow:
    condition: str
    seed: int
    success_rate: float
    steps_to_flag_success_mean: float
    semantic_gain_mean: float
    repeat_rate_mean: float
    error_rate_mean: float
    prophecy_error_mean: float
    total_reward_mean: float
    episodes: int
    successes: int


@dataclass(frozen=True)
class LearningCurveRow:
    condition: str
    window_start: int
    window_end: int
    success_rate: float
    episode_count: int


def analyze_results(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    learning_window: int = 20,
) -> dict[str, list[Any]]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episodes = read_episode_rows(input_path)
    seed_stats = condition_seed_stats(episodes)
    summary = summary_table(seed_stats, bootstrap_samples=bootstrap_samples)
    learning = learning_curve_rows(episodes, window_size=learning_window)

    write_csv(output_path / "summary_table.csv", summary)
    write_csv(output_path / "condition_stats.csv", seed_stats)
    write_csv(output_path / "learning_curve.csv", learning)
    write_analysis_plots(
        summary_rows=summary,
        condition_stats=seed_stats,
        learning_curve=learning,
        output_dir=output_path,
    )
    write_report(
        output_path / "report.md",
        input_dir=input_path,
        output_dir=output_path,
        summary_rows=summary,
    )
    return {
        "summary": summary,
        "condition_stats": seed_stats,
        "learning_curve": learning,
    }


def read_episode_rows(input_dir: str | Path) -> list[dict[str, Any]]:
    input_path = Path(input_dir)
    rows: list[dict[str, Any]] = []
    condition_dirs = [
        path for path in sorted(input_path.iterdir()) if path.is_dir()
    ] if input_path.exists() else []
    for condition_dir in condition_dirs:
        path = condition_dir / "gridworld_episodes.csv"
        if path.exists():
            rows.extend(_read_csv(path))
    if rows:
        return rows
    fallback = input_path / "gridworld_episodes.csv"
    if fallback.exists():
        return _read_csv(fallback)
    raise FileNotFoundError(f"No gridworld_episodes.csv found under {input_path}")


def condition_seed_stats(episodes: list[dict[str, Any]]) -> list[ConditionStatRow]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in episodes:
        grouped.setdefault((str(row["condition"]), int(row["seed"])), []).append(row)

    stats = []
    for (condition, seed), rows in sorted(grouped.items()):
        successes = [row for row in rows if _bool(row["success"])]
        total_steps = sum(_float(row["steps_to_flag"]) for row in rows)
        stats.append(
            ConditionStatRow(
                condition=condition,
                seed=seed,
                success_rate=_safe_mean(1.0 if _bool(row["success"]) else 0.0 for row in rows),
                steps_to_flag_success_mean=_safe_mean(_float(row["steps_to_flag"]) for row in successes),
                semantic_gain_mean=_safe_mean(_float(row["semantic_gain_total"]) for row in rows),
                repeat_rate_mean=_safe_mean(_float(row["repeat_count"]) / _float(row["steps_to_flag"]) for row in rows if _float(row["steps_to_flag"]) > 0),
                error_rate_mean=_safe_mean(_float(row["error_count"]) / _float(row["steps_to_flag"]) for row in rows if _float(row["steps_to_flag"]) > 0),
                prophecy_error_mean=_safe_mean(_float(row["prophecy_error_mean"]) for row in rows),
                total_reward_mean=_safe_mean(_float(row["total_reward"]) for row in rows),
                episodes=len(rows),
                successes=len(successes),
            )
        )
        _ = total_steps
    return stats


def summary_table(seed_stats: list[ConditionStatRow], *, bootstrap_samples: int = 1000) -> list[SummaryRow]:
    by_condition: dict[str, list[ConditionStatRow]] = {}
    for row in seed_stats:
        by_condition.setdefault(row.condition, []).append(row)

    summary = []
    for condition in sorted(by_condition):
        rows = by_condition[condition]
        success_values = [row.success_rate for row in rows]
        step_values = [
            row.steps_to_flag_success_mean
            for row in rows
            if row.successes > 0
        ]
        success_ci = bootstrap_ci(success_values, samples=bootstrap_samples)
        step_ci = bootstrap_ci(step_values, samples=bootstrap_samples)
        summary.append(
            SummaryRow(
                condition=condition,
                success_rate_mean=_safe_mean(success_values),
                success_rate_ci95_low=success_ci[0],
                success_rate_ci95_high=success_ci[1],
                steps_to_flag_mean=_safe_mean(step_values),
                steps_to_flag_ci95_low=step_ci[0],
                steps_to_flag_ci95_high=step_ci[1],
                semantic_gain_mean=_safe_mean(row.semantic_gain_mean for row in rows),
                repeat_rate_mean=_safe_mean(row.repeat_rate_mean for row in rows),
                error_rate_mean=_safe_mean(row.error_rate_mean for row in rows),
                prophecy_error_mean=_safe_mean(row.prophecy_error_mean for row in rows),
                total_reward_mean=_safe_mean(row.total_reward_mean for row in rows),
            )
        )
    return summary


def learning_curve_rows(
    episodes: list[dict[str, Any]],
    *,
    window_size: int = 20,
) -> list[LearningCurveRow]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in episodes:
        condition = str(row["condition"])
        episode = int(row["episode"])
        window_start = (episode // window_size) * window_size
        grouped.setdefault((condition, window_start), []).append(row)

    rows = []
    for (condition, window_start), window_rows in sorted(grouped.items()):
        rows.append(
            LearningCurveRow(
                condition=condition,
                window_start=window_start,
                window_end=window_start + window_size - 1,
                success_rate=_safe_mean(1.0 if _bool(row["success"]) else 0.0 for row in window_rows),
                episode_count=len(window_rows),
            )
        )
    return rows


def bootstrap_ci(values: list[float], *, samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    boot_means = []
    for _ in range(samples):
        boot_sample = [values[rng.randrange(len(values))] for _ in values]
        boot_means.append(mean(boot_sample))
    boot_means.sort()
    low_index = int(0.025 * (len(boot_means) - 1))
    high_index = int(0.975 * (len(boot_means) - 1))
    return float(boot_means[low_index]), float(boot_means[high_index])


def write_report(
    path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    summary_rows: list[SummaryRow],
) -> None:
    condition_labels = ", ".join(condition_label(row.condition) for row in summary_rows)
    lines = [
        "# AASSR GridWorld Analysis",
        "",
        "## Experiment Setting",
        "",
        f"- Input: `{input_dir}`",
        f"- Output: `{output_dir}`",
        f"- Conditions: {condition_labels}",
        "- Confidence interval: seed-level bootstrap 95% CI",
        "- Steps to FLAG are averaged over successful episodes only.",
        "",
        "## Summary",
        "",
        "| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {success:.3f} [{success_low:.3f}, {success_high:.3f}] | "
            "{steps:.3f} [{steps_low:.3f}, {steps_high:.3f}] | {semantic:.3f} | "
            "{repeat:.3f} | {error:.3f} | {prophecy:.3f} | {reward:.3f} |".format(
                condition=condition_label(row.condition),
                success=row.success_rate_mean,
                success_low=row.success_rate_ci95_low,
                success_high=row.success_rate_ci95_high,
                steps=row.steps_to_flag_mean,
                steps_low=row.steps_to_flag_ci95_low,
                steps_high=row.steps_to_flag_ci95_high,
                semantic=row.semantic_gain_mean,
                repeat=row.repeat_rate_mean,
                error=row.error_rate_mean,
                prophecy=row.prophecy_error_mean,
                reward=row.total_reward_mean,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. "
            "The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. "
            "C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. "
            "The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.",
            "",
            "DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. "
            "ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.",
            "",
            "## Figures",
            "",
            "- `figure_success_rate.png`",
            "- `figure_steps_to_flag.png`",
            "- `figure_semantic_gain.png`",
            "- `figure_repeat_error_rate.png`",
            "- `figure_learning_curve.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(rows[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(mean(materialized))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: Any) -> float:
    if value in {"", None}:
        return 0.0
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AASSR GridWorld experiment CSV outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--learning-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_results(
        input_dir=args.input,
        output_dir=args.output,
        bootstrap_samples=args.bootstrap_samples,
        learning_window=args.learning_window,
    )
    print(f"wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
