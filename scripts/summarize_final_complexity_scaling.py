from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_PAIR = ("imagination_v2", "dqn")
SECONDARY_PAIR = ("imagination_v2", "neural_policy_only")
DISPLAY_NAMES = {
    "dqn": "DQN",
    "legacy_aassr": "Legacy AASSR",
    "neural_policy_only": "Neural Policy-only",
    "imagination_v2": "Imagination v2",
}


def _read_recursive(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.rglob(filename))
    if not paths:
        raise FileNotFoundError(f"no {filename} files under {root}")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    draws: int = 20_000,
) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    indices = rng.integers(0, array.size, size=(draws, array.size))
    samples = array[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return mean, float(lower), float(upper)


def _wilcoxon_greater(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    nonzero = array[~np.isclose(array, 0.0)]
    if nonzero.size == 0:
        return 1.0
    return float(
        stats.wilcoxon(
            nonzero,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * p_values[int(original_index)]
        running = max(running, candidate)
        adjusted[int(original_index)] = min(1.0, running)
    return adjusted.tolist()


def _seed_level_rates(final_eval: pd.DataFrame) -> pd.DataFrame:
    return (
        final_eval.groupby(["condition", "seed", "phase", "level"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_steps=("steps", "mean"),
            mean_path_efficiency=("path_efficiency", "mean"),
            episodes=("success", "size"),
        )
        .sort_values(["phase", "condition", "seed", "level"])
    )


def _effect_by_level(
    seed_rates: pd.DataFrame,
    *,
    phase: str,
    numerator: str,
    denominator: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = seed_rates[
        (seed_rates["phase"] == phase)
        & seed_rates["condition"].isin([numerator, denominator])
    ]
    pivot = selected.pivot_table(
        index=["seed", "level"],
        columns="condition",
        values="success_rate",
    ).reset_index()
    missing = {numerator, denominator} - set(pivot.columns)
    if missing:
        raise ValueError(f"missing conditions for paired effect: {sorted(missing)}")
    pivot["delta"] = pivot[numerator] - pivot[denominator]

    summary_rows: list[dict[str, Any]] = []
    for level, group in pivot.groupby("level"):
        mean, lower, upper = _bootstrap_mean_ci(group["delta"], rng=rng)
        summary_rows.append(
            {
                "phase": phase,
                "numerator": numerator,
                "denominator": denominator,
                "level": int(level),
                "mean_success_rate_difference": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "seed_count": int(group["seed"].nunique()),
            }
        )

    slope_rows: list[dict[str, Any]] = []
    for seed, group in pivot.groupby("seed"):
        ordered = group.sort_values("level")
        if ordered["level"].nunique() < 2:
            continue
        slope = float(
            np.polyfit(
                ordered["level"].to_numpy(dtype=float),
                ordered["delta"].to_numpy(dtype=float),
                1,
            )[0]
        )
        slope_rows.append({"seed": int(seed), "slope": slope})
    slopes = pd.DataFrame(slope_rows)
    mean_slope, slope_lower, slope_upper = _bootstrap_mean_ci(
        slopes["slope"],
        rng=rng,
    )
    p_value = _wilcoxon_greater(slopes["slope"])
    level_summary = pd.DataFrame(summary_rows).sort_values("level")
    crossover_levels = level_summary[
        level_summary["mean_success_rate_difference"] >= 0.0
    ]["level"].tolist()
    strict_positive_levels = level_summary[
        level_summary["ci95_lower"] > 0.0
    ]["level"].tolist()
    hypothesis = {
        "phase": phase,
        "numerator": numerator,
        "denominator": denominator,
        "seed_count": int(slopes["seed"].nunique()),
        "mean_seed_slope_per_level": mean_slope,
        "slope_ci95_lower": slope_lower,
        "slope_ci95_upper": slope_upper,
        "wilcoxon_one_sided_p": p_value,
        "first_nonnegative_mean_delta_level": (
            int(crossover_levels[0]) if crossover_levels else None
        ),
        "levels_with_ci_strictly_above_zero": [
            int(item) for item in strict_positive_levels
        ],
        "supported": bool(slope_lower > 0.0 and p_value < 0.05),
    }
    return level_summary, slopes, hypothesis


def _condition_success_summary(
    seed_rates: pd.DataFrame,
    *,
    phase: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = seed_rates[seed_rates["phase"] == phase]
    for (condition, level), group in selected.groupby(["condition", "level"]):
        mean, lower, upper = _bootstrap_mean_ci(
            group["success_rate"],
            rng=rng,
        )
        rows.append(
            {
                "phase": phase,
                "condition": condition,
                "level": int(level),
                "mean_success_rate": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "seed_count": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "level"])


def _mcnemar_by_level(
    final_eval: pd.DataFrame,
    *,
    phase: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    selected = final_eval[
        (final_eval["phase"] == phase)
        & final_eval["condition"].isin([left, right])
    ][["condition", "seed", "level", "map_seed", "success"]]
    pivot = selected.pivot_table(
        index=["seed", "level", "map_seed"],
        columns="condition",
        values="success",
        aggfunc="first",
    ).reset_index()
    if left not in pivot or right not in pivot:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for level, group in pivot.groupby("level"):
        left_only = int(((group[left] == 1) & (group[right] == 0)).sum())
        right_only = int(((group[left] == 0) & (group[right] == 1)).sum())
        discordant = left_only + right_only
        p_value = (
            float(
                stats.binomtest(
                    min(left_only, right_only),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        rows.append(
            {
                "phase": phase,
                "left": left,
                "right": right,
                "level": int(level),
                "left_only_success": left_only,
                "right_only_success": right_only,
                "discordant": discordant,
                "exact_mcnemar_p": p_value,
            }
        )
    frame = pd.DataFrame(rows).sort_values("level")
    if not frame.empty:
        frame["holm_adjusted_p"] = _holm_adjust(
            frame["exact_mcnemar_p"].tolist()
        )
    return frame


def _deduplicated_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "seed",
        "split",
        "level",
        "map_seed",
        "oracle_shortest_steps",
        "reachable_nonterminal_states",
        "max_graph_depth",
        "irreversible_failure_ratio",
        "mean_nonfailure_actions",
        "success_edge_ratio",
    ]
    available = [column for column in columns if column in manifest.columns]
    return manifest[available].drop_duplicates(
        subset=[
            column
            for column in ["seed", "split", "level", "map_seed"]
            if column in available
        ]
    )


def _complexity_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    deduplicated = _deduplicated_manifest(manifest)
    numeric = [
        "oracle_shortest_steps",
        "reachable_nonterminal_states",
        "max_graph_depth",
        "irreversible_failure_ratio",
        "mean_nonfailure_actions",
        "success_edge_ratio",
    ]
    frame = (
        deduplicated.groupby(["split", "level"], as_index=False)[numeric]
        .mean()
        .sort_values(["split", "level"])
    )
    return frame


def _plot_success(summary: pd.DataFrame, output: Path, phase: str) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for condition, group in summary.groupby("condition"):
        ordered = group.sort_values("level")
        axis.errorbar(
            ordered["level"],
            ordered["mean_success_rate"],
            yerr=np.vstack(
                [
                    ordered["mean_success_rate"] - ordered["ci95_lower"],
                    ordered["ci95_upper"] - ordered["mean_success_rate"],
                ]
            ),
            marker="o",
            capsize=3,
            label=DISPLAY_NAMES.get(condition, condition),
        )
    axis.set_xlabel("Complexity level")
    axis.set_ylabel("Success rate")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"Success by frozen map complexity ({phase})")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_effects(effect_frames: list[pd.DataFrame], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for frame in effect_frames:
        if frame.empty:
            continue
        ordered = frame.sort_values("level")
        label = (
            f"{DISPLAY_NAMES.get(ordered.iloc[0]['numerator'], ordered.iloc[0]['numerator'])}"
            f" − {DISPLAY_NAMES.get(ordered.iloc[0]['denominator'], ordered.iloc[0]['denominator'])}"
        )
        axis.errorbar(
            ordered["level"],
            ordered["mean_success_rate_difference"],
            yerr=np.vstack(
                [
                    ordered["mean_success_rate_difference"] - ordered["ci95_lower"],
                    ordered["ci95_upper"] - ordered["mean_success_rate_difference"],
                ]
            ),
            marker="o",
            capsize=3,
            label=label,
        )
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Complexity level")
    axis.set_ylabel("Paired success-rate difference")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_title("Relative advantage by complexity (unseen maps)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_slopes(slopes: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    ordered = slopes.sort_values("slope").reset_index(drop=True)
    axis.bar(np.arange(len(ordered)), ordered["slope"])
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Independent seed (sorted)")
    axis.set_ylabel("Slope of Imagination v2 − DQN")
    axis.set_title("Seed-level complexity interaction slopes")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report(
    output: Path,
    *,
    primary: dict[str, Any],
    secondary: dict[str, Any],
    success_summary: pd.DataFrame,
    primary_effect: pd.DataFrame,
    complexity_summary: pd.DataFrame,
) -> None:
    decision = "지지됨" if primary["supported"] else "지지되지 않음"
    lines = [
        "# AASSR 최종 복잡도 스케일링 결과",
        "",
        "## 주가설 판정",
        "",
        f"- 판정: **{decision}**",
        (
            "- 검정 대상: unseen 맵에서 복잡도 Level이 증가할 때 "
            "`Imagination v2 − DQN` 성공률 차이의 seed별 기울기가 양수인지"
        ),
        f"- 독립 seed 수: {primary['seed_count']}",
        f"- 평균 기울기/level: {primary['mean_seed_slope_per_level']:.6f}",
        (
            f"- seed bootstrap 95% CI: "
            f"[{primary['slope_ci95_lower']:.6f}, {primary['slope_ci95_upper']:.6f}]"
        ),
        f"- one-sided Wilcoxon p: {primary['wilcoxon_one_sided_p']:.6g}",
        (
            "- 평균 차이가 처음 0 이상이 된 Level: "
            f"{primary['first_nonnegative_mean_delta_level']}"
        ),
        "",
        "판정 규칙은 평균 기울기의 seed-bootstrap 95% CI 하한이 0보다 크고, "
        "one-sided Wilcoxon p<0.05인 경우로 사전 고정했다.",
        "",
        "## Level별 Imagination v2 − DQN",
        "",
        "| Level | 평균 차이 | 95% CI | seeds |",
        "|---:|---:|---:|---:|",
    ]
    for row in primary_effect.itertuples(index=False):
        lines.append(
            f"| {row.level} | {_format_percent(row.mean_success_rate_difference)} | "
            f"[{_format_percent(row.ci95_lower)}, {_format_percent(row.ci95_upper)}] | "
            f"{row.seed_count} |"
        )

    lines.extend(
        [
            "",
            "## 보조가설: Imagination v2 − Neural Policy-only",
            "",
            f"- 지지 여부: **{'지지됨' if secondary['supported'] else '지지되지 않음'}**",
            f"- 평균 기울기/level: {secondary['mean_seed_slope_per_level']:.6f}",
            (
                f"- 95% CI: [{secondary['slope_ci95_lower']:.6f}, "
                f"{secondary['slope_ci95_upper']:.6f}]"
            ),
            f"- one-sided Wilcoxon p: {secondary['wilcoxon_one_sided_p']:.6g}",
            "",
            "## 조건별 unseen 성공률",
            "",
            "| 조건 | Level | 성공률 | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in success_summary.itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES.get(row.condition, row.condition)} | {row.level} | "
            f"{_format_percent(row.mean_success_rate)} | "
            f"[{_format_percent(row.ci95_lower)}, {_format_percent(row.ci95_upper)}] |"
        )

    lines.extend(
        [
            "",
            "## 복잡도 Level 검증",
            "",
            "Level은 에이전트 결과를 보지 않고 frozen strict GridPush의 상태 그래프만으로 정했다. "
            "정렬 우선순위는 Oracle 최단 성공 길이, 즉시 비가역 실패 행동 비율, "
            "도달 가능한 비종료 상태 수, 최대 그래프 깊이다.",
            "",
            "| split | Level | L* | reachable states | failure ratio | max depth |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in complexity_summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.level} | {row.oracle_shortest_steps:.3f} | "
            f"{row.reachable_nonterminal_states:.3f} | "
            f"{row.irreversible_failure_ratio:.4f} | {row.max_graph_depth:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "이 실험은 환경 규칙을 바꾸지 않고 동일 생성기 안의 구조적 난이도 분위만 비교한다. "
            "따라서 KPDE의 모든 구성요소를 독립 조작한 실험은 아니며, 의존 깊이·부분관측·"
            "지식-행동 결합의 개별 인과효과를 따로 주장하지 않는다.",
            "",
            "포기 기능, 고정 tick 제한, 중간 외부 보상은 사용하지 않았다.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    evaluation = _read_recursive(args.input, "evaluation_episodes.csv")
    manifests = _read_recursive(args.input, "map_manifest.csv")
    summaries = sorted(args.input.rglob("summary.json"))
    if not summaries:
        raise FileNotFoundError("no summary.json files found")

    # Derive seed from the evaluation files and attach it to manifests whose
    # rows are otherwise identical across conditions.
    if "seed" not in manifests.columns:
        manifest_frames: list[pd.DataFrame] = []
        for path in sorted(args.input.rglob("map_manifest.csv")):
            frame = pd.read_csv(path)
            parent_summary = path.parent / "summary.json"
            payload = json.loads(parent_summary.read_text(encoding="utf-8"))
            frame["seed"] = int(payload["config"]["seed"])
            frame["condition"] = payload["config"]["condition"]
            manifest_frames.append(frame)
        manifests = pd.concat(manifest_frames, ignore_index=True)

    final_checkpoint = (
        evaluation.groupby(["condition", "seed"])["checkpoint_transition_target"]
        .max()
        .rename("final_checkpoint")
        .reset_index()
    )
    final_eval = evaluation.merge(final_checkpoint, on=["condition", "seed"])
    final_eval = final_eval[
        final_eval["checkpoint_transition_target"] == final_eval["final_checkpoint"]
    ].copy()

    expected_conditions = set(DISPLAY_NAMES)
    observed_conditions = set(final_eval["condition"].unique())
    missing = expected_conditions - observed_conditions
    if missing:
        raise ValueError(f"final aggregation is incomplete; missing {sorted(missing)}")

    rng = np.random.default_rng(20260805)
    seed_rates = _seed_level_rates(final_eval)
    unseen_success = _condition_success_summary(
        seed_rates,
        phase="evaluation_unseen",
        rng=rng,
    )
    seen_success = _condition_success_summary(
        seed_rates,
        phase="evaluation_seen",
        rng=rng,
    )
    primary_effect, primary_slopes, primary_hypothesis = _effect_by_level(
        seed_rates,
        phase="evaluation_unseen",
        numerator=PRIMARY_PAIR[0],
        denominator=PRIMARY_PAIR[1],
        rng=rng,
    )
    secondary_effect, secondary_slopes, secondary_hypothesis = _effect_by_level(
        seed_rates,
        phase="evaluation_unseen",
        numerator=SECONDARY_PAIR[0],
        denominator=SECONDARY_PAIR[1],
        rng=rng,
    )
    mcnemar = _mcnemar_by_level(
        final_eval,
        phase="evaluation_unseen",
        left=PRIMARY_PAIR[0],
        right=PRIMARY_PAIR[1],
    )
    complexity = _complexity_summary(manifests)

    seed_rates.to_csv(args.output / "seed_level_rates.csv", index=False)
    unseen_success.to_csv(args.output / "unseen_success_by_level.csv", index=False)
    seen_success.to_csv(args.output / "seen_success_by_level.csv", index=False)
    primary_effect.to_csv(args.output / "primary_relative_effect.csv", index=False)
    secondary_effect.to_csv(args.output / "secondary_relative_effect.csv", index=False)
    primary_slopes.to_csv(args.output / "primary_seed_slopes.csv", index=False)
    secondary_slopes.to_csv(args.output / "secondary_seed_slopes.csv", index=False)
    mcnemar.to_csv(args.output / "mcnemar_by_level.csv", index=False)
    complexity.to_csv(args.output / "complexity_by_level.csv", index=False)

    hypothesis_payload = {
        "primary": primary_hypothesis,
        "secondary": secondary_hypothesis,
        "analysis_unit": "independent seed",
        "primary_split": "evaluation_unseen",
        "alpha": 0.05,
        "bootstrap_draws": 20_000,
        "multiple_comparison_correction": "Holm for level-wise McNemar tests",
        "episode_pseudoreplication_used_for_primary_test": False,
    }
    (args.output / "hypothesis_test.json").write_text(
        json.dumps(hypothesis_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _plot_success(
        unseen_success,
        args.output / "unseen_success_by_level.png",
        "unseen",
    )
    _plot_success(
        seen_success,
        args.output / "seen_success_by_level.png",
        "seen",
    )
    _plot_effects(
        [primary_effect, secondary_effect],
        args.output / "relative_advantage_by_level.png",
    )
    _plot_slopes(
        primary_slopes,
        args.output / "primary_seed_slopes.png",
    )
    _write_report(
        args.output / "final_report_ko.md",
        primary=primary_hypothesis,
        secondary=secondary_hypothesis,
        success_summary=unseen_success,
        primary_effect=primary_effect,
        complexity_summary=complexity[complexity["split"] == "unseen"],
    )
    print(json.dumps(hypothesis_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
