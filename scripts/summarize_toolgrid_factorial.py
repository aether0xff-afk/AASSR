from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int = 20_000) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    randomizer = np.random.default_rng(seed)
    samples = randomizer.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _load_csvs(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if not paths:
        raise FileNotFoundError(f"no {name} files below {root}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ToolGrid factorial artifacts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    episodes = _load_csvs(root, "evaluation_episodes.csv")
    manifests = _load_csvs(root, "map_manifest.csv")
    final_target = episodes["checkpoint_transition_target"].max()
    unseen = episodes[
        (episodes["phase"] == "evaluation_unseen")
        & (episodes["checkpoint_transition_target"] == final_target)
    ].copy()

    seed_rates = (
        unseen.groupby(["seed", "condition", "grid_size", "action_count"], as_index=False)
        .agg(success_rate=("success", "mean"), mean_steps=("steps", "mean"))
    )
    seed_rates.to_csv(output / "seed_cell_rates.csv", index=False)

    summaries = []
    for keys, group in seed_rates.groupby(["condition", "grid_size", "action_count"]):
        condition, grid_size, action_count = keys
        values = group["success_rate"].to_numpy(dtype=float)
        low, high = _bootstrap_mean(
            values,
            seed=int(grid_size) * 10_000 + int(action_count) * 100 + len(condition),
        )
        summaries.append(
            {
                "condition": condition,
                "grid_size": int(grid_size),
                "action_count": int(action_count),
                "seed_count": int(values.size),
                "mean_success_rate": float(values.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["condition", "grid_size", "action_count"]
    )
    summary.to_csv(output / "condition_cell_summary.csv", index=False)

    pivot = seed_rates.pivot_table(
        index=["seed", "grid_size", "action_count"],
        columns="condition",
        values="success_rate",
    ).reset_index()
    for baseline in ("dqn", "neural_policy_only"):
        if baseline in pivot and "imagination_v2" in pivot:
            pivot[f"imagination_v2_minus_{baseline}"] = (
                pivot["imagination_v2"] - pivot[baseline]
            )
    pivot.to_csv(output / "paired_cell_effects.csv", index=False)

    coefficient_rows = []
    if "imagination_v2_minus_dqn" in pivot:
        for seed, group in pivot.groupby("seed"):
            x_size = (group["grid_size"].to_numpy(dtype=float) - 5.0) / 2.0
            x_branch = (group["action_count"].to_numpy(dtype=float) - 10.0) / 2.0
            design = np.column_stack(
                [
                    np.ones(len(group)),
                    x_size,
                    x_branch,
                    x_size * x_branch,
                ]
            )
            target = group["imagination_v2_minus_dqn"].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            coefficient_rows.append(
                {
                    "seed": int(seed),
                    "intercept": float(coefficients[0]),
                    "map_size_effect": float(coefficients[1]),
                    "action_branch_effect": float(coefficients[2]),
                    "interaction_effect": float(coefficients[3]),
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(output / "seed_factorial_coefficients.csv", index=False)

    manipulation = (
        manifests[manifests["split"] == "unseen"]
        .groupby(["grid_size", "action_count"], as_index=False)
        .agg(
            mean_oracle_steps=("oracle_shortest_steps", "mean"),
            mean_effective_branching=("effective_branching_factor", "mean"),
            unique_tools=("tool_count", "mean"),
        )
    )
    manipulation.to_csv(output / "manipulation_check.csv", index=False)

    result: dict[str, object] = {
        "final_checkpoint": int(final_target),
        "seed_count": int(seed_rates["seed"].nunique()),
        "manipulation": manipulation.to_dict(orient="records"),
    }
    if not coefficients.empty:
        result["mean_coefficients"] = {
            column: float(coefficients[column].mean())
            for column in (
                "intercept",
                "map_size_effect",
                "action_branch_effect",
                "interaction_effect",
            )
        }
        result["coefficient_sign_counts"] = {
            column: int((coefficients[column] > 0).sum())
            for column in (
                "map_size_effect",
                "action_branch_effect",
                "interaction_effect",
            )
        }
    (output / "factorial_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# ToolGrid 팩토리얼 파일럿 결과",
        "",
        f"- 최종 체크포인트: {int(final_target):,} transitions",
        f"- 독립 seed: {seed_rates['seed'].nunique()}개",
        "- 요인: map size 3/5/7 × action count 8/12",
        "- 조건: DQN / Neural Policy-only / Imagination v2",
        "",
        "## 조작 점검",
        "",
        manipulation.to_markdown(index=False),
        "",
        "## 조건별 unseen 성공률",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    if not coefficients.empty:
        lines.extend(
            [
                "## Imagination v2 − DQN 요인 계수",
                "",
                coefficients.to_markdown(index=False),
                "",
                "양의 `map_size_effect`는 맵 크기가 커질수록 Imagination v2의 상대 우위가 증가함을,",
                "양의 `action_branch_effect`는 행동 가지 수가 늘수록 상대 우위가 증가함을 뜻한다.",
                "`interaction_effect`는 두 요인이 동시에 커질 때의 추가 효과다.",
                "",
            ]
        )
    (output / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
