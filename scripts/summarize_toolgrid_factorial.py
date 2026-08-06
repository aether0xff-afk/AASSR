from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAJECTORY_KEYS = [
    "seed",
    "grid_size",
    "action_count",
    "checkpoint_transition_target",
    "episode",
    "map_seed",
]
TRAJECTORY_FIELDS = [
    "success",
    "steps",
    "environment_steps_total",
    "termination",
]


def _bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    randomizer = np.random.default_rng(seed)
    samples = randomizer.choice(
        values,
        size=(draws, values.size),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _load_csvs(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if not paths:
        raise FileNotFoundError(f"no {name} files below {root}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def _verify_protocol(
    training: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> dict[str, object]:
    final_target = int(checkpoints["checkpoint_transition_target"].max())
    final = checkpoints[
        checkpoints["checkpoint_transition_target"] == final_target
    ].copy()

    budget_mismatch = final[
        final["actual_training_transitions"]
        != final["checkpoint_transition_target"]
    ]
    if not budget_mismatch.empty:
        cells = budget_mismatch[
            ["condition", "seed", "grid_size", "action_count"]
        ].to_dict(orient="records")
        raise RuntimeError(
            "ToolGrid cells did not respect the exact real-transition budget: "
            f"{cells}"
        )

    imagination_training = training[
        training["condition"] == "imagination_v2"
    ]
    training_imagination_runs = int(
        imagination_training["imagination_runs"].sum()
    )
    if training_imagination_runs:
        raise RuntimeError(
            "Imagination v2 changed the training trajectory: "
            f"{training_imagination_runs} training-time imagination runs found"
        )

    policy = training[
        training["condition"] == "neural_policy_only"
    ][TRAJECTORY_KEYS + TRAJECTORY_FIELDS].copy()
    imagined = training[
        training["condition"] == "imagination_v2"
    ][TRAJECTORY_KEYS + TRAJECTORY_FIELDS].copy()

    policy = policy.sort_values(TRAJECTORY_KEYS).reset_index(drop=True)
    imagined = imagined.sort_values(TRAJECTORY_KEYS).reset_index(drop=True)
    if len(policy) != len(imagined):
        raise RuntimeError(
            "Policy-only and Imagination training trajectories have different "
            f"row counts: {len(policy)} != {len(imagined)}"
        )
    if not policy[TRAJECTORY_KEYS].equals(imagined[TRAJECTORY_KEYS]):
        raise RuntimeError(
            "Policy-only and Imagination training trajectories use different "
            "maps or checkpoint segments"
        )

    mismatches = []
    for field in TRAJECTORY_FIELDS:
        unequal = policy[field].astype(str) != imagined[field].astype(str)
        if unequal.any():
            mismatches.append(
                {
                    "field": field,
                    "count": int(unequal.sum()),
                }
            )
    if mismatches:
        raise RuntimeError(
            "Policy-only and Imagination no longer share a matched training "
            f"trajectory: {mismatches}"
        )

    return {
        "exact_transition_budget": True,
        "training_imagination_runs": training_imagination_runs,
        "matched_policy_imagination_training_trajectory": True,
        "matched_training_rows": int(len(policy)),
    }


def _required_tool_set(values: pd.Series) -> set[int]:
    observed: set[int] = set()
    for value in values:
        parsed = json.loads(value) if isinstance(value, str) else value
        observed.update(int(item) for item in parsed)
    return observed


def _manipulation_check(manifests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unseen = manifests[manifests["split"] == "unseen"]
    for (grid_size, action_count), group in unseen.groupby(
        ["grid_size", "action_count"]
    ):
        expected = set(range(int(action_count) - 4))
        observed = _required_tool_set(group["required_tools"])
        rows.append(
            {
                "grid_size": int(grid_size),
                "action_count": int(action_count),
                "mean_oracle_steps": float(
                    group["oracle_shortest_steps"].mean()
                ),
                "mean_effective_branching": float(
                    group["effective_branching_factor"].mean()
                ),
                "semantic_tool_count": len(expected),
                "observed_required_tool_count": len(observed),
                "observed_required_tools": json.dumps(sorted(observed)),
                "tool_coverage_complete": observed == expected,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["grid_size", "action_count"]
    )
    incomplete = result[~result["tool_coverage_complete"]]
    if not incomplete.empty:
        raise RuntimeError(
            "Unseen ToolGrid pools do not cover every semantic tool: "
            f"{incomplete.to_dict(orient='records')}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and validate ToolGrid factorial artifacts."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    episodes = _load_csvs(root, "evaluation_episodes.csv")
    training = _load_csvs(root, "training_episodes.csv")
    checkpoints = _load_csvs(root, "checkpoints.csv")
    manifests = _load_csvs(root, "map_manifest.csv")

    protocol = _verify_protocol(training, checkpoints)
    manipulation = _manipulation_check(manifests)
    manipulation.to_csv(output / "manipulation_check.csv", index=False)

    final_target = int(episodes["checkpoint_transition_target"].max())
    unseen = episodes[
        (episodes["phase"] == "evaluation_unseen")
        & (episodes["checkpoint_transition_target"] == final_target)
    ].copy()

    seed_rates = (
        unseen.groupby(
            ["seed", "condition", "grid_size", "action_count"],
            as_index=False,
        )
        .agg(
            success_rate=("success", "mean"),
            mean_steps=("steps", "mean"),
            mean_imagined_nodes=("imagined_nodes", "mean"),
            imagination_use_rate=(
                "imagination_runs",
                lambda values: float((values > 0).mean()),
            ),
        )
    )
    seed_rates.to_csv(output / "seed_cell_rates.csv", index=False)

    summaries = []
    for keys, group in seed_rates.groupby(
        ["condition", "grid_size", "action_count"]
    ):
        condition, grid_size, action_count = keys
        values = group["success_rate"].to_numpy(dtype=float)
        low, high = _bootstrap_mean(
            values,
            seed=(
                int(grid_size) * 10_000
                + int(action_count) * 100
                + len(condition)
            ),
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
                "mean_imagined_nodes": float(
                    group["mean_imagined_nodes"].mean()
                ),
                "mean_imagination_use_rate": float(
                    group["imagination_use_rate"].mean()
                ),
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
            x_size = (
                group["grid_size"].to_numpy(dtype=float) - 5.0
            ) / 2.0
            x_branch = (
                group["action_count"].to_numpy(dtype=float) - 10.0
            ) / 2.0
            design = np.column_stack(
                [
                    np.ones(len(group)),
                    x_size,
                    x_branch,
                    x_size * x_branch,
                ]
            )
            target = group[
                "imagination_v2_minus_dqn"
            ].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(
                design,
                target,
                rcond=None,
            )
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
    coefficients.to_csv(
        output / "seed_factorial_coefficients.csv",
        index=False,
    )

    result: dict[str, object] = {
        "final_checkpoint": final_target,
        "seed_count": int(seed_rates["seed"].nunique()),
        "protocol_validation": protocol,
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
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# ToolGrid 팩토리얼 파일럿 결과",
        "",
        f"- 최종 체크포인트: {final_target:,} transitions",
        f"- 독립 seed: {seed_rates['seed'].nunique()}개",
        "- 요인: map size 3/5/7 × semantic tool count 4/8",
        "- 조건: DQN / Neural Policy-only / Imagination v2",
        "",
        "## 프로토콜 검증",
        "",
        f"- 정확한 transition budget: {protocol['exact_transition_budget']}",
        (
            "- 학습 중 Imagination 실행 수: "
            f"{protocol['training_imagination_runs']}"
        ),
        (
            "- Policy-only/Imagination 동일 학습 궤적: "
            f"{protocol['matched_policy_imagination_training_trajectory']}"
        ),
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
                (
                    "양의 `map_size_effect`는 맵 크기가 커질수록 "
                    "Imagination v2의 상대 우위가 증가함을 뜻한다."
                ),
                (
                    "양의 `action_branch_effect`는 의미적 도구 가지 수가 "
                    "늘수록 상대 우위가 증가함을 뜻한다."
                ),
                (
                    "`interaction_effect`는 두 요인이 동시에 커질 때의 "
                    "추가 효과다."
                ),
                "",
            ]
        )
    (output / "report_ko.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
