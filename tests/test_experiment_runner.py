from __future__ import annotations

import csv
import json

from aassr_v2.experiment_runner import (
    load_config,
    planned_run_count,
    read_rows,
    run_experiment,
)
from aassr_v2.experiment_statistics import (
    cross_seed_summary,
    regenerate_seed_level_summary,
    seed_level_rows,
)


def test_pilot_config_has_expected_size() -> None:
    config = load_config("configs/pilot.json")
    assert planned_run_count(config) == 252


def test_tiny_experiment_writes_all_artifacts(tmp_path) -> None:
    config = {
        "name": "tiny",
        "seeds": [1],
        "suites": [
            {
                "kind": "prophecy",
                "models": ["tabular"],
                "train_episodes": 2,
                "eval_episodes": 1,
            },
            {
                "kind": "imagination",
                "train_episodes": 2,
                "eval_episodes": 1,
                "conditions": [
                    {
                        "name": "depth1",
                        "mode": "imagination",
                        "model": "tabular",
                        "maximum_depth": 1,
                    },
                    {
                        "name": "depth2",
                        "mode": "imagination",
                        "model": "tabular",
                        "maximum_depth": 2,
                    },
                ],
            },
            {
                "kind": "dependency",
                "lengths": [2],
                "train_episodes": 1,
                "eval_episodes": 1,
                "conditions": [
                    {
                        "name": "policy",
                        "mode": "policy",
                        "model": "tabular",
                    }
                ],
            },
            {
                "kind": "skills",
                "length": 2,
                "episodes": 3,
                "conditions": [
                    {
                        "name": "skill",
                        "use_skills": True,
                        "promotion_successes": 2,
                    }
                ],
            },
            {
                "kind": "information_value",
                "noise_levels": [0],
                "steps": 2,
            },
        ],
    }
    artifacts = run_experiment(
        config,
        output_dir=tmp_path / "run",
        overwrite=True,
    )
    expected = planned_run_count(config)
    assert artifacts.row_count == expected
    assert artifacts.episodes_csv.exists()
    assert artifacts.summary_csv.exists()
    assert artifacts.report_md.exists()
    assert artifacts.resolved_config_json.exists()
    seed_csv, summary_csv, report = regenerate_seed_level_summary(
        artifacts.output_dir
    )
    assert seed_csv.exists()
    assert summary_csv.exists()
    assert report.exists()
    assert len(read_rows(artifacts.episodes_csv)) == expected


def test_depth_two_imagination_escapes_immediate_reward_trap(tmp_path) -> None:
    config = {
        "name": "choice",
        "seeds": [3],
        "suites": [
            {
                "kind": "imagination",
                "train_episodes": 3,
                "eval_episodes": 2,
                "conditions": [
                    {
                        "name": "depth1",
                        "mode": "imagination",
                        "model": "tabular",
                        "maximum_depth": 1,
                        "minimum_path_confidence": 0.0,
                        "uncertainty_penalty": 0.0,
                    },
                    {
                        "name": "depth2",
                        "mode": "imagination",
                        "model": "tabular",
                        "maximum_depth": 2,
                        "minimum_path_confidence": 0.0,
                        "uncertainty_penalty": 0.0,
                    },
                ],
            }
        ],
    }
    artifacts = run_experiment(
        config,
        output_dir=tmp_path / "choice",
        overwrite=True,
    )
    rows = read_rows(artifacts.episodes_csv)
    depth1 = [row for row in rows if row["condition"] == "depth1"]
    depth2 = [row for row in rows if row["condition"] == "depth2"]
    assert {row["action_family"] for row in depth1} == {"shortcut"}
    assert {row["success"] for row in depth1} == {"0"}
    assert {row["action_family"] for row in depth2} == {"setup"}
    assert {row["success"] for row in depth2} == {"1"}


def test_skill_reduces_high_level_planning_after_promotion(tmp_path) -> None:
    config = {
        "name": "skill",
        "seeds": [0],
        "suites": [
            {
                "kind": "skills",
                "length": 3,
                "episodes": 4,
                "conditions": [
                    {
                        "name": "enabled",
                        "use_skills": True,
                        "promotion_successes": 2,
                    }
                ],
            }
        ],
    }
    artifacts = run_experiment(
        config,
        output_dir=tmp_path / "skill",
        overwrite=True,
    )
    rows = sorted(
        read_rows(artifacts.episodes_csv),
        key=lambda row: int(row["episode"]),
    )
    assert [int(row["high_level_steps"]) for row in rows[:2]] == [6, 6]
    assert [int(row["high_level_steps"]) for row in rows[2:]] == [1, 1]
    assert [int(row["skill_uses"]) for row in rows[2:]] == [1, 1]


def test_statistics_average_inside_seed_first() -> None:
    rows = [
        {
            "suite": "x",
            "condition": "full",
            "environment": "e",
            "model": "m",
            "action_family": "",
            "seed": 1,
            "success": 1,
            "steps": 2,
        },
        {
            "suite": "x",
            "condition": "full",
            "environment": "e",
            "model": "m",
            "action_family": "",
            "seed": 1,
            "success": 1,
            "steps": 4,
        },
        {
            "suite": "x",
            "condition": "full",
            "environment": "e",
            "model": "m",
            "action_family": "",
            "seed": 2,
            "success": 0,
            "steps": 10,
        },
    ]
    per_seed = seed_level_rows(rows)
    assert len(per_seed) == 2
    summary = cross_seed_summary(per_seed)
    assert len(summary) == 1
    assert summary[0]["success_mean"] == 0.5
    assert summary[0]["steps_mean"] == 6.5
