from __future__ import annotations

import csv
import json

from aassr_v2.paper_artifacts import validate_paper_artifacts
from aassr_v2.paper_protocol import (
    ExperimentPhase,
    planned_paper_run_count,
)
from aassr_v2.paper_runner import run_paper_suite


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "paper_smoke_all",
        "runner": "paper_suite",
        "protocol_version": "paper-smoke-all-v1",
        "study_stage": "pilot",
        "research_seeds": [1, 2, 3, 4, 5],
        "world_seeds": {
            "train": [101],
            "seen": [151],
            "unseen": [201],
        },
        "budgets": {
            "train_episodes": 1,
            "eval_episodes": 1,
            "real_transitions_per_episode": 9,
            "adaptation_episodes": [0, 1, 4, 16, 64],
        },
        "phases": [phase.value for phase in ExperimentPhase],
        "phase_learning": {
            phase.value: phase.permits_learning for phase in ExperimentPhase
        },
        "execution": {
            "workers": 1,
            "cuda_workers": 1,
            "device": "cpu",
            "allow_cpu_fallback": True,
        },
        "safe_application": {
            "opt_in": True,
            "internal_network": True,
            "allowed_hosts": ["paper-target"],
            "compose_file": "docker/paper_safe_application/docker-compose.yml",
        },
        "suites": [
            {
                "kind": "autonomy",
                "lengths": [2],
                "conditions": [{"name": "full_aassr"}],
            },
            {
                "kind": "ablation",
                "lengths": [2],
                "conditions": [{"name": "full_aassr"}],
            },
            {
                "kind": "transfer",
                "length": 2,
                "conditions": [{"name": "full_transfer"}],
            },
            {
                "kind": "creativity",
                "episodes": 1,
                "conditions": [
                    {"name": "random"},
                    {"name": "full_aassr"},
                ],
            },
            {"kind": "safe_application", "episodes": 1},
        ],
    }


def test_full_p0_through_p5_smoke_and_resume(tmp_path) -> None:
    config = _config()
    assert planned_paper_run_count(config) == 530
    artifacts = run_paper_suite(
        config, output_dir=tmp_path / "paper", overwrite=True
    )
    assert artifacts.row_count == 530
    assert validate_paper_artifacts(artifacts.output_dir) == []
    rows = list(
        csv.DictReader(artifacts.episodes_csv.open(encoding="utf-8-sig"))
    )
    assert {row["suite"] for row in rows} == {
        "autonomy",
        "ablation",
        "transfer",
        "creativity",
        "safe_application",
    }
    frozen = [
        row
        for row in rows
        if row["phase"].startswith("evaluation")
        and row["checkpoint_fingerprint_before"]
    ]
    assert frozen
    assert all(
        row["checkpoint_fingerprint_before"]
        == row["checkpoint_fingerprint_after"]
        for row in frozen
    )
    creative_transitions = [
        json.loads(line)
        for line in artifacts.transitions_jsonl.read_text(
            encoding="utf-8"
        ).splitlines()
        if '"suite": "creativity"' in line
    ]
    agent_visible = json.dumps(
        [
            {
                "before": row["before"],
                "action": row["action"],
                "after": row["after"],
            }
            for row in creative_transitions
        ]
    )
    for secret in (
        "information_route",
        "resource_route",
        "bypass_route",
        "tool_route",
        "emergent_combination",
    ):
        assert secret not in agent_visible
    transfer_rows = [
        row for row in rows if row["suite"] == "transfer"
    ]
    starts: dict[tuple[str, str, str], set[str]] = {}
    for row in transfer_rows:
        if not row["branch_start_fingerprint"]:
            continue
        key = (
            row["condition"],
            row["research_seed"],
            row["world_seed"],
        )
        starts.setdefault(key, set()).add(
            row["branch_start_fingerprint"]
        )
    assert starts
    assert all(len(values) == 1 for values in starts.values())
    adaptation = list(
        csv.DictReader(
            (
                artifacts.output_dir
                / "statistics"
                / "adaptation_summary.csv"
            ).open(encoding="utf-8-sig")
        )
    )
    assert adaptation
    assert all(
        "unseen_prediction_calibration_error" in row
        for row in adaptation
    )
    resumed = run_paper_suite(
        config, output_dir=tmp_path / "paper", resume=True
    )
    assert resumed.row_count == artifacts.row_count
