from __future__ import annotations

from aassr_v2.experiment_runner import read_rows
from aassr_v2.final_pilot import (
    InformationChoiceWorld,
    load_final_config,
    planned_final_run_count,
    run_final_pilot,
)
from aassr_v2.knowledge import KnowledgeStore
from aassr_v2.tabular_prophecy import TabularProphecy
from aassr_v2.types import Action
from aassr_v2.validated_learning import (
    FixedHoldoutTransitionEvaluator,
    FixedReplayBuffer,
)


def test_final_pilot_config_has_expected_size() -> None:
    config = load_final_config("configs/pilot.json")
    assert planned_final_run_count(config) == 1980


def test_current_holdout_sample_is_not_used_to_grade_itself() -> None:
    evaluator = FixedHoldoutTransitionEvaluator(
        TabularProphecy(),
        replay=FixedReplayBuffer(holdout_stride=2),
        minimum_holdout_count=1,
        samples=1,
    )
    knowledge = KnowledgeStore()

    first_world = InformationChoiceWorld(noise_facts=4, seed=1)
    evaluator.execute(first_world, Action("useful_probe"), knowledge)

    second_world = InformationChoiceWorld(noise_facts=4, seed=2)
    second = evaluator.execute(second_world, Action("useful_probe"), knowledge)

    assert second.effect.holdout_before == 0.0
    assert second.effect.holdout_after == 0.0
    assert second.effect.holdout_gain == 0.0
    assert len(evaluator.replay.holdout()) == 1


def test_skill_rows_preserve_real_seed_values(tmp_path) -> None:
    config = {
        "name": "skill_seed_check",
        "runner": "final_pilot",
        "seeds": [7, 13],
        "suites": [
            {
                "kind": "skills",
                "length": 2,
                "episodes": 3,
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
    artifacts = run_final_pilot(
        config,
        output_dir=tmp_path / "skills",
        overwrite=True,
    )
    rows = read_rows(artifacts.episodes_csv)
    assert {row["seed"] for row in rows} == {"7", "13"}
    assert len(rows) == 6


def test_trap_dependency_requires_full_chain_imagination(tmp_path) -> None:
    config = {
        "name": "trap_check",
        "runner": "final_pilot",
        "seeds": [7],
        "suites": [
            {
                "kind": "dependency",
                "lengths": [4],
                "train_episodes": 2,
                "eval_episodes": 1,
                "conditions": [
                    {
                        "name": "shallow",
                        "mode": "imagination",
                        "model": "tabular",
                        "maximum_depth": 2,
                    },
                    {
                        "name": "full",
                        "mode": "imagination",
                        "model": "tabular",
                        "depth_multiplier": 2,
                    },
                ],
            }
        ],
    }
    artifacts = run_final_pilot(
        config,
        output_dir=tmp_path / "trap",
        overwrite=True,
    )
    rows = read_rows(artifacts.episodes_csv)
    shallow = next(row for row in rows if row["condition"] == "shallow")
    full = next(row for row in rows if row["condition"] == "full")
    assert shallow["action_family"] == "advance_greedy"
    assert shallow["success"] == "0"
    assert full["action_family"] == "advance_safe"
    assert full["success"] == "1"


def test_validated_information_beats_raw_novelty_selection(tmp_path) -> None:
    config = {
        "name": "information_check",
        "runner": "final_pilot",
        "seeds": [7],
        "suites": [
            {
                "kind": "information_value",
                "noise_facts": 8,
                "train_episodes": 60,
                "eval_episodes": 5,
                "epsilon": 0.2,
                "novelty_weight": 0.2,
                "conditions": [
                    {
                        "name": "raw",
                        "mode": "novelty_count",
                        "lr": 0.2,
                    },
                    {
                        "name": "validated",
                        "mode": "validated_information",
                        "lr": 0.2,
                        "minimum_holdout_count": 2,
                        "samples": 2,
                    },
                ],
            }
        ],
    }
    artifacts = run_final_pilot(
        config,
        output_dir=tmp_path / "information",
        overwrite=True,
    )
    rows = read_rows(artifacts.episodes_csv)
    raw = [row for row in rows if row["condition"] == "raw"]
    validated = [row for row in rows if row["condition"] == "validated"]
    assert {row["action_family"] for row in raw} == {"noise_probe"}
    assert {row["success"] for row in raw} == {"0"}
    assert {row["action_family"] for row in validated} == {"useful_probe"}
    assert {row["success"] for row in validated} == {"1"}
