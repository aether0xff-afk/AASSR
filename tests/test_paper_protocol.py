from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.autonomous_benchmarks import OpaqueDependencyWorld
from aassr_v2.autonomous_experiment import run_autonomous_experiment
from aassr_v2.paper_protocol import (
    capture_checkpoint_parts,
    checkpoint_fingerprint,
    expand_suite_conditions,
    load_paper_config,
    planned_paper_run_count,
    restore_checkpoint_parts,
    validate_paper_config,
)
from aassr_v2.paper_statistics import (
    bootstrap_confidence_interval,
    holm_correction,
    paired_permutation_test,
    trapezoid_auc,
)
from aassr_v2.paper_runner import (
    _apply_effect_transfer,
    _profile_representation,
    run_paper_suite,
)
from aassr_v2.paper_types import BudgetLedger, ExperimentPhase
from aassr_v2.tabular_prophecy import TabularProphecy


def tiny_paper_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "tiny_paper",
        "runner": "paper_suite",
        "protocol_version": "tiny-paper-v1",
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
            "real_transitions_per_episode": 4,
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
        "suites": [
            {
                "kind": "autonomy",
                "lengths": [2],
                "conditions": [{"name": "full_aassr"}],
            }
        ],
    }


def test_paper_config_rejects_seed_and_phase_leakage() -> None:
    config = tiny_paper_config()
    validate_paper_config(config)
    duplicate = copy.deepcopy(config)
    duplicate["world_seeds"]["unseen"] = [101]  # type: ignore[index]
    with pytest.raises(ValueError, match="disjoint"):
        validate_paper_config(duplicate)
    seen_leak = copy.deepcopy(config)
    seen_leak["world_seeds"]["seen"] = [101]  # type: ignore[index]
    with pytest.raises(ValueError, match="disjoint"):
        validate_paper_config(seen_leak)
    mutation = copy.deepcopy(config)
    mutation["phase_learning"]["evaluation_seen"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="must be False"):
        validate_paper_config(mutation)


def test_all_committed_paper_configs_validate() -> None:
    paths = sorted(
        [
            *Path("configs").glob("paper_*_pilot_v1.json"),
            *Path("configs").glob("paper_*_final_v1.json"),
        ]
    )
    assert len(paths) == 10
    for path in paths:
        assert load_paper_config(path)["runner"] == "paper_suite"


def test_committed_pilot_and_final_seeds_are_disjoint() -> None:
    for family in (
        "autonomy",
        "ablation",
        "transfer",
        "creativity",
        "safe_application",
    ):
        pilot = load_paper_config(
            f"configs/paper_{family}_pilot_v1.json"
        )
        final = load_paper_config(
            f"configs/paper_{family}_final_v1.json"
        )
        assert not (
            set(pilot["research_seeds"]) & set(final["research_seeds"])
        )
        pilot_worlds = {
            seed
            for values in pilot["world_seeds"].values()
            for seed in values
        }
        final_worlds = {
            seed
            for values in final["world_seeds"].values()
            for seed in values
        }
        assert not (pilot_worlds & final_worlds)


def test_ablation_matrix_declares_full_sensitivity_grid() -> None:
    config = load_paper_config("configs/paper_ablation_final_v1.json")
    conditions = expand_suite_conditions(config["suites"][0])
    matrix = [
        item for item in conditions if item["name"].startswith("full_depth")
    ]
    assert len(matrix) == 4 * 3 * 3
    assert {item["imagination_depth"] for item in matrix} == {1, 2, 4, 6}
    assert {
        item["imagination_branching_factor"] for item in matrix
    } == {1, 2, 4}
    assert {item["imagination_aggregation"] for item in matrix} == {
        "max",
        "mean",
        "risk-adjusted",
    }


def test_opaque_world_does_not_reveal_wrong_branch_before_terminal() -> None:
    viable = OpaqueDependencyWorld(3, seed=77)
    wrong = OpaqueDependencyWorld(3, seed=77)
    correct_action = viable.oracle_action()
    wrong_action = next(
        item
        for item in wrong.snapshot().available_actions
        if item.signature != correct_action.signature
    )
    viable_step = viable.step(correct_action)
    wrong_step = wrong.step(wrong_action)
    assert viable_step.reward == wrong_step.reward == 0.0
    assert viable_step.snapshot == wrong_step.snapshot


def test_budget_ledger_fails_closed() -> None:
    ledger = BudgetLedger(2, action_proposal_limit=2)
    ledger.consume_real()
    ledger.record_imagined(10)
    ledger.record_proposals(2)
    ledger.consume_real()
    assert ledger.real_remaining == 0
    assert ledger.imagined_transitions == 10
    with pytest.raises(RuntimeError, match="transition budget"):
        ledger.consume_real()
    with pytest.raises(RuntimeError, match="proposal budget"):
        ledger.record_proposals()


def test_checkpoint_parts_can_reset_policy_and_retain_prophecy() -> None:
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            minimum_holdout_count=2,
            use_imagination=False,
        ),
        seed=4,
    )
    for episode in range(4):
        world = OpaqueDependencyWorld(2, seed=10)
        while not world.terminal:
            state = world.snapshot()
            decision = agent.select_action(state, episode=episode)
            outcome = world.step(decision.action)
            agent.observe(state, decision.action, outcome)
        agent.finish_episode(final_return=outcome.reward)
    parts = capture_checkpoint_parts(
        agent,
        episode=4,
        effect_representation={"profiles": [{"executions": 8}]},
    )
    fresh = AutonomousLearningAgent(TabularProphecy(), seed=99)
    retained = restore_checkpoint_parts(
        fresh,
        parts.selected(
            policy=False,
            prophecy=True,
            holdout=False,
            effect_representation=True,
        ),
        retain_policy=False,
        retain_prophecy=True,
        retain_holdout=False,
    )
    assert not fresh.policy._local
    assert fresh.prophecy._exact
    assert retained == {"profiles": [{"executions": 8}]}
    assert checkpoint_fingerprint(fresh, effect_representation=retained)


def test_effect_transfer_representation_is_name_free_and_changes_policy() -> None:
    world = OpaqueDependencyWorld(2, seed=10)
    before = world.snapshot()
    action = before.available_actions[0]
    outcome = world.step(action)
    transition = {
        "world_seed": 10,
        "episode": 0,
        "before": {
            "vector": list(before.vector),
            "facts": sorted(before.facts),
            "available_actions": [
                item.signature for item in before.available_actions
            ],
            "goal_progress": before.goal_progress,
        },
        "action": action.signature,
        "after": {
            "vector": list(outcome.snapshot.vector),
            "facts": sorted(outcome.snapshot.facts),
            "available_actions": [
                item.signature
                for item in outcome.snapshot.available_actions
            ],
            "goal_progress": outcome.snapshot.goal_progress,
        },
        "reward": 1.0,
        "error": False,
    }
    representation = _profile_representation([transition])
    encoded = json.dumps(representation)
    assert "op_" not in encoded
    agent = AutonomousLearningAgent(TabularProphecy(), seed=2)
    assert agent.policy.value(before, action) == 0.0
    _apply_effect_transfer(agent, [transition], representation)
    assert agent.policy.value(before, action) != 0.0


def test_seed_statistics_are_deterministic_and_paired() -> None:
    values = [0.1, 0.3, 0.8, 0.9]
    assert bootstrap_confidence_interval(
        values, samples=500, seed=7
    ) == bootstrap_confidence_interval(values, samples=500, seed=7)
    assert paired_permutation_test([1, 1, 1], [0, 0, 0]) < 0.3
    corrected = holm_correction([0.01, 0.04, 0.03])
    assert corrected == [0.03, 0.06, 0.06]
    assert trapezoid_auc([(0, 0.0), (4, 1.0)]) == 0.5


def test_dqn_baseline_runs_and_evaluation_is_frozen(tmp_path) -> None:
    pytest.importorskip("torch")
    config = {
        "name": "dqn_smoke",
        "runner": "autonomous_main",
        "seeds": [1],
        "train_episodes": 2,
        "eval_episodes": 1,
        "evaluation_modes": ["evaluation_seen"],
        "execution": {
            "workers": 1,
            "cuda_workers": 1,
            "device": "cpu",
        },
        "environments": [{"name": "opaque_l2", "length": 2}],
        "conditions": [{"name": "dqn", "algorithm": "dqn"}],
    }
    artifacts = run_autonomous_experiment(
        config, output_dir=tmp_path / "dqn", overwrite=True
    )
    rows = list(
        __import__("csv").DictReader(
            artifacts.episodes_csv.open(encoding="utf-8-sig")
        )
    )
    evaluation = [
        row for row in rows if row["phase"] == "evaluation_seen"
    ]
    assert len(evaluation) == 1
    assert (
        evaluation[0]["checkpoint_fingerprint_before"]
        == evaluation[0]["checkpoint_fingerprint_after"]
    )


def test_planned_count_matches_autonomy_contract() -> None:
    config = tiny_paper_config()
    assert planned_paper_run_count(config) == 15


def test_final_run_requires_frozen_external_gate_manifest(tmp_path) -> None:
    config = tiny_paper_config()
    config["study_stage"] = "final"
    config["research_seeds"] = list(range(20))
    config["acceptance_gates"] = {
        "p0": True,
        "p1": True,
        "p2": True,
        "p3": True,
    }
    config["acceptance_gate_manifest"] = str(
        tmp_path / "missing-gates.json"
    )
    config["statistics"] = {
        "unit": "research_seed",
        "confidence": 0.95,
        "bootstrap_samples": 100,
        "permutation_samples": 100,
        "test": "paired_permutation",
        "multiple_comparisons": "holm",
    }
    validate_paper_config(config)
    with pytest.raises(FileNotFoundError, match="acceptance-gate"):
        run_paper_suite(config, output_dir=tmp_path / "final")
