from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_dqn_baseline import (
    RAW_DQN_CONDITION,
    RELATIONAL_DQN_CONDITION,
    BareRelationalDQNAgent,
    HardwareRawDynamicActionDQN,
    RawDQNAgent,
    build_raw_dqn_agent,
    build_relational_dqn_agent,
)
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_generation import (
    RelationalGRUBranchCritic,
    RelationalInvariantDQN,
)
from aassr_v2.current_hardware import (
    HardwareRelationalGRUBranchCritic,
    HardwareRelationalInvariantDQN,
)
from aassr_v2.current_protocol import run_current_episode
from aassr_v2.pentest_agent_main_test import DynamicActionDQN
from aassr_v2.pentest_current_generation_main import (
    CURRENT_EXPERIMENT_CONDITIONS,
    run_current_generation_condition,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.replay import ReplayTransition


def test_hardware_relational_dqn_matches_initial_q_values_on_cpu() -> None:
    seed = 12345
    reference = RelationalInvariantDQN(seed, train_transitions=256)
    hardware = HardwareRelationalInvariantDQN(
        seed,
        train_transitions=256,
        device="cpu",
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions[:8])
    assert actions

    assert hardware.score_actions(state, actions) == pytest.approx(
        reference.score_actions(state, actions),
        abs=1e-7,
        rel=1e-7,
    )
    stats = hardware.model_stats()
    assert stats["device"] == "cpu"
    assert stats["hardware_optimized"] == 1
    assert stats["per_row_target_item_syncs"] == 0
    assert stats["fused_next_action_reduce"] == 1


def test_hardware_raw_dqn_matches_plain_dynamic_dqn_initial_q_values_on_cpu() -> None:
    seed = 54321
    reference = DynamicActionDQN(seed, train_transitions=256)
    hardware = HardwareRawDynamicActionDQN(
        seed,
        train_transitions=256,
        device="cpu",
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions[:8])
    assert actions

    assert hardware.score_actions(state, actions) == pytest.approx(
        reference.score_actions(state, actions),
        abs=1e-7,
        rel=1e-7,
    )
    stats = hardware.model_stats()
    assert stats["representation"] == "raw-v3-vector+raw-signature-action-hash"
    assert stats["per_row_target_item_syncs"] == 0
    assert stats["fused_next_action_reduce"] == 1


def test_hardware_critic_matches_relational_critic_initial_forward_on_cpu() -> None:
    seed = 777
    reference = RelationalGRUBranchCritic(seed)
    hardware = HardwareRelationalGRUBranchCritic(seed, device="cpu")
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    action = state.available_actions[0]

    left = reference.score_step(
        state,
        action,
        state,
        prophecy_confidence=0.5,
    )
    right = hardware.score_step(
        state,
        action,
        state,
        prophecy_confidence=0.5,
    )
    assert right.value == pytest.approx(left.value, abs=1e-7, rel=1e-7)
    assert hardware.hardware_stats()["device"] == "cpu"


def test_current_aassr_and_dqn_controls_use_hardware_paths() -> None:
    aassr = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=128,
        use_imagination=True,
        device="cpu",
    )
    relational = build_relational_dqn_agent(
        seed=7,
        train_transitions=128,
        device="cpu",
    )
    raw = build_raw_dqn_agent(
        seed=7,
        train_transitions=128,
        device="cpu",
    )

    assert isinstance(aassr.dqn, HardwareRelationalInvariantDQN)
    assert isinstance(relational, BareRelationalDQNAgent)
    assert isinstance(relational.dqn, HardwareRelationalInvariantDQN)
    assert isinstance(raw, RawDQNAgent)
    assert isinstance(raw.dqn, HardwareRawDynamicActionDQN)
    assert isinstance(aassr.critic, HardwareRelationalGRUBranchCritic)
    assert aassr.dqn.model_stats()["device"] == "cpu"
    assert relational.dqn.model_stats()["device"] == "cpu"
    assert raw.dqn.model_stats()["device"] == "cpu"
    assert aassr.critic.hardware_stats()["device"] == "cpu"
    assert str(aassr.base_neural_prophecy.device) == "cpu"
    assert aassr.planner.scorer is aassr.critic
    assert aassr.current_depth_batching is True


def test_current_calibration_refresh_batches_selected_holdout_rows() -> None:
    agent = build_current_pentest_aassr_core(
        seed=9,
        train_transitions=128,
        use_imagination=True,
        device="cpu",
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    action = state.available_actions[0]
    replay = agent.evaluator.replay
    for index in range(40):
        replay.add(
            ReplayTransition(
                state,
                action,
                state,
                trace_id=f"calibration-{index}",
            )
        )
    assert len(replay.holdout()) == 8

    before_calls = agent.base_neural_prophecy.batch_prediction_calls
    agent.calibrated_prophecy._calibration(state, action)
    diagnostics = agent.calibrated_prophecy.diagnostics()
    assert agent.base_neural_prophecy.batch_prediction_calls - before_calls == 1
    assert diagnostics["calibration_batch_refreshes"] == 1
    assert diagnostics["calibration_batch_rows"] == 8
    assert diagnostics["calibration_refresh_batching"] == 1


@pytest.mark.parametrize(
    ("condition", "builder"),
    (
        (RAW_DQN_CONDITION, build_raw_dqn_agent),
        (RELATIONAL_DQN_CONDITION, build_relational_dqn_agent),
    ),
)
def test_dqn_controls_are_dqn_only_and_run_current_protocol(condition, builder) -> None:
    agent = builder(
        seed=42,
        train_transitions=64,
        device="cpu",
    )
    diagnostics = agent.diagnostics()
    assert diagnostics["condition"] == condition
    assert diagnostics["dqn_only"] is True
    assert diagnostics["aseq"]["guard_events"] == 0
    assert diagnostics["imagination"]["runs"] == 0
    assert diagnostics["skill_uses"] == 0
    assert set(diagnostics["modules_absent"]) == {
        "aseq",
        "knowledge",
        "prophecy",
        "imagination",
        "skills",
        "branch_critic",
        "feature_memory",
        "information_value_residual",
    }

    before = agent.learning_counters()
    row, consumed = run_current_episode(
        agent,
        condition=condition,
        research_seed=42,
        stage_index=0,
        scenario_seed=90_001,
        phase="train",
        block=0,
        episode=0,
        focus_level=0,
        transition_start=0,
        transition_cap=8,
        transition_budget=64,
        training=True,
    )
    after = agent.learning_counters()
    assert consumed > 0
    assert row.condition == condition
    assert after[0] - before[0] == consumed
    assert row.aseq_guard_events == 0
    assert row.imagination_runs == 0


def test_current_experiment_condition_contract_has_four_conditions() -> None:
    assert CURRENT_EXPERIMENT_CONDITIONS == (
        "dqn_raw",
        "dqn_relational",
        "aassr_current_no_imagination",
        "aassr_current_full",
    )


def test_tiny_current_main_materializes_all_four_conditions(tmp_path) -> None:
    result = run_current_generation_condition(
        tmp_path,
        research_seed=7,
        transition_budget=16,
        block_target=8,
        train_seeds=(90_001,),
        validation_seeds=(93_001,),
        diagnostic_seeds=(92_001,),
        diagnostic_stage_indices=(0,),
        device="cpu",
        allow_tf32=False,
    )
    assert result["diagnostic_full_stage_sweep"] is False
    assert result["training_checkpoint_count"] == 3
    assert result["nominal_total_training_transitions"] == 48
    assert set(result["diagnostic_successes"]) == set(CURRENT_EXPERIMENT_CONDITIONS)
    assert result["dqn_raw"]["condition"] == RAW_DQN_CONDITION
    assert result["dqn_relational"]["condition"] == RELATIONAL_DQN_CONDITION
    assert result["dqn_raw"]["exact_budget"] is True
    assert result["dqn_relational"]["exact_budget"] is True
    assert result["aassr"]["exact_budget"] is True
    assert result["aassr"]["same_checkpoint_comparison"] is True
    assert result["validation_learning_frozen"] is True
    assert result["diagnostic_learning_frozen"] is True
    assert (tmp_path / "diagnostic_dqn_raw.csv").exists()
    assert (tmp_path / "diagnostic_dqn_relational.csv").exists()
    assert (tmp_path / "diagnostic_aassr_current_no_imagination.csv").exists()
    assert (tmp_path / "diagnostic_aassr_current_full.csv").exists()
    assert (tmp_path / "summary.json").exists()
