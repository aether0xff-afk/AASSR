from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_dqn_baseline import (
    BARE_DQN_CONDITION,
    BareRelationalDQNAgent,
    build_bare_dqn_agent,
)
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_generation import RelationalInvariantDQN
from aassr_v2.current_hardware import (
    HardwareRelationalGRUBranchCritic,
    HardwareRelationalInvariantDQN,
)
from aassr_v2.current_protocol import run_current_episode
from aassr_v2.pentest_current_generation_main import CURRENT_EXPERIMENT_CONDITIONS
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def test_hardware_dqn_matches_current_relational_dqn_initial_q_values_on_cpu() -> None:
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


def test_current_aassr_and_bare_dqn_share_hardware_backend() -> None:
    aassr = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=128,
        use_imagination=True,
        device="cpu",
    )
    bare = build_bare_dqn_agent(
        seed=7,
        train_transitions=128,
        device="cpu",
    )

    assert isinstance(aassr.dqn, HardwareRelationalInvariantDQN)
    assert isinstance(bare.dqn, HardwareRelationalInvariantDQN)
    assert isinstance(aassr.critic, HardwareRelationalGRUBranchCritic)
    assert aassr.dqn.model_stats()["device"] == "cpu"
    assert bare.dqn.model_stats()["device"] == "cpu"
    assert aassr.critic.hardware_stats()["device"] == "cpu"
    assert str(aassr.base_neural_prophecy.device) == "cpu"
    assert aassr.planner.scorer is aassr.critic
    assert aassr.current_depth_batching is True


def test_bare_dqn_is_actually_dqn_only_and_runs_current_protocol() -> None:
    agent = build_bare_dqn_agent(
        seed=42,
        train_transitions=64,
        device="cpu",
    )
    assert isinstance(agent, BareRelationalDQNAgent)
    diagnostics = agent.diagnostics()
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
        condition=BARE_DQN_CONDITION,
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
    assert row.condition == BARE_DQN_CONDITION
    assert after[0] - before[0] == consumed
    assert row.aseq_guard_events == 0
    assert row.imagination_runs == 0


def test_current_experiment_condition_contract_has_three_conditions() -> None:
    assert CURRENT_EXPERIMENT_CONDITIONS == (
        "dqn_bare",
        "aassr_current_no_imagination",
        "aassr_current_full",
    )
