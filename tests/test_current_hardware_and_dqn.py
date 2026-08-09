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
from aassr_v2.current_hardware import HardwareRelationalInvariantDQN
from aassr_v2.pentest_current_generation_main import (
    CURRENT_EXPERIMENT_CONDITIONS,
    run_current_generation_condition,
)
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


def test_current_aassr_and_bare_dqn_share_hardware_dqn_backend() -> None:
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
    assert aassr.dqn.model_stats()["device"] == "cpu"
    assert bare.dqn.model_stats()["device"] == "cpu"
    assert str(aassr.base_neural_prophecy.device) == "cpu"
    assert aassr.current_depth_batching is True


def test_bare_dqn_is_actually_dqn_only() -> None:
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


def test_current_main_contains_bare_dqn_and_same_checkpoint_aassr(tmp_path) -> None:
    result = run_current_generation_condition(
        tmp_path,
        research_seed=7,
        transition_budget=32,
        block_target=16,
        train_seeds=(90_001,),
        validation_seeds=(93_001,),
        diagnostic_seeds=(92_001,),
        device="cpu",
        allow_tf32=False,
    )

    assert tuple(result["experiment_conditions"]) == CURRENT_EXPERIMENT_CONDITIONS
    assert result["dqn_bare"]["condition"] == BARE_DQN_CONDITION
    assert result["dqn_bare"]["exact_budget"] is True
    assert result["dqn_bare"]["learning_frozen_in_evaluation"] is True
    assert result["aassr"]["exact_budget"] is True
    assert result["aassr"]["same_checkpoint_comparison"] is True
    assert result["validation_learning_frozen"] is True
    assert result["diagnostic_learning_frozen"] is True
    assert result["hardware_execution_contract"][
        "same_dqn_backend_for_bare_and_aassr"
    ] is True
    assert result["hardware_execution_contract"][
        "dqn_target_host_sync_per_row"
    ] is False
    assert set(result["diagnostic_successes"]) == {
        "dqn_bare",
        "aassr_current_no_imagination",
        "aassr_current_full",
    }
    assert (tmp_path / "diagnostic_dqn_bare.csv").exists()
    assert (tmp_path / "diagnostic_aassr_current_no_imagination.csv").exists()
    assert (tmp_path / "diagnostic_aassr_current_full.csv").exists()
    assert (tmp_path / "summary.json").exists()
