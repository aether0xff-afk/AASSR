from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.current_planner import CurrentFullyBatchedImaginationTree
from aassr_v2.native_batching import DepthBatchedImaginationTree
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def test_current_depth_batcher_executes_relational_stochastic_batch_path() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions[:2])
    assert actions

    states = tuple(state for _ in actions)
    memories = tuple(None for _ in actions)
    rows = agent.current_batched_prophecy.predict_step_batch(
        states,
        actions,
        memories,
        samples=1,
    )

    assert len(rows) == len(actions)
    assert all(row.predictions for row in rows)
    diagnostics = agent.current_batched_prophecy.runtime_diagnostics()
    neural = agent.base_neural_prophecy.diagnostics()
    assert diagnostics["current_imagination_batch_calls"] == 1
    assert diagnostics["current_imagination_batch_rows"] == len(actions)
    assert diagnostics["current_imagination_skill_fallback_rows"] == 0
    assert neural["batch_prediction_calls"] >= 1
    assert neural["batch_prediction_rows"] >= len(actions)
    assert neural["state_input_relational"] == 1
    assert neural["action_input_relational"] == 1
    # The canonical model no longer exposes the historical ensemble-mean flag.
    # Its stronger multimodal contract is an explicit conditional mixture whose
    # stochastic outcome mass is separate from epistemic reliability.
    assert neural["conditional_mixture_components"] == 3
    assert neural["mixture_training_objective"] == "soft-mixture-likelihood"
    assert neural["reliability_outcome_probability_separated"] == 1


def test_current_parallel_universe_batches_all_neural_stages() -> None:
    agent = build_current_pentest_aassr_core(
        seed=42,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()

    assert isinstance(agent.planner, CurrentFullyBatchedImaginationTree)
    prophecy_before = agent.current_batched_prophecy.runtime_diagnostics()[
        "current_imagination_batch_calls"
    ]
    result = agent.planner.plan(state, maximum_depth=2)
    prophecy_after = agent.current_batched_prophecy.runtime_diagnostics()[
        "current_imagination_batch_calls"
    ]
    planner = agent.planner.runtime_diagnostics()
    critic = agent.critic.hardware_stats()
    dqn = agent.dqn.model_stats()

    assert result.nodes
    assert result.root_evaluations
    assert result.maximum_depth_reached >= 1
    assert planner["policy_batch_calls"] > 0
    assert planner["policy_batch_rows"] > 0
    assert planner["policy_scalar_fallback_rows"] == 0
    assert dqn["pair_score_batch_calls"] == planner["policy_batch_calls"]
    assert prophecy_after > prophecy_before
    assert planner["critic_batch_calls"] > 0
    assert planner["critic_batch_rows"] > 0
    assert planner["critic_scalar_fallback_rows"] == 0
    assert critic["batch_score_calls"] == planner["critic_batch_calls"]
    assert critic["batch_score_rows"] == planner["critic_batch_rows"]
    assert critic["scalar_score_calls"] == 0
    assert agent.planner.scorer is agent.critic


def test_fully_batched_planner_preserves_scalar_policy_and_critic_result() -> None:
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()

    scalar_agent = build_current_pentest_aassr_core(
        seed=100,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    scalar = DepthBatchedImaginationTree(
        scalar_agent.policy,
        scalar_agent.current_batched_prophecy,
        config=scalar_agent.planner.config,
        scorer=scalar_agent.critic,
    )
    scalar._state_key = scalar_agent.planner._state_key

    batch_agent = build_current_pentest_aassr_core(
        seed=100,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    batched = batch_agent.planner

    left = scalar.plan(state, maximum_depth=2)
    right = batched.plan(state, maximum_depth=2)
    assert left.chosen_action.signature == right.chosen_action.signature

    left_values = {
        item.action.signature: item.aggregate_value
        for item in left.root_evaluations
    }
    right_values = {
        item.action.signature: item.aggregate_value
        for item in right.root_evaluations
    }
    assert set(left_values) == set(right_values)
    for signature in left_values:
        assert right_values[signature] == pytest.approx(
            left_values[signature],
            abs=1e-7,
            rel=1e-7,
        )
