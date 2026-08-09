from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_agent import build_current_standalone_pentest_aassr_core
from aassr_v2.current_decision_optimization import (
    _coverage_cache_key,
    install_current_decision_optimizations,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def _agent(seed: int):
    return build_current_standalone_pentest_aassr_core(
        seed=seed,
        train_transitions=256,
        use_imagination=True,
        device="cpu",
    )


def test_training_suppressed_short_circuit_preserves_actions_and_gate_reason() -> None:
    reference = _agent(7)
    optimized = _agent(7)
    install_current_decision_optimizations(optimized)
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()

    for episode in range(20):
        left = reference._core_select_action(state, episode=episode, explore=True)
        right = optimized._core_select_action(state, episode=episode, explore=True)
        assert right.action.signature == left.action.signature
        assert right.imagination_gate_reason == left.imagination_gate_reason
        assert right.imagination_eligible == left.imagination_eligible
        assert right.used_imagination == left.used_imagination

    assert optimized.current_coverage_skipped_decisions == 20


def test_disabled_imagination_short_circuit_preserves_greedy_action() -> None:
    reference = _agent(42)
    optimized = _agent(42)
    reference.requested_imagination = False
    optimized.requested_imagination = False
    install_current_decision_optimizations(optimized)
    state = TransferDiagnosticWorld(90_002, stage=TRANSFER_STAGES[0]).snapshot()

    left = reference._core_select_action(state, episode=0, explore=False)
    right = optimized._core_select_action(state, episode=0, explore=False)
    assert right.action.signature == left.action.signature
    assert left.imagination_gate_reason == right.imagination_gate_reason == "disabled"


def test_relational_coverage_memoization_preserves_exact_average() -> None:
    agent = _agent(100)
    state = TransferDiagnosticWorld(90_003, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions)
    assert actions

    keys = tuple(_coverage_cache_key(state, action) for action in actions)
    assert len(set(keys)) < len(keys), "test state needs repeated relational action keys"

    expected = sum(agent.skill_prophecy.confidence(state, action) for action in actions) / len(actions)
    install_current_decision_optimizations(agent)
    actual = agent.skill_prophecy.coverage(state, actions)
    assert actual == pytest.approx(expected, abs=0.0, rel=0.0)
