from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.current_semantic_evaluator import (
    RelationalActionUnlockValueEstimator,
    RelationalAdvancedTransitionEvaluator,
    semantic_prediction_uncertainty,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, Prediction, StateSnapshot


def _renamed(route: str) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request",
        parameters={"route_id": route, "profile_id": "profile-browse"},
    )
    state = StateSnapshot(
        (0.0,) * AGENT_STATE_SIZE,
        facts=frozenset(
            {
                f"known_route:{route}",
                "known_profile:profile-browse",
            }
        ),
        available_actions=(action,),
    )
    return state, action


def test_current_builder_replaces_raw_vector_information_evaluator() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    assert isinstance(agent.evaluator, RelationalAdvancedTransitionEvaluator)
    assert agent.current_semantic_evaluator is True
    diagnostics = agent.diagnostics()["current_repairs"]
    assert diagnostics["semantic_information_evaluator"] is True
    assert diagnostics["relational_repeat_unlock_value"] is True


def test_unlock_value_transfers_across_renamed_structural_actions() -> None:
    left_state, left_action = _renamed("route-05")
    right_state, right_action = _renamed("route-28")
    estimator = RelationalActionUnlockValueEstimator()

    estimator.observe_future_return(left_state, (left_action,), 0.75)
    assert estimator.estimate(right_state, (right_action,)) == pytest.approx(0.75)


def test_semantic_uncertainty_ignores_raw_id_slot_movement() -> None:
    left_state, _ = _renamed("route-05")
    right_state, _ = _renamed("route-28")
    left_vector = list(left_state.vector)
    right_vector = list(right_state.vector)
    left_vector[100] = 1.0
    right_vector[300] = 1.0
    left_state = StateSnapshot(
        tuple(left_vector),
        facts=left_state.facts,
        available_actions=left_state.available_actions,
    )
    right_state = StateSnapshot(
        tuple(right_vector),
        facts=right_state.facts,
        available_actions=right_state.available_actions,
    )

    uncertainty = semantic_prediction_uncertainty(
        (
            Prediction(left_state, 0.8, source="left"),
            Prediction(right_state, 0.8, source="right"),
        )
    )
    # Same relational future, so only confidence uncertainty remains: 0.5 * 0.2.
    assert uncertainty == pytest.approx(0.1)
