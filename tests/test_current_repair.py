from __future__ import annotations

from types import SimpleNamespace

import pytest

from aassr_v2.current_relational_state import install_relational_state_contract

install_relational_state_contract()

from aassr_v2.branch_critic import CriticTransition
from aassr_v2.current_relational_codec import (
    decode_relational_state,
    descriptor,
    legal_action_mask,
    semantic_prediction_score,
    transition_target,
)
from aassr_v2.current_relational_model import (
    RelationalProphecyConfig,
    RelationalStochasticProphecy,
)
from aassr_v2.current_repair import preserve_root_evaluations
from aassr_v2.current_return_critic import (
    ReturnAwareHardwareRelationalGRUBranchCritic,
)
from aassr_v2.imagination_tree import (
    ImaginationNode,
    ImaginationResult,
    RootActionEvaluation,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.policy import PolicyMemory
from aassr_v2.types import Action, Prediction, StateSnapshot


def _vector(*, marker_index: int | None = None) -> tuple[float, ...]:
    values = [0.0] * AGENT_STATE_SIZE
    if marker_index is not None:
        values[marker_index] = 1.0
    return tuple(values)


def _transition(route_id: str, *, marker: int) -> tuple[StateSnapshot, Action, StateSnapshot]:
    action = Action(
        "request",
        parameters={"route_id": route_id, "profile_id": "profile-browse"},
    )
    before = StateSnapshot(
        _vector(marker_index=marker),
        facts=frozenset(
            {
                f"known_route:{route_id}",
                "known_profile:profile-browse",
            }
        ),
        available_actions=(action,),
    )
    after = StateSnapshot(
        _vector(marker_index=marker),
        facts=before.facts | {f"observed_route_role:{route_id}:catalog"},
        available_actions=(action,),
    )
    return before, action, after


def test_relational_world_model_target_is_rename_invariant() -> None:
    left = _transition("route-05", marker=100)
    right = _transition("route-28", marker=300)

    assert left[0].vector != right[0].vector
    assert left[2].vector != right[2].vector
    assert transition_target(*left) == transition_target(*right)


def test_relational_decode_preserves_semantics_and_legal_surface() -> None:
    before, action, actual = _transition("route-05", marker=100)
    _, next_descriptor, next_mask, next_terminal = transition_target(
        before, action, actual
    )
    decoded = decode_relational_state(
        next_descriptor,
        next_mask,
        scaffold=before,
        predicted_terminal=next_terminal,
        source="unit-test",
    )

    assert legal_action_mask(decoded) == next_mask
    assert len(decoded.available_actions) == int(sum(next_mask))
    assert any(
        fact.endswith(":catalog") and fact.startswith("observed_route_role:")
        for fact in decoded.facts
    )
    assert descriptor(decoded) == pytest.approx(next_descriptor)


def test_relational_prophecy_returns_members_not_ensemble_mean() -> None:
    pytest.importorskip("torch")
    before, action, actual = _transition("route-05", marker=100)
    prophecy = RelationalStochasticProphecy(
        seed=7,
        device="cpu",
        config=RelationalProphecyConfig(
            hidden_units=16,
            ensemble_size=3,
            replay_capacity=16,
            batch_size=1,
            warmup_steps=1,
            gradient_steps_per_observation=1,
        ),
    )
    prophecy.learn(before, action, actual)
    predictions = prophecy.predict(before, action, samples=3)

    assert len(predictions) == 3
    assert len({prediction.source for prediction in predictions}) == 3
    assert all("member-" in prediction.source for prediction in predictions)
    assert prophecy.diagnostics()["ensemble_outcomes_not_mean_collapsed"] == 1


def test_semantic_score_ignores_irrelevant_concrete_raw_slots() -> None:
    before, action, actual = _transition("route-05", marker=100)
    changed_vector = list(actual.vector)
    changed_vector[400] = 1.0
    renamed_raw_only = StateSnapshot(
        tuple(changed_vector),
        facts=actual.facts,
        available_actions=actual.available_actions,
        goal_progress=actual.goal_progress,
        metadata=actual.metadata,
    )
    score = semantic_prediction_score(
        (Prediction(renamed_raw_only, 1.0, source="unit-test"),),
        actual,
    )
    assert score == pytest.approx(1.0)


def test_return_critic_distinguishes_failure_truncation_and_success_from_every_suffix() -> None:
    pytest.importorskip("torch")
    before, action, after = _transition("route-05", marker=100)
    trajectory = (
        CriticTransition(before, action, after, 1.0),
        CriticTransition(after, action, after, 1.0),
    )
    critic = ReturnAwareHardwareRelationalGRUBranchCritic(7, device="cpu")
    assert critic.value_center == 0.0

    critic.set_episode_return(1.0, 0.9)
    critic.observe_episode(trajectory, success=True)
    assert critic.replay[-2][1] == pytest.approx((0.9, 0.9))
    assert critic.replay[-1][1] == pytest.approx((1.0,))

    critic.set_episode_return(0.0, 0.9)
    critic.observe_episode(trajectory, success=False)
    assert critic.replay[-2][1] == pytest.approx((0.0, 0.0))
    assert critic.replay[-1][1] == pytest.approx((0.0,))

    critic.set_episode_return(-1.0, 0.9)
    critic.observe_episode(trajectory, success=False)
    assert critic.replay[-2][1] == pytest.approx((-0.9, -0.9))
    assert critic.replay[-1][1] == pytest.approx((-1.0,))
    assert critic.suffix_sequences == 6


def test_root_preservation_restores_beam_pruned_root() -> None:
    action_a = Action("request", parameters={"route_id": "a", "profile_id": "p"})
    action_b = Action("request", parameters={"route_id": "b", "profile_id": "p"})
    state = StateSnapshot(_vector(), available_actions=(action_a, action_b))
    memory = PolicyMemory.empty()
    node_a = ImaginationNode(
        node_id=1,
        parent_id=0,
        depth=1,
        state=state,
        root_action=action_a,
        action_from_parent=action_a,
        state_path=("root", "a"),
        action_path=(action_a.signature,),
        cumulative_value=0.7,
        step_confidence=1.0,
        cumulative_confidence=1.0,
        policy_memory=memory,
    )
    node_b = ImaginationNode(
        node_id=2,
        parent_id=0,
        depth=1,
        state=state,
        root_action=action_b,
        action_from_parent=action_b,
        state_path=("root", "b"),
        action_path=(action_b.signature,),
        cumulative_value=0.4,
        step_confidence=1.0,
        cumulative_confidence=1.0,
        policy_memory=memory,
    )
    result = ImaginationResult(
        chosen_action=action_a,
        root_evaluations=(
            RootActionEvaluation(
                action=action_a,
                leaf_values=(0.7,),
                aggregate_value=0.7,
                best_path=(action_a.signature,),
                best_leaf_id=1,
            ),
        ),
        nodes=(node_a, node_b),
        expanded_nodes=1,
        maximum_depth_reached=1,
    )
    planner = SimpleNamespace(
        _adjusted_value=lambda node: node.cumulative_value,
        _aggregate=lambda values: sum(values) / len(values),
    )

    repaired = preserve_root_evaluations(planner, state, result)
    assert {item.action.signature for item in repaired.root_evaluations} == {
        action_a.signature,
        action_b.signature,
    }
    assert len(repaired.root_evaluations) == len(state.available_actions)
