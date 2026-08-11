from __future__ import annotations

from types import SimpleNamespace

import pytest

from aassr_v2.current_planner import CurrentFullyBatchedImaginationTree
from aassr_v2.current_relational_codec import (
    TERMINAL_TRUNCATION,
    terminal_class,
)
from aassr_v2.imagination_tree import ImaginationNode
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.policy import PolicyMemory
from aassr_v2.types import Action, StateSnapshot


def _state(*, facts: frozenset[str] = frozenset(), actions=()) -> StateSnapshot:
    return StateSnapshot(
        (0.0,) * AGENT_STATE_SIZE,
        facts=facts,
        available_actions=tuple(actions),
    )


def _node(
    node_id: int,
    *,
    parent_id: int | None,
    depth: int,
    value: float,
    action: Action | None,
    state: StateSnapshot,
    terminal_reason: str | None = None,
) -> ImaginationNode:
    path = () if action is None else (action.signature,)
    return ImaginationNode(
        node_id=node_id,
        parent_id=parent_id,
        depth=depth,
        state=state,
        root_action=action,
        action_from_parent=action,
        state_path=tuple(str(index) for index in range(depth + 1)),
        action_path=path,
        cumulative_value=float(value),
        step_confidence=1.0,
        cumulative_confidence=1.0,
        policy_memory=PolicyMemory.empty(),
        terminal_reason=terminal_reason,
    )


def _planner(aggregation: str = "mean") -> CurrentFullyBatchedImaginationTree:
    planner = object.__new__(CurrentFullyBatchedImaginationTree)
    planner.config = SimpleNamespace(
        aggregation=aggregation,
        top_mean_count=2,
    )
    planner.chance_backup_groups = 0
    planner.decision_backup_nodes = 0
    planner.task_truncation_leaves = 0
    planner.task_failure_leaves = 0
    return planner


def test_stochastic_outcomes_are_probability_weighted_not_equal_votes() -> None:
    planner = _planner("mean")
    value = planner._aggregate_outcomes((1.0, -1.0), (0.9, 0.1))
    assert value == pytest.approx(0.8)


def test_risk_adjusted_chance_backup_uses_weighted_variance() -> None:
    planner = _planner("risk_adjusted")
    value = planner._aggregate_outcomes((1.0, -1.0), (0.9, 0.1))
    # Weighted mean 0.8, weighted std 0.6.
    assert value == pytest.approx(0.2)


def test_future_agent_actions_use_max_not_average() -> None:
    planner = _planner("mean")
    root_action = Action(
        "request",
        parameters={"route_id": "root", "profile_id": "profile-browse"},
    )
    good = Action(
        "request",
        parameters={"route_id": "good", "profile_id": "profile-browse"},
    )
    bad = Action(
        "request",
        parameters={"route_id": "bad", "profile_id": "profile-browse"},
    )
    state = _state(actions=(good, bad))
    root = _node(0, parent_id=None, depth=0, value=0.0, action=None, state=state)
    parent = _node(1, parent_id=0, depth=1, value=0.0, action=root_action, state=state)
    good_child = _node(2, parent_id=1, depth=2, value=1.0, action=good, state=state)
    bad_child = _node(3, parent_id=1, depth=2, value=-1.0, action=bad, state=state)

    backed, _, _, _ = planner._bellman_backups(
        (root, parent, good_child, bad_child),
        {1: 1.0, 2: 1.0, 3: 1.0},
    )

    # The bad action is optional. Its existence must not reduce the state value.
    assert backed[1] == pytest.approx(1.0)


def test_chance_outcomes_for_one_action_are_expected_before_decision_max() -> None:
    planner = _planner("mean")
    root_action = Action(
        "request",
        parameters={"route_id": "root", "profile_id": "profile-browse"},
    )
    next_action = Action(
        "request",
        parameters={"route_id": "next", "profile_id": "profile-browse"},
    )
    state = _state(actions=(next_action,))
    root = _node(0, parent_id=None, depth=0, value=0.0, action=None, state=state)
    parent = _node(1, parent_id=0, depth=1, value=0.0, action=root_action, state=state)
    common = _node(2, parent_id=1, depth=2, value=1.0, action=next_action, state=state)
    rare = _node(3, parent_id=1, depth=2, value=-1.0, action=next_action, state=state)

    backed, _, _, _ = planner._bellman_backups(
        (root, parent, common, rare),
        {1: 1.0, 2: 0.9, 3: 0.1},
    )

    assert backed[1] == pytest.approx(0.8)


def test_rate_limit_is_task_truncation_even_when_actions_remain() -> None:
    action = Action(
        "request",
        parameters={"route_id": "route-01", "profile_id": "profile-browse"},
    )
    vector = [0.0] * AGENT_STATE_SIZE
    vector[6] = 1.0
    state = StateSnapshot(
        tuple(vector),
        facts=frozenset({"rate_limited"}),
        available_actions=(action,),
    )
    planner = _planner("mean")

    assert terminal_class(state) == TERMINAL_TRUNCATION
    assert planner._task_terminal_reason(state) == "truncation"
