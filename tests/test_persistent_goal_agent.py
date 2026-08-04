from __future__ import annotations

from typing import cast

from aassr_v2.goal_gridpush_experiment import (
    GoalGridPushWorld,
    GoalProposal,
)
from aassr_v2.goals import GoalSet
from aassr_v2.imagination_tree import ImaginationResult
from aassr_v2.persistent_goal_agent import PersistentGoalSeparatedAgent
from aassr_v2.types import StateSnapshot


def test_active_goal_survives_a_reality_step_when_not_reached() -> None:
    agent = PersistentGoalSeparatedAgent(7)
    world = GoalGridPushWorld(7)
    before = world.snapshot()
    desired = StateSnapshot(
        tuple(value + 10.0 for value in before.vector),
        frozenset({"unreached_target"}),
        before.available_actions,
        0.0,
    )
    proposal = GoalProposal(
        GoalSet(),
        desired,
        cast(ImaginationResult, object()),
        1.0,
    )
    agent.active_goal = proposal

    action = before.available_actions[0]
    outcome = world.step(action)
    agent.observe(before, action, outcome)

    assert agent.active_goal is proposal
    assert agent.active_goal_age == 1


def test_reached_goal_is_cleared_after_observation() -> None:
    agent = PersistentGoalSeparatedAgent(7)
    world = GoalGridPushWorld(7)
    before = world.snapshot()
    action = before.available_actions[0]
    outcome = world.step(action)
    proposal = GoalProposal(
        GoalSet(),
        outcome.snapshot,
        cast(ImaginationResult, object()),
        1.0,
    )
    agent.active_goal = proposal

    agent.observe(before, action, outcome)

    assert agent.active_goal is None
    assert agent.goal_completions == 1


def test_goal_age_limit_is_configurable() -> None:
    agent = PersistentGoalSeparatedAgent(7, maximum_goal_age=1)
    assert agent.maximum_goal_age == 1
