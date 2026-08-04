from __future__ import annotations

from aassr_v2.hierarchical_code_experiment_setup import (
    HierarchicalCodeGoalAgent,
    make_code_direct_agent,
)


def test_direct_conditions_only_differ_in_imagination_depth() -> None:
    policy = make_code_direct_agent("policy_only", 7)
    short = make_code_direct_agent("short_imagination", 7)
    deep = make_code_direct_agent("deep_imagination", 7)

    assert not policy.config.use_imagination
    assert short.config.use_imagination
    assert deep.config.use_imagination
    assert short.planner.config.maximum_depth == 2
    assert deep.planner.config.maximum_depth == 4
    assert policy.prophecy.name == short.prophecy.name == deep.prophecy.name


def test_goal_uses_four_step_maker_and_two_step_waypoint() -> None:
    agent = HierarchicalCodeGoalAgent(7, room_length=4)

    assert agent.maker.planner.config.maximum_depth == 4
    assert agent.maker.waypoint_depth == 2
    assert agent.maximum_goal_age == 2
    assert agent.state_executor.samples == 2
