from __future__ import annotations

from types import SimpleNamespace

from aassr_v2.hierarchical_code_experiment_setup import (
    CheckpointWaypointGoalMaker,
    HierarchicalCodeGoalAgent,
    make_code_direct_agent,
)
from aassr_v2.types import StateSnapshot


def test_direct_conditions_only_differ_in_imagination_depth() -> None:
    policy = make_code_direct_agent("policy_only", 7)
    short = make_code_direct_agent("short_imagination", 7)
    deep = make_code_direct_agent("deep_imagination", 7)

    assert not policy.config.use_imagination
    assert short.config.use_imagination
    assert deep.config.use_imagination
    assert short.planner.config.maximum_depth == 2
    assert deep.planner.config.maximum_depth == 4
    assert short.planner.config.aggregation == "max"
    assert deep.planner.config.aggregation == "max"
    assert policy.prophecy.name == short.prophecy.name == deep.prophecy.name


def test_goal_uses_four_step_maker_and_two_step_waypoint() -> None:
    agent = HierarchicalCodeGoalAgent(7, room_length=4)

    assert agent.maker.planner.config.maximum_depth == 4
    assert agent.maker.waypoint_depth == 2
    assert agent.maximum_goal_age == 2
    assert agent.state_executor.samples == 2


def test_checkpoint_waypoint_stays_before_state_reset() -> None:
    root_state = StateSnapshot((0.0,), frozenset({"stage:0"}), (), 0.0)
    before_reset = StateSnapshot((0.75,), frozenset({"stage:0"}), (), 0.0)
    after_reset = StateSnapshot(
        (0.0,),
        frozenset({"stage:1", "checkpoint_transition"}),
        (),
        0.0,
    )
    plan = SimpleNamespace(
        nodes=(
            SimpleNamespace(node_id=0, parent_id=None, state=root_state),
            SimpleNamespace(node_id=1, parent_id=0, state=before_reset),
            SimpleNamespace(node_id=2, parent_id=1, state=after_reset),
        )
    )
    maker = object.__new__(CheckpointWaypointGoalMaker)
    maker.waypoint_depth = 2

    waypoint = maker._waypoint_for(2, plan)

    assert waypoint is before_reset
