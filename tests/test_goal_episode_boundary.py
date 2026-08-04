from __future__ import annotations

from typing import cast

from aassr_v2.collapsing_gridpush_world import CollapsingGridPushWorld
from aassr_v2.goal_gridpush_diagnostic_setup import DiagnosticPersistentGoalAgent
from aassr_v2.goal_gridpush_experiment import GoalProposal


def test_active_goal_does_not_cross_episode_boundary() -> None:
    agent = DiagnosticPersistentGoalAgent(7)
    state = CollapsingGridPushWorld(700001).snapshot()
    marker = cast(GoalProposal, object())
    agent.active_goal = marker
    agent.active_goal_age = 3
    agent._active_episode_token = (0, 700000, False)

    agent.select_action(state, episode=1, explore=False)

    assert agent.active_goal is None
    assert agent.active_goal_age == 0
    assert agent._active_episode_token == (1, 700001, False)
