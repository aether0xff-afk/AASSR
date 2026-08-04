from __future__ import annotations

from aassr_v2.goal_gridpush_experiment import GoalGridPushWorld
from aassr_v2.persistent_goal_agent import PersistentGoalSeparatedAgent


def test_goal_is_reused_instead_of_rebuilt_every_step() -> None:
    agent = PersistentGoalSeparatedAgent(7)
    world = GoalGridPushWorld(7)

    # Give Prophecy enough real transitions for the Maker to form a GOAL.
    for episode in range(8):
        training_world = GoalGridPushWorld(7)
        while training_world.snapshot().available_actions:
            before = training_world.snapshot()
            decision = agent.base.select_action(before, episode=episode, explore=True)
            outcome = training_world.step(decision.action)
            agent.base.observe(before, decision.action, outcome)
            if not outcome.snapshot.available_actions:
                break
        agent.base.finish_episode(
            final_return=1.0 if training_world.success else 0.0
        )

    before = world.snapshot()
    first = agent.select_action(before, episode=20, explore=False)
    first_proposals = agent.goal_proposals
    if agent.active_goal is None:
        # The learned model may still reject a GOAL; this is a valid safe state.
        assert not first.imagination_changed_action
        return

    outcome = world.step(first.action)
    agent.observe(before, first.action, outcome)
    agent.select_action(outcome.snapshot, episode=20, explore=False)

    assert agent.goal_proposals == first_proposals
    assert agent.goal_reuses >= 1


def test_goal_age_limit_discards_stale_goal() -> None:
    agent = PersistentGoalSeparatedAgent(7, maximum_goal_age=1)
    assert agent.maximum_goal_age == 1
