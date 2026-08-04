from __future__ import annotations

from aassr_v2.hierarchical_code_experiment_setup import (
    HierarchicalCodeGoalAgent,
    make_code_direct_agent,
)
from aassr_v2.hierarchical_code_proof_experiment import (
    pretrain_prophecy_from_random_actions,
)


def test_policy_and_goal_receive_same_random_transition_stream() -> None:
    policy = make_code_direct_agent("policy_only", 7)
    goal = HierarchicalCodeGoalAgent(7, room_length=4)

    policy_stats = pretrain_prophecy_from_random_actions(
        policy,
        seed=7,
        episodes=40,
        map_count=16,
        stage_count=4,
        room_length=4,
    )
    goal_stats = pretrain_prophecy_from_random_actions(
        goal,
        seed=7,
        episodes=40,
        map_count=16,
        stage_count=4,
        room_length=4,
    )

    assert policy_stats == goal_stats
    assert policy.prophecy.effect_observations == goal.base.prophecy.effect_observations
    assert policy.prophecy.effect_bucket_count == goal.base.prophecy.effect_bucket_count


def test_random_pretraining_does_not_update_policy_values() -> None:
    agent = make_code_direct_agent("policy_only", 13)
    before = (
        dict(agent.policy._local),
        dict(agent.policy._global),
        dict(agent.policy._state_visits),
    )

    pretrain_prophecy_from_random_actions(
        agent,
        seed=13,
        episodes=40,
        map_count=16,
        stage_count=4,
        room_length=4,
    )

    after = (
        dict(agent.policy._local),
        dict(agent.policy._global),
        dict(agent.policy._state_visits),
    )
    assert after == before == ({}, {}, {})
