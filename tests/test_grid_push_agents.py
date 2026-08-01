from __future__ import annotations

from aassr_v2.causal_agent_v2 import CausalAASSRAgent
from aassr_v2.causal_representation import RepresentedReturnAgent
from aassr_v2.grid_push_agents import (
    GridRelationalEffectEncoder,
    checkpoint_contains_private_solver_data,
    grid_observable_transition,
    run_small_grid_diagnostic,
)
from aassr_v2.grid_push_world import GridPushWorld, ProceduralGridPushGenerator


def test_grid_relational_value_survives_first_observed_effect() -> None:
    spec, _, _ = ProceduralGridPushGenerator().generate(
        7001, maximum_actions=30, random_rollouts=10
    )
    world = GridPushWorld(spec)
    before = world.observe()
    agent = RepresentedReturnAgent(
        GridRelationalEffectEncoder(), seed=1, learning_rate=1.0
    )
    action = before.available_actions[0]
    old_key = agent.encoder.action_key(before, action)
    outcome = world.step(action)
    agent.observe_transition(grid_observable_transition(before, action, outcome))
    agent.finish_episode(True, gamma=1.0)
    new_key = agent.encoder.action_key(before, action)
    assert old_key != new_key
    assert agent.q_value(before, action) == 1.0
    assert agent.select_action(before, epsilon=0.0) == action
    assert agent.key_migration_count >= 1


def test_agent_checkpoint_contains_no_solver_or_private_link_data() -> None:
    agent = CausalAASSRAgent(GridRelationalEffectEncoder, seed=3)
    assert not checkpoint_contains_private_solver_data(agent.export_full_checkpoint())


def test_small_diagnostic_executes_real_policy_prophecy_imagination_path() -> None:
    specs = [
        ProceduralGridPushGenerator().generate(
            seed, maximum_actions=30, random_rollouts=10
        )[0]
        for seed in (7002, 7003)
    ]
    summaries, episodes, decisions = run_small_grid_diagnostic(
        specs=specs,
        research_seeds=[17],
        training_episodes=8,
        evaluation_episodes=2,
        maximum_steps=20,
    )
    assert {summary.condition for summary in summaries} == {
        "random", "contextual_policy", "full_aassr"
    }
    assert len(episodes) == 6
    assert decisions  # CausalImaginationPlanner was called.
    for summary in summaries:
        assert summary.evaluation_learning_calls == 0
        if summary.checkpoint_before_evaluation:
            assert summary.checkpoint_before_evaluation == summary.checkpoint_after_evaluation
