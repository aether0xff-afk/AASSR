from __future__ import annotations

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
    ContextualPolicy,
)
from aassr_v2.autonomous_benchmarks import OpaqueDependencyWorld
from aassr_v2.autonomous_experiment import (
    planned_autonomous_run_count,
    run_autonomous_experiment,
)
from aassr_v2.experiment_runner import read_rows
from aassr_v2.tabular_prophecy import TabularProphecy


def test_opaque_world_contains_no_answer_words() -> None:
    environment = OpaqueDependencyWorld(4, seed=7)
    text = " ".join(
        action.verb_name
        for stage_actions in environment._actions
        for action in stage_actions
    ).lower()
    assert "safe" not in text
    assert "trap" not in text
    assert "greedy" not in text
    assert "finish" not in text
    assert environment.snapshot().goal_progress == 0.0


def test_contextual_policy_values_same_action_by_state() -> None:
    policy = ContextualPolicy(learning_rate=1.0)
    first_world = OpaqueDependencyWorld(2, seed=3)
    first = first_world.snapshot()
    action = first.available_actions[0]
    first_world.step(action)
    second = first_world.snapshot()
    policy.observe_return(first, action, 1.0)
    policy.observe_return(second, action, -1.0)
    assert policy.value(first, action) == 1.0
    assert policy.value(second, action) == -1.0


def test_agent_discovers_path_without_demonstration() -> None:
    config = AutonomousAgentConfig(
        epsilon_decay_episodes=200,
        imagination_depth=4,
        imagination_interval=2,
        minimum_holdout_count=2,
        validation_interval=8,
    )
    agent = AutonomousLearningAgent(TabularProphecy(), config=config, seed=11)
    outcomes = []
    for episode in range(400):
        environment = OpaqueDependencyWorld(4, seed=101)
        while not environment.terminal:
            state = environment.snapshot()
            decision = agent.select_action(state, episode=episode)
            outcome = environment.step(decision.action)
            agent.observe(state, decision.action, outcome)
        agent.finish_episode(final_return=outcome.reward)
        outcomes.append(outcome.reward)
    assert sum(outcomes[-50:]) / 50.0 >= 0.7


def test_autonomous_runner_writes_expected_rows(tmp_path) -> None:
    config = {
        "name": "tiny_autonomous",
        "runner": "autonomous_main",
        "seeds": [1],
        "train_episodes": 20,
        "eval_episodes": 5,
        "environments": [{"name": "opaque_l3", "length": 3}],
        "conditions": [
            {
                "name": "full",
                "use_imagination": True,
                "minimum_holdout_count": 2,
            }
        ],
    }
    assert planned_autonomous_run_count(config) == 25
    artifacts = run_autonomous_experiment(
        config, output_dir=tmp_path / "run", overwrite=True
    )
    rows = read_rows(artifacts.episodes_csv)
    assert len(rows) == 25
    assert {row["phase"] for row in rows} == {"training", "evaluation"}
    assert {row["action_family"] for row in rows} == {"opaque"}
    assert (artifacts.output_dir / "protocol_manifest.json").exists()
