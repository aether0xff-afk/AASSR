from __future__ import annotations

import csv
import json

import pytest

from aassr_v2.baseline_efficiency_benchmark import (
    CHOICE_ACTIONS,
    GRIDPUSH_OBSERVATION_SIZE,
    BenchmarkGridPushWorld,
    DQNBenchmarkAgent,
    TabularQLearningAgent,
    choice_index,
    encode_gridpush_state,
    oracle_shortest_steps,
    run_gridpush_baseline_benchmark,
    solvable_map_seeds,
)


def test_fixed_action_world_and_encoding() -> None:
    world = BenchmarkGridPushWorld(7)
    state = world.snapshot()

    assert len(state.available_actions) == 4
    assert [choice_index(action) for action in state.available_actions] == [0, 1, 2, 3]
    assert len(encode_gridpush_state(state)) == GRIDPUSH_OBSERVATION_SIZE
    assert "energy" not in state.metadata
    assert state.metadata["termination"] == "fixed_choice_irreversible_path"


def test_oracle_filters_to_solvable_maps() -> None:
    seeds = solvable_map_seeds(100, 5)

    assert len(seeds) == 5
    assert len(set(seeds)) == 5
    assert all(oracle_shortest_steps(seed) for seed in seeds)


def test_q_learning_updates_terminal_reward() -> None:
    seed = solvable_map_seeds(200, 1)[0]
    world = BenchmarkGridPushWorld(seed)
    agent = TabularQLearningAgent(7, train_episodes=10)
    before = world.snapshot()
    action = CHOICE_ACTIONS[0]
    outcome = world.step(action)

    agent.observe(before, action, outcome)

    assert agent.updates == 1
    assert agent.model_stats()["model_units"] == 1


def test_dqn_smoke_when_torch_available() -> None:
    pytest.importorskip("torch")
    seed = solvable_map_seeds(300, 1)[0]
    world = BenchmarkGridPushWorld(seed)
    agent = DQNBenchmarkAgent(7, train_episodes=10, warmup_steps=2, batch_size=2)

    for _ in range(2):
        before = world.snapshot()
        if not before.available_actions:
            break
        decision = agent.select_action(before, episode=0, training=True)
        outcome = world.step(decision.action)
        agent.observe(before, decision.action, outcome)

    assert agent.model_stats()["model_units"] > 0


def test_small_benchmark_writes_reproducible_outputs(tmp_path) -> None:
    payload = run_gridpush_baseline_benchmark(
        tmp_path,
        condition="q_learning",
        seed=7,
        train_episodes=4,
        train_map_count=2,
        evaluation_episodes=2,
        checkpoints=(0, 2, 4),
    )

    assert payload["config"]["only_solvable_maps"] is True
    assert payload["final"]["checkpoint_episode"] == 4
    assert (tmp_path / "training_episodes.csv").exists()
    assert (tmp_path / "evaluation_episodes.csv").exists()
    assert (tmp_path / "checkpoints.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["final"] == payload["final"]

    with (tmp_path / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2 * 2
