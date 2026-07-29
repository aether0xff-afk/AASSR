from __future__ import annotations

from aassr_v2.escape_gridworld import (
    EscapeGridWorld,
    generate_escape_grid,
    oracle_plan,
)
from aassr_v2.escape_training import (
    EscapeTrainingConfig,
    TrainingMode,
    epsilon_for_episode,
    train_escape_agent,
)
from aassr_v2.types import Action


def _execute_oracle(seed: int, color_count: int = 3) -> EscapeGridWorld:
    spec = generate_escape_grid(
        seed,
        color_count=color_count,
        distractor_boxes=2,
        max_steps=250,
    )
    environment = EscapeGridWorld(spec)
    plan = oracle_plan(spec)
    assert plan
    for action in plan:
        outcome = environment.step(action)
        assert not outcome.error
    return environment


def test_generated_escape_worlds_are_solvable() -> None:
    for seed in range(10):
        environment = _execute_oracle(seed)
        assert environment.success
        assert environment.position == environment.spec.goal


def test_box_reveals_key_and_matching_door_opens() -> None:
    spec = generate_escape_grid(11, color_count=1, distractor_boxes=0)
    environment = EscapeGridWorld(spec)
    events: list[str] = []

    for action in oracle_plan(spec):
        outcome = environment.step(action)
        events.append(outcome.event)

    assert "found_key:red" in events
    assert "opened_door:red" in events
    assert "red" in environment.inventory
    assert spec.doors[0][0] in environment.open_doors
    assert environment.success


def test_door_rejects_agent_without_matching_key() -> None:
    spec = generate_escape_grid(3, color_count=1, distractor_boxes=0)
    environment = EscapeGridWorld(spec)
    door_position = spec.doors[0][0]
    environment.position = (door_position[0] - 1, door_position[1])
    outcome = environment.step(
        Action("interact", target=f"door:{door_position[0]},{door_position[1]}")
    )
    assert outcome.error
    assert outcome.event == "missing_key:red"
    assert door_position not in environment.open_doors


def test_only_terminal_escape_has_external_reward() -> None:
    spec = generate_escape_grid(5, color_count=2, distractor_boxes=1)
    environment = EscapeGridWorld(spec)
    rewards = []
    for action in oracle_plan(spec):
        outcome = environment.step(action)
        rewards.append(outcome.reward)
    assert rewards[-1] == 1.0
    assert all(reward == 0.0 for reward in rewards[:-1])


def test_live_and_fast_modes_run_the_same_learning_configuration() -> None:
    config = EscapeTrainingConfig(
        episodes=8,
        seed=19,
        color_count=1,
        distractor_boxes=0,
        max_steps=60,
        live_step_delay=0.0,
        fast_progress_interval=2,
    )
    live = train_escape_agent(config, mode=TrainingMode.LIVE)
    fast = train_escape_agent(config, mode=TrainingMode.FAST)
    assert live.episodes == fast.episodes == 8
    assert live.successes == fast.successes
    assert live.q_entries == fast.q_entries
    assert live.oracle_steps == fast.oracle_steps


def test_epsilon_decay_is_monotonic() -> None:
    config = EscapeTrainingConfig(
        episodes=100,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_episodes=100,
    )
    values = [epsilon_for_episode(config, episode) for episode in range(101)]
    assert values[0] == 1.0
    assert values[-1] == 0.1
    assert all(left >= right for left, right in zip(values, values[1:]))
