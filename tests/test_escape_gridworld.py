from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aassr_v2.escape_gridworld import (
    EscapeGridWorld,
    generate_escape_grid,
    oracle_plan,
)
from aassr_v2.escape_reporting import describe, rolling_mean
from aassr_v2.escape_training import (
    EscapeTrainingConfig,
    TrainingMode,
    TrainingRuntime,
    success_score_multiplier,
    train_escape_agent,
)
from aassr_v2.types import Action


def _execute_oracle(seed: int, color_count: int = 3) -> EscapeGridWorld:
    spec = generate_escape_grid(
        seed,
        color_count=color_count,
        distractor_boxes=2,
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


def test_episode_has_no_automatic_tick_timeout() -> None:
    spec = generate_escape_grid(2, color_count=1, distractor_boxes=0)
    environment = EscapeGridWorld(spec)
    for _ in range(500):
        environment.step(Action("move", destination="west"))
    assert not environment.success
    assert not environment.done
    assert environment.steps == 500


def test_success_score_rewards_shorter_completed_routes() -> None:
    assert success_score_multiplier(20, 20) == pytest.approx(2.0)
    assert success_score_multiplier(20, 40) == pytest.approx(1.5)
    assert success_score_multiplier(20, 200) == pytest.approx(1.1)
    assert success_score_multiplier(20, 40) > success_score_multiplier(20, 200)


def test_runtime_mode_can_change_without_restarting() -> None:
    runtime = TrainingRuntime(TrainingMode.FAST)
    assert runtime.mode is TrainingMode.FAST
    runtime.set_mode(TrainingMode.LIVE)
    assert runtime.mode is TrainingMode.LIVE
    runtime.set_mode(TrainingMode.FAST)
    assert runtime.mode is TrainingMode.FAST


def test_live_and_fast_modes_run_the_same_learning_configuration(tmp_path: Path) -> None:
    config = EscapeTrainingConfig(
        episodes=8,
        seed=19,
        color_count=1,
        distractor_boxes=0,
        use_imagination=False,
        live_step_delay=0.0,
        fast_progress_interval=2,
        minimum_holdout_count=2,
        save_episode_checkpoints=False,
    )
    live = train_escape_agent(
        config,
        mode=TrainingMode.LIVE,
        output_dir=tmp_path / "live",
    )
    fast = train_escape_agent(
        config,
        mode=TrainingMode.FAST,
        output_dir=tmp_path / "fast",
    )
    assert live.episodes == fast.episodes == 8
    assert live.successes == fast.successes
    assert live.policy_entries == fast.policy_entries
    assert live.oracle_steps == fast.oracle_steps
    assert live.imagination_decisions == fast.imagination_decisions == 0
    assert [record.steps for record in live.episode_records] == [
        record.steps for record in fast.episode_records
    ]
    assert [record.score for record in live.episode_records] == [
        record.score for record in fast.episode_records
    ]


def test_complete_session_persists_steps_episodes_checkpoints_and_charts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recorded"
    config = EscapeTrainingConfig(
        episodes=2,
        seed=19,
        color_count=1,
        distractor_boxes=0,
        use_imagination=False,
        live_step_delay=0.0,
        minimum_holdout_count=2,
        save_episode_checkpoints=True,
    )
    summary = train_escape_agent(
        config,
        mode=TrainingMode.FAST,
        output_dir=output,
    )

    expected_files = (
        "session.json",
        "world.json",
        "steps.jsonl",
        "episodes.csv",
        "episodes.jsonl",
        "mode_switches.jsonl",
        "summary.json",
        "summary.txt",
        "statistics.json",
        "session.log",
    )
    for name in expected_files:
        assert (output / name).exists(), name

    step_lines = (output / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(step_lines) == summary.total_steps
    first_step = json.loads(step_lines[0])
    assert {
        "timestamp_utc",
        "episode",
        "step",
        "session_elapsed_seconds",
        "episode_elapsed_seconds",
        "tick_wall_seconds",
        "compute_seconds",
        "mode",
        "epsilon",
        "action",
        "event",
        "decision",
        "metrics",
        "before",
        "after",
    } <= first_step.keys()
    assert "available_actions" in first_step["before"]
    assert "vector" in first_step["after"]

    with (output / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(int(row["steps"]) > 0 for row in rows)
    assert all(float(row["duration_seconds"]) >= 0.0 for row in rows)
    assert all(json.loads(row["action_counts"]) for row in rows)

    for episode in range(1, 3):
        assert (output / "checkpoints" / f"episode_{episode:06d}.json.gz").exists()
    assert (output / "checkpoints" / "latest.json.gz").exists()
    assert (output / "checkpoints" / "final.json.gz").exists()

    expected_charts = (
        "episode_steps.svg",
        "episode_scores.svg",
        "episode_duration.svg",
        "prediction_and_holdout.svg",
        "intrinsic_value.svg",
        "imagination_usage.svg",
        "errors_and_repeats.svg",
        "action_distribution.svg",
        "event_distribution.svg",
    )
    for name in expected_charts:
        path = output / "charts" / name
        assert path.exists(), name
        assert "<svg" in path.read_text(encoding="utf-8")

    saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["episodes"] == 2
    assert saved_summary["total_steps"] == summary.total_steps
    assert saved_summary["statistics"]["steps"]["count"] == 2


def test_descriptive_statistics_and_rolling_mean() -> None:
    result = describe([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["minimum"] == 1.0
    assert result["maximum"] == 4.0
    assert rolling_mean([1.0, 2.0, 3.0], window=2) == pytest.approx(
        [1.0, 1.5, 2.5]
    )


def test_epsilon_decay_is_monotonic() -> None:
    from aassr_v2.escape_training import epsilon_for_episode

    config = EscapeTrainingConfig(
        episodes=100,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_episodes=100,
        save_episode_checkpoints=False,
    )
    values = [epsilon_for_episode(config, episode) for episode in range(101)]
    assert values[0] == 1.0
    assert values[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(values, values[1:]))
