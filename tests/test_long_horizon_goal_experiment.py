from __future__ import annotations

import csv

from aassr_v2.long_horizon_goal_experiment import (
    LongHorizonDependencyWorld,
    run_long_horizon_goal_experiment,
)


def _take_direction(world: LongHorizonDependencyWorld, direction: str) -> float:
    reward = 0.0
    for _ in range(world.room_length):
        action = next(
            item
            for item in world.snapshot().available_actions
            if item.parameters.get("direction") == direction
        )
        reward = world.step(action).reward
    return reward


def test_only_final_checkpoint_emits_external_reward() -> None:
    world = LongHorizonDependencyWorld(7, stage_count=3, room_length=5)
    assert "energy" not in world.snapshot().metadata

    for stage in range(world.stage_count):
        reward = _take_direction(world, world.room.target_direction)
        if stage < world.stage_count - 1:
            assert reward == 0.0
            assert not world.success
        else:
            assert reward == 1.0
            assert world.success

    assert world.snapshot().goal_progress == 1.0
    assert not world.snapshot().available_actions


def test_wrong_branch_ends_only_at_dependency_dead_end() -> None:
    world = LongHorizonDependencyWorld(13, stage_count=3, room_length=5)
    wrong = next(
        direction
        for direction in world.room.choices
        if direction != world.room.target_direction
    )

    for step in range(world.room_length):
        reward = _take_direction(world, wrong) if step == 0 else 0.0
        break

    assert reward == 0.0
    assert world.failed
    assert not world.success
    assert world.path_step == world.room_length
    assert not world.snapshot().available_actions


def test_small_long_horizon_experiment_writes_all_conditions(tmp_path) -> None:
    payload = run_long_horizon_goal_experiment(
        tmp_path,
        seeds=(7,),
        train_episodes=2,
        train_map_count=2,
        evaluation_episodes=1,
        training_tail=1,
        stage_count=2,
        room_length=5,
    )

    conditions = {
        row["condition"]
        for row in payload["summary"]
        if row["phase"] == "evaluation_unseen"
    }
    assert conditions == {
        "policy_only",
        "short_imagination",
        "deep_imagination",
        "goal_maker_executor",
    }

    with (tmp_path / "episodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert (tmp_path / "summary.json").exists()
