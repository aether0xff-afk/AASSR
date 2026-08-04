from __future__ import annotations

import json

from aassr_v2.goal_gridpush_experiment import (
    DIRECTIONS,
    GoalGridPushWorld,
    GoalSeparatedAgent,
    run_goal_gridpush_experiment,
)
from aassr_v2.types import Action


def _direction_toward(
    source: tuple[int, int],
    target: tuple[int, int],
) -> str:
    if source[0] < target[0]:
        return "east"
    if source[0] > target[0]:
        return "west"
    if source[1] < target[1]:
        return "south"
    if source[1] > target[1]:
        return "north"
    raise ValueError("source already equals target")


def _oracle_action(world: GoalGridPushWorld) -> Action:
    if world.phase == 0:
        return next(
            action
            for action in world.snapshot().available_actions
            if action.parameters.get("direction")
            == _direction_toward(world.agent, world.crate)
        )
    if world.phase == 1:
        return next(
            action
            for action in world.snapshot().available_actions
            if action.parameters.get("direction")
            == _direction_toward(world.crate, world.pit)
        )
    if world.phase == 2:
        direction = _direction_toward(world.agent, world.key)
    elif world.phase == 4:
        direction = _direction_toward(world.agent, world.door)
    elif world.phase == 6:
        direction = _direction_toward(world.agent, world.exit)
    else:
        return world.snapshot().available_actions[0]
    return next(
        action
        for action in world.snapshot().available_actions
        if action.parameters.get("direction") == direction
    )


def test_gridpush_uses_only_sparse_final_reward() -> None:
    world = GoalGridPushWorld(7)
    rewards = []
    while world.snapshot().available_actions:
        outcome = world.step(_oracle_action(world))
        rewards.append(outcome.reward)
    assert world.success
    assert rewards[-1] == 1.0
    assert all(value == 0.0 for value in rewards[:-1])
    assert len(rewards) == world.optimal_steps


def test_wrong_moves_consume_resource_but_world_remains_finite() -> None:
    world = GoalGridPushWorld(13, slack=0)
    wrong = Action("move", parameters={"direction": next(iter(DIRECTIONS))})
    for _ in range(world.initial_energy + 1):
        if not world.snapshot().available_actions:
            break
        world.step(wrong)
    assert not world.snapshot().available_actions
    assert world.success or world.failed


def test_goal_agent_keeps_maker_and_executor_separate() -> None:
    agent = GoalSeparatedAgent(7)
    assert agent.maker.planner is not agent.executor
    assert agent.maker.planner.scorer.__class__.__name__ == "StateDeltaScorer"


def test_goal_gridpush_smoke_writes_all_conditions(tmp_path) -> None:
    payload = run_goal_gridpush_experiment(
        tmp_path,
        seeds=(7,),
        train_episodes=4,
        train_map_count=2,
        evaluation_episodes=2,
        training_tail=2,
    )
    assert (tmp_path / "episodes.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["goal_maker_executor_separated"] is True
    assert {row["condition"] for row in summary["summary"]} == {
        "policy_only",
        "prophecy_one_step",
        "full_imagination",
        "goal_maker_executor",
    }
