from __future__ import annotations

from aassr_v2.collapsing_gridpush_world import CollapsingGridPushWorld
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


def _oracle_action(world: CollapsingGridPushWorld) -> Action:
    if world.phase == 0:
        direction = _direction_toward(world.agent, world.crate)
    elif world.phase == 1:
        direction = _direction_toward(world.crate, world.pit)
    elif world.phase == 2:
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


def test_world_has_no_energy_or_step_limit() -> None:
    world = CollapsingGridPushWorld(7)
    metadata = world.snapshot().metadata
    assert "energy" not in metadata
    assert metadata["termination"] == "irreversible_path_only"


def test_oracle_path_receives_only_final_reward() -> None:
    world = CollapsingGridPushWorld(13)
    rewards: list[float] = []
    while world.snapshot().available_actions:
        outcome = world.step(_oracle_action(world))
        rewards.append(outcome.reward)
    assert world.success
    assert rewards[-1] == 1.0
    assert all(reward == 0.0 for reward in rewards[:-1])
    assert len(rewards) == world.optimal_steps


def test_irreversible_paths_terminate_without_a_tick_counter() -> None:
    for seed in range(20):
        world = CollapsingGridPushWorld(seed)
        actions_taken = 0
        while world.snapshot().available_actions:
            outcome = world.step(world.snapshot().available_actions[-1])
            actions_taken += 1
            assert actions_taken <= world.grid_size * world.grid_size * 5 + 2
            if not outcome.snapshot.available_actions:
                break
        assert world.success or world.failed
