from __future__ import annotations

from aassr_v2.compositional_long_horizon_world import (
    CompositionalLongHorizonWorld,
)
from aassr_v2.effect_prophecy import effect_context_key
from aassr_v2.long_horizon_goal_experiment import CorridorRoom


def _advance_room(
    world: CompositionalLongHorizonWorld,
    direction: str,
) -> None:
    for _ in range(world.room_length):
        action = next(
            item
            for item in world.snapshot().available_actions
            if item.parameters.get("direction") == direction
        )
        world.step(action)


def test_same_local_room_reuses_prophecy_context_across_stages() -> None:
    world = CompositionalLongHorizonWorld(7, stage_count=2, room_length=5)
    repeated = CorridorRoom(("east", "west"), "east")
    world.rooms = (repeated, repeated)

    stage_zero = world.snapshot()
    context_zero = effect_context_key(stage_zero)
    assert "stage:0" in stage_zero.facts

    _advance_room(world, "east")
    stage_one = world.snapshot()

    assert not world.success
    assert "stage:1" in stage_one.facts
    assert "stage:0" not in stage_one.facts
    assert effect_context_key(stage_one) == context_zero


def test_checkpoint_is_observed_without_intermediate_reward() -> None:
    world = CompositionalLongHorizonWorld(13, stage_count=2, room_length=5)
    reward_before_final = 0.0

    for _ in range(world.room_length):
        direction = world.room.target_direction
        action = next(
            item
            for item in world.snapshot().available_actions
            if item.parameters.get("direction") == direction
        )
        reward_before_final = world.step(action).reward

    snapshot = world.snapshot()
    assert reward_before_final == 0.0
    assert "checkpoint_transition" in snapshot.facts
    assert snapshot.goal_progress == 0.0
    assert not world.success
