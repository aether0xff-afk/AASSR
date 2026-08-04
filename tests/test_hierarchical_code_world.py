from __future__ import annotations

from aassr_v2.effect_prophecy import effect_context_key
from aassr_v2.hierarchical_code_world import HierarchicalCodeWorld


def _action_for_bit(world: HierarchicalCodeWorld, bit: int):
    return next(
        action
        for action in world.snapshot().available_actions
        if action.parameters.get("bit") == bit
    )


def _correct_action(world: HierarchicalCodeWorld):
    return _action_for_bit(world, world.code[world.position])


def test_only_last_of_twenty_checkpoints_emits_reward() -> None:
    world = HierarchicalCodeWorld(7, stage_count=20, room_length=4)
    rewards = []

    while world.snapshot().available_actions:
        rewards.append(world.step(_correct_action(world)).reward)

    assert world.success
    assert not world.failed
    assert len(rewards) == 80
    assert rewards[:-1] == [0.0] * 79
    assert rewards[-1] == 1.0
    assert world.snapshot().goal_progress == 1.0


def test_wrong_choice_is_hidden_until_fourth_entry() -> None:
    world = HierarchicalCodeWorld(13, stage_count=20, room_length=4)
    wrong = 1 - world.code[0]

    first = world.step(_action_for_bit(world, wrong))

    assert not first.error
    assert not world.failed
    assert "failed" not in first.snapshot.facts
    assert first.reward == 0.0

    while world.position < world.room_length:
        world.step(_correct_action(world))

    assert world.failed
    assert not world.success
    assert not world.snapshot().available_actions
    assert "failed" in world.snapshot().facts


def test_local_prophecy_context_is_reused_across_stage_identity() -> None:
    world = HierarchicalCodeWorld(21, stage_count=2, room_length=4)
    repeated_code = (1, 0, 1, 1)
    world.codes = (repeated_code, repeated_code)

    stage_zero = world.snapshot()
    context_zero = effect_context_key(stage_zero)
    assert "stage:0" in stage_zero.facts

    for _ in range(world.room_length):
        world.step(_correct_action(world))

    stage_one = world.snapshot()
    assert "stage:1" in stage_one.facts
    assert "stage:0" not in stage_one.facts
    assert effect_context_key(stage_one) == context_zero
    assert "checkpoint_transition" in stage_one.facts
