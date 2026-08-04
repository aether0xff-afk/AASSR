from __future__ import annotations

from aassr_v2.effect_prophecy import effect_context_key
from aassr_v2.hierarchical_code_world import (
    HierarchicalCodeWorld,
    next_code,
)


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


def test_checkpoint_uses_one_reusable_code_transition_rule() -> None:
    world = HierarchicalCodeWorld(21, stage_count=20, room_length=4)
    before_code = world.code

    for _ in range(world.room_length):
        world.step(_correct_action(world))

    assert world.code == next_code(before_code)
    assert world.snapshot().metadata["stage"] == 1
    assert "checkpoint_transition" in world.snapshot().facts
    assert not any(fact.startswith("stage:") for fact in world.snapshot().facts)


def test_same_local_state_has_same_prophecy_context_at_different_stages() -> None:
    world = HierarchicalCodeWorld(42, stage_count=20, room_length=4)
    first = world.snapshot()
    first_context = effect_context_key(first)

    # The binary-increment rule cycles after all sixteen four-bit codes.
    world.stage = 16
    world.entered.clear()
    world.just_opened_checkpoint = False
    repeated = world.snapshot()

    assert repeated.metadata["stage"] == 16
    assert repeated.vector == first.vector
    assert repeated.facts == first.facts
    assert effect_context_key(repeated) == first_context
