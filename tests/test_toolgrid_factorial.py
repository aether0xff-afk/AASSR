from __future__ import annotations

from statistics import fmean

import pytest

from aassr_v2.toolgrid_factorial import (
    ACTION_COUNTS,
    GRID_SIZES,
    TOOLGRID_STATE_SIZE,
    ToolGridCodec,
    ToolGridWorld,
    build_actions,
    encode_toolgrid_state,
    run_toolgrid_factorial,
)


def test_action_spaces_have_expected_semantic_tools() -> None:
    for action_count in ACTION_COUNTS:
        actions = build_actions(action_count)
        assert len(actions) == action_count
        assert [item.verb_name for item in actions[:4]] == [
            "move_north",
            "move_south",
            "move_west",
            "move_east",
        ]
        assert len({item.signature for item in actions}) == action_count


def test_oracle_solution_reaches_success_without_tick_limit() -> None:
    for grid_size in GRID_SIZES:
        for action_count in ACTION_COUNTS:
            world = ToolGridWorld(17, grid_size=grid_size, action_count=action_count)
            actions = world.oracle_actions()
            assert len(actions) == world.optimal_steps
            for action in actions:
                world.step(action)
            assert world.success
            assert not world.failed
            assert world.steps == world.optimal_steps


def test_larger_maps_raise_mean_solution_horizon() -> None:
    means = {}
    for grid_size in GRID_SIZES:
        values = [
            ToolGridWorld(seed, grid_size=grid_size, action_count=8).optimal_steps
            for seed in range(200)
        ]
        means[grid_size] = fmean(values)
    assert means[3] < means[5] < means[7]


def test_all_tools_are_required_across_balanced_map_pool() -> None:
    for action_count in ACTION_COUNTS:
        observed = {
            tool
            for seed in range(200)
            for tool in ToolGridWorld(
                seed, grid_size=5, action_count=action_count
            ).required_tools
        }
        assert observed == set(range(action_count - 4))


def test_codec_preserves_dimension_and_action_count() -> None:
    world = ToolGridWorld(3, grid_size=5, action_count=12)
    state = world.snapshot()
    codec = ToolGridCodec(12)
    assert len(encode_toolgrid_state(state)) == TOOLGRID_STATE_SIZE
    decoded = codec.decode(
        codec.encode(state),
        scaffold=state,
        terminal_class=0,
        source="test",
    )
    assert len(decoded.vector) == TOOLGRID_STATE_SIZE
    assert len(decoded.available_actions) == 12
    assert f"phase:{world.phase}" in decoded.facts


def test_wrong_tool_is_irreversible_but_contextually_meaningful() -> None:
    world = ToolGridWorld(11, grid_size=5, action_count=12)
    for action in world.oracle_actions():
        if action.verb_name.startswith("tool_"):
            correct = action
            break
        world.step(action)
    else:  # pragma: no cover
        raise AssertionError("oracle path did not contain a tool action")
    wrong = next(
        action
        for action in world.actions[4:]
        if action.signature != correct.signature
    )
    world.step(wrong)
    assert world.failed
    assert not world.success


def test_small_dqn_run_writes_complete_outputs(tmp_path) -> None:
    pytest.importorskip("torch")
    payload = run_toolgrid_factorial(
        tmp_path,
        condition="dqn",
        seed=7,
        grid_size=3,
        action_count=8,
        transition_budget=32,
        train_map_count=4,
        evaluation_map_count=3,
        checkpoints=(0, 16, 32),
    )
    assert payload["final"]["actual_training_transitions"] >= 32
    assert (tmp_path / "map_manifest.csv").exists()
    assert (tmp_path / "training_episodes.csv").exists()
    assert (tmp_path / "evaluation_episodes.csv").exists()
    assert (tmp_path / "checkpoints.csv").exists()
    assert (tmp_path / "summary.json").exists()
