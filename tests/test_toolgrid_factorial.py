from __future__ import annotations

import csv
from statistics import fmean

import pytest

from aassr_v2.autonomous_agent_core import HoldoutTransition
from aassr_v2.toolgrid_factorial_masked import (
    ACTION_COUNTS,
    GRID_SIZES,
    TOOLGRID_STATE_SIZE,
    MaskAwareToolGridDQNAgent,
    OutcomeAwareCalibratedProphecy,
    ProductionToolGridHybridAgent,
    ToolGridCodec,
    ToolGridWorld,
    build_actions,
    encode_toolgrid_state,
    run_toolgrid_factorial,
)
from aassr_v2.types import Prediction, StateSnapshot


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
                assert action in world.snapshot().available_actions
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
            for seed in range(300)
            for tool in ToolGridWorld(
                seed, grid_size=5, action_count=action_count
            ).required_tools
        }
        assert observed == set(range(action_count - 4))


def test_context_masking_separates_navigation_and_tool_branching() -> None:
    world = ToolGridWorld(11, grid_size=5, action_count=12)
    assert 1 <= len(world.snapshot().available_actions) <= 4
    for action in world.oracle_actions():
        if action.verb_name.startswith("tool_"):
            assert len(world.snapshot().available_actions) == 8
            break
        world.step(action)
    else:  # pragma: no cover
        raise AssertionError("oracle path did not contain a tool action")


def test_codec_uses_categorical_tool_identity_and_decodes_raw_schema() -> None:
    world = ToolGridWorld(3, grid_size=5, action_count=12)
    state = world.snapshot()
    codec = ToolGridCodec(12)
    encoded = codec.encode(state)

    assert len(encode_toolgrid_state(state)) == TOOLGRID_STATE_SIZE
    assert codec.dimension == TOOLGRID_STATE_SIZE - 1 + 8
    category = encoded[6:14]
    assert sum(category) == pytest.approx(1.0)
    assert set(category) <= {0.0, 1.0}

    decoded = codec.decode(
        encoded,
        scaffold=state,
        terminal_class=0,
        source="test",
    )
    assert len(decoded.vector) == TOOLGRID_STATE_SIZE
    assert tuple(item.signature for item in decoded.available_actions) == tuple(
        item.signature for item in state.available_actions
    )
    assert decoded.facts == state.facts


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
        for action in world.snapshot().available_actions
        if action.signature != correct.signature
    )
    world.step(wrong)
    assert world.failed
    assert not world.success


class _Holdout:
    def __init__(self, items):
        self._items = list(items)


class _ExactBase:
    def __init__(self, predicted: StateSnapshot) -> None:
        self.predicted = predicted
        self.gradient_updates = 0

    def predict(self, state, action, *, samples):
        del state, action, samples
        return (Prediction(self.predicted, 1.0, "dummy"),)

    def confidence(self, state, action):
        del state, action
        return 1.0

    def learn(self, state, action, actual_next_state):
        del state, action, actual_next_state


def test_calibration_does_not_cache_pre_ready_zero() -> None:
    world = ToolGridWorld(2, grid_size=3, action_count=8)
    before = world.snapshot()
    action = before.available_actions[0]
    after = world.step(action).snapshot
    transition = HoldoutTransition(before, action, after)
    holdout = _Holdout([transition] * 7)
    calibrated = OutcomeAwareCalibratedProphecy(
        _ExactBase(after),
        holdout,
        build_actions(8),
        minimum_count=8,
    )

    assert calibrated._calibration(action) == 0.0
    holdout._items.append(transition)
    assert calibrated._calibration(action) > 0.9


def test_calibration_distinguishes_terminal_success_from_failure() -> None:
    world = ToolGridWorld(3, grid_size=3, action_count=8)
    before = world.snapshot()
    action = before.available_actions[0]
    vector = before.vector
    success = StateSnapshot(
        vector=vector,
        facts=frozenset({"success"}),
        available_actions=(),
        goal_progress=1.0,
        metadata={},
    )
    failure = StateSnapshot(
        vector=vector,
        facts=frozenset({"failed"}),
        available_actions=(),
        goal_progress=0.0,
        metadata={},
    )
    transition = HoldoutTransition(before, action, success)
    calibrated = OutcomeAwareCalibratedProphecy(
        _ExactBase(failure),
        _Holdout([transition] * 8),
        build_actions(8),
        minimum_count=8,
    )
    assert calibrated._calibration(action) == 0.0


def test_dqn_bellman_target_masks_context_invalid_actions() -> None:
    torch = pytest.importorskip("torch")
    agent = MaskAwareToolGridDQNAgent(
        7,
        action_count=8,
        train_transitions=32,
        batch_size=1,
        warmup_steps=1,
    )
    final_layer = agent.target[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.copy_(
            torch.tensor([100.0, 90.0, 80.0, 70.0, 1.0, 2.0, 3.0, 4.0])
        )
    observations = torch.zeros((1, TOOLGRID_STATE_SIZE))
    mask = torch.tensor([[False, False, False, False, True, False, False, False]])
    nonterminal = torch.tensor([0.0])
    terminal = torch.tensor([1.0])

    assert agent._masked_next_values(observations, mask, nonterminal).item() == 1.0
    assert agent._masked_next_values(observations, mask, terminal).item() == 0.0


def test_policy_only_and_imagination_share_training_trajectory() -> None:
    pytest.importorskip("torch")
    policy_only = ProductionToolGridHybridAgent(
        7,
        action_count=8,
        train_transitions=64,
        use_imagination=False,
    )
    imagination = ProductionToolGridHybridAgent(
        7,
        action_count=8,
        train_transitions=64,
        use_imagination=True,
    )
    left = ToolGridWorld(7001, grid_size=3, action_count=8)
    right = ToolGridWorld(7001, grid_size=3, action_count=8)
    policy_only.begin_episode(training=True)
    imagination.begin_episode(training=True)

    for step in range(12):
        left_state = left.snapshot()
        right_state = right.snapshot()
        assert left_state == right_state
        left_decision = policy_only.select_action(
            left_state,
            episode=step,
            training=True,
        )
        right_decision = imagination.select_action(
            right_state,
            episode=step,
            training=True,
        )
        assert left_decision.action.signature == right_decision.action.signature
        assert not right_decision.used_imagination
        left_outcome = left.step(left_decision.action)
        right_outcome = right.step(right_decision.action)
        policy_only.observe(left_state, left_decision.action, left_outcome)
        imagination.observe(right_state, right_decision.action, right_outcome)
        if left.success or left.failed:
            break

    policy_only.end_episode(success=left.success, training=True)
    imagination.end_episode(success=right.success, training=True)
    assert left.success == right.success
    assert left.failed == right.failed
    assert policy_only.dqn.environment_steps == imagination.dqn.environment_steps
    assert policy_only.base_prophecy.observations == imagination.base_prophecy.observations


def test_small_dqn_run_uses_exact_budget_and_corrected_metrics(tmp_path) -> None:
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
    assert payload["final"]["actual_training_transitions"] == 32
    assert payload["config"]["exact_transition_budget"] is True
    assert payload["config"]["dqn_target_action_masking"] is True

    with (tmp_path / "training_episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        training = list(csv.DictReader(handle))
    assert all(
        int(row["environment_steps_total"])
        <= int(row["checkpoint_transition_target"])
        for row in training
    )

    with (tmp_path / "map_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        manifest = list(csv.DictReader(handle))
    assert all(
        row["effective_branching_factor"] == row["tool_count"]
        for row in manifest
    )
    assert all(
        row["semantic_branching_factor"] == row["tool_count"]
        for row in manifest
    )

    for name in (
        "map_manifest.csv",
        "training_episodes.csv",
        "evaluation_episodes.csv",
        "checkpoints.csv",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
