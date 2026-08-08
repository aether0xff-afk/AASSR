from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from aassr_v2.effect_prophecy import EffectComposedProphecy
from aassr_v2.gru_prophecy import OnlineGRUProphecy
from aassr_v2.imagination_tree import (
    ImaginationConfig,
    ImaginationTree,
    StateDeltaScorer,
)
from aassr_v2.integrated_agent import (
    ContextualSkillAwareProphecy,
    IntegratedProphecyView,
)
from aassr_v2.local_acceleration import BatchedIntegratedProphecyView
from aassr_v2.metrics import expected_prediction_vector
from aassr_v2.native_batching import (
    DepthBatchedImaginationTree,
    DepthBatchedProphecyView,
)
from aassr_v2.policy import WeightedPolicy
from aassr_v2.replay import PredictionValidator, ReplayTransition
from aassr_v2.skills import SkillLibrary
from aassr_v2.torch_gru_prophecy import TorchGRUProphecy
from aassr_v2.types import Action, StateSnapshot


def _state(vector, *, facts=(), actions=(), progress=0.0):
    return StateSnapshot(
        tuple(float(value) for value in vector),
        facts=frozenset(facts),
        available_actions=tuple(actions),
        goal_progress=float(progress),
    )


def _assert_close(left, right, tolerance=1e-9):
    assert len(left) == len(right)
    if not left:
        return
    assert max(abs(a - b) for a, b in zip(left, right, strict=True)) <= tolerance


def test_torch_float64_matches_canonical_custom_gru_update() -> None:
    action = Action("advance", parameters={"slot": "x"})
    state = _state((0.2, -0.4, 0.7), actions=(action,))
    next_state = _state((0.5, -0.1, 0.9), facts=("seen",), actions=(action,))

    python_model = OnlineGRUProphecy(3, seed=23)
    torch_model = TorchGRUProphecy(
        3,
        seed=23,
        device="cpu",
        dtype="float64",
        allow_cpu_fallback=False,
    )

    _assert_close(
        python_model.predict_vector(state, action),
        torch_model.predict_vector(state, action),
        tolerance=1e-11,
    )
    python_model.learn(state, action, next_state)
    torch_model.learn(state, action, next_state)
    _assert_close(
        python_model.predict_vector(state, action),
        torch_model.predict_vector(state, action),
        tolerance=1e-9,
    )
    assert python_model.training_stats.updates == torch_model.training_stats.updates == 1
    assert math.isclose(
        python_model.training_stats.last_loss,
        torch_model.training_stats.last_loss,
        rel_tol=1e-9,
        abs_tol=1e-11,
    )


def test_torch_float64_matches_recurrent_advance_then_learning() -> None:
    a = Action("a")
    b = Action("b")
    c = Action("c")
    s0 = _state((0.1, -0.2, 0.3), actions=(a, b, c))
    s1 = _state((0.4, 0.2, -0.1), facts=("one",), actions=(a, b, c))
    s2 = _state((-0.2, 0.5, 0.6), facts=("one", "held"), actions=(a, b, c))
    s3 = _state((0.7, -0.3, 0.2), facts=("done",), actions=(a, b, c))

    python_model = OnlineGRUProphecy(3, seed=31)
    torch_model = TorchGRUProphecy(
        3,
        seed=31,
        device="cpu",
        dtype="float64",
        allow_cpu_fallback=False,
    )

    python_model.learn(s0, a, s1)
    torch_model.learn(s0, a, s1)
    python_model.advance_sequence(s1, b)
    torch_model.advance_sequence(s1, b)
    python_model.learn(s2, c, s3)
    torch_model.learn(s2, c, s3)

    _assert_close(
        python_model.predict_vector(s2, c),
        torch_model.predict_vector(s2, c),
        tolerance=1e-9,
    )
    assert python_model.training_stats.updates == torch_model.training_stats.updates == 2
    assert math.isclose(
        python_model.training_stats.last_loss,
        torch_model.training_stats.last_loss,
        rel_tol=1e-9,
        abs_tol=1e-11,
    )


def test_torch_predict_batch_matches_scalar_prediction() -> None:
    a = Action("a")
    b = Action("b")
    s0 = _state((0.0, 0.0), actions=(a, b))
    s1 = _state((1.0, 0.0), facts=("a",), actions=(a, b))
    s2 = _state((0.0, 1.0), facts=("b",), actions=(a, b))
    model = TorchGRUProphecy(2, seed=5, device="cpu", dtype="float64")
    model.reset_sequence()
    model.learn(s0, a, s1)
    model.reset_sequence()
    model.learn(s0, b, s2)

    batch = model.predict_batch((s0, s0), (a, b), samples=2)
    scalar = (
        model.predict(s0, a, samples=2),
        model.predict(s0, b, samples=2),
    )
    for left, right in zip(batch, scalar, strict=True):
        _assert_close(
            expected_prediction_vector(left),
            expected_prediction_vector(right),
            tolerance=1e-12,
        )
        assert [item.source for item in left] == [item.source for item in right]
        _assert_close(
            tuple(item.probability for item in left),
            tuple(item.probability for item in right),
            tolerance=1e-12,
        )


def test_validator_batches_and_reuses_identical_model_holdout_score() -> None:
    action = Action("advance")
    state = _state((0.0, 0.0), actions=(action,))
    next_state = _state((1.0, 0.0), facts=("done",), actions=(action,))
    model = TorchGRUProphecy(2, seed=7, device="cpu", dtype="float64")
    transition = ReplayTransition(state, action, next_state, "h-1")
    validator = PredictionValidator(samples=1, recent_limit=64)

    first = validator.evaluate(model, (transition,))
    second = validator.evaluate(model, (transition,))
    assert first == second
    assert validator.cache_hits == 1
    assert validator.cache_misses == 1
    assert validator.batch_calls == 1

    model.learn(state, action, next_state)
    validator.evaluate(model, (transition,))
    assert validator.cache_misses == 2
    assert validator.batch_calls == 2


def test_batched_integrated_view_preserves_effect_composition() -> None:
    action = Action("advance")
    before = _state((0.0, 0.0), facts=("ready",), actions=(action,))
    after = _state((1.0, 0.0), facts=("ready", "done"), actions=(action,), progress=0.5)
    base = TorchGRUProphecy(2, seed=9, device="cpu", dtype="float64")
    skills = SkillLibrary()
    contextual = ContextualSkillAwareProphecy(base, skills)
    effect = EffectComposedProphecy(contextual, minimum_samples=1)
    effect.learn(before, action, after)

    scalar = IntegratedProphecyView(effect, contextual)
    batched = BatchedIntegratedProphecyView(effect, contextual)
    expected = scalar.predict(before, action, samples=2)
    actual = batched.predict_batch((before,), (action,), samples=2)[0]

    _assert_close(
        expected_prediction_vector(expected),
        expected_prediction_vector(actual),
        tolerance=1e-12,
    )
    assert [item.source for item in expected] == [item.source for item in actual]
    _assert_close(
        tuple(item.probability for item in expected),
        tuple(item.probability for item in actual),
        tolerance=1e-12,
    )


def test_depth_batched_imagination_matches_scalar_tree() -> None:
    a = Action("a")
    b = Action("b")
    c = Action("c")
    actions = (a, b, c)
    s0 = _state((0.0, 0.0), facts=("root",), actions=actions)
    s1 = _state((0.7, 0.1), facts=("root", "a"), actions=actions, progress=0.1)
    s2 = _state((0.2, 0.8), facts=("root", "b"), actions=actions, progress=0.2)
    s3 = _state((0.5, 0.5), facts=("root", "c"), actions=actions, progress=0.15)

    base = TorchGRUProphecy(2, seed=19, device="cpu", dtype="float64")
    skills = SkillLibrary()
    contextual = ContextualSkillAwareProphecy(base, skills)
    effect = EffectComposedProphecy(contextual, minimum_samples=1)
    for _ in range(2):
        base.reset_sequence()
        effect.learn(s0, a, s1)
        base.reset_sequence()
        effect.learn(s0, b, s2)
        base.reset_sequence()
        effect.learn(s0, c, s3)

    scalar_view = IntegratedProphecyView(effect, contextual)
    batch_base = BatchedIntegratedProphecyView(effect, contextual)
    batch_view = DepthBatchedProphecyView(batch_base)
    config = ImaginationConfig(
        branching_factor=2,
        maximum_depth=2,
        beam_width=4,
        outcome_samples=1,
        minimum_path_confidence=0.0,
        update_policy=False,
        expand_all_root_actions=True,
    )
    scorer = StateDeltaScorer()
    scalar_policy = WeightedPolicy()
    batch_policy = WeightedPolicy()
    scalar_tree = ImaginationTree(
        scalar_policy,
        scalar_view,
        config=config,
        scorer=scorer,
    )
    batch_tree = DepthBatchedImaginationTree(
        batch_policy,
        batch_view,
        config=config,
        scorer=scorer,
    )

    expected = scalar_tree.plan(s0)
    actual = batch_tree.plan(s0)

    assert actual.chosen_action.signature == expected.chosen_action.signature
    assert actual.expanded_nodes == expected.expanded_nodes
    assert actual.maximum_depth_reached == expected.maximum_depth_reached
    assert len(actual.nodes) == len(expected.nodes)
    assert len(actual.root_evaluations) == len(expected.root_evaluations)
    for left, right in zip(actual.root_evaluations, expected.root_evaluations, strict=True):
        assert left.action.signature == right.action.signature
        assert left.best_path == right.best_path
        assert left.best_leaf_id == right.best_leaf_id
        assert math.isclose(left.aggregate_value, right.aggregate_value, abs_tol=1e-12)
        _assert_close(left.leaf_values, right.leaf_values, tolerance=1e-12)
    for left, right in zip(actual.nodes, expected.nodes, strict=True):
        assert left.node_id == right.node_id
        assert left.parent_id == right.parent_id
        assert left.depth == right.depth
        assert left.action_path == right.action_path
        assert left.state_path == right.state_path
        assert left.terminal_reason == right.terminal_reason
        assert math.isclose(left.cumulative_value, right.cumulative_value, abs_tol=1e-12)
        assert math.isclose(left.cumulative_confidence, right.cumulative_confidence, abs_tol=1e-12)

    runtime = batch_view.runtime_diagnostics()
    assert runtime["imagination_batch_calls"] >= 1
    assert runtime["imagination_batch_rows"] >= len(actions)
