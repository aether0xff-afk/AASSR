from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from aassr_v2.effect_prophecy import EffectComposedProphecy
from aassr_v2.gru_prophecy import OnlineGRUProphecy
from aassr_v2.integrated_agent import (
    ContextualSkillAwareProphecy,
    IntegratedProphecyView,
)
from aassr_v2.local_acceleration import BatchedIntegratedProphecyView
from aassr_v2.metrics import expected_prediction_vector
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
