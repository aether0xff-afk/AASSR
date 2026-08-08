from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from aassr_v2.effect_prophecy import EffectComposedProphecy
from aassr_v2.integrated_agent import ContextualSkillAwareProphecy
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


def _assert_close(left, right, tolerance=1e-10):
    assert len(left) == len(right)
    assert max(abs(a - b) for a, b in zip(left, right, strict=True)) <= tolerance


def test_expected_vector_fast_path_matches_symbolic_effect_prediction() -> None:
    a = Action("a")
    b = Action("b")
    actions = (a, b)
    s0 = _state((0.0, 0.0, 0.2), facts=("root",), actions=actions)
    s1 = _state((0.8, 0.1, 0.3), facts=("root", "a"), actions=actions, progress=0.2)
    s2 = _state((0.2, 0.9, 0.4), facts=("root", "b"), actions=actions, progress=0.3)

    base = TorchGRUProphecy(3, seed=41, device="cpu", dtype="float64")
    skills = SkillLibrary()
    contextual = ContextualSkillAwareProphecy(base, skills)
    effect = EffectComposedProphecy(contextual, minimum_samples=1)
    for _ in range(3):
        base.reset_sequence()
        effect.learn(s0, a, s1)
        base.reset_sequence()
        effect.learn(s0, b, s2)

    view = BatchedIntegratedProphecyView(effect, contextual)
    symbolic = view.predict_batch((s0, s0), (a, b), samples=2)
    expected_symbolic = tuple(expected_prediction_vector(row) for row in symbolic)
    expected_fast = view.expected_vector_batch((s0, s0), (a, b), samples=2)

    for left, right in zip(expected_fast, expected_symbolic, strict=True):
        _assert_close(left, right, tolerance=1e-10)
    assert view.expected_vector_batch_calls == 1
    assert view.expected_vector_batch_rows == 2


def test_validator_prefers_expected_vector_batch() -> None:
    action = Action("advance")
    before = _state((0.0, 0.1), actions=(action,))
    after = _state((0.7, 0.2), facts=("seen",), actions=(action,))
    base = TorchGRUProphecy(2, seed=43, device="cpu", dtype="float64")
    skills = SkillLibrary()
    contextual = ContextualSkillAwareProphecy(base, skills)
    effect = EffectComposedProphecy(contextual, minimum_samples=1)
    effect.learn(before, action, after)
    view = BatchedIntegratedProphecyView(effect, contextual)

    validator = PredictionValidator(samples=2, recent_limit=64)
    transition = ReplayTransition(before, action, after, "v")
    score = validator.evaluate(view, (transition,))
    assert math.isfinite(score.mean_similarity)
    assert validator.batch_calls == 1
    assert validator.expected_vector_calls == 1
    assert view.expected_vector_batch_calls == 1
