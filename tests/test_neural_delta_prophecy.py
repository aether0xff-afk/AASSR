from __future__ import annotations

from types import SimpleNamespace

import pytest

from aassr_v2.autonomous_agent_core import HoldoutTransition
from aassr_v2.baseline_efficiency_portable import (
    CHOICE_ACTIONS,
    BenchmarkGridPushWorld,
    solvable_map_seeds,
)
from aassr_v2.benchmark_neural_prophecy import (
    BenchmarkGridPushCodec,
    EmpiricallyCalibratedProphecy,
)
from aassr_v2.bottleneck_sota_portable import BenchmarkOracleProphecy
from aassr_v2.metrics import structured_prediction_similarity
from aassr_v2.neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy
from aassr_v2.types import Prediction, StateSnapshot


def test_benchmark_codec_roundtrip_preserves_explicit_state() -> None:
    seed = solvable_map_seeds(9100, 1)[0]
    state = BenchmarkGridPushWorld(seed).snapshot()
    codec = BenchmarkGridPushCodec()
    encoded = codec.encode(state)
    decoded = codec.decode(
        encoded,
        scaffold=state,
        terminal_class=0,
        source="test",
    )
    assert decoded.vector == state.vector
    assert decoded.facts == state.facts
    assert decoded.available_actions == state.available_actions


def test_structured_similarity_penalizes_wrong_terminal_structure() -> None:
    seed = solvable_map_seeds(9150, 1)[0]
    actual = BenchmarkGridPushWorld(seed).snapshot()
    exact = Prediction(actual, 1.0, "exact")
    wrong = StateSnapshot(
        vector=actual.vector,
        facts=frozenset({*actual.facts, "failed"}),
        available_actions=(),
        goal_progress=0.0,
        metadata=actual.metadata,
    )
    wrong_prediction = Prediction(wrong, 1.0, "wrong-terminal")
    assert structured_prediction_similarity((exact,), actual) == pytest.approx(1.0)
    assert structured_prediction_similarity(
        (wrong_prediction,),
        actual,
    ) < 0.25


def test_holdout_calibration_requires_untrained_evidence() -> None:
    seed = solvable_map_seeds(9170, 1)[0]
    world = BenchmarkGridPushWorld(seed)
    action = CHOICE_ACTIONS[0]
    before = world.snapshot()
    after = world.step(action).snapshot
    base = BenchmarkOracleProphecy()
    empty = EmpiricallyCalibratedProphecy(
        base,
        SimpleNamespace(_items=[]),
        minimum_count=2,
        refresh_stride=1,
    )
    assert empty.coverage(before, (action,)) == 0.0

    holdout = SimpleNamespace(
        _items=[
            HoldoutTransition(before, action, after),
            HoldoutTransition(before, action, after),
        ]
    )
    calibrated = EmpiricallyCalibratedProphecy(
        base,
        holdout,
        minimum_count=2,
        refresh_stride=1,
    )
    prediction = calibrated.predict(before, action, samples=1)[0]
    assert prediction.probability == pytest.approx(1.0)
    assert calibrated.coverage(before, (action,)) == pytest.approx(1.0)


def test_neural_delta_prophecy_learns_and_predicts_snapshot() -> None:
    pytest.importorskip("torch")
    seed = solvable_map_seeds(9200, 1)[0]
    prophecy = NeuralDeltaProphecy(
        BenchmarkGridPushCodec(),
        config=NeuralDeltaConfig(
            hidden_units=32,
            ensemble_size=2,
            replay_capacity=128,
            batch_size=8,
            warmup_steps=8,
            gradient_steps_per_observation=1,
            confidence_prior=8.0,
        ),
        seed=7,
    )
    for index in range(24):
        world = BenchmarkGridPushWorld(seed + index)
        before = world.snapshot()
        action = CHOICE_ACTIONS[index % len(CHOICE_ACTIONS)]
        after = world.step(action).snapshot
        prophecy.learn(before, action, after)
    prediction = prophecy.predict(
        BenchmarkGridPushWorld(seed).snapshot(),
        CHOICE_ACTIONS[0],
        samples=1,
    )[0]
    assert len(prediction.next_state.vector) == 16
    assert 0.0 < prediction.probability <= 1.0
    assert prediction.source == "neural-delta:ensemble"
    stats = prophecy.stats()
    assert stats.observations == 24
    assert stats.gradient_updates > 0
    assert stats.parameter_count > 0