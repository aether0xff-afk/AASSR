from __future__ import annotations

import pytest

from aassr_v2.baseline_efficiency_portable import (
    CHOICE_ACTIONS,
    BenchmarkGridPushWorld,
    solvable_map_seeds,
)
from aassr_v2.benchmark_neural_prophecy import BenchmarkGridPushCodec
from aassr_v2.neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy


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
