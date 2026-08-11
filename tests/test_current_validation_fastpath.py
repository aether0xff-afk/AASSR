from __future__ import annotations

from statistics import fmean

import pytest

pytest.importorskip("torch")

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_semantic_calibration import (
    SemanticPredictionValidator,
    probability_weighted_semantic_score,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.replay import ReplayTransition


def _rows(count: int = 8):
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0])
    output = []
    for index in range(count):
        before = world.snapshot()
        action = before.available_actions[index % len(before.available_actions)]
        outcome = world.step(action)
        output.append(ReplayTransition(before, action, outcome.snapshot, f"row-{index}"))
        if world.success or world.failed or world.rate_limited:
            world = TransferDiagnosticWorld(90_001 + index + 1, stage=TRANSFER_STAGES[0])
    return tuple(output)


def test_current_builder_installs_semantic_probability_holdout_validator() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        device="cpu",
    )
    assert isinstance(agent.evaluator.validator, SemanticPredictionValidator)
    assert agent.current_semantic_validation is True
    assert not hasattr(agent.evaluator.validator, "_decoded_expected_vectors")


def test_current_semantic_validator_matches_explicit_probability_weighted_score() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        device="cpu",
    )
    rows = _rows()
    states = tuple(item.state for item in rows)
    actions = tuple(item.action for item in rows)
    validator = agent.evaluator.validator

    # Exercise the learned stochastic path without paying optimizer warmup here.
    # The comparison is semantic: no concrete raw-vector reconstruction is used.
    agent.base_neural_prophecy.observations = agent.base_neural_prophecy.config.warmup_steps
    predicted = agent.prophecy.predict_batch(
        states,
        actions,
        samples=validator.samples,
    )
    expected = fmean(
        probability_weighted_semantic_score(predictions, item.next_state)
        for item, predictions in zip(rows, predicted, strict=True)
    )

    actual = validator.evaluate(agent.prophecy, rows)
    assert actual.count == len(rows)
    assert actual.mean_similarity == pytest.approx(expected, abs=0.0, rel=0.0)
    diagnostics = validator.runtime_diagnostics()
    assert diagnostics["batch_calls"] == 1
    assert diagnostics["expected_vector_calls"] == 0


def test_current_semantic_validator_warmup_uses_semantic_batch_not_raw_vectors() -> None:
    agent = build_current_pentest_aassr_core(
        seed=11,
        train_transitions=256,
        device="cpu",
    )
    rows = _rows()
    validator = agent.evaluator.validator

    score = validator.evaluate(agent.prophecy, rows)
    predictions = agent.prophecy.predict_batch(
        tuple(item.state for item in rows),
        tuple(item.action for item in rows),
        samples=validator.samples,
    )
    expected = fmean(
        probability_weighted_semantic_score(items, row.next_state)
        for row, items in zip(rows, predictions, strict=True)
    )

    assert score.count == len(rows)
    assert score.mean_similarity == pytest.approx(expected, abs=0.0, rel=0.0)
    diagnostics = validator.runtime_diagnostics()
    assert diagnostics["batch_calls"] == 1
    assert diagnostics["cache_misses"] == 1
    assert diagnostics["expected_vector_calls"] == 0
