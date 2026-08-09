from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_validation import CurrentVectorPredictionValidator
from aassr_v2.metrics import expected_prediction_vector
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


def test_current_builder_installs_vector_holdout_validator() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        device="cpu",
    )
    assert isinstance(agent.evaluator.validator, CurrentVectorPredictionValidator)
    assert agent.current_fast_validation is True


def test_current_vector_fast_path_matches_symbolic_decoded_vectors() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        device="cpu",
    )
    rows = _rows()
    states = tuple(item.state for item in rows)
    actions = tuple(item.action for item in rows)

    # Exercise the post-warmup Neural Delta decode without paying for optimizer
    # warmup in this unit test. Model weights remain the exact same weights for
    # both paths below.
    agent.base_neural_prophecy.observations = agent.base_neural_prophecy.config.warmup_steps

    fast = agent.evaluator.validator._decoded_expected_vectors(states, actions)
    symbolic_rows = agent.prophecy.predict_batch(states, actions, samples=4)
    symbolic = tuple(expected_prediction_vector(row) for row in symbolic_rows)

    assert len(fast) == len(symbolic)
    for left, right in zip(fast, symbolic, strict=True):
        assert left == pytest.approx(right, abs=0.0, rel=0.0)


def test_current_vector_validator_matches_generic_similarity_score() -> None:
    agent = build_current_pentest_aassr_core(
        seed=11,
        train_transitions=256,
        device="cpu",
    )
    rows = _rows()
    for item in rows:
        agent.evaluator.replay._holdout.append(item)

    # Warmup behavior is also part of the exact contract: both paths predict the
    # current state itself until Neural Delta has enough observations.
    fast = agent.evaluator.validator.evaluate(agent.prophecy, rows)

    from aassr_v2.replay import PredictionValidator

    reference = PredictionValidator(samples=4, recent_limit=64)
    slow = reference.evaluate(agent.prophecy, rows)
    assert fast.count == slow.count
    assert fast.mean_similarity == pytest.approx(slow.mean_similarity, abs=0.0, rel=0.0)
    diagnostics = agent.evaluator.validator.runtime_diagnostics()
    assert diagnostics["current_vector_fast_calls"] == 0  # warmup uses state vectors directly
    assert diagnostics["current_symbolic_fallback_calls"] == 0
