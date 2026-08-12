from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from aassr_v2.current_relational_codec import ACTION_SLOT_COUNT, TERMINAL_CLASSES
from aassr_v2.current_runtime_performance import (
    _install_indexed_calibration,
    _install_status_mixture_fast_path,
    PERFORMANCE_CONTRACT,
)
from aassr_v2.current_semantic_calibration import SemanticCalibratedProphecy
from aassr_v2.current_status_models import (
    STATUS_SIZE,
    STATUS_START_INDEX,
    StatusAwareConditionalMixtureRelationalProphecy,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.replay import ReplayBuffer, ReplayTransition


def _populate_model(model: StatusAwareConditionalMixtureRelationalProphecy) -> None:
    descriptor_size = model.component_size - ACTION_SLOT_COUNT - TERMINAL_CLASSES
    for row_index in range(max(model.config.batch_size, model.config.warmup_steps)):
        inputs = tuple(
            ((row_index + column) % 11) / 10.0
            for column in range(model.input_size)
        )
        descriptor = [0.0] * descriptor_size
        descriptor[STATUS_START_INDEX + (row_index % STATUS_SIZE)] = 1.0
        if STATUS_START_INDEX:
            descriptor[row_index % STATUS_START_INDEX] = 1.0
        mask = [0.0] * ACTION_SLOT_COUNT
        mask[row_index % ACTION_SLOT_COUNT] = 1.0
        terminal = row_index % TERMINAL_CLASSES
        model.replay.append((inputs, tuple(descriptor), tuple(mask), terminal))
        model._status_observation_counts[row_index % STATUS_SIZE] += 1
    model.observations = max(model.config.batch_size, model.config.warmup_steps)


def test_status_mixture_training_fast_path_preserves_parameter_update() -> None:
    import torch

    baseline = StatusAwareConditionalMixtureRelationalProphecy(seed=1234, device="cpu")
    optimized = StatusAwareConditionalMixtureRelationalProphecy(seed=1234, device="cpu")
    _populate_model(baseline)
    _populate_model(optimized)
    _install_status_mixture_fast_path(optimized)

    baseline._train_step()
    optimized._train_step()

    assert baseline.gradient_updates == optimized.gradient_updates == 1
    for baseline_model, optimized_model in zip(
        baseline.models,
        optimized.models,
        strict=True,
    ):
        for left, right in zip(
            baseline_model.parameters(),
            optimized_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7)

    assert optimized.performance_training_metric_syncs == 1


def test_vectorized_status_mixture_decode_matches_reference() -> None:
    import torch

    baseline = StatusAwareConditionalMixtureRelationalProphecy(seed=44, device="cpu")
    optimized = StatusAwareConditionalMixtureRelationalProphecy(seed=44, device="cpu")
    _install_status_mixture_fast_path(optimized)

    generator = torch.Generator().manual_seed(77)
    outputs = torch.randn(
        baseline.config.ensemble_size,
        5,
        baseline.output_size,
        generator=generator,
    )
    reference = baseline._decoded_outputs(outputs)
    fast = optimized._decoded_outputs(outputs)
    for left, right in zip(reference, fast, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7)


def test_calibration_holdout_index_preserves_value_and_reuses_scan() -> None:
    base = StatusAwareConditionalMixtureRelationalProphecy(seed=91, device="cpu")
    replay = ReplayBuffer(capacity=64, holdout_stride=5)
    calibrated = SemanticCalibratedProphecy(base, replay)
    _install_indexed_calibration(calibrated)

    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    action = state.available_actions[0]
    # Keep this below minimum_count so the test isolates indexing/cache behavior
    # without invoking neural prediction.
    replay._holdout.extend(
        ReplayTransition(state, action, state, trace_id=str(index))
        for index in range(4)
    )

    first = calibrated._calibration(state, action)
    second = calibrated._calibration(state, action)

    assert first == second == 0.0
    assert calibrated.performance_holdout_index_rebuilds == 1
    assert calibrated.performance_holdout_index_hits >= 1


def test_performance_contract_is_explicitly_semantics_preserving() -> None:
    assert PERFORMANCE_CONTRACT == "semantics-preserving-no-training-schedule-change-v1"
