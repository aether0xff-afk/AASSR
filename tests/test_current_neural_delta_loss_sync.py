from __future__ import annotations

import random
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from aassr_v2.current_generation import CurrentNeuralDeltaProphecy
from aassr_v2.neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy
from aassr_v2.pentest_agent_main_test import ACTION_FEATURE_SIZE, HttpAgentCodec
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_STAGES,
    TransferDiagnosticWorld,
)


class _ScalarLossReference(CurrentNeuralDeltaProphecy):
    """Current inputs/device with the inherited per-model loss extraction."""

    def _train_step(self) -> None:
        NeuralDeltaProphecy._train_step(self)


def _small_prophecy(prophecy_type: type[CurrentNeuralDeltaProphecy]):
    return prophecy_type(
        HttpAgentCodec(),
        config=NeuralDeltaConfig(
            action_feature_size=ACTION_FEATURE_SIZE,
            hidden_units=8,
            ensemble_size=3,
            replay_capacity=16,
            batch_size=4,
            warmup_steps=8,
            gradient_steps_per_observation=1,
        ),
        seed=20260809,
        device="cpu",
    )


def _training_transitions(count: int):
    rows = []
    for index in range(count):
        world = TransferDiagnosticWorld(
            91_000 + index,
            stage=TRANSFER_STAGES[0],
        )
        before = world.snapshot()
        action = before.available_actions[index % len(before.available_actions)]
        after = world.step(action).snapshot
        rows.append((before, action, after))
    return tuple(rows)


def _assert_exact(left: Any, right: Any, *, path: str = "root") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_exact(left_item, right_item, path=f"{path}[{index}]")
        return
    assert left == right, path


def test_current_neural_delta_bulk_loss_transfer_is_scalar_reference_exact() -> None:
    reference = _small_prophecy(_ScalarLossReference)
    optimized = _small_prophecy(CurrentNeuralDeltaProphecy)
    transitions = _training_transitions(6)

    for before, action, after in transitions:
        reference.learn(before, action, after)
        optimized.learn(before, action, after)

    # Exercise the deque's sequential maxlen truncation as well as loss order.
    prior_losses = tuple(index / 1_000.0 for index in range(255))
    reference._losses.extend(prior_losses)
    optimized._losses.extend(prior_losses)

    assert tuple(reference.replay) == tuple(optimized.replay)
    assert reference.observations == optimized.observations == len(transitions)
    for reference_model, optimized_model in zip(
        reference.models,
        optimized.models,
        strict=True,
    ):
        _assert_exact(reference_model.state_dict(), optimized_model.state_dict())

    random.seed(8675309)
    python_rng_before = random.getstate()
    torch.manual_seed(112358)
    torch_rng_before = torch.random.get_rng_state().clone()

    reference._train_step()
    reference_python_rng_after = random.getstate()
    reference_torch_rng_after = torch.random.get_rng_state().clone()

    random.setstate(python_rng_before)
    torch.random.set_rng_state(torch_rng_before)
    optimized._train_step()
    optimized_python_rng_after = random.getstate()
    optimized_torch_rng_after = torch.random.get_rng_state().clone()

    assert reference_python_rng_after == optimized_python_rng_after == python_rng_before
    assert torch.equal(reference_torch_rng_after, optimized_torch_rng_after)
    assert torch.equal(reference_torch_rng_after, torch_rng_before)
    assert reference.randomizer.getstate() == optimized.randomizer.getstate()
    assert tuple(reference.replay) == tuple(optimized.replay)
    assert reference.observations == optimized.observations
    assert reference.gradient_updates == optimized.gradient_updates == 1
    assert tuple(reference._losses) == tuple(optimized._losses)

    for model_index, (reference_model, optimized_model) in enumerate(
        zip(reference.models, optimized.models, strict=True)
    ):
        _assert_exact(
            reference_model.state_dict(),
            optimized_model.state_dict(),
            path=f"model[{model_index}]",
        )
        for parameter_index, (reference_parameter, optimized_parameter) in enumerate(
            zip(reference_model.parameters(), optimized_model.parameters(), strict=True)
        ):
            _assert_exact(
                reference_parameter.grad,
                optimized_parameter.grad,
                path=f"model[{model_index}].grad[{parameter_index}]",
            )

    for optimizer_index, (reference_optimizer, optimized_optimizer) in enumerate(
        zip(reference.optimizers, optimized.optimizers, strict=True)
    ):
        _assert_exact(
            reference_optimizer.state_dict(),
            optimized_optimizer.state_dict(),
            path=f"optimizer[{optimizer_index}]",
        )

    diagnostics = optimized.diagnostics()
    assert diagnostics["training_loss_bulk_host_transfer_calls"] == 1
    assert diagnostics["training_loss_bulk_host_transfer_rows"] == 3
    assert diagnostics["per_model_training_loss_item_syncs"] == 0

