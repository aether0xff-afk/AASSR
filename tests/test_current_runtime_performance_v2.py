from __future__ import annotations

from collections import deque

import torch
from torch import nn

from aassr_v2.current_return_critic import (
    CRITIC_REPLAY_CAPACITY,
    ReturnAwareHardwareRelationalGRUBranchCritic,
)
from aassr_v2.current_runtime_performance_v2 import (
    _stacked_linear_forward,
    install_prepacked_critic_training,
)


def _mlp(input_size: int, hidden: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_size),
    )


def test_stacked_ensemble_forward_matches_independent_modules() -> None:
    torch.manual_seed(314159)
    models = [_mlp(11, 17, 13) for _ in range(3)]
    values = torch.randn(19, 11)

    with torch.no_grad():
        reference = torch.stack([model(values) for model in models], dim=0)
        fused = _stacked_linear_forward(torch, models, values)

    assert fused.shape == reference.shape
    assert torch.allclose(fused, reference, rtol=1e-6, atol=1e-6)


def _synthetic_critic(seed: int) -> ReturnAwareHardwareRelationalGRUBranchCritic:
    critic = ReturnAwareHardwareRelationalGRUBranchCritic(seed, device="cpu")
    rows = []
    feature_size = critic.encoder.feature_size
    for row_index in range(24):
        length = 1 + row_index % 5
        encoded = tuple(
            tuple(
                ((row_index + 1) * (step + 1) * (column + 3) % 37) / 37.0
                for column in range(feature_size)
            )
            for step in range(length)
        )
        target = (-1.0, 0.0, 1.0)[row_index % 3]
        targets = (target,) * length
        rows.append((encoded, targets))
    critic.replay = deque(rows, maxlen=CRITIC_REPLAY_CAPACITY)
    critic.suffix_sequences = len(rows)
    return critic


def _maximum_parameter_difference(left: object, right: object) -> float:
    maximum = 0.0
    for left_parameter, right_parameter in zip(
        left.parameters(),
        right.parameters(),
        strict=True,
    ):
        maximum = max(
            maximum,
            float((left_parameter - right_parameter).abs().max().item()),
        )
    return maximum


def test_prepacked_critic_training_preserves_one_update() -> None:
    reference = _synthetic_critic(2718)
    optimized = _synthetic_critic(2718)

    optimized.performance_full_replay_copies = 0
    optimized._performance_replay_snapshot_revision = -1
    optimized._performance_replay_snapshot = ()
    install_prepacked_critic_training(optimized)

    reference._train_step()
    optimized._train_step()

    assert reference.gradient_updates == optimized.gradient_updates == 1
    assert list(reference._losses) == list(optimized._losses)
    assert _maximum_parameter_difference(reference.gru, optimized.gru) <= 1e-7
    assert _maximum_parameter_difference(reference.output, optimized.output) <= 1e-7
    assert optimized.performance_prepacked_sequence_batches == 1
    assert optimized.performance_prepacked_sequence_rows > 0
