from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

torch = pytest.importorskip("torch")

from aassr_v2.branch_critic import GRUBranchCritic, ParentTransitionCritic
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy
from aassr_v2.pentest_agent_main_test import DynamicActionDQN, HttpAgentCodec


@pytest.fixture
def non_default_torch_thread_count() -> Iterator[int]:
    """Exercise constructors with a value that the old hard-coded policy changed."""

    original = torch.get_num_threads()
    configured = 2 if original != 2 else 3
    torch.set_num_threads(configured)
    try:
        yield configured
    finally:
        torch.set_num_threads(original)


def _dynamic_dqn() -> DynamicActionDQN:
    return DynamicActionDQN(
        7,
        train_transitions=8,
        hidden_units=8,
        replay_capacity=8,
        batch_size=2,
        warmup_steps=2,
        target_update_interval=4,
    )


def _parent_transition_critic() -> ParentTransitionCritic:
    return ParentTransitionCritic(
        lambda _state: (0.0,),
        1,
        hidden_units=8,
        replay_capacity=8,
        batch_size=2,
        gradient_steps_per_episode=1,
        seed=7,
    )


def _gru_branch_critic() -> GRUBranchCritic:
    return GRUBranchCritic(
        lambda _state: (0.0,),
        1,
        hidden_units=8,
        replay_capacity=8,
        batch_size=2,
        gradient_steps_per_episode=1,
        seed=7,
    )


def _neural_delta_prophecy() -> NeuralDeltaProphecy:
    return NeuralDeltaProphecy(
        HttpAgentCodec(),
        config=NeuralDeltaConfig(
            hidden_units=8,
            ensemble_size=1,
            replay_capacity=8,
            batch_size=2,
            warmup_steps=2,
        ),
        seed=7,
    )


@pytest.mark.parametrize(
    "factory",
    (
        _dynamic_dqn,
        _parent_transition_critic,
        _gru_branch_critic,
        _neural_delta_prophecy,
    ),
    ids=(
        "dynamic-action-dqn",
        "parent-transition-critic",
        "gru-branch-critic",
        "neural-delta-prophecy",
    ),
)
def test_model_constructor_preserves_process_torch_thread_policy(
    factory: Callable[[], object],
    non_default_torch_thread_count: int,
) -> None:
    before = torch.get_num_threads()

    constructed = factory()

    assert constructed is not None
    assert before == non_default_torch_thread_count
    assert torch.get_num_threads() == before


def test_current_runtime_construction_preserves_process_torch_thread_policy(
    non_default_torch_thread_count: int,
) -> None:
    before = torch.get_num_threads()

    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=8,
        use_imagination=True,
        device="cpu",
    )

    assert agent is not None
    assert before == non_default_torch_thread_count
    assert torch.get_num_threads() == before
