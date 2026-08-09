from __future__ import annotations

from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_hardware import install_hardware_dqn
from .current_planner import CurrentFullyBatchedImaginationTree
from .current_validation import install_current_fast_validation


def _enable_current_full_batching(agent: CurrentStandalonePentestAASSRAgent) -> None:
    old = agent.planner
    planner = CurrentFullyBatchedImaginationTree(
        old.policy,
        agent.current_batched_prophecy,
        config=old.config,
        scorer=old.scorer,
    )
    planner._state_key = old._state_key
    agent.planner = planner
    agent.core.planner = planner
    agent.current_depth_batching = True
    agent.current_critic_batching = True


def build_current_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
    enable_batching: bool = True,
    allow_tf32: bool = True,
) -> CurrentStandalonePentestAASSRAgent:
    """Build the sole active current-generation pentest AASSR runtime.

    The current runtime is standalone and uses mandatory depth batching for both
    Neural Delta prediction and learned GRU branch scoring. DQN, Neural Delta and
    the GRU Critic share the requested torch device. Holdout validation keeps the
    same samples/frequency/signal while skipping symbolic StateSnapshot/action
    reconstruction that the cosine validator never consumes.
    """

    if not enable_batching:
        raise ValueError(
            "current-generation pentest AASSR requires semantics-preserving "
            "Prophecy/Critic depth batching; use an explicit legacy/research "
            "entrypoint for historical scalar reproduction"
        )
    agent = build_current_standalone_pentest_aassr_core(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
    _enable_current_full_batching(agent)
    install_hardware_dqn(
        agent,
        seed=int(seed),
        train_transitions=int(train_transitions),
        device=device,
        allow_tf32=bool(allow_tf32),
    )
    install_current_fast_validation(agent)
    return agent
