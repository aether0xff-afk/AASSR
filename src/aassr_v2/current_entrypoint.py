from __future__ import annotations

from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_hardware import install_hardware_dqn


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

    The current runtime is standalone: it does not construct
    `IntegratedAASSRAgent`, `AutonomousLearningAgent`, the v0.4 contextual Policy,
    the old online GRU Prophecy, or the hand-written Goal scorer. Historical
    builders stay importable under explicit legacy names for reproduction only.

    Depth batching is mandatory. The requested torch device is also applied to
    both the Neural Delta world model and the active relational DQN Policy. The
    DQN backend keeps Bellman next-action reductions on device so CUDA execution
    does not introduce one host synchronization per replay row.
    """

    if not enable_batching:
        raise ValueError(
            "current-generation pentest AASSR requires its semantics-preserving "
            "depth-batched Neural Delta planner; use an explicit legacy/research "
            "entrypoint for historical scalar reproduction"
        )
    agent = build_current_standalone_pentest_aassr_core(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
    install_hardware_dqn(
        agent,
        seed=int(seed),
        train_transitions=int(train_transitions),
        device=device,
        allow_tf32=bool(allow_tf32),
    )
    return agent
