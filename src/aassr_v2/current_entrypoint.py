from __future__ import annotations

from dataclasses import replace

from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_confidence_gate import install_current_confidence_gate
from .current_decision_optimization import install_current_decision_optimizations
from .current_hardware import install_hardware_dqn
from .current_hot_path_profile import install_current_hot_path_profiler
from .current_planner import CurrentFullyBatchedImaginationTree
from .current_relational_state import install_relational_state_contract
from .current_repair import install_current_repairs
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
    profile_hot_path: bool = False,
) -> CurrentStandalonePentestAASSRAgent:
    """Build the sole active current-generation pentest AASSR runtime.

    The active runtime uses one permutation-invariant contract end-to-end:
    relational Policy input, relational stochastic Prophecy input/output, semantic
    holdout calibration, and a relational GRU Critic trained on the real sparse
    {-1, 0, +1} episode return. The public relational state retains only audited
    observable state: terminal lockout/rate-limit controls, self-counted request
    usage, self-observed workflow progress, response semantics, and legal action
    structure. Exact audit pressure, hidden session TTL remaining, hidden stage
    depth, and concrete route/profile/object identity are not learner inputs.

    Imagination keeps every observable root action, carries multiple futures,
    backs stochastic outcomes up by their probability, chooses among future agent
    actions with max, and treats Prophecy confidence only as reliability.
    """

    if not enable_batching:
        raise ValueError(
            "current-generation pentest AASSR requires semantics-preserving "
            "Prophecy/Critic depth batching; use an explicit legacy/research "
            "entrypoint for historical scalar reproduction"
        )

    install_relational_state_contract()

    agent = build_current_standalone_pentest_aassr_core(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
    _enable_current_full_batching(agent)
    hardware = install_hardware_dqn(
        agent,
        seed=int(seed),
        train_transitions=int(train_transitions),
        device=device,
        allow_tf32=bool(allow_tf32),
    )

    # Construct the compatibility validator first. The repaired stack then
    # replaces it with semantic validation while preserving the evaluator/replay
    # partitioning and same-transition anti-hindsight contract.
    install_current_fast_validation(agent)
    install_current_repairs(
        agent,
        seed=int(seed),
        device=hardware.resolved_device,
    )
    # The external objective is expected sparse return. Probability already
    # represents real aleatoric risk; subtracting an additional outcome standard
    # deviation would introduce a new risk preference not present in the task.
    agent.planner.config = replace(agent.planner.config, aggregation="mean")
    agent.core.planner = agent.planner
    install_current_confidence_gate(agent)
    install_current_decision_optimizations(agent)
    if profile_hot_path:
        install_current_hot_path_profiler(agent)
    return agent
