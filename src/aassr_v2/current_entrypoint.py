from __future__ import annotations

from dataclasses import replace

from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_confidence_gate import install_current_confidence_gate
from .current_critic_support import install_critic_support_gate
from .current_decision_optimization import install_current_decision_optimizations
from .current_hardware import install_hardware_dqn
from .current_hot_path_profile import install_current_hot_path_profiler
from .current_planner import CurrentFullyBatchedImaginationTree
from .current_relational_state_v3 import install_status_aware_relational_contract
from .current_repair import install_current_repairs
from .current_root_dedup import install_structural_root_dedup
from .current_status_models import install_status_supervised_world_model
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

    The active public relational state is v3: it keeps the latest actually
    observed HTTP status (200/302/400/401/403/404/409/429) while exact audit
    pressure, hidden session-TTL remaining, hidden stage depth, and concrete
    route/profile/object identity remain unavailable to learners.

    The world model gives the public status slice an explicit supervised loss so
    dangerous response errors cannot be diluted by the rest of the descriptor.
    Prophecy reliability and local Critic training support are fail-closed gates,
    never value bonuses. Expensive root prediction/Critic work is deduplicated by
    structural relational action identity while final execution remains concrete.
    """

    if not enable_batching:
        raise ValueError(
            "current-generation pentest AASSR requires semantics-preserving "
            "Prophecy/Critic depth batching; use an explicit legacy/research "
            "entrypoint for historical scalar reproduction"
        )

    install_status_aware_relational_contract()

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

    install_current_fast_validation(agent)
    install_current_repairs(
        agent,
        seed=int(seed),
        device=hardware.resolved_device,
    )
    install_status_supervised_world_model(
        agent,
        seed=int(seed),
        device=hardware.resolved_device,
    )
    agent.planner.config = replace(agent.planner.config, aggregation="mean")
    agent.core.planner = agent.planner
    install_current_confidence_gate(agent)
    install_critic_support_gate(agent)
    install_structural_root_dedup(agent)
    install_current_decision_optimizations(agent)
    if profile_hot_path:
        install_current_hot_path_profiler(agent)
    return agent
