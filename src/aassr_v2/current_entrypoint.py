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
from .current_manifest import CURRENT_COMPONENTS
from .current_planner import CurrentFullyBatchedImaginationTree
from .current_relational_state_v3 import install_status_aware_relational_contract
from .current_repair import install_current_repairs
from .current_root_dedup import install_structural_root_dedup
from .current_runtime_performance_policy import (
    install_current_runtime_performance_device_aware,
)
from .current_status_models import (
    StatusAwareConditionalMixtureRelationalProphecy,
    install_status_supervised_world_model,
)
from .current_validation import install_current_fast_validation


# The repaired 2k diagnostic established this gate threshold. Scaling the training
# budget must not silently change the intervention criterion at the same time.
CURRENT_INTERVENTION_MARGIN = 0.05


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
    enable_performance_optimizations: bool = True,
) -> CurrentStandalonePentestAASSRAgent:
    """Build the sole active current-generation pentest AASSR runtime.

    This is the same status-balanced conditional-mixture runtime exercised by the
    latest repaired CUDA validation. Historical builders remain importable only
    for reproduction; current runners should resolve through this function.

    Public relational state v3 preserves the latest observed HTTP status while
    exact audit pressure, hidden session-TTL remaining, hidden stage depth, and
    concrete route/profile/object identity remain unavailable to learners.
    Prophecy confidence and local Critic support are fail-closed reliability gates,
    never value bonuses. Structural aliases share expensive model/Critic compute
    while final execution remains a concrete real action.

    ``enable_performance_optimizations`` changes only implementation mechanics:
    indexing, redundant state encoding, accelerator synchronization and tensor
    packing. It does not alter seeds, replay samples, update cadence, batch size,
    losses, exploration or action/value semantics. CPU keeps only the Python
    indexing fast path; accelerator-specific packing/synchronization changes are
    enabled only when the resolved model device is CUDA.
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
    agent.config = replace(
        agent.config,
        imagination_intervention_margin=CURRENT_INTERVENTION_MARGIN,
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
        model_class=StatusAwareConditionalMixtureRelationalProphecy,
    )
    if enable_performance_optimizations:
        install_current_runtime_performance_device_aware(agent)
    agent.planner.config = replace(agent.planner.config, aggregation="mean")
    agent.core.planner = agent.planner
    agent.current_components = dict(CURRENT_COMPONENTS)
    install_current_confidence_gate(agent)
    install_critic_support_gate(agent)
    install_structural_root_dedup(agent)
    install_current_decision_optimizations(agent)
    if profile_hot_path:
        install_current_hot_path_profiler(agent)
    return agent
