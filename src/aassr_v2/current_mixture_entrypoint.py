from __future__ import annotations

from dataclasses import replace

from . import current_repair as repair_module
from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_confidence_gate import install_current_confidence_gate
from .current_critic_support import install_critic_support_gate
from .current_decision_optimization import install_current_decision_optimizations
from .current_entrypoint import _enable_current_full_batching
from .current_hardware import install_hardware_dqn
from .current_hot_path_profile import install_current_hot_path_profiler
from .current_manifest import CURRENT_COMPONENTS
from .current_relational_mixture_model import ConditionalMixtureRelationalProphecy
from .current_relational_state_v3 import install_status_aware_relational_contract
from .current_root_dedup import install_structural_root_dedup
from .current_validation import install_current_fast_validation


MIXTURE_CURRENT_COMPONENTS = {
    **dict(CURRENT_COMPONENTS),
    "observation": "response-causal-relational-public-state-v3+latest-http-status",
    "prophecy": "relational-conditional-mixture-ensemble-v4-status-aware",
    "prophecy_output": (
        "relational-descriptor-v3+latest-http-status+legal-action-mask+"
        "active-success-failure-truncation-mixture-v4"
    ),
    "imagination": (
        "root-concrete-execution+structural-compute-dedup+"
        "probability-chance-max-decision-depth-batched"
    ),
    "critic_gate": "local-real-training-support-fail-closed-v1",
}


def build_current_mixture_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
    enable_batching: bool = True,
    allow_tf32: bool = True,
    profile_hot_path: bool = False,
) -> CurrentStandalonePentestAASSRAgent:
    """Build the repaired status-aware candidate after the first 2k audit.

    The conditional relational mixture predicts public response status as part of
    relational v3. Prophecy confidence and local Critic training support are
    reliability gates only. Structural concrete aliases share model/Critic
    computation while final execution remains a real concrete environment action.
    """
    if not enable_batching:
        raise ValueError("repaired mixture current generation requires batching")

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

    previous_model = repair_module.RelationalStochasticProphecy
    repair_module.RelationalStochasticProphecy = ConditionalMixtureRelationalProphecy
    try:
        repair_module.install_current_repairs(
            agent,
            seed=int(seed),
            device=hardware.resolved_device,
        )
    finally:
        repair_module.RelationalStochasticProphecy = previous_model

    agent.planner.config = replace(agent.planner.config, aggregation="mean")
    agent.core.planner = agent.planner
    agent.current_components = dict(MIXTURE_CURRENT_COMPONENTS)
    install_current_confidence_gate(agent)
    install_critic_support_gate(agent)
    install_structural_root_dedup(agent)
    install_current_decision_optimizations(agent)
    if profile_hot_path:
        install_current_hot_path_profiler(agent)
    return agent
