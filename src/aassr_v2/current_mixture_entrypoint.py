from __future__ import annotations

from dataclasses import replace

from . import current_repair as repair_module
from .current_agent import (
    CurrentStandalonePentestAASSRAgent,
    build_current_standalone_pentest_aassr_core,
)
from .current_confidence_gate import install_current_confidence_gate
from .current_decision_optimization import install_current_decision_optimizations
from .current_entrypoint import _enable_current_full_batching
from .current_hardware import install_hardware_dqn
from .current_hot_path_profile import install_current_hot_path_profiler
from .current_manifest import CURRENT_COMPONENTS
from .current_relational_mixture_model import ConditionalMixtureRelationalProphecy
from .current_relational_state import install_relational_state_contract
from .current_validation import install_current_fast_validation


MIXTURE_CURRENT_COMPONENTS = {
    **dict(CURRENT_COMPONENTS),
    "prophecy": "relational-conditional-mixture-ensemble-v3",
    "prophecy_output": (
        "relational-descriptor-v2+legal-action-mask+"
        "active-success-failure-truncation-mixture-v3"
    ),
    "imagination": (
        "root-preserving-parallel-universe-tree-v5+"
        "probability-chance-max-decision-depth-batched"
    ),
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
    """Build the frozen repaired candidate used by the one-shot validation.

    This is the current-generation stack with the conditional relational mixture
    world model. Chance outcomes are probability-weighted; future action choices
    are max-backed; confidence remains a reliability gate only.
    """
    if not enable_batching:
        raise ValueError("repaired mixture current generation requires batching")

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

    # Match the actual optimization objective: expected sparse return. Real
    # failure risk is already present in {-1,0,+1} outcome values and must not be
    # double-penalized by an extra variance preference.
    agent.planner.config = replace(agent.planner.config, aggregation="mean")
    agent.core.planner = agent.planner
    agent.current_components = dict(MIXTURE_CURRENT_COMPONENTS)
    install_current_confidence_gate(agent)
    install_current_decision_optimizations(agent)
    if profile_hot_path:
        install_current_hot_path_profiler(agent)
    return agent
