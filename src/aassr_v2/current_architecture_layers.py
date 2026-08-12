from __future__ import annotations

from enum import Enum
from typing import Mapping


class CurrentArchitectureLayer(str, Enum):
    CORE = "core"
    PLUGIN = "plugin"
    ASSEMBLY = "assembly"
    PERFORMANCE = "performance"


# This registry is deliberately module-oriented. It does not pretend every legacy
# filename has already been physically moved; instead it makes ownership explicit
# now, so future refactors can move files without changing the scientific contract.
CURRENT_MODULE_OWNERSHIP: Mapping[str, CurrentArchitectureLayer] = {
    # Domain-independent algorithm/contracts.
    "aassr_v2.current_core_manifest": CurrentArchitectureLayer.CORE,
    "aassr_v2.current_plugin_api": CurrentArchitectureLayer.CORE,
    "aassr_v2.types": CurrentArchitectureLayer.CORE,
    "aassr_v2.knowledge": CurrentArchitectureLayer.CORE,
    "aassr_v2.skills": CurrentArchitectureLayer.CORE,
    "aassr_v2.semantic_control": CurrentArchitectureLayer.CORE,
    "aassr_v2.policy": CurrentArchitectureLayer.CORE,
    "aassr_v2.prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.imagination_tree": CurrentArchitectureLayer.CORE,
    "aassr_v2.branch_critic": CurrentArchitectureLayer.CORE,

    # Pentest/HTTP domain binding. Some implementation files retain historical
    # paths for compatibility but are plugin-owned, not universal AASSR core.
    "aassr_v2.plugins.current_pentest": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_state": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_state_v3": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_codec": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_decode_v2": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_model": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_mixture_model": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_status_models": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_agent_main_test": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_causal": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_env": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_schedule": CurrentArchitectureLayer.PLUGIN,

    # Composition/orchestration. These may depend on both core and one plugin.
    "aassr_v2.current_agent": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_entrypoint": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_manifest": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_repair": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_planner": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_validation": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_confidence_gate": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_critic_support": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_root_dedup": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_decision_optimization": CurrentArchitectureLayer.ASSEMBLY,

    # Execution mechanics only; never scientific/environment semantics.
    "aassr_v2.current_runtime_performance": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_runtime_performance_policy": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_runtime_performance_v2": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_replay_performance": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_hot_path_profile": CurrentArchitectureLayer.PERFORMANCE,
}


def modules_for_layer(layer: CurrentArchitectureLayer) -> tuple[str, ...]:
    return tuple(
        module
        for module, owner in CURRENT_MODULE_OWNERSHIP.items()
        if owner is layer
    )
