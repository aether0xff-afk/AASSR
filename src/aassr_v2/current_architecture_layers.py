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
    "aassr_v2.current_architecture_layers": CurrentArchitectureLayer.CORE,
    "aassr_v2.types": CurrentArchitectureLayer.CORE,
    "aassr_v2.knowledge": CurrentArchitectureLayer.CORE,
    "aassr_v2.skills": CurrentArchitectureLayer.CORE,
    "aassr_v2.semantic_control": CurrentArchitectureLayer.CORE,
    "aassr_v2.policy": CurrentArchitectureLayer.CORE,
    "aassr_v2.prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.imagination_tree": CurrentArchitectureLayer.CORE,
    "aassr_v2.branch_critic": CurrentArchitectureLayer.CORE,
    "aassr_v2.action_plugins": CurrentArchitectureLayer.CORE,
    "aassr_v2.autonomous_agent": CurrentArchitectureLayer.CORE,
    "aassr_v2.autonomous_agent_core": CurrentArchitectureLayer.CORE,
    "aassr_v2.effect_prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.empirical_confidence": CurrentArchitectureLayer.CORE,
    "aassr_v2.feature_memory": CurrentArchitectureLayer.CORE,
    "aassr_v2.goals": CurrentArchitectureLayer.CORE,
    "aassr_v2.gru_prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.imagination": CurrentArchitectureLayer.CORE,
    "aassr_v2.learning": CurrentArchitectureLayer.CORE,
    "aassr_v2.metrics": CurrentArchitectureLayer.CORE,
    "aassr_v2.neural_delta_prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.persistent_effect_prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.replay": CurrentArchitectureLayer.CORE,
    "aassr_v2.serialization": CurrentArchitectureLayer.CORE,
    "aassr_v2.tabular_prophecy": CurrentArchitectureLayer.CORE,
    "aassr_v2.torch_gru_prophecy": CurrentArchitectureLayer.CORE,

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
    "aassr_v2.current_generation": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_runtime": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_relational_skill_prophecy": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_semantic_calibration": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.current_semantic_evaluator": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_agent_main_test": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_causal": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_env": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_schedule": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_curriculum_dedup": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_transfer_stages": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_http_benchmark": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.pentest_http_lab": CurrentArchitectureLayer.PLUGIN,
    "aassr_v2.dreamerv3_baseline": CurrentArchitectureLayer.PLUGIN,

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
    "aassr_v2.current_checkpoint": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_dqn_baseline": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_experiment_suite": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_mixture_entrypoint": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_protocol": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_return_critic": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.current_run_artifacts": CurrentArchitectureLayer.ASSEMBLY,
    "aassr_v2.integrated_agent": CurrentArchitectureLayer.ASSEMBLY,

    # Execution mechanics only; never scientific/environment semantics.
    "aassr_v2.current_runtime_performance": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_runtime_performance_policy": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_runtime_performance_v2": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_replay_performance": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_hot_path_profile": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_hardware": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.current_performance": CurrentArchitectureLayer.PERFORMANCE,
    "aassr_v2.native_batching": CurrentArchitectureLayer.PERFORMANCE,
}


def modules_for_layer(layer: CurrentArchitectureLayer) -> tuple[str, ...]:
    return tuple(
        module
        for module, owner in CURRENT_MODULE_OWNERSHIP.items()
        if owner is layer
    )
