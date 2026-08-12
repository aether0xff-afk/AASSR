from __future__ import annotations

from typing import Mapping


CURRENT_GENERATION_VERSION = "aassr-current-generation-v2"

# Sole source of truth for the active post-v0.4 runtime. Historical modules stay
# importable for reproduction, but current builders/runners must use this stack.
#
# IMPORTANT: this manifest describes the runtime exercised by the latest repaired
# CUDA validation. Do not infer the active generation from README version text or
# from historical runner names.
CURRENT_COMPONENTS: Mapping[str, str] = {
    "builder": "build_current_pentest_aassr_core",
    "observation": "response-causal-relational-public-state-v3+latest-http-status",
    "aseq": "semantic-self-loop-empirical-v3",
    "policy": "relational-invariant-dqn+information-residual-v1",
    "policy_state_input": "relational-public-structural-v3+latest-http-status",
    "policy_action_input": "relational-role-features-v1",
    "policy_hardware": "frontier-batched-dqn+fused-sync-free-bellman-v2",
    "prophecy": "relational-conditional-mixture-ensemble-v5-status-balanced",
    "prophecy_output": (
        "relational-descriptor-v3+latest-http-status+legal-action-mask+"
        "active-success-failure-truncation-mixture-v5"
    ),
    "prophecy_status_objective": (
        "class-balanced-categorical-public-http-status-v2"
    ),
    "calibration": "semantic-probability-holdout-calibration-v3-status-aware",
    "information_evaluator": "semantic-relational-probability-aware-v3-status-aware",
    "knowledge": "episode-local-response-knowledge-context-v1",
    "imagination": (
        "root-concrete-execution+structural-compute-dedup+"
        "probability-chance-max-decision-depth-batched"
    ),
    "hardware": (
        "dqn+relational-world-model+return-gru-critic-same-device+"
        "full-depth-batching-v6"
    ),
    "critic": (
        "relational-gru-discounted-sparse-return+"
        "zero-memory-decision-suffixes+batched-train-v3"
    ),
    "critic_support_gate": "local-real-training-support-fail-closed-v1",
    "skills": "relational-aseq-template-v1",
    "goals": "external-final-goal+relational-skill-promotion-v1",
    "effect_composition": "superseded-by-relational-world-model-disabled",
    "training_imagination": "disabled-same-checkpoint",
    "current_protocol": "standalone-current-protocol-v2",
    "hidden_pressure_contract": "audit-and-session-countdown-masked",
    "public_response_contract": "latest-http-status-preserved",
    "chance_objective": "expected-external-sparse-return",
    "planner_discount": "same-as-agent-gamma",
    "imagined_action_identity": "one-action-per-relational-legal-slot",
    "root_execution_identity": "concrete-execution-structural-compute-dedup",
    "unknown_role_contract": "known-unobserved-entities-explicitly-unknown",
}

LEGACY_COMPONENTS_ACTIVE: tuple[str, ...] = ()
