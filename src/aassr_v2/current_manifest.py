from __future__ import annotations

from typing import Mapping


CURRENT_GENERATION_VERSION = "aassr-current-generation-v2"

# Sole source of truth for the active post-v0.4 runtime. Historical modules stay
# importable for reproduction, but current builders/runners must use this stack.
CURRENT_COMPONENTS: Mapping[str, str] = {
    "observation": "response_causal_observation_v3",
    "aseq": "semantic-self-loop-empirical-v3",
    "policy": "relational-invariant-dqn+information-residual-v1",
    "policy_state_input": "relational-public-structural-v2",
    "policy_action_input": "relational-role-features-v1",
    "policy_hardware": "frontier-batched-dqn+fused-sync-free-bellman-v2",
    "prophecy": "relational-stochastic-world-model-v2",
    "prophecy_output": (
        "relational-descriptor-v2+legal-action-mask+"
        "active-success-failure-truncation-v2"
    ),
    "calibration": "semantic-probability-holdout-calibration-v2",
    "information_evaluator": "semantic-relational-probability-aware-v2",
    "knowledge": "episode-local-response-knowledge-context-v1",
    "imagination": (
        "root-preserving-parallel-universe-tree-v5+"
        "probability-chance-max-decision-depth-batched"
    ),
    "hardware": (
        "dqn+relational-world-model+return-gru-critic-same-device+"
        "full-depth-batching-v5"
    ),
    "critic": (
        "relational-gru-discounted-sparse-return+"
        "zero-memory-decision-suffixes+batched-train-v3"
    ),
    "skills": "relational-aseq-template-v1",
    "goals": "external-final-goal+relational-skill-promotion-v1",
    "effect_composition": "superseded-by-relational-world-model-disabled",
    "training_imagination": "disabled-same-checkpoint",
    "current_protocol": "standalone-current-protocol-v2",
    "hidden_pressure_contract": "audit-and-session-countdown-masked",
    "chance_objective": "expected-external-sparse-return",
    "planner_discount": "same-as-agent-gamma",
    "imagined_action_identity": "one-action-per-relational-legal-slot",
    "unknown_role_contract": "known-unobserved-entities-explicitly-unknown",
}

LEGACY_COMPONENTS_ACTIVE: tuple[str, ...] = ()
