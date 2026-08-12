from __future__ import annotations

from typing import Mapping


CURRENT_GENERATION_VERSION = "aassr-current-generation-v3-pre10k-audited"

# Sole source of truth for the active post-v0.4 runtime. Historical modules stay
# importable for reproduction, but current builders/runners must use this stack.
CURRENT_COMPONENTS: Mapping[str, str] = {
    "builder": "build_current_pentest_aassr_core",
    "observation": "response-causal-relational-public-state-v3+latest-http-status",
    "aseq": "semantic-self-loop-empirical-v3",
    "policy": "relational-invariant-dqn+information-residual-v1",
    "policy_state_input": "relational-public-structural-v3+latest-http-status",
    "policy_action_input": "relational-role-features-v1",
    "policy_hardware": "frontier-batched-dqn+fused-sync-free-bellman-v2",
    "relational_dqn_control": "fresh-process-public-state-v3+latest-http-status",
    "prophecy": "relational-conditional-mixture-ensemble-v6-status-categorical",
    "prophecy_output": (
        "relational-descriptor-v3+categorical-latest-http-status+legal-action-mask+"
        "active-success-failure-truncation-complete-mixture-v6"
    ),
    "prophecy_status_objective": "class-balanced-categorical-public-http-status-v2",
    "prophecy_status_inference": "softmax-categorical-realized-one-hot",
    "prophecy_outcome_mass": "complete-before-expected-return-backup",
    "prophecy_mode_identity": "lossless-exact-duplicate-only-learned-modes",
    "prophecy_epistemic_confidence": "ensemble-mode-set-sparse-max-disagreement",
    "calibration": "semantic-state-local-probability-holdout-v3",
    "information_evaluator": "semantic-relational-probability-aware-v3-status-aware",
    "knowledge": "episode-local-response-knowledge-context-v1",
    "imagination": (
        "root-concrete-execution+structural-compute-dedup+"
        "complete-probability-chance-max-decision-depth-batched"
    ),
    "imagination_intervention_margin": "fixed-0.05-scaling-contract",
    "imagination_coverage": "unique-structural-action-mean",
    "hardware": (
        "dqn+relational-world-model+return-gru-critic-same-device+"
        "full-depth-batching-v6"
    ),
    "critic": (
        "relational-gru-discounted-sparse-return+zero-memory-decision-suffixes+"
        "recent-signed-window128+batched-train-v4"
    ),
    "critic_readiness": "cumulative+recent-positive-negative-signed-support-v3",
    "critic_support_gate": "local-real-training-support-fail-closed-v1",
    "skills": "relational-aseq-template-v1",
    "skill_prophecy": "stochastic-complete-up-to-64+explicit-unresolved-tail-fail-closed-v3",
    "goals": "external-final-goal+relational-skill-promotion-v1",
    "effect_composition": "superseded-by-relational-world-model-disabled",
    "training_imagination": "disabled-same-checkpoint",
    "current_protocol": "standalone-current-protocol-v3-durable-checkpoint",
    "hidden_pressure_contract": "audit-and-session-countdown-masked",
    "public_response_contract": "latest-http-status-preserved",
    "chance_objective": "expected-external-sparse-return",
    "planner_discount": "same-as-agent-gamma",
    "planner_probability_contract": "complete-mass-required-before-expected-backup",
    "imagined_action_identity": "one-action-per-relational-legal-slot",
    "root_execution_identity": "concrete-execution-structural-compute-dedup",
    "unknown_role_contract": "known-unobserved-entities-explicitly-unknown",
    "checkpoint_contract": "fresh-process-portable-frozen-evaluation-v2",
    "provenance_contract": "architecture-version+exact-git-commit",
    "exploration_scaling_contract": "budget-normalized-explicitly-reported",
}

LEGACY_COMPONENTS_ACTIVE: tuple[str, ...] = ()
