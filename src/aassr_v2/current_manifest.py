from __future__ import annotations

from typing import Mapping


CURRENT_GENERATION_VERSION = "aassr-current-generation-v2"

# Sole source of truth for the active post-v0.4 runtime. Historical modules stay
# importable for reproduction, but current builders/runners must use this stack.
CURRENT_COMPONENTS: Mapping[str, str] = {
    "observation": "response_causal_observation_v3",
    "aseq": "semantic-self-loop-empirical-v3",
    "policy": "relational-invariant-dqn+information-residual-v1",
    "policy_state_input": "relational-structural-v1",
    "policy_action_input": "relational-role-features-v1",
    "policy_hardware": "frontier-batched-dqn+fused-sync-free-bellman-v2",
    "prophecy": "relational-stochastic-ensemble-v1",
    "prophecy_output": "relational-descriptor+legal-action-mask+terminal-v1",
    "calibration": "semantic-frozen-replay-relational-holdout-v1",
    "information_evaluator": "semantic-relational-repeat-unlock-v1",
    "knowledge": "episode-local-response-knowledge-context-v1",
    "imagination": "root-preserving-parallel-universe-tree-v3+multi-outcome-depth-batched",
    "hardware": "dqn+relational-world-model+return-gru-critic-same-device+full-depth-batching-v4",
    "critic": "relational-gru-discounted-sparse-return-root-scale+batched-train-v4",
    "skills": "relational-aseq-template-v1",
    "goals": "external-final-goal+relational-skill-promotion-v1",
    "effect_composition": "superseded-by-relational-world-model-disabled",
    "training_imagination": "disabled-same-checkpoint",
    "current_protocol": "standalone-current-protocol-v2",
}

LEGACY_COMPONENTS_ACTIVE: tuple[str, ...] = ()
