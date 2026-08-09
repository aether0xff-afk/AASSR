from __future__ import annotations

from typing import Mapping


CURRENT_GENERATION_VERSION = "aassr-current-generation-v1"

# Sole source of truth for the active post-v0.4 runtime. Historical modules may
# contain their own archived metadata, but new runners, package exports and CI
# must import this manifest instead of inferring the active stack from filenames.
CURRENT_COMPONENTS: Mapping[str, str] = {
    "observation": "response_causal_observation_v3",
    "aseq": "semantic-self-loop-empirical-v3",
    "policy": "relational-invariant-dqn+information-residual-v1",
    "policy_state_input": "relational-structural-v1",
    "policy_action_input": "relational-role-features-v1",
    "policy_hardware": "shared-device-hardware-dqn-fused-sync-free-targets-v1",
    "prophecy": "neural-delta-ensemble+relational-state-action-v2",
    "prophecy_output": "concrete-scaffold-delta-v1",
    "calibration": "frozen-replay-relational-holdout-v1",
    "knowledge": "episode-local-response-knowledge-context-v1",
    "imagination": "parallel-universe-tree-v2+depth-batched",
    "hardware": "dqn+neural-delta+gru-critic-same-device+depth-batching-v2",
    "critic": "relational-gru-branch-critic-final-outcome-v1",
    "skills": "relational-aseq-template-v1",
    "goals": "external-final-goal+relational-skill-promotion-v1",
    "effect_composition": "superseded-by-neural-delta-disabled",
    "training_imagination": "disabled-same-checkpoint",
    "current_protocol": "standalone-current-protocol-v1",
}

# Legacy code remains importable for exact reproduction, but none is reachable
# from the package-level current pentest builder or current experiment runners.
LEGACY_COMPONENTS_ACTIVE: tuple[str, ...] = ()
