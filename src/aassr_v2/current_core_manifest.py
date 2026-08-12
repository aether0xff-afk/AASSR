from __future__ import annotations

from typing import Mapping


CURRENT_CORE_VERSION = "aassr-current-core-v1"

# Domain-independent algorithmic contract. A plugin may bind concrete observation,
# action, state-codec, outcome and environment semantics, but it may not redefine
# these responsibilities without becoming a different AASSR core generation.
CURRENT_CORE_COMPONENTS: Mapping[str, str] = {
    "agent_loop": "closed-loop-select-execute-observe-learn-v1",
    "aseq": "semantic-exact-self-loop-guard-v3",
    "policy": "learned-policy+information-residual-contract-v1",
    "prophecy": "stochastic-conditional-mixture-world-model-contract-v1",
    "prophecy_confidence": "epistemic-reliability-not-value-bonus-v1",
    "calibration": "local-heldout-prediction-calibration-v1",
    "knowledge": "episode-local-observed-knowledge-store-v1",
    "skills": "learned-aseq-template-library-v1",
    "imagination": "multi-step-chance-decision-tree-v1",
    "imagination_backup": "complete-probability-chance+max-agent-decision-v1",
    "critic": "learned-signed-sparse-return-sequence-critic-v1",
    "critic_support_gate": "local-real-training-support-fail-closed-v1",
    "goals": "external-final-goal+learned-skill-promotion-v1",
    "training_imagination": "disabled-same-checkpoint-evaluation-contract",
    "runtime_performance_contract": (
        "same-seeds+same-replay-rows+same-update-cadence+same-batch-size+"
        "same-loss+same-exploration+same-action-semantics"
    ),
}

# Tokens that belong to an environment/plugin binding rather than the core. The
# boundary regression intentionally rejects these from CURRENT_CORE_COMPONENTS.
CORE_FORBIDDEN_DOMAIN_TOKENS: tuple[str, ...] = (
    "http",
    "pentest",
    "route",
    "profile",
    "csrf",
    "status-code",
)
