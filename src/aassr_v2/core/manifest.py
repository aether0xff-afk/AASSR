from __future__ import annotations

from typing import Mapping


CORE_VERSION = "aassr-core-v2-minimal-plugin"
PLUGIN_CONTRACT_VERSION = "minimal-syntax-public-data-v1"

CORE_COMPONENTS: Mapping[str, str] = {
    "plugin_boundary": PLUGIN_CONTRACT_VERSION,
    "representation": "core-owned-schema-driven-hashed-relational-v1",
    "experience_memory": "core-owned-action-outcome-evidence-v1",
    "policy": "external-reward-dqn+separate-information-residual-v1",
    "prophecy": "core-owned-neural-delta-ensemble-v1",
    "calibration": "real-transition-holdout-v1",
    "knowledge": "public-observation-provenance-store-v1",
    "aseq": "semantic-exact-self-loop-guard-v3",
    "skills": "structural-action-template-v1",
    "critic": "signed-sparse-return-gru-v1",
    "imagination": "multi-step-chance-decision-planning-v1",
    "imagination_gate": "coverage+observed-return-support-fail-closed-v1",
    "external_reward": "plugin-passthrough-only",
}

PLUGIN_ALLOWED_AUTHORITIES: tuple[str, ...] = (
    "action-syntax",
    "observation-data-types",
    "real-io",
    "external-reward-passthrough",
    "termination-passthrough",
)

PLUGIN_FORBIDDEN_AUTHORITIES: tuple[str, ...] = (
    "semantic-state-encoding",
    "action-value-ranking",
    "task-relation-labeling",
    "world-model-installation",
    "planning-score-definition",
    "strategic-action-filtering",
    "reward-shaping",
)
