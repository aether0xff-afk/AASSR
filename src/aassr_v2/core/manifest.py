from __future__ import annotations

from typing import Mapping


CORE_VERSION = "aassr-core-v2-minimal-plugin"
CORE_ARCHITECTURE_VERSION = "aassr-core-architecture-v1"
PLUGIN_CONTRACT_VERSION = "minimal-syntax-public-data-v1"

# Canonical architecture.  These are responsibility boundaries, not model
# choices.  DQN/GRU/ensemble implementations may change without changing the
# architecture version as long as these ownership and dependency rules hold.
CORE_ARCHITECTURE: Mapping[str, tuple[str, ...]] = {
    "boundary": (
        "plugin_contract",
        "transition",
    ),
    "world_state": (
        "knowledge",
        "action_surface",
        "representation",
    ),
    "decision_learning": (
        "policy",
        "prophecy",
        "critic",
        "skills",
        "aseq",
        "imagination",
    ),
    "orchestration": (
        "runtime",
    ),
}

# Allowed conceptual dependency graph.  Runtime is the only component allowed
# to orchestrate the whole graph.  Plugins are deliberately absent: the Core
# depends only on the minimal plugin contract and public transition data.
CORE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "transition": ("plugin_contract",),
    "knowledge": ("plugin_contract", "transition"),
    "action_surface": ("plugin_contract", "knowledge"),
    "representation": ("plugin_contract", "knowledge", "action_surface"),
    "policy": ("representation",),
    "prophecy": ("representation",),
    "critic": ("representation",),
    "skills": ("representation", "policy"),
    "aseq": ("representation",),
    "imagination": ("policy", "prophecy", "critic", "skills"),
    "runtime": (
        "transition",
        "knowledge",
        "action_surface",
        "representation",
        "policy",
        "prophecy",
        "critic",
        "skills",
        "aseq",
        "imagination",
    ),
}

# Current algorithm implementations.  This table is intentionally separate from
# CORE_ARCHITECTURE: replacing one of these implementations is an experiment or
# optimization, not an architectural rewrite.
CORE_IMPLEMENTATIONS: Mapping[str, str] = {
    "plugin_boundary": PLUGIN_CONTRACT_VERSION,
    "representation": "core-owned-schema-driven-hashed-relational-v1",
    "experience_memory": "core-owned-action-outcome-evidence-v1",
    "policy": "external-reward-dqn+separate-information-residual-v1",
    "prophecy": "core-owned-neural-delta-ensemble-v1",
    "calibration": "real-transition-holdout-v1",
    "knowledge": "core-owned-public-typed-memory+candidate-reuse-v2",
    "aseq": "semantic-exact-self-loop-guard-v3",
    "skills": "structural-action-template-v1",
    "critic": "signed-sparse-return-gru-v1",
    "imagination": "multi-step-chance-decision-planning-v1",
    "imagination_gate": "coverage+observed-return-support-fail-closed-v1",
    "external_reward": "plugin-passthrough-only",
}

# Backward-compatible name used by existing diagnostics/tests.
CORE_COMPONENTS = CORE_IMPLEMENTATIONS

PLUGIN_ALLOWED_AUTHORITIES: tuple[str, ...] = (
    "action-syntax",
    "observation-data-types",
    "mechanical-value-space",
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
    "problem-solving-memory",
)
