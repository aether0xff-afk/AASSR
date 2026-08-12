from __future__ import annotations

from aassr_v2.current_core_manifest import (
    CORE_FORBIDDEN_DOMAIN_TOKENS,
    CURRENT_CORE_COMPONENTS,
    CURRENT_CORE_VERSION,
)
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.plugins.current_pentest import (
    CURRENT_PENTEST_PLUGIN,
    PENTEST_PLUGIN_COMPONENTS,
    PENTEST_PLUGIN_ID,
)


def test_current_core_manifest_is_domain_independent() -> None:
    rendered = " ".join(
        f"{key} {value}" for key, value in CURRENT_CORE_COMPONENTS.items()
    ).lower()
    for token in CORE_FORBIDDEN_DOMAIN_TOKENS:
        assert token.lower() not in rendered


def test_pentest_plugin_owns_http_and_environment_binding() -> None:
    assert CURRENT_PENTEST_PLUGIN.plugin_id == PENTEST_PLUGIN_ID
    assert CURRENT_PENTEST_PLUGIN.components == PENTEST_PLUGIN_COMPONENTS
    rendered = " ".join(PENTEST_PLUGIN_COMPONENTS.values()).lower()
    assert "http" in rendered
    assert "pentest" in rendered
    assert "observation" in PENTEST_PLUGIN_COMPONENTS
    assert "world_model_binding" in PENTEST_PLUGIN_COMPONENTS
    assert "reward_semantics" in PENTEST_PLUGIN_COMPONENTS


def test_current_builder_exposes_core_and_plugin_as_separate_runtime_objects() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )

    assert agent.current_core_plugin_boundary is True
    assert agent.current_core_version == CURRENT_CORE_VERSION
    assert agent.current_core_components == dict(CURRENT_CORE_COMPONENTS)
    assert agent.current_plugin_id == PENTEST_PLUGIN_ID
    assert agent.current_plugin_components == dict(PENTEST_PLUGIN_COMPONENTS)

    assert agent.aassr_core.policy is agent.policy
    assert agent.aassr_core.prophecy is agent.prophecy
    assert agent.aassr_core.planner is agent.planner
    assert agent.aassr_core.critic is agent.critic
    assert agent.aassr_core.knowledge is agent.knowledge
    assert agent.aassr_core.skills is agent.skills
    assert agent.aassr_core.aseq is agent.aseq
    assert agent.aassr_core.goals is agent.goals

    old_knowledge = agent.knowledge
    agent.begin_episode(clear_knowledge=True)
    assert agent.knowledge is not old_knowledge
    assert agent.aassr_core.knowledge is agent.knowledge

    diagnostics = agent.diagnostics()
    assert diagnostics["core_plugin_boundary"] is True
    assert diagnostics["aassr_core"]["version"] == CURRENT_CORE_VERSION
    assert diagnostics["runtime_plugin"]["id"] == PENTEST_PLUGIN_ID
