from __future__ import annotations

from dataclasses import replace

import pytest

from aassr_v2.current_core_manifest import (
    CORE_FORBIDDEN_DOMAIN_TOKENS,
    CURRENT_CORE_COMPONENTS,
    CURRENT_CORE_VERSION,
)
from aassr_v2.current_entrypoint import (
    build_current_aassr_core,
    build_current_pentest_aassr_core,
)
from aassr_v2.branch_critic import CriticTransition
from aassr_v2.current_relational_state_v3 import (
    relational_state_key_v3,
    relational_state_vector_v3,
)
from aassr_v2.plugins.current_pentest import (
    CURRENT_PENTEST_PLUGIN,
    PENTEST_PLUGIN_COMPONENTS,
    PENTEST_PLUGIN_ID,
)
from aassr_v2.replay import ReplayTransition


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
    assert CURRENT_PENTEST_PLUGIN.environment_factory is not None
    assert CURRENT_PENTEST_PLUGIN.representation.state_codec_factory is not None


def test_current_builder_exposes_core_and_plugin_as_separate_runtime_objects() -> None:
    pytest.importorskip("torch")
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
    assert diagnostics["runtime_plugin"]["representation"]["id"] == (
        CURRENT_PENTEST_PLUGIN.representation.binding_id
    )
    assert diagnostics["runtime_plugin"]["environment_factory"] == "plugin-provided"


def test_two_runtime_bindings_are_isolated_without_module_monkeypatching() -> None:
    pytest.importorskip("torch")
    from aassr_v2 import current_generation, current_hardware, current_relational_codec

    original_globals = (
        current_generation.relational_state_vector,
        current_generation.relational_state_key,
        current_hardware.relational_state_key,
        current_relational_codec.REL_DESCRIPTOR_SIZE,
    )
    primary = CURRENT_PENTEST_PLUGIN.representation
    alternate_score_calls = 0

    def alternate_prediction_score(predictions, actual) -> float:
        nonlocal alternate_score_calls
        del predictions, actual
        alternate_score_calls += 1
        return 0.375

    alternate = replace(
        primary,
        binding_id="test-isolated-binding",
        state_vector=lambda state: tuple(-value for value in primary.state_vector(state)),
        state_key=lambda state: ("alternate", primary.state_key(state)),
        action_structure=lambda state, action: tuple(
            1.0 - value for value in primary.action_structure(state, action)
        ),
        prediction_score=alternate_prediction_score,
    )
    alternate_plugin = replace(
        CURRENT_PENTEST_PLUGIN,
        plugin_id="test-isolated-plugin",
        representation=alternate,
    )

    first = build_current_aassr_core(
        plugin=CURRENT_PENTEST_PLUGIN,
        seed=101,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
        enable_performance_optimizations=True,
    )
    second = build_current_aassr_core(
        plugin=alternate_plugin,
        seed=103,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
        enable_performance_optimizations=True,
    )

    assert first.representation is primary
    assert second.representation is alternate
    assert first.runtime_plugin.representation is primary
    assert second.runtime_plugin.representation is alternate
    assert first.dqn.encode_state is primary.state_vector
    assert second.dqn.encode_state is alternate.state_vector
    from aassr_v2.types import Action, StateSnapshot

    action = Action("request", parameters={"route_id": "r", "profile_id": "p"})
    state = StateSnapshot(
        vector=(0.25,) + (0.0,) * (primary.state_size - 1),
        facts=frozenset(
            {
                "known_route:r",
                "known_profile:p",
                "observed_route_role:r:catalog",
            }
        ),
        available_actions=(action,),
        metadata={"observation_contract": primary.observation_contract},
    )
    assert first.dqn.encode_state(state) == primary.state_vector(state)
    assert second.dqn.encode_state(state) == alternate.state_vector(state)
    assert first.dqn.action_features(state, action) == primary.action_structure(
        state, action
    )
    assert second.dqn.action_features(state, action) == alternate.action_structure(
        state, action
    )
    assert first.dqn.encode_state(state) != second.dqn.encode_state(state)
    assert first.dqn.action_features(state, action) != second.dqn.action_features(
        state, action
    )
    assert first.base_neural_prophecy._input(state, action) == (
        primary.state_vector(state) + primary.action_structure(state, action)
    )
    assert second.base_neural_prophecy._input(state, action) == (
        alternate.state_vector(state) + alternate.action_structure(state, action)
    )
    calibration_values = []
    for agent in (first, second):
        agent.calibrated_prophecy.minimum_count = 1
        agent.calibrated_prophecy.evaluation_limit = 1
        agent.calibrated_prophecy.replay._holdout.append(
            ReplayTransition(state, action, state, trace_id="runtime-isolation")
        )
        calibration_values.append(
            agent.calibrated_prophecy._calibration(state, action)
        )

    assert calibration_values[1] == pytest.approx(0.375)
    assert alternate_score_calls == 1

    primary_action_key = primary.action_structure(state, action)
    alternate_action_key = alternate.action_structure(state, action)
    assert set(first.calibrated_prophecy._performance_holdout_index) == {
        primary_action_key
    }
    assert set(second.calibrated_prophecy._performance_holdout_index) == {
        alternate_action_key
    }
    assert next(iter(first.calibrated_prophecy._cache))[:2] == (
        primary_action_key,
        primary.state_key(state),
    )
    assert next(iter(second.calibrated_prophecy._cache))[:2] == (
        alternate_action_key,
        alternate.state_key(state),
    )
    assert (
        current_generation.relational_state_vector,
        current_generation.relational_state_key,
        current_hardware.relational_state_key,
        current_relational_codec.REL_DESCRIPTOR_SIZE,
    ) == original_globals


def test_pentest_runtime_binding_matches_canonical_v3_representation() -> None:
    pytest.importorskip("torch")
    from aassr_v2.types import StateSnapshot

    binding = CURRENT_PENTEST_PLUGIN.representation
    state = StateSnapshot(vector=(0.0,) * binding.state_size, metadata={})
    assert binding.state_vector(state) == relational_state_vector_v3(state)
    assert binding.state_key(state) == relational_state_key_v3(state)


def test_current_critic_uses_the_runtime_scoped_v3_binding() -> None:
    pytest.importorskip("torch")
    from aassr_v2.types import Action, StateSnapshot

    agent = build_current_pentest_aassr_core(
        seed=107,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
        enable_performance_optimizations=False,
    )
    binding = agent.representation
    action = Action("request", parameters={"route_id": "r", "profile_id": "p"})
    state = StateSnapshot(
        vector=(0.0,) * binding.state_size,
        available_actions=(action,),
        metadata={"observation_contract": binding.observation_contract},
    )
    encoded = agent.critic.encoder.encode(
        CriticTransition(state, action, state, 0.25)
    )
    assert agent.dqn.encode_state(state) == binding.state_vector(state)
    assert agent.dqn.action_features(state, action) == binding.action_structure(
        state, action
    )
    assert agent.base_neural_prophecy._input(state, action) == (
        binding.state_vector(state) + binding.action_structure(state, action)
    )
    assert encoded == (
        binding.state_vector(state)
        + binding.action_structure(state, action)
        + binding.state_vector(state)
        + (1.0,)
    )
