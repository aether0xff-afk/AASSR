from __future__ import annotations

from aassr_v2.aassr_core import FULL_CORE_EVIDENCE, AASSRCoreConfig
from aassr_v2.grid_push_core_comparison import (
    CONDITIONS,
    _fixture_spec,
    _module_fixture,
    _observable_transition,
)
from aassr_v2.grid_push_plugin import GridPushEnvironmentPlugin
from aassr_v2.imagination_tree import ImaginationConfig


def _fixture_config() -> AASSRCoreConfig:
    return AASSRCoreConfig(
        epsilon_start=0.0,
        epsilon_end=0.0,
        skill_promotion_successes=1,
        imagination_minimum_coverage=0.0,
        imagination=ImaginationConfig(
            branching_factor=2,
            maximum_depth=2,
            beam_width=8,
            outcome_samples=1,
            minimum_path_confidence=0.0,
            uncertainty_penalty=0.0,
            update_policy=False,
        ),
    )


def test_comparison_condition_names_are_exact_and_nonoverlapping() -> None:
    assert CONDITIONS == (
        "random",
        "contextual_policy",
        "reduced_causal_agent",
        "tabular_fixed_goal_core",
        "full_aassr_no_imagination",
        "full_aassr",
    )


def test_module_fixture_records_every_required_connection() -> None:
    evidence, traces = _module_fixture(_fixture_config(), 6)
    assert evidence["all_required_evidence_positive"]
    assert all(
        evidence["audit"]["evidence"][name] > 0
        for name in FULL_CORE_EVIDENCE
    )
    assert evidence["goal_generator_trace"]
    assert evidence["gru_training_trace"]["prediction_loss_after"] > 0.0
    assert evidence["information_flow_trace"]["predicted_information_value"] != 0.0
    assert evidence["frozen_checkpoint_immutable"]
    assert evidence["frozen_learning_updates_zero"]
    assert traces


def test_reduced_agent_adapter_uses_only_observable_plugin_transition() -> None:
    spec = _fixture_spec()
    plugin = GridPushEnvironmentPlugin(lambda _seed: spec)
    before = plugin.reset(88001)
    action = next(
        schema for schema in plugin.action_schemas() if schema.action_id == "MOVE_EAST"
    ).build({})
    visible = plugin.execute(action)
    adapted = _observable_transition(visible)
    assert adapted.before == before
    assert adapted.after == visible.after
    assert adapted.action_succeeded
    assert adapted.terminal_reward == 1.0

