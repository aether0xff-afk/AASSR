from __future__ import annotations

from dataclasses import dataclass

import pytest

from aassr_v2.core.plugin_contract import (
    ActionCommand,
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    TemporalKind,
    ValueKind,
    validate_minimal_plugin,
)
from aassr_v2.core.representation import SchemaDrivenRepresentation


SCHEMA = PluginSchema(
    plugin_id="toy-minimal",
    version="v1",
    observations=(
        ObservationField("place", ValueKind.ENTITY),
        ObservationField(
            "choices",
            ValueKind.SET,
            item_kind=ValueKind.ENTITY,
        ),
        ObservationField(
            "turn_count",
            ValueKind.SCALAR,
            TemporalKind.COUNTER,
        ),
    ),
    actions=(
        ActionSpec(
            "choose",
            parameters=(
                ActionParameter("target", ValueKind.ENTITY),
            ),
        ),
    ),
)


@dataclass
class GoodPlugin:
    schema = SCHEMA

    def reset(self, *, seed=None):
        del seed
        return PluginStepResult(
            PluginObservation(
                {
                    "place": "start",
                    "choices": ("a", "b"),
                    "turn_count": 0,
                }
            )
        )

    def step(self, command):
        return PluginStepResult(
            PluginObservation(
                {
                    "place": str(command.arguments["target"]),
                    "choices": ("a", "b"),
                    "turn_count": 1,
                }
            )
        )


class TooPowerfulPlugin(GoodPlugin):
    def install_world_model(self):
        raise AssertionError("must never be called")


def test_minimal_plugin_rejects_core_authority() -> None:
    validate_minimal_plugin(GoodPlugin())
    with pytest.raises(TypeError, match="Core-owned authority"):
        validate_minimal_plugin(TooPowerfulPlugin())


def test_core_synthesizes_actions_from_data_types() -> None:
    representation = SchemaDrivenRepresentation(SCHEMA)
    observation = PluginObservation(
        {
            "place": "start",
            "choices": ("a", "b"),
            "turn_count": 0,
        }
    )
    commands = representation.synthesize_commands(observation)
    assert {
        (item.action_id, item.arguments["target"])
        for item in commands
    } == {
        ("choose", "start"),
        ("choose", "a"),
        ("choose", "b"),
    }


def test_counter_is_not_a_semantic_identity_shortcut() -> None:
    representation = SchemaDrivenRepresentation(SCHEMA)
    left = PluginObservation(
        {
            "place": "same",
            "choices": ("a", "b"),
            "turn_count": 1,
        }
    )
    right = PluginObservation(
        {
            "place": "same",
            "choices": ("a", "b"),
            "turn_count": 99,
        }
    )
    assert (
        representation.semantic_observation_identity(left)
        == representation.semantic_observation_identity(right)
    )


def test_plugin_command_remains_syntax_only() -> None:
    action = ActionCommand("choose", {"target": "a"}).to_action(SCHEMA)
    assert action.verb_name == "choose"
    assert action.parameters["target"] == "a"
    assert action.metadata["plugin_id"] == "toy-minimal"


def test_core_separates_structure_from_concrete_execution_identity() -> None:
    representation = SchemaDrivenRepresentation(SCHEMA)
    observation = PluginObservation(
        {
            "place": "start",
            "choices": ("a", "b"),
            "turn_count": 0,
        }
    )
    left = ActionCommand("choose", {"target": "a"}).to_action(SCHEMA)
    right = ActionCommand("choose", {"target": "b"}).to_action(SCHEMA)

    assert (
        representation.action_structure_from_observation(observation, left)
        == representation.action_structure_from_observation(observation, right)
    )
    assert (
        representation.action_features_from_observation(observation, left)
        == representation.action_features_from_observation(observation, right)
    )
    assert left.signature != right.signature


def test_transfer_state_ignores_concrete_entity_renaming() -> None:
    representation = SchemaDrivenRepresentation(SCHEMA)
    left = PluginObservation(
        {"place": "start", "choices": ("a", "b"), "turn_count": 0}
    )
    right = PluginObservation(
        {"place": "origin", "choices": ("x", "y"), "turn_count": 0}
    )
    assert representation.observation_vector(left) == representation.observation_vector(right)
    assert (
        representation.semantic_observation_identity(left)
        != representation.semantic_observation_identity(right)
    )


def test_plugin_observation_has_no_action_candidate_channel() -> None:
    observation = PluginObservation({"place": "start", "choices": (), "turn_count": 0})
    assert not hasattr(observation, "commands")
