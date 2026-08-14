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
from aassr_v2.core.public_memory import MemoryBackedRepresentation
from aassr_v2.core.representation import SchemaDrivenRepresentation
from aassr_v2.semantic_control import SemanticSelfLoopASEQ


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
            parameters=(ActionParameter("target", ValueKind.ENTITY),),
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
        {"place": "same", "choices": ("a", "b"), "turn_count": 1}
    )
    right = PluginObservation(
        {"place": "same", "choices": ("a", "b"), "turn_count": 99}
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
        {"place": "start", "choices": ("a", "b"), "turn_count": 0}
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
    observation = PluginObservation(
        {"place": "start", "choices": (), "turn_count": 0}
    )
    assert not hasattr(observation, "commands")


def _targets(snapshot) -> set[str]:
    return {
        str(action.parameters["target"])
        for action in snapshot.available_actions
        if action.verb_name == "choose"
    }


def test_core_not_plugin_retains_public_entity_history() -> None:
    representation = MemoryBackedRepresentation(SCHEMA)
    representation.begin_episode()

    first = representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {"place": "start", "choices": ("a",), "turn_count": 0}
            )
        )
    )
    assert "a" in _targets(first)

    second = representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {"place": "next", "choices": ("b",), "turn_count": 1}
            )
        )
    )
    # "a" is no longer in the current observation, but the Core remembers that
    # it was publicly observed earlier and may still choose it.
    assert {"a", "b"} <= _targets(second)

    representation.begin_episode(preserve=False)
    third = representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {"place": "fresh", "choices": ("b",), "turn_count": 0}
            )
        )
    )
    assert "a" not in _targets(third)


def test_repeated_counter_jitter_does_not_defeat_aseq_semantics() -> None:
    representation = MemoryBackedRepresentation(SCHEMA)
    representation.begin_episode()
    first = representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {"place": "same", "choices": ("a",), "turn_count": 1}
            )
        )
    )
    second = representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {"place": "same", "choices": ("a",), "turn_count": 999}
            )
        )
    )
    semantic_first = representation.semantic_state_identity(first)
    semantic_second = representation.semantic_state_identity(second)
    assert semantic_first == semantic_second

    action = next(
        item
        for item in first.available_actions
        if item.parameters.get("target") == "a"
    )
    aseq = SemanticSelfLoopASEQ(repeat_threshold=2)
    aseq.observe(semantic_first, action, semantic_second)
    aseq.observe(semantic_first, action, semantic_second)
    filtered, guarded, fallback = aseq.filter_state(first, semantic_first)
    assert guarded == 1
    assert fallback is False
    assert action.signature not in {
        item.signature for item in filtered.available_actions
    }


def test_event_mapping_value_jitter_does_not_create_fake_new_knowledge() -> None:
    schema = PluginSchema(
        plugin_id="event-jitter",
        version="v1",
        observations=(
            ObservationField("place", ValueKind.ENTITY, TemporalKind.STATE),
            ObservationField("headers", ValueKind.MAPPING, TemporalKind.EVENT),
            ObservationField("latency", ValueKind.SCALAR, TemporalKind.MEASUREMENT),
        ),
        actions=(),
    )
    representation = MemoryBackedRepresentation(schema)
    representation.begin_episode()
    representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {
                    "place": "same",
                    "headers": {"date": "A", "kind": "x"},
                    "latency": 1.0,
                }
            )
        )
    )
    revision = representation.public_knowledge.revision
    representation.to_snapshot(
        PluginStepResult(
            PluginObservation(
                {
                    "place": "same",
                    "headers": {"date": "B", "kind": "y"},
                    "latency": 999.0,
                }
            )
        )
    )
    # Same public header keys and same persistent state: changing volatile values
    # must not make every request look like semantic progress.
    assert representation.public_knowledge.revision == revision
