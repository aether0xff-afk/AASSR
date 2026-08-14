from __future__ import annotations

from aassr_v2.core import (
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    ValueKind,
)
from aassr_v2.core.public_memory import MemoryBackedRepresentation


SCHEMA = PluginSchema(
    plugin_id="value-space-test",
    version="v1",
    observations=(
        ObservationField(
            "page_text",
            ValueKind.TEXT,
            value_space="response-body",
        ),
        ObservationField(
            "payload_templates",
            ValueKind.SET,
            item_kind=ValueKind.TEXT,
            value_space="form-payload",
        ),
        ObservationField(
            "links",
            ValueKind.SET,
            item_kind=ValueKind.ENTITY,
            value_space="url",
        ),
        ObservationField(
            "object_ids",
            ValueKind.SET,
            item_kind=ValueKind.ENTITY,
            value_space="object-id",
        ),
    ),
    actions=(
        ActionSpec(
            "submit",
            parameters=(
                ActionParameter(
                    "body",
                    ValueKind.TEXT,
                    value_space="form-payload",
                ),
            ),
        ),
        ActionSpec(
            "visit",
            parameters=(
                ActionParameter(
                    "url",
                    ValueKind.ENTITY,
                    value_space="url",
                ),
            ),
        ),
    ),
)


def _pairs(representation, observation):
    return {
        (command.action_id, tuple(sorted(command.arguments.items())))
        for command in representation.synthesize_commands(
            observation,
            per_parameter_limit=16,
            total_limit=64,
        )
    }


def test_same_python_kind_does_not_cross_mechanical_value_spaces() -> None:
    representation = MemoryBackedRepresentation(SCHEMA, candidate_seed=7)
    representation.begin_episode()
    commands = _pairs(
        representation,
        PluginObservation(
            {
                "page_text": "THIS IS A RESPONSE, NOT A REQUEST PAYLOAD",
                "payload_templates": ("name=&token=",),
                "links": ("/alpha",),
                "object_ids": ("object-17",),
            }
        ),
    )

    assert ("submit", (("body", "name=&token="),)) in commands
    assert (
        "submit",
        (("body", "THIS IS A RESPONSE, NOT A REQUEST PAYLOAD"),),
    ) not in commands
    assert ("visit", (("url", "/alpha"),)) in commands
    assert ("visit", (("url", "object-17"),)) not in commands


def test_core_public_memory_preserves_value_space_when_reusing_old_values() -> None:
    representation = MemoryBackedRepresentation(SCHEMA, candidate_seed=11)
    representation.begin_episode()

    first = PluginObservation(
        {
            "page_text": "first page",
            "payload_templates": ("q=",),
            "links": ("/remembered",),
            "object_ids": ("entity-that-is-not-a-url",),
        }
    )
    representation.public_knowledge.observe(SCHEMA, first)

    second = PluginObservation(
        {
            "page_text": "second page",
            "payload_templates": (),
            "links": (),
            "object_ids": (),
        }
    )
    commands = _pairs(representation, second)

    assert ("visit", (("url", "/remembered"),)) in commands
    assert (
        "visit",
        (("url", "entity-that-is-not-a-url"),),
    ) not in commands
    assert ("submit", (("body", "q="),)) in commands
    assert ("submit", (("body", "first page"),)) not in commands
