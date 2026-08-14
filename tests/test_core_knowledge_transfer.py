from __future__ import annotations

from aassr_v2.core import (
    ObservationField,
    PluginObservation,
    PluginSchema,
    TemporalKind,
    ValueKind,
)
from aassr_v2.core.public_memory import CorePublicKnowledge, MemoryBackedRepresentation


SCHEMA = PluginSchema(
    plugin_id="knowledge-transfer-test",
    version="v1",
    observations=(
        ObservationField(
            "mode",
            ValueKind.CATEGORICAL,
            TemporalKind.EVENT,
            value_space="mode",
        ),
        ObservationField(
            "message",
            ValueKind.TEXT,
            TemporalKind.EVENT,
            value_space="message",
        ),
        ObservationField(
            "entity",
            ValueKind.ENTITY,
            TemporalKind.STATE,
            value_space="object-id",
        ),
        ObservationField(
            "objects",
            ValueKind.SET,
            TemporalKind.STATE,
            item_kind=ValueKind.ENTITY,
            value_space="object-id",
        ),
        ObservationField(
            "counter",
            ValueKind.SCALAR,
            TemporalKind.COUNTER,
            value_space="counter",
        ),
        ObservationField(
            "latency",
            ValueKind.SCALAR,
            TemporalKind.MEASUREMENT,
            value_space="latency",
        ),
    ),
    actions=(),
)


def _observation(
    *,
    mode: str = "alpha",
    message: str = "door locked",
    entity: str = "object-a",
    objects: tuple[str, ...] = ("object-b", "object-c"),
    counter: float = 1.0,
    latency: float = 5.0,
) -> PluginObservation:
    return PluginObservation(
        {
            "mode": mode,
            "message": message,
            "entity": entity,
            "objects": objects,
            "counter": counter,
            "latency": latency,
        }
    )


def test_remembered_categorical_and_text_content_reaches_transfer_vector() -> None:
    left = CorePublicKnowledge()
    right = CorePublicKnowledge()
    left.observe(SCHEMA, _observation(mode="alpha", message="door locked"))
    right.observe(SCHEMA, _observation(mode="beta", message="door opened"))

    # Both memories have the same number/types of values. A count-only memory
    # would therefore be identical; remembered public content must now survive.
    assert left.diagnostics()["values:categorical"] == right.diagnostics()[
        "values:categorical"
    ]
    assert left.diagnostics()["values:text"] == right.diagnostics()["values:text"]
    assert left.structural_vector(256) != right.structural_vector(256)
    assert left.diagnostics()["transfer_evidence"] > 0


def test_concrete_entity_renaming_does_not_change_transfer_knowledge_vector() -> None:
    left = CorePublicKnowledge()
    right = CorePublicKnowledge()
    left.observe(
        SCHEMA,
        _observation(
            entity="object-a",
            objects=("object-b", "object-c"),
        ),
    )
    right.observe(
        SCHEMA,
        _observation(
            entity="renamed-x",
            objects=("renamed-y", "renamed-z"),
        ),
    )

    # Exact values remain available inside each episode for command execution,
    # but learner-facing structural memory must not encode the concrete names.
    assert left.all_values(ValueKind.ENTITY, value_space="object-id") != right.all_values(
        ValueKind.ENTITY,
        value_space="object-id",
    )
    assert left.structural_vector(256) == right.structural_vector(256)


def test_counter_and_measurement_history_is_not_persisted_as_knowledge() -> None:
    schema = PluginSchema(
        plugin_id="volatile-only",
        version="v1",
        observations=(
            ObservationField("count", ValueKind.SCALAR, TemporalKind.COUNTER),
            ObservationField("latency", ValueKind.SCALAR, TemporalKind.MEASUREMENT),
        ),
        actions=(),
    )
    knowledge = CorePublicKnowledge()
    knowledge.observe(schema, PluginObservation({"count": 1.0, "latency": 5.0}))
    knowledge.observe(schema, PluginObservation({"count": 2.0, "latency": 9.0}))

    diagnostics = knowledge.diagnostics()
    assert diagnostics["values:scalar"] == 0
    assert diagnostics["semantic_evidence"] == 0
    assert diagnostics["transfer_evidence"] == 0
    assert diagnostics["revision"] == 0


def test_current_counter_and_measurement_are_still_visible_to_current_state() -> None:
    schema = PluginSchema(
        plugin_id="volatile-current-state",
        version="v1",
        observations=(
            ObservationField("count", ValueKind.SCALAR, TemporalKind.COUNTER),
            ObservationField("latency", ValueKind.SCALAR, TemporalKind.MEASUREMENT),
        ),
        actions=(),
    )
    representation = MemoryBackedRepresentation(schema)

    first = representation.observation_vector(
        PluginObservation({"count": 1.0, "latency": 5.0})
    )
    second = representation.observation_vector(
        PluginObservation({"count": 8.0, "latency": 50.0})
    )

    assert first != second
    assert representation.public_knowledge.diagnostics()["transfer_evidence"] == 0
