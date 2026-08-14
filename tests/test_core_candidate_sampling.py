from __future__ import annotations

from aassr_v2.core.plugin_contract import (
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    ValueKind,
)
from aassr_v2.core.public_memory import MemoryBackedRepresentation


SCHEMA = PluginSchema(
    plugin_id="candidate-sampling",
    version="v1",
    observations=(
        ObservationField("place", ValueKind.ENTITY),
        ObservationField("choices", ValueKind.SET, item_kind=ValueKind.ENTITY),
    ),
    actions=(
        ActionSpec(
            "choose",
            parameters=(ActionParameter("target", ValueKind.ENTITY),),
        ),
    ),
)


def _targets(commands) -> tuple[str, ...]:
    return tuple(sorted(str(item.arguments["target"]) for item in commands))


def test_bounded_surface_is_not_lexicographic_prefix() -> None:
    representation = MemoryBackedRepresentation(SCHEMA, candidate_seed=17)
    representation.begin_episode()
    choices = tuple(f"id-{index:02d}" for index in range(20))
    observation = PluginObservation({"place": "start", "choices": choices})
    representation.public_knowledge.observe(SCHEMA, observation)

    commands = representation.synthesize_commands(
        observation,
        per_parameter_limit=4,
        total_limit=4,
    )
    selected = _targets(commands)
    all_values = tuple(sorted(("start", *choices)))
    assert len(selected) == 4
    assert selected != all_values[:4]
    assert representation.candidate_sampling_events > 0


def test_same_episode_same_public_evidence_has_stable_surface() -> None:
    representation = MemoryBackedRepresentation(SCHEMA, candidate_seed=23)
    representation.begin_episode()
    observation = PluginObservation(
        {
            "place": "start",
            "choices": tuple(f"item-{index:02d}" for index in range(20)),
        }
    )
    representation.public_knowledge.observe(SCHEMA, observation)
    first = _targets(
        representation.synthesize_commands(
            observation,
            per_parameter_limit=5,
            total_limit=5,
        )
    )
    second = _targets(
        representation.synthesize_commands(
            observation,
            per_parameter_limit=5,
            total_limit=5,
        )
    )
    assert first == second
