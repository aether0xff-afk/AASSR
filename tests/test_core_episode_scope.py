from __future__ import annotations

from aassr_v2.core.plugin_contract import (
    ActionCommand,
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    ValueKind,
)
from aassr_v2.core.public_memory import MemoryBackedRepresentation


SCHEMA = PluginSchema(
    plugin_id="episode-scope",
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


def _result(place: str, choices: tuple[str, ...], *, reward: float = 0.0):
    return PluginStepResult(
        PluginObservation({"place": place, "choices": choices}),
        reward=reward,
    )


def test_concrete_candidate_experience_resets_with_episode_knowledge() -> None:
    representation = MemoryBackedRepresentation(SCHEMA)
    representation.begin_episode()
    before = _result("start", ("a", "b"))
    after = _result("next", ("a", "b"), reward=-1.0)
    snapshot = representation.to_snapshot(before)
    action = ActionCommand("choose", {"target": "a"}).to_action(SCHEMA)
    representation.observe_real_transition(
        before_result=before,
        action=action,
        after_result=after,
    )
    assert representation.experience.features(action.signature)[0] > 0.0

    representation.begin_episode(preserve=False)
    assert representation.experience.features(action.signature)[0] == 0.0
    assert representation.public_knowledge.revision == 0


def test_explicit_preserve_keeps_concrete_evidence() -> None:
    representation = MemoryBackedRepresentation(SCHEMA)
    representation.begin_episode()
    before = _result("start", ("a",))
    after = _result("same", ("a",), reward=1.0)
    action = ActionCommand("choose", {"target": "a"}).to_action(SCHEMA)
    representation.observe_real_transition(
        before_result=before,
        action=action,
        after_result=after,
    )
    uses = representation.experience.features(action.signature)[0]

    representation.begin_episode(preserve=True)
    assert representation.experience.features(action.signature)[0] == uses
