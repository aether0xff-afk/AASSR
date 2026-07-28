from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import ActionVerb


class CurriculumStage(str, Enum):
    BASIC_CONTROL = "basic_control"
    STATE_CHANGE = "state_change"
    INFORMATION_ENABLES_ACTION = "information_enables_action"
    SEQUENCE_COMPOSITION = "sequence_composition"
    ATTRIBUTE_RELATION = "attribute_relation"
    OPEN_GENERALIZATION = "open_generalization"


@dataclass(frozen=True, slots=True)
class CurriculumSpecification:
    stage: CurriculumStage
    allowed_verbs: frozenset[ActionVerb]
    final_goal: str
    concepts_exposed: tuple[str, ...]
    demonstrations_allowed: bool = False


DEFAULT_CURRICULUM: tuple[CurriculumSpecification, ...] = (
    CurriculumSpecification(
        stage=CurriculumStage.BASIC_CONTROL,
        allowed_verbs=frozenset({ActionVerb.MOVE, ActionVerb.OBSERVE}),
        final_goal="move to the visible goal",
        concepts_exposed=("movement", "observation", "goal completion"),
        demonstrations_allowed=True,
    ),
    CurriculumSpecification(
        stage=CurriculumStage.STATE_CHANGE,
        allowed_verbs=frozenset(
            {ActionVerb.MOVE, ActionVerb.OBSERVE, ActionVerb.USE}
        ),
        final_goal="reach a goal after changing an object's state",
        concepts_exposed=("action changes state", "obstacle constrains movement"),
    ),
    CurriculumSpecification(
        stage=CurriculumStage.INFORMATION_ENABLES_ACTION,
        allowed_verbs=frozenset(
            {ActionVerb.MOVE, ActionVerb.OBSERVE, ActionVerb.PICKUP, ActionVerb.USE}
        ),
        final_goal="reach the goal using information or an acquired object",
        concepts_exposed=("information unlocks action", "object enables action"),
    ),
    CurriculumSpecification(
        stage=CurriculumStage.SEQUENCE_COMPOSITION,
        allowed_verbs=frozenset(ActionVerb),
        final_goal="compose multiple actions to reach the goal",
        concepts_exposed=("ordered dependency", "delayed usefulness"),
    ),
    CurriculumSpecification(
        stage=CurriculumStage.ATTRIBUTE_RELATION,
        allowed_verbs=frozenset(ActionVerb),
        final_goal="infer which attribute-matched objects should interact",
        concepts_exposed=("attribute binding", "relation inference"),
    ),
    CurriculumSpecification(
        stage=CurriculumStage.OPEN_GENERALIZATION,
        allowed_verbs=frozenset(ActionVerb),
        final_goal="solve an unseen environment from the final goal only",
        concepts_exposed=("self-directed information search", "novel composition"),
    ),
)
