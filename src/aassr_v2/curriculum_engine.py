from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .action_plugins import (
    ActionRegistry,
    ParameterLesson,
    parameter_lessons,
)


class AcademyStage(str, Enum):
    BASIC_CONTROL = "basic_control"
    OBSTACLE_NAVIGATION = "obstacle_navigation"
    ACQUISITION = "acquisition"
    STATE_CHANGE = "state_change"
    DEPENDENCY = "dependency"
    ATTRIBUTE_RELATION = "attribute_relation"
    OPEN_GENERALIZATION = "open_generalization"
    PLUGIN_PARAMETERS = "plugin_parameters"


@dataclass(frozen=True, slots=True)
class AcademyLesson:
    stage: AcademyStage
    goal_description: str
    concepts: tuple[str, ...]
    demonstrations_allowed: bool
    parameter_lessons: tuple[ParameterLesson, ...] = ()


DEFAULT_ACADEMY: tuple[AcademyLesson, ...] = (
    AcademyLesson(
        AcademyStage.BASIC_CONTROL,
        "move and observe",
        ("control", "observation"),
        True,
    ),
    AcademyLesson(
        AcademyStage.OBSTACLE_NAVIGATION,
        "reach a goal around obstacles",
        ("blocked movement", "alternate path"),
        False,
    ),
    AcademyLesson(
        AcademyStage.ACQUISITION,
        "obtain a visible resource",
        ("acquisition changes state",),
        False,
    ),
    AcademyLesson(
        AcademyStage.STATE_CHANGE,
        "change an object state before proceeding",
        ("causal state change",),
        False,
    ),
    AcademyLesson(
        AcademyStage.DEPENDENCY,
        "solve a multi-step dependency",
        ("delayed usefulness", "ordered dependency"),
        False,
    ),
    AcademyLesson(
        AcademyStage.ATTRIBUTE_RELATION,
        "infer a relation from attributes",
        ("role binding", "functional similarity"),
        False,
    ),
    AcademyLesson(
        AcademyStage.OPEN_GENERALIZATION,
        "solve an unseen composition from the final goal only",
        ("self-directed exploration",),
        False,
    ),
)


@dataclass(slots=True)
class CurriculumTeacher:
    lessons: tuple[AcademyLesson, ...] = DEFAULT_ACADEMY
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.25
    window: int = 10
    _index: int = 0
    _recent: deque[float] = field(default_factory=deque)

    @property
    def current(self) -> AcademyLesson:
        return self.lessons[self._index]

    @property
    def completed(self) -> bool:
        return (
            self._index == len(self.lessons) - 1
            and len(self._recent) >= self.window
            and sum(self._recent) / len(self._recent)
            >= self.promotion_threshold
        )

    def observe(self, success: bool) -> AcademyLesson:
        self._recent.append(1.0 if success else 0.0)
        while len(self._recent) > self.window:
            self._recent.popleft()
        if len(self._recent) < self.window:
            return self.current
        rate = sum(self._recent) / len(self._recent)
        if (
            rate >= self.promotion_threshold
            and self._index < len(self.lessons) - 1
        ):
            self._index += 1
            self._recent.clear()
        elif (
            rate <= self.demotion_threshold
            and self._index > 0
        ):
            self._index -= 1
            self._recent.clear()
        return self.current

    @classmethod
    def from_registry(
        cls,
        registry: ActionRegistry,
    ) -> CurriculumTeacher:
        parameter_material = tuple(
            lesson
            for schema in registry.schemas()
            for lesson in parameter_lessons(schema)
        )
        lessons = DEFAULT_ACADEMY + (
            AcademyLesson(
                AcademyStage.PLUGIN_PARAMETERS,
                (
                    "use required and optional command parameters "
                    "without receiving optimal combinations"
                ),
                (
                    "syntax",
                    "required slots",
                    "optional slots",
                    "defaults",
                ),
                False,
                parameter_material,
            ),
        )
        return cls(lessons=lessons)
