from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .prophecy import ProphecyStep
from .types import Action, Prediction, StateSnapshot, TransitionTrace

SKILL_VERB = "skill"


@dataclass(frozen=True, slots=True)
class Skill:
    skill_id: str
    primitive_actions: tuple[Action, ...]
    achieved_goal_ids: tuple[str, ...]
    required_facts: frozenset[str]
    added_facts: frozenset[str]
    removed_facts: frozenset[str]
    successes: int
    failures: int = 0

    @property
    def reliability(self) -> float:
        return self.successes / max(
            1,
            self.successes + self.failures,
        )

    def applicable(
        self,
        state: StateSnapshot,
        *,
        minimum_reliability: float = 0.5,
    ) -> bool:
        return (
            self.reliability >= minimum_reliability
            and self.required_facts <= state.facts
        )

    def as_action(self) -> Action:
        return Action(
            SKILL_VERB,
            target=self.skill_id,
            metadata={
                "skill_id": self.skill_id,
                "skill_length": len(self.primitive_actions),
            },
        )


@dataclass(slots=True)
class _SkillCandidate:
    primitive_actions: tuple[Action, ...]
    achieved_goal_ids: tuple[str, ...]
    required_facts: frozenset[str]
    added_facts: frozenset[str]
    removed_facts: frozenset[str]
    successes: int = 0
    failures: int = 0


class SkillLibrary:
    """Promote repeated goal-solving ASeq fragments into reusable actions."""

    def __init__(
        self,
        *,
        promotion_successes: int = 2,
        maximum_length: int = 12,
    ) -> None:
        if promotion_successes <= 0 or maximum_length <= 0:
            raise ValueError(
                "promotion_successes and maximum_length must be positive"
            )
        self.promotion_successes = promotion_successes
        self.maximum_length = maximum_length
        self._candidates: dict[
            tuple[str, ...],
            _SkillCandidate,
        ] = {}
        self._skills: dict[str, Skill] = {}
        self._next_id = 1

    @staticmethod
    def _sequence_key(
        actions: Iterable[Action],
    ) -> tuple[str, ...]:
        return tuple(action.signature for action in actions)

    def observe_goal_completion(
        self,
        traces: Iterable[TransitionTrace],
        *,
        achieved_goal_ids: Iterable[str],
    ) -> Skill | None:
        materialized = tuple(traces)[-self.maximum_length :]
        goals = tuple(sorted(set(achieved_goal_ids)))
        if not materialized or not goals:
            return None
        actions = tuple(trace.action for trace in materialized)
        key = self._sequence_key(actions)
        required = materialized[0].before.facts
        added = frozenset().union(
            *(trace.added_facts for trace in materialized)
        )
        removed = frozenset().union(
            *(trace.removed_facts for trace in materialized)
        )
        candidate = self._candidates.get(key)
        if candidate is None:
            candidate = _SkillCandidate(
                actions,
                goals,
                required,
                added,
                removed,
            )
            self._candidates[key] = candidate
        candidate.successes += 1
        if candidate.successes < self.promotion_successes:
            return None
        existing = next(
            (
                skill
                for skill in self._skills.values()
                if self._sequence_key(skill.primitive_actions) == key
            ),
            None,
        )
        if existing is not None:
            updated = replace(
                existing,
                successes=candidate.successes,
                failures=candidate.failures,
            )
            self._skills[existing.skill_id] = updated
            return updated
        skill_id = f"skill-{self._next_id:04d}"
        self._next_id += 1
        skill = Skill(
            skill_id,
            actions,
            goals,
            required,
            added,
            removed,
            candidate.successes,
            candidate.failures,
        )
        self._skills[skill_id] = skill
        return skill

    def record_failure(self, skill_id: str) -> None:
        skill = self._skills[skill_id]
        self._skills[skill_id] = replace(
            skill,
            failures=skill.failures + 1,
        )

    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def all(self) -> tuple[Skill, ...]:
        return tuple(
            sorted(
                self._skills.values(),
                key=lambda skill: skill.skill_id,
            )
        )

    def actions_for(
        self,
        state: StateSnapshot,
    ) -> tuple[Action, ...]:
        return tuple(
            skill.as_action()
            for skill in self.all()
            if skill.applicable(state)
        )

    def augment_state(
        self,
        state: StateSnapshot,
    ) -> StateSnapshot:
        existing = {
            action.signature: action
            for action in state.available_actions
        }
        for action in self.actions_for(state):
            existing[action.signature] = action
        return state.with_actions(
            tuple(
                sorted(
                    existing.values(),
                    key=lambda action: action.signature,
                )
            )
        )


class SkillAwareProphecy:
    """Roll primitive predictions for one learned skill in imagination."""

    def __init__(
        self,
        base: object,
        library: SkillLibrary,
    ) -> None:
        self.base = base
        self.library = library

    @property
    def name(self) -> str:
        return (
            f"skill-aware:{getattr(self.base, 'name', 'prophecy')}"
        )

    def initial_memory(self) -> Any:
        factory = getattr(self.base, "initial_memory", None)
        return factory() if callable(factory) else None

    def _base_step(
        self,
        state: StateSnapshot,
        action: Action,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        predict_step = getattr(self.base, "predict_step", None)
        if callable(predict_step):
            return predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )
        return ProphecyStep(
            self.base.predict(
                state,
                action,
                samples=samples,
            ),
            memory,
        )

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        if action.verb_name != SKILL_VERB:
            step = self._base_step(
                state,
                action,
                memory,
                samples,
            )
            return ProphecyStep(
                tuple(
                    replace(
                        prediction,
                        next_state=self.library.augment_state(
                            prediction.next_state
                        ),
                    )
                    for prediction in step.predictions
                ),
                step.memory,
            )
        skill = self.library.get(str(action.target))
        current = state
        branch_memory = memory
        probability = 1.0
        for primitive in skill.primitive_actions:
            step = self._base_step(
                current,
                primitive,
                branch_memory,
                1,
            )
            best = max(
                step.predictions,
                key=lambda item: item.probability,
            )
            current = best.next_state
            branch_memory = step.memory
            probability *= best.probability
        current = self.library.augment_state(current)
        return ProphecyStep(
            (
                Prediction(
                    current,
                    max(0.0, min(1.0, probability)),
                    source=f"{self.name}:skill",
                ),
            ),
            branch_memory,
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.predict_step(
            state,
            action,
            memory=self.initial_memory(),
            samples=samples,
        ).predictions

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        """Reuse primitive model confidence instead of re-running predictions.

        Effect-composed coverage asks its wrapped model for confidence for every
        available action. Most actions are primitives, so delegating the base
        model's calibrated confidence avoids an O(action_count) collection of
        redundant GRU rollouts while preserving the same confidence source.
        Learned Skill actions are rare and retain the prediction-derived path.
        """

        if action.verb_name != SKILL_VERB:
            confidence = getattr(self.base, "confidence", None)
            if callable(confidence):
                return max(0.0, min(1.0, float(confidence(state, action))))

        predictions = self.predict(state, action, samples=1)
        value = 0.0
        for prediction in predictions:
            source = prediction.source.lower()
            probability = float(prediction.probability)
            if source.endswith(":unseen"):
                probability = 0.0
            elif source.endswith(":action-family"):
                probability *= 0.5
            value = max(value, probability)
        return max(0.0, min(1.0, value))

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(self.confidence(state, action) for action in materialized) / len(
            materialized
        )

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if action.verb_name == SKILL_VERB:
            return
        self.base.learn(
            state,
            action,
            actual_next_state,
        )


@dataclass(frozen=True, slots=True)
class SkillExecutionResult:
    outcomes: tuple[Any, ...]
    completed: bool


class SkillExecutor:
    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def execute(
        self,
        environment: object,
        action: Action,
    ) -> SkillExecutionResult:
        if action.verb_name != SKILL_VERB:
            return SkillExecutionResult(
                (environment.step(action),),
                True,
            )
        skill = self.library.get(str(action.target))
        outcomes = []
        for primitive in skill.primitive_actions:
            outcome = environment.step(primitive)
            outcomes.append(outcome)
            if getattr(outcome, "error", False):
                self.library.record_failure(skill.skill_id)
                return SkillExecutionResult(
                    tuple(outcomes),
                    False,
                )
        return SkillExecutionResult(
            tuple(outcomes),
            True,
        )
