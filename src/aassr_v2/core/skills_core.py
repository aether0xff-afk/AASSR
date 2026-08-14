from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Callable, Sequence

from ..skills import SKILL_VERB, Skill
from ..types import Action, Prediction, StateSnapshot, TransitionTrace
from .representation import SchemaDrivenRepresentation


PrimitiveValueFn = Callable[[StateSnapshot, Action], float]


@dataclass(slots=True)
class _CoreSkillCandidate:
    actions: tuple[Action, ...]
    templates: tuple[tuple[float, ...], ...]
    goals: tuple[str, ...]
    added: frozenset[str]
    removed: frozenset[str]
    successes: int = 0
    failures: int = 0


class CoreRelationalSkillLibrary:
    """Reuse learned action *structure* rather than environment identifiers.

    A structural Skill may match several concrete actions in a new state.  The
    library must not silently ground that ambiguity by lexicographic identifier.
    When a Core Policy value function is supplied, the currently learned value
    chooses the concrete grounding. Exact value ties use a deterministic seeded
    rank whose seed contains no concrete action signature.
    """

    def __init__(
        self,
        representation: SchemaDrivenRepresentation,
        *,
        primitive_value: PrimitiveValueFn | None = None,
        seed: int = 0,
        promotion_successes: int = 2,
        maximum_length: int = 12,
    ) -> None:
        if promotion_successes <= 0 or maximum_length <= 0:
            raise ValueError("skill thresholds must be positive")
        self.representation = representation
        self.primitive_value = primitive_value
        self.seed = int(seed)
        self.promotion_successes = int(promotion_successes)
        self.maximum_length = int(maximum_length)
        self._candidates: dict[
            tuple[tuple[float, ...], ...],
            _CoreSkillCandidate,
        ] = {}
        self._templates: dict[str, tuple[tuple[float, ...], ...]] = {}
        self._skills: dict[str, Skill] = {}
        self._next_id = 1
        self.ambiguous_groundings = 0
        self.value_groundings = 0
        self.symmetric_groundings = 0

    def _trace_templates(
        self,
        traces: Sequence[TransitionTrace],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            self.representation.action_structure(trace.before, trace.action)
            for trace in traces
        )

    def observe_goal_completion(
        self,
        traces: Sequence[TransitionTrace],
        *,
        achieved_goal_ids: Sequence[str],
    ) -> Skill | None:
        materialized = tuple(traces)[-self.maximum_length :]
        goals = tuple(sorted(set(achieved_goal_ids)))
        if not materialized or not goals:
            return None
        templates = self._trace_templates(materialized)
        candidate = self._candidates.get(templates)
        if candidate is None:
            candidate = _CoreSkillCandidate(
                actions=tuple(trace.action for trace in materialized),
                templates=templates,
                goals=goals,
                added=frozenset().union(
                    *(trace.added_facts for trace in materialized)
                ),
                removed=frozenset().union(
                    *(trace.removed_facts for trace in materialized)
                ),
            )
            self._candidates[templates] = candidate
        candidate.successes += 1
        if candidate.successes < self.promotion_successes:
            return None

        existing = next(
            (
                skill_id
                for skill_id, stored in self._templates.items()
                if stored == templates
            ),
            None,
        )
        if existing is not None:
            old = self._skills[existing]
            updated = replace(
                old,
                successes=candidate.successes,
                failures=candidate.failures,
            )
            self._skills[existing] = updated
            return updated

        skill_id = f"skill-{self._next_id:04d}"
        self._next_id += 1
        skill = Skill(
            skill_id=skill_id,
            primitive_actions=candidate.actions,
            achieved_goal_ids=goals,
            required_facts=frozenset(),
            added_facts=candidate.added,
            removed_facts=candidate.removed,
            successes=candidate.successes,
            failures=candidate.failures,
        )
        self._skills[skill_id] = skill
        self._templates[skill_id] = templates
        return skill

    def record_failure(self, skill_id: str) -> None:
        old = self._skills[skill_id]
        self._skills[skill_id] = replace(old, failures=old.failures + 1)

    def template_length(self, skill_id: str) -> int:
        return len(self._templates[skill_id])

    def _symmetric_tie_index(
        self,
        *,
        skill_id: str,
        step_index: int,
        state: StateSnapshot,
        candidate_count: int,
    ) -> int:
        semantic = self.representation.semantic_state_identity(state)
        payload = (
            self.seed,
            skill_id,
            int(step_index),
            semantic,
            int(candidate_count),
        )
        digest = hashlib.blake2b(
            repr(payload).encode("utf-8"),
            digest_size=16,
        ).digest()
        return int.from_bytes(digest, "big") % max(1, int(candidate_count))

    def resolve_primitive(
        self,
        skill_id: str,
        index: int,
        state: StateSnapshot,
    ) -> Action | None:
        templates = self._templates.get(skill_id)
        if templates is None or not 0 <= index < len(templates):
            return None
        target = templates[index]
        candidates = tuple(
            action
            for action in state.available_actions
            if action.verb_name != SKILL_VERB
            and self.representation.action_structure(state, action) == target
        )
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        self.ambiguous_groundings += 1
        ordered = tuple(sorted(candidates, key=lambda action: action.signature))
        tied = ordered
        if self.primitive_value is not None:
            scored = tuple(
                (action, float(self.primitive_value(state, action)))
                for action in ordered
            )
            best = max(value for _, value in scored)
            tied = tuple(
                action
                for action, value in scored
                if abs(value - best) <= 1e-12
            )
            if len(tied) < len(ordered):
                self.value_groundings += 1
        if len(tied) == 1:
            return tied[0]

        self.symmetric_groundings += 1
        choice = self._symmetric_tie_index(
            skill_id=skill_id,
            step_index=index,
            state=state,
            candidate_count=len(tied),
        )
        return tied[choice]

    def actions_for(self, state: StateSnapshot) -> tuple[Action, ...]:
        rows = []
        for skill in self.all():
            if skill.reliability < 0.5:
                continue
            if self.resolve_primitive(skill.skill_id, 0, state) is not None:
                rows.append(skill.as_action())
        return tuple(rows)

    def augment_state(self, state: StateSnapshot) -> StateSnapshot:
        actions = {item.signature: item for item in state.available_actions}
        for item in self.actions_for(state):
            actions[item.signature] = item
        return state.with_actions(
            tuple(actions[key] for key in sorted(actions))
        )

    def all(self) -> tuple[Skill, ...]:
        return tuple(
            sorted(self._skills.values(), key=lambda item: item.skill_id)
        )

    def diagnostics(self) -> dict[str, int]:
        return {
            "candidates": len(self._candidates),
            "templates": len(self._templates),
            "promoted": len(self._skills),
            "ambiguous_groundings": self.ambiguous_groundings,
            "value_groundings": self.value_groundings,
            "symmetric_groundings": self.symmetric_groundings,
        }


class CoreSkillAwareProphecy:
    """Resolve structural Skill steps against each imagined state."""

    name = "core-skill-aware-prophecy-v1"

    def __init__(
        self,
        base: object,
        library: CoreRelationalSkillLibrary,
    ) -> None:
        self.base = base
        self.library = library

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _skill_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        skill_id = str(action.target)
        current = state
        probability = 1.0
        for index in range(self.library.template_length(skill_id)):
            primitive = self.library.resolve_primitive(skill_id, index, current)
            if primitive is None:
                return (
                    Prediction(
                        current,
                        0.0,
                        source=f"{self.name}:unavailable",
                    ),
                )
            predictions = self.base.predict(
                current,
                primitive,
                samples=max(1, samples if index == 0 else 1),
            )
            best = max(predictions, key=lambda item: item.probability)
            current = best.next_state
            probability *= float(best.probability)
        return (
            Prediction(
                self.library.augment_state(current),
                max(0.0, min(1.0, probability)),
                source=f"{self.name}:skill",
            ),
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if action.verb_name == SKILL_VERB:
            return self._skill_predictions(state, action, samples=samples)
        return tuple(
            replace(
                prediction,
                next_state=self.library.augment_state(prediction.next_state),
            )
            for prediction in self.base.predict(state, action, samples=samples)
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        if action.verb_name != SKILL_VERB:
            return float(self.base.confidence(state, action))
        return max(
            (
                float(item.probability)
                for item in self._skill_predictions(state, action, samples=1)
            ),
            default=0.0,
        )

    def coverage(
        self,
        state: StateSnapshot,
        actions: Sequence[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(
            self.confidence(state, action)
            for action in materialized
        ) / len(materialized)

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if action.verb_name != SKILL_VERB:
            self.base.learn(state, action, actual_next_state)
