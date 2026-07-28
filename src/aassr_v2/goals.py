from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Iterable

from .types import Action, StateSnapshot


class GoalKind(str, Enum):
    FACT_PRESENT = "fact_present"
    FACT_ABSENT = "fact_absent"
    ACTION_AVAILABLE = "action_available"
    GOAL_PROGRESS = "goal_progress"
    VECTOR_TARGET = "vector_target"
    KNOWLEDGE_PRESENT = "knowledge_present"


@dataclass(frozen=True, slots=True)
class Goal:
    goal_id: str
    kind: GoalKind
    target: str | tuple[float, ...] | float
    priority: float = 1.0
    threshold: float = 1.0
    parent_goal_id: str | None = None
    source: str = "external"
    final: bool = False


@dataclass(frozen=True, slots=True)
class GoalEvaluation:
    goal_id: str
    satisfaction: float
    achieved: bool


@dataclass(slots=True)
class GoalSet:
    _goals: dict[str, Goal] = field(default_factory=dict)

    def add(self, goal: Goal) -> None:
        self._goals[goal.goal_id] = goal

    def remove(self, goal_id: str) -> None:
        self._goals.pop(goal_id, None)

    def all(self) -> tuple[Goal, ...]:
        return tuple(
            sorted(
                self._goals.values(),
                key=lambda item: (-item.priority, item.goal_id),
            )
        )

    def clone(self) -> GoalSet:
        return GoalSet(dict(self._goals))

    def evaluate(
        self,
        state: StateSnapshot,
        *,
        knowledge_keys: Iterable[str] = (),
    ) -> tuple[GoalEvaluation, ...]:
        action_signatures = {
            action.signature for action in state.available_actions
        }
        knowledge = set(knowledge_keys)
        results = []
        for goal in self.all():
            satisfaction = goal_satisfaction(
                goal,
                state,
                action_signatures,
                knowledge,
            )
            results.append(
                GoalEvaluation(
                    goal.goal_id,
                    satisfaction,
                    satisfaction >= goal.threshold,
                )
            )
        return tuple(results)

    def achieved_ids(
        self,
        state: StateSnapshot,
        *,
        knowledge_keys: Iterable[str] = (),
    ) -> tuple[str, ...]:
        return tuple(
            item.goal_id
            for item in self.evaluate(
                state,
                knowledge_keys=knowledge_keys,
            )
            if item.achieved
        )


def goal_satisfaction(
    goal: Goal,
    state: StateSnapshot,
    action_signatures: set[str] | None = None,
    knowledge_keys: set[str] | None = None,
) -> float:
    action_signatures = action_signatures or {
        action.signature for action in state.available_actions
    }
    knowledge_keys = knowledge_keys or set()
    if goal.kind is GoalKind.FACT_PRESENT:
        return 1.0 if str(goal.target) in state.facts else 0.0
    if goal.kind is GoalKind.FACT_ABSENT:
        return 1.0 if str(goal.target) not in state.facts else 0.0
    if goal.kind is GoalKind.ACTION_AVAILABLE:
        return 1.0 if str(goal.target) in action_signatures else 0.0
    if goal.kind is GoalKind.KNOWLEDGE_PRESENT:
        return 1.0 if str(goal.target) in knowledge_keys else 0.0
    if goal.kind is GoalKind.GOAL_PROGRESS:
        target = float(goal.target)
        if target <= 0.0:
            return 1.0
        return max(0.0, min(1.0, state.goal_progress / target))
    if goal.kind is GoalKind.VECTOR_TARGET:
        target = tuple(float(value) for value in goal.target)  # type: ignore[arg-type]
        if len(target) != len(state.vector):
            return 0.0
        distance = sqrt(
            sum(
                (left - right) ** 2
                for left, right in zip(
                    state.vector,
                    target,
                    strict=True,
                )
            )
        )
        return 1.0 / (1.0 + distance)
    raise ValueError(f"unsupported goal kind: {goal.kind}")


class GoalGenerator:
    """Create internal desires from state gaps, not action demonstrations."""

    @staticmethod
    def from_desired_state(
        current: StateSnapshot,
        desired: StateSnapshot,
        *,
        parent_goal_id: str | None = None,
        prefix: str = "internal",
    ) -> tuple[Goal, ...]:
        goals: list[Goal] = []
        for fact in sorted(desired.facts - current.facts):
            goals.append(
                Goal(
                    f"{prefix}:fact:{len(goals)}",
                    GoalKind.FACT_PRESENT,
                    fact,
                    parent_goal_id=parent_goal_id,
                    source="state_gap",
                )
            )
        current_actions = {
            action.signature for action in current.available_actions
        }
        for action in desired.available_actions:
            if action.signature not in current_actions:
                goals.append(
                    Goal(
                        f"{prefix}:action:{len(goals)}",
                        GoalKind.ACTION_AVAILABLE,
                        action.signature,
                        parent_goal_id=parent_goal_id,
                        source="state_gap",
                    )
                )
        if desired.goal_progress > current.goal_progress:
            goals.append(
                Goal(
                    f"{prefix}:progress:{len(goals)}",
                    GoalKind.GOAL_PROGRESS,
                    desired.goal_progress,
                    parent_goal_id=parent_goal_id,
                    source="state_gap",
                    final=desired.goal_progress >= 1.0,
                )
            )
        return tuple(goals)

    @staticmethod
    def from_blocked_action(
        action: Action,
        missing_facts: Iterable[str],
        *,
        parent_goal_id: str | None = None,
    ) -> tuple[Goal, ...]:
        goals = [
            Goal(
                f"need:{action.signature}:fact:{index}",
                GoalKind.FACT_PRESENT,
                fact,
                parent_goal_id=parent_goal_id,
                source="blocked_action",
            )
            for index, fact in enumerate(
                sorted(set(missing_facts))
            )
        ]
        if not goals:
            goals.append(
                Goal(
                    f"need:{action.signature}:knowledge",
                    GoalKind.KNOWLEDGE_PRESENT,
                    f"precondition:{action.signature}",
                    parent_goal_id=parent_goal_id,
                    source="unknown_precondition",
                )
            )
        return tuple(goals)


@dataclass(frozen=True, slots=True)
class GoalStateScorer:
    goals: GoalSet
    final_goal_bonus: float = 20.0
    internal_goal_weight: float = 2.0
    step_cost: float = 0.01

    def score(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
    ) -> float:
        del action
        before_values = {
            item.goal_id: item.satisfaction
            for item in self.goals.evaluate(before)
        }
        after_values = {
            item.goal_id: item.satisfaction
            for item in self.goals.evaluate(after)
        }
        value = -self.step_cost
        for goal in self.goals.all():
            gain = (
                after_values.get(goal.goal_id, 0.0)
                - before_values.get(goal.goal_id, 0.0)
            )
            weight = (
                self.final_goal_bonus
                if goal.final
                else self.internal_goal_weight * goal.priority
            )
            value += weight * gain
        return value


def choose_goal(
    goals: GoalSet,
    state: StateSnapshot,
) -> Goal | None:
    evaluations = {
        item.goal_id: item for item in goals.evaluate(state)
    }
    unfinished = [
        goal
        for goal in goals.all()
        if not evaluations[goal.goal_id].achieved
    ]
    return max(
        unfinished,
        key=lambda goal: goal.priority,
        default=None,
    )
