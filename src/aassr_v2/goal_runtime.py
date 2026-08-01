from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

from .goals import Goal, GoalGenerator, GoalSet, GoalStateScorer, choose_goal
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class GoalLifecycleRecord:
    goal_id: str
    created_episode: int
    created_step: int
    evidence_observation_sha256: str
    evidence_observation: Mapping[str, Any]
    kind: str
    target: str
    source: str
    selected: bool = False
    achieved: bool = False
    discarded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservableGoalProgressEstimator:
    """Translate GOAL satisfaction deltas into a domain-neutral value."""

    scorer: GoalStateScorer

    def __call__(self, before: StateSnapshot, after: StateSnapshot) -> float:
        return self.scorer.score(
            before,
            Action("__observable_goal_evaluation__"),
            after,
        )


class ObservableGoalRuntime:
    """Drive the existing GoalGenerator only from observed experience.

    The adapter does not inspect a world object.  State gaps come from two
    snapshots surrounding an executed action, and blocked goals come only from
    the public success/failure result.
    """

    def __init__(
        self,
        generator: GoalGenerator,
        goals: GoalSet,
        *,
        maximum_internal_goals: int = 256,
    ) -> None:
        if maximum_internal_goals <= 0:
            raise ValueError("maximum_internal_goals must be positive")
        self.generator = generator
        self.goals = goals
        self.maximum_internal_goals = maximum_internal_goals
        self._records: dict[str, GoalLifecycleRecord] = {}
        self._target_index: dict[tuple[str, str], str] = {}
        self.generator_calls = 0
        self.selection_calls = 0

    @staticmethod
    def _evidence(
        observation: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any]]:
        copied = json.loads(json.dumps(dict(observation), sort_keys=True))
        encoded = json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), copied

    @staticmethod
    def _target(goal: Goal) -> str:
        if isinstance(goal.target, tuple):
            return json.dumps(goal.target, separators=(",", ":"))
        return str(goal.target)

    def _add_generated(
        self,
        generated: Iterable[Goal],
        *,
        episode: int,
        step: int,
        evidence_observation: Mapping[str, Any],
    ) -> tuple[Goal, ...]:
        evidence_hash, copied_evidence = self._evidence(evidence_observation)
        added: list[Goal] = []
        for candidate in generated:
            target = self._target(candidate)
            if target in {"terminal", "terminal_success"}:
                continue
            key = (candidate.kind.value, target)
            existing_id = self._target_index.get(key)
            if existing_id is not None:
                continue
            goal_id = f"{candidate.goal_id}:{evidence_hash[:10]}"
            goal = replace(candidate, goal_id=goal_id)
            self.goals.add(goal)
            self._target_index[key] = goal_id
            self._records[goal_id] = GoalLifecycleRecord(
                goal_id=goal_id,
                created_episode=int(episode),
                created_step=int(step),
                evidence_observation_sha256=evidence_hash,
                evidence_observation=copied_evidence,
                kind=goal.kind.value,
                target=target,
                source=goal.source,
            )
            added.append(goal)
        self._enforce_limit()
        return tuple(added)

    def _enforce_limit(self) -> None:
        internal = [
            goal
            for goal in self.goals.all()
            if not goal.final and goal.goal_id in self._records
        ]
        excess = len(internal) - self.maximum_internal_goals
        if excess <= 0:
            return
        ordered = sorted(
            internal,
            key=lambda item: (
                not self._records[item.goal_id].achieved,
                self._records[item.goal_id].created_episode,
                self._records[item.goal_id].created_step,
                item.goal_id,
            ),
        )
        for goal in ordered[:excess]:
            record = self._records[goal.goal_id]
            self._records[goal.goal_id] = replace(record, discarded=True)
            self.goals.remove(goal.goal_id)

    def observe_transition(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
        *,
        action_succeeded: bool,
        episode: int,
        step: int,
        evidence_observation: Mapping[str, Any],
    ) -> tuple[Goal, ...]:
        if action_succeeded:
            self.generator_calls += 1
            generated = self.generator.from_desired_state(
                before,
                after,
                parent_goal_id="terminal_success",
                prefix=f"internal:e{episode}:s{step}",
            )
        else:
            self.generator_calls += 1
            generated = self.generator.from_blocked_action(
                action,
                (),
                parent_goal_id="terminal_success",
            )
        return self._add_generated(
            generated,
            episode=episode,
            step=step,
            evidence_observation=evidence_observation,
        )

    def select(
        self,
        state: StateSnapshot,
    ) -> Goal | None:
        self.selection_calls += 1
        selected = choose_goal(self.goals, state)
        if selected is not None and selected.goal_id in self._records:
            record = self._records[selected.goal_id]
            self._records[selected.goal_id] = replace(record, selected=True)
        return selected

    def mark_achieved(
        self,
        state: StateSnapshot,
        *,
        knowledge_keys: Iterable[str] = (),
    ) -> tuple[str, ...]:
        achieved = self.goals.achieved_ids(
            state,
            knowledge_keys=knowledge_keys,
        )
        for goal_id in achieved:
            if goal_id in self._records:
                record = self._records[goal_id]
                self._records[goal_id] = replace(record, achieved=True)
        return achieved

    def records(self) -> tuple[GoalLifecycleRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda item: (
                    item.created_episode,
                    item.created_step,
                    item.goal_id,
                ),
            )
        )

    def internal_goal_count(self) -> int:
        return sum(
            not record.discarded for record in self._records.values()
        )
