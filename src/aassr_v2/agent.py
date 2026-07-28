from __future__ import annotations

from dataclasses import dataclass

from .goals import GoalSet
from .learning import AdvancedEvaluation, AdvancedTransitionEvaluator
from .skills import SKILL_VERB, SkillLibrary
from .types import Action, StateSnapshot, TransitionTrace


@dataclass(frozen=True, slots=True)
class AgentStep:
    action: Action
    evaluation: AdvancedEvaluation
    used_skill: bool


class LearningAgent:
    """Integrate planning, real learning, GOALs and autonomous Skill creation."""

    def __init__(
        self,
        planner: object,
        evaluator: AdvancedTransitionEvaluator,
        goals: GoalSet,
        skills: SkillLibrary,
        *,
        policy: object | None = None,
    ) -> None:
        self.planner = planner
        self.evaluator = evaluator
        self.goals = goals
        self.skills = skills
        self.policy = policy
        self.episode_evaluations: list[AdvancedEvaluation] = []
        self.episode_traces: list[TransitionTrace] = []
        self._previous_goal_ids: set[str] = set()

    def _augmented_snapshot(
        self,
        environment: object,
    ) -> StateSnapshot:
        return self.skills.augment_state(
            environment.snapshot()
        )

    def step(
        self,
        environment: object,
        knowledge: object,
    ) -> AgentStep:
        plan = self.planner.plan(
            self._augmented_snapshot(environment)
        )
        action = plan.chosen_action
        used_skill = action.verb_name == SKILL_VERB
        if used_skill:
            skill = self.skills.get(str(action.target))
            latest = None
            for primitive in skill.primitive_actions:
                latest = self.evaluator.execute(
                    environment,
                    primitive,
                    knowledge,
                )
                self.episode_evaluations.append(latest)
                self.episode_traces.append(latest.trace)
                if latest.trace.error:
                    self.skills.record_failure(
                        skill.skill_id
                    )
                    break
            if latest is None:
                raise RuntimeError("empty skill")
            evaluation = latest
        else:
            evaluation = self.evaluator.execute(
                environment,
                action,
                knowledge,
            )
            self.episode_evaluations.append(evaluation)
            self.episode_traces.append(evaluation.trace)

        achieved = set(
            self.goals.achieved_ids(
                environment.snapshot(),
                knowledge_keys=(
                    entry.key
                    for entry in knowledge.values()
                ),
            )
        )
        newly_achieved = (
            achieved - self._previous_goal_ids
        )
        if newly_achieved:
            self.skills.observe_goal_completion(
                self.episode_traces,
                achieved_goal_ids=newly_achieved,
            )
        self._previous_goal_ids = achieved
        return AgentStep(
            action,
            evaluation,
            used_skill,
        )

    def finish_episode(
        self,
        *,
        final_return: float,
    ) -> None:
        self.evaluator.finish_episode(
            self.episode_evaluations,
            final_return=final_return,
            policy=self.policy,
        )
        self.episode_evaluations.clear()
        self.episode_traces.clear()
        self._previous_goal_ids.clear()
