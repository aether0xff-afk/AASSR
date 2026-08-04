from __future__ import annotations

from .autonomous_agent_core import ActionDecision
from .goal_gridpush_experiment import (
    GoalExecutor,
    GoalProposal,
    GoalSeparatedAgent,
    GridPushStep,
    ImaginedGoalMaker,
)
from .types import Action, StateSnapshot


class PersistentGoalSeparatedAgent(GoalSeparatedAgent):
    """Separate GOAL creation from execution and keep one GOAL across steps.

    The Maker runs only when there is no active GOAL. The Executor then works
    on that GOAL until the imagined target state is reached, the episode ends,
    or repeated execution fails to reach it within a small number of reality
    steps. This matches the intended Maker/Executor separation and avoids
    rebuilding a different GOAL every tick.
    """

    def __init__(self, seed: int, *, maximum_goal_age: int = 6) -> None:
        if maximum_goal_age <= 0:
            raise ValueError("maximum_goal_age must be positive")
        super().__init__(seed)
        self.maker = ImaginedGoalMaker(
            self.base.policy,
            self.base.prophecy,
            depth=5,
        )
        self.executor = GoalExecutor(
            self.base.policy,
            self.base.prophecy,
            depth=3,
        )
        self.maximum_goal_age = maximum_goal_age
        self.active_goal: GoalProposal | None = None
        self.active_goal_age = 0
        self.goal_reuses = 0
        self.goal_completions = 0
        self.goal_abandons = 0

    @staticmethod
    def _target_reached(proposal: GoalProposal, state: StateSnapshot) -> bool:
        target = proposal.desired_state.vector
        if len(target) != len(state.vector):
            return False
        squared_distance = sum(
            (left - right) ** 2
            for left, right in zip(state.vector, target, strict=True)
        )
        return squared_distance <= 1e-12

    def _clear_goal(self, *, completed: bool) -> None:
        if self.active_goal is None:
            return
        if completed:
            self.goal_completions += 1
        else:
            self.goal_abandons += 1
        self.active_goal = None
        self.active_goal_age = 0

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ) -> ActionDecision:
        policy_decision = self.base.select_action(
            state,
            episode=episode,
            explore=explore,
        )
        if policy_decision.imagination_gate_reason in {
            "epsilon_random",
            "random_policy",
        }:
            return policy_decision
        if self.base.model_coverage(state) <= 0.0:
            return policy_decision

        if self.active_goal is not None and self._target_reached(
            self.active_goal,
            state,
        ):
            self._clear_goal(completed=True)
        if self.active_goal is not None and (
            self.active_goal_age >= self.maximum_goal_age
        ):
            self._clear_goal(completed=False)

        if self.active_goal is None:
            self.active_goal = self.maker.propose(state)
            self.active_goal_age = 0
            if self.active_goal is not None:
                self.goal_proposals += 1
        else:
            self.goal_reuses += 1

        proposal = self.active_goal
        if proposal is None:
            return policy_decision

        plan = self.executor.plan(state, proposal)
        preferred = plan.root_evaluations[0]
        policy_evaluation = next(
            (
                item
                for item in plan.root_evaluations
                if item.action.signature == policy_decision.action.signature
            ),
            None,
        )
        if policy_evaluation is None:
            return policy_decision

        advantage = preferred.aggregate_value - policy_evaluation.aggregate_value
        changed = (
            preferred.action.signature != policy_decision.action.signature
            and advantage >= 0.02
        )
        if changed:
            self.goal_switches += 1
        action = preferred.action if changed else policy_decision.action
        return ActionDecision(
            action,
            True,
            imagined_nodes=len(proposal.maker_plan.nodes) + len(plan.nodes),
            imagination_depth=max(
                proposal.maker_plan.maximum_depth_reached,
                plan.maximum_depth_reached,
            ),
            root_imagined_value=(
                preferred.aggregate_value
                if changed
                else policy_evaluation.aggregate_value
            ),
            policy_action_signature=policy_decision.action.signature,
            imagination_opportunity=True,
            imagination_eligible=True,
            imagination_gate_reason=(
                "goal_intervention" if changed else "goal_policy_agreement"
            ),
            imagination_changed_action=changed,
            model_coverage=self.base.model_coverage(state),
            imagination_preferred_action_signature=preferred.action.signature,
            imagination_policy_value=policy_evaluation.aggregate_value,
            imagination_preferred_value=preferred.aggregate_value,
            imagination_advantage=advantage,
            imagination_required_advantage=0.02,
            imagination_switch_candidate=(
                preferred.action.signature != policy_decision.action.signature
            ),
            imagination_intervention_allowed=changed,
        )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> object:
        metrics = self.base.observe(before, action, outcome)
        if self.active_goal is not None:
            self.active_goal_age += 1
            if self._target_reached(self.active_goal, outcome.snapshot):
                self._clear_goal(completed=True)
            elif not outcome.snapshot.available_actions:
                self._clear_goal(completed=False)
        return metrics

    def finish_episode(self, *, final_return: float) -> None:
        self.base.finish_episode(final_return=final_return)
        self.active_goal = None
        self.active_goal_age = 0


def install_persistent_goal_agent() -> None:
    """Install the persistent implementation in the experiment factory."""

    from . import goal_gridpush_experiment

    goal_gridpush_experiment.GoalSeparatedAgent = PersistentGoalSeparatedAgent
