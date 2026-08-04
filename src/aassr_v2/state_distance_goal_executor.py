from __future__ import annotations

from dataclasses import dataclass

from .autonomous_agent_core import ActionDecision
from .goal_gridpush_experiment import GoalProposal
from .long_horizon_goal_experiment import HierarchicalGoalAgent, WaypointGoalMaker
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class GoalActionEvaluation:
    action: Action
    distance: float
    confidence: float


class StateDistanceGoalExecutor:
    """Execute a Maker GOAL using only Prophecy and the desired state.

    For every currently available action, predict the immediate next state and
    measure its distance from the waypoint state supplied by the GOAL Maker.
    No direction name, map rule, correct action, reward hint, or environment
    callback is used here.
    """

    def __init__(self, prophecy: object, *, samples: int = 2) -> None:
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.prophecy = prophecy
        self.samples = samples

    @staticmethod
    def _distance(left: StateSnapshot, right: StateSnapshot) -> float:
        if len(left.vector) != len(right.vector):
            return float("inf")
        squared = sum(
            (a - b) ** 2
            for a, b in zip(left.vector, right.vector, strict=True)
        )
        missing_facts = len(right.facts - left.facts)
        failed = 1 if "failed" in left.facts else 0
        return squared + 4.0 * missing_facts + 100.0 * failed

    def evaluate(
        self,
        state: StateSnapshot,
        action: Action,
        proposal: GoalProposal,
    ) -> GoalActionEvaluation:
        predict = getattr(self.prophecy, "predict")
        predictions = tuple(predict(state, action, samples=self.samples))
        if not predictions:
            return GoalActionEvaluation(action, float("inf"), 0.0)
        total_probability = sum(item.probability for item in predictions)
        if total_probability <= 0.0:
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            weights = [
                item.probability / total_probability for item in predictions
            ]
        distance = sum(
            weight * self._distance(item.next_state, proposal.desired_state)
            for weight, item in zip(weights, predictions, strict=True)
        )
        confidence = sum(
            weight * item.probability
            for weight, item in zip(weights, predictions, strict=True)
        )
        return GoalActionEvaluation(action, distance, confidence)

    def choose(
        self,
        state: StateSnapshot,
        proposal: GoalProposal,
    ) -> tuple[GoalActionEvaluation, tuple[GoalActionEvaluation, ...]]:
        evaluations = tuple(
            self.evaluate(state, action, proposal)
            for action in state.available_actions
        )
        if not evaluations:
            raise ValueError("state has no available actions")
        best_distance = min(item.distance for item in evaluations)
        candidates = [
            item
            for item in evaluations
            if abs(item.distance - best_distance) <= 1e-12
        ]
        chosen = max(
            candidates,
            key=lambda item: (item.confidence, item.action.signature),
        )
        return chosen, evaluations


class StateDistanceHierarchicalGoalAgent(HierarchicalGoalAgent):
    """GOAL Maker/Executor separation with an explicit state-only handoff."""

    def __init__(self, seed: int, *, room_length: int = 6) -> None:
        super().__init__(seed, room_length=room_length)
        self.maker = WaypointGoalMaker(
            self.base.policy,
            self.base.prophecy,
            search_depth=room_length,
            waypoint_depth=3,
        )
        self.state_executor = StateDistanceGoalExecutor(
            self.base.prophecy,
            samples=2,
        )

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

        maker_nodes = 0
        if self.active_goal is None:
            self.active_goal = self.maker.propose(state)
            self.active_goal_age = 0
            if self.active_goal is not None:
                self.goal_proposals += 1
                maker_nodes = len(self.active_goal.maker_plan.nodes)
        else:
            self.goal_reuses += 1

        proposal = self.active_goal
        if proposal is None:
            return policy_decision

        preferred, evaluations = self.state_executor.choose(state, proposal)
        by_signature = {
            item.action.signature: item for item in evaluations
        }
        policy_evaluation = by_signature.get(policy_decision.action.signature)
        if policy_evaluation is None:
            return policy_decision

        changed = preferred.action.signature != policy_decision.action.signature
        if changed:
            self.goal_switches += 1
        self.active_goal_age += 1
        executed = preferred.action if changed else policy_decision.action
        advantage = policy_evaluation.distance - preferred.distance
        return ActionDecision(
            executed,
            True,
            imagined_nodes=maker_nodes + self.state_executor.samples * len(evaluations),
            imagination_depth=(
                proposal.maker_plan.maximum_depth_reached
                if maker_nodes
                else 1
            ),
            root_imagined_value=-preferred.distance,
            policy_action_signature=policy_decision.action.signature,
            imagination_opportunity=True,
            imagination_eligible=True,
            imagination_gate_reason=(
                "goal_intervention" if changed else "goal_policy_agreement"
            ),
            imagination_changed_action=changed,
            model_coverage=self.base.model_coverage(state),
            imagination_preferred_action_signature=preferred.action.signature,
            imagination_policy_value=-policy_evaluation.distance,
            imagination_preferred_value=-preferred.distance,
            imagination_advantage=advantage,
            imagination_required_advantage=0.0,
            imagination_switch_candidate=changed,
            imagination_intervention_allowed=changed,
        )


def install_state_distance_goal_executor() -> None:
    """Install the explicit state-only Executor in the long-horizon runner."""

    from . import long_horizon_goal_experiment

    long_horizon_goal_experiment.HierarchicalGoalAgent = (
        StateDistanceHierarchicalGoalAgent
    )
