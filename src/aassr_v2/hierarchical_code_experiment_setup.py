from __future__ import annotations

from .autonomous_agent_core import AutonomousAgentConfig, AutonomousLearningAgent
from .effect_prophecy import EffectComposedProphecy
from .imagination_tree import ImaginationConfig, ImaginationTree, StateDeltaScorer
from .long_horizon_goal_experiment import HierarchicalGoalAgent, WaypointGoalMaker
from .state_distance_goal_executor import StateDistanceGoalExecutor
from .tabular_prophecy import TabularProphecy


def _code_prophecy() -> EffectComposedProphecy:
    return EffectComposedProphecy(
        TabularProphecy(),
        minimum_samples=2,
        capacity_per_bucket=16,
    )


def _code_config(*, use_imagination: bool, depth: int) -> AutonomousAgentConfig:
    return AutonomousAgentConfig(
        gamma=0.99,
        epsilon_start=0.90,
        epsilon_end=0.05,
        epsilon_decay_episodes=500,
        exploration_bonus=0.15,
        use_imagination=use_imagination,
        imagination_depth=depth,
        imagination_branching_factor=2,
        imagination_beam_width=32,
        imagination_outcome_samples=2,
        imagination_minimum_coverage=0.0,
        imagination_intervention_margin=0.0,
        imagination_uncertainty_margin=0.10,
        imagination_aggregation="risk-adjusted",
        validated_gain_weight=0.0,
        repeat_penalty=0.0,
        error_penalty=0.0,
        effect_novelty_weight=0.0,
        extrinsic_reward_weight=1.0,
        effect_minimum_samples=2,
    )


def _code_scorer() -> StateDeltaScorer:
    return StateDeltaScorer(
        goal_progress_weight=100.0,
        new_fact_weight=8.0,
        unlocked_action_weight=2.0,
        step_cost=0.01,
    )


def make_code_direct_agent(
    condition: str,
    seed: int,
) -> AutonomousLearningAgent:
    if condition == "policy_only":
        use_imagination = False
        depth = 1
    elif condition == "short_imagination":
        use_imagination = True
        depth = 2
    elif condition == "deep_imagination":
        use_imagination = True
        depth = 4
    else:
        raise ValueError(f"unknown condition: {condition}")

    agent = AutonomousLearningAgent(
        _code_prophecy(),
        config=_code_config(use_imagination=use_imagination, depth=depth),
        seed=seed,
    )
    if use_imagination:
        agent.planner = ImaginationTree(
            agent.policy,
            agent.prophecy,
            config=ImaginationConfig(
                branching_factor=2,
                maximum_depth=depth,
                beam_width=32,
                outcome_samples=2,
                minimum_path_confidence=0.0,
                uncertainty_penalty=0.10,
                aggregation="risk_adjusted",
                update_policy=False,
                expand_all_root_actions=True,
            ),
            scorer=_code_scorer(),
        )
    return agent


class HierarchicalCodeGoalAgent(HierarchicalGoalAgent):
    """GOAL Maker depth four, reusable two-step waypoint Executor."""

    def __init__(self, seed: int, *, room_length: int = 4) -> None:
        if room_length != 4:
            raise ValueError("HierarchicalCodeGoalAgent requires four-step checkpoints")

        # Build the same public counters and episode lifecycle as the generic
        # hierarchical agent, then replace every learned component with the
        # matched code-task versions below.
        super().__init__(seed, room_length=room_length)
        self.base = AutonomousLearningAgent(
            _code_prophecy(),
            config=_code_config(use_imagination=False, depth=1),
            seed=seed,
        )
        self.maker = WaypointGoalMaker(
            self.base.policy,
            self.base.prophecy,
            search_depth=4,
            waypoint_depth=2,
        )
        self.state_executor = StateDistanceGoalExecutor(
            self.base.prophecy,
            samples=2,
        )
        self.active_goal = None
        self.active_goal_age = 0
        self.maximum_goal_age = 2
        self.goal_proposals = 0
        self.goal_switches = 0
        self.goal_reuses = 0
        self.goal_completions = 0
        self.goal_abandons = 0

    def select_action(self, state, *, episode: int, explore: bool):
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

        from .autonomous_agent_core import ActionDecision

        changed = preferred.action.signature != policy_decision.action.signature
        if changed:
            self.goal_switches += 1
        self.active_goal_age += 1
        executed = preferred.action if changed else policy_decision.action
        advantage = policy_evaluation.distance - preferred.distance
        return ActionDecision(
            executed,
            True,
            imagined_nodes=(
                maker_nodes
                + self.state_executor.samples * len(evaluations)
            ),
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


def install_hierarchical_code_agents() -> None:
    """Install matched Policy, Imagination and GOAL agents for the code task."""

    from . import long_horizon_goal_experiment

    long_horizon_goal_experiment.make_direct_agent = make_code_direct_agent
    long_horizon_goal_experiment.HierarchicalGoalAgent = (
        HierarchicalCodeGoalAgent
    )
