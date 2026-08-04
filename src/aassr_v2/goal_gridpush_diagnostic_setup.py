from __future__ import annotations

from .autonomous_agent_core import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from .collapsing_gridpush_world import CollapsingGridPushWorld
from .goal_gridpush_experiment import GoalExecutor, ImaginedGoalMaker
from .imagination_tree import StateDeltaScorer
from .persistent_goal_agent import PersistentGoalSeparatedAgent
from .tabular_prophecy import TabularProphecy
from .types import StateSnapshot


class DiagnosticPersistentGoalAgent(PersistentGoalSeparatedAgent):
    """Persistent GOAL path with a bounded but genuinely multi-step search."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed, maximum_goal_age=6)
        self._active_episode_token: tuple[int, int, bool] | None = None
        self.maker = ImaginedGoalMaker(
            self.base.policy,
            self.base.prophecy,
            depth=4,
        )
        self.executor = GoalExecutor(
            self.base.policy,
            self.base.prophecy,
            depth=3,
        )
        # Preserve full root comparison while bounding later expansion.
        self.maker.planner.config = self.maker.planner.config.__class__(
            branching_factor=2,
            maximum_depth=4,
            beam_width=16,
            outcome_samples=2,
            discount=self.maker.planner.config.discount,
            minimum_path_confidence=0.0,
            uncertainty_penalty=0.25,
            goal_threshold=1.0,
            aggregation="risk_adjusted",
            top_mean_count=2,
            update_policy=False,
            expand_all_root_actions=True,
        )

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ) -> ActionDecision:
        token = (
            int(episode),
            int(state.metadata.get("map_seed", -1)),
            bool(explore),
        )
        if token != self._active_episode_token:
            self.active_goal = None
            self.active_goal_age = 0
            self._active_episode_token = token
        return super().select_action(state, episode=episode, explore=explore)

    def finish_episode(self, *, final_return: float) -> None:
        super().finish_episode(final_return=final_return)
        self._active_episode_token = None


def _diagnostic_standard_agent(
    condition: str,
    seed: int,
) -> AutonomousLearningAgent:
    if condition == "policy_only":
        depth = 1
        use_imagination = False
    elif condition == "prophecy_one_step":
        depth = 1
        use_imagination = True
    elif condition == "full_imagination":
        depth = 5
        use_imagination = True
    else:
        raise ValueError(f"unknown condition: {condition}")

    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            use_imagination=use_imagination,
            imagination_depth=depth,
            imagination_branching_factor=2,
            imagination_beam_width=16,
            imagination_outcome_samples=2,
            imagination_minimum_coverage=0.0,
            imagination_intervention_margin=0.02,
            imagination_uncertainty_margin=0.25,
            imagination_aggregation="risk-adjusted",
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay_episodes=250,
            effect_minimum_samples=2,
        ),
        seed=seed,
    )
    agent.planner.scorer = StateDeltaScorer(
        goal_progress_weight=50.0,
        new_fact_weight=4.0,
        unlocked_action_weight=2.0,
        step_cost=0.01,
    )
    return agent


def install_goal_gridpush_diagnostic_agents() -> None:
    """Install matched bounded agents and the no-tick-limit world."""

    from . import goal_gridpush_experiment

    goal_gridpush_experiment.GoalGridPushWorld = CollapsingGridPushWorld
    goal_gridpush_experiment.GoalSeparatedAgent = DiagnosticPersistentGoalAgent
    goal_gridpush_experiment._standard_agent = _diagnostic_standard_agent
