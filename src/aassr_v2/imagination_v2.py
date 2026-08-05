from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from typing import Any

from .baseline_efficiency_benchmark import (
    BenchmarkAgent,
    encode_gridpush_state,
    make_benchmark_agent,
)
from .benchmark_neural_prophecy import NeuralProphecyHybridAgent
from .bottleneck_sota_diagnostic import HybridDQNImaginationAgent, _current_scorer
from .branch_critic import CriticTransition, GRUBranchCritic
from .tabular_prophecy import TabularProphecy
from .types import Action, StateSnapshot


class _OnlineGRUCriticMixin:
    """Train one shared GRU critic from real episode outcomes.

    The critic receives only ordered real transitions and the final success bit.
    It is not allowed to prune imagined branches until it has observed both
    successful and failed real episodes. This prevents an untrained 0.5-valued
    network from changing behavior at the beginning of training.
    """

    critic: GRUBranchCritic

    def _initialize_online_critic(
        self,
        critic: GRUBranchCritic,
        *,
        minimum_episodes: int = 64,
        minimum_class_episodes: int = 4,
    ) -> None:
        self.critic = critic
        self._critic_minimum_episodes = int(minimum_episodes)
        self._critic_minimum_class_episodes = int(minimum_class_episodes)
        self._critic_trajectory: list[CriticTransition] = []
        self._critic_counts: Counter[str] = Counter()

    @property
    def critic_ready(self) -> bool:
        stats = self.critic.stats()
        return (
            self._critic_counts["episodes"] >= self._critic_minimum_episodes
            and self._critic_counts["successes"]
            >= self._critic_minimum_class_episodes
            and self._critic_counts["failures"]
            >= self._critic_minimum_class_episodes
            and stats.gradient_updates > 0
        )

    def begin_episode(self, *, training: bool) -> None:
        self._critic_trajectory.clear()
        super().begin_episode(training=training)

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> Any:
        if self.critic_ready:
            return super().select_action(
                state,
                episode=episode,
                training=training,
            )

        original = self.agent.config
        self.agent.config = replace(original, use_imagination=False)
        try:
            return super().select_action(
                state,
                episode=episode,
                training=training,
            )
        finally:
            self.agent.config = original

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: Any,
    ) -> None:
        confidence_fn = getattr(self.agent.prophecy, "confidence", None)
        confidence = (
            float(confidence_fn(before, action))
            if callable(confidence_fn)
            else 1.0
        )
        self._critic_trajectory.append(
            CriticTransition(
                before=before,
                action=action,
                after=outcome.snapshot,
                prophecy_confidence=max(0.0, min(1.0, confidence)),
            )
        )
        super().observe(before, action, outcome)

    def end_episode(self, *, success: bool, training: bool) -> None:
        if training:
            self.critic.observe_episode(
                tuple(self._critic_trajectory),
                success=success,
            )
            self._critic_counts["episodes"] += 1
            self._critic_counts["successes" if success else "failures"] += 1
        super().end_episode(success=success, training=training)
        self._critic_trajectory.clear()

    def model_stats(self) -> dict[str, int | float]:
        stats = dict(super().model_stats())
        critic = self.critic.stats()
        stats["model_units"] = int(stats.get("model_units", 0)) + int(
            critic.parameter_count
        )
        stats["model_bytes"] = int(stats.get("model_bytes", 0)) + int(
            critic.model_bytes
        )
        stats["gradient_updates"] = int(stats.get("gradient_updates", 0)) + int(
            critic.gradient_updates
        )
        stats["critic_gradient_updates"] = critic.gradient_updates
        stats["critic_episodes"] = critic.episodes
        stats["critic_transitions"] = critic.transitions
        stats["critic_ready"] = int(self.critic_ready)
        return stats

    def diagnostics(self) -> dict[str, Any]:
        parent = super()
        diagnostics_fn = getattr(parent, "diagnostics", None)
        base = dict(diagnostics_fn()) if callable(diagnostics_fn) else {}
        base["critic"] = {
            **asdict(self.critic.stats()),
            **dict(self._critic_counts),
            "ready": self.critic_ready,
            "minimum_episodes": self._critic_minimum_episodes,
            "minimum_class_episodes": self._critic_minimum_class_episodes,
        }
        return base


class ControlledLegacyImaginationAgent(HybridDQNImaginationAgent):
    """Current Prophecy and hand scorer with the same DQN Policy as v2."""

    def __init__(self, seed: int, *, train_episodes: int) -> None:
        super().__init__(
            seed,
            train_episodes=train_episodes,
            prophecy=TabularProphecy(),
            scorer=_current_scorer(),
            depth=5,
            branching_factor=2,
            beam_width=16,
            outcome_samples=2,
            imagination_interval=1,
            use_effect_composition=True,
            name="controlled_legacy_imagination",
        )


class NeuralPolicyOnlyAgent(NeuralProphecyHybridAgent):
    """Matched control: v2 Policy and Prophecy with Imagination always disabled."""

    def __init__(self, seed: int, *, train_episodes: int) -> None:
        super().__init__(
            seed,
            train_episodes=train_episodes,
            calibrated=True,
            conservative=True,
        )
        self.name = "neural_policy_only"
        self.agent.config = replace(self.agent.config, use_imagination=False)


class LegacyProphecyGRUCriticAgent(
    _OnlineGRUCriticMixin,
    HybridDQNImaginationAgent,
):
    """Ablation: legacy Prophecy with the learned GRU branch critic."""

    def __init__(self, seed: int, *, train_episodes: int) -> None:
        critic = GRUBranchCritic(
            encode_gridpush_state,
            25,
            hidden_units=64,
            batch_size=16,
            replay_capacity=4_000,
            gradient_steps_per_episode=2,
            seed=seed ^ 0x43524954,
        )
        HybridDQNImaginationAgent.__init__(
            self,
            seed,
            train_episodes=train_episodes,
            prophecy=TabularProphecy(),
            scorer=critic,
            depth=5,
            branching_factor=2,
            beam_width=16,
            outcome_samples=2,
            imagination_interval=1,
            use_effect_composition=True,
            name="legacy_prophecy_gru_critic",
        )
        self._initialize_online_critic(critic)


class ImaginationV2Agent(
    _OnlineGRUCriticMixin,
    NeuralProphecyHybridAgent,
):
    """Neural Delta Prophecy plus branch-local GRU Critic pruning."""

    def __init__(self, seed: int, *, train_episodes: int) -> None:
        NeuralProphecyHybridAgent.__init__(
            self,
            seed,
            train_episodes=train_episodes,
            calibrated=True,
            conservative=True,
        )
        self.name = "imagination_v2"
        critic = GRUBranchCritic(
            encode_gridpush_state,
            25,
            hidden_units=64,
            batch_size=16,
            replay_capacity=4_000,
            gradient_steps_per_episode=2,
            seed=seed ^ 0x56324352,
        )
        self.agent.planner.scorer = critic
        self._initialize_online_critic(critic)


IMAGINATION_V2_CONDITIONS = (
    "dqn",
    "legacy_aassr",
    "controlled_legacy",
    "neural_policy_only",
    "neural_manual",
    "legacy_gru_critic",
    "imagination_v2",
)


def make_imagination_v2_agent(
    condition: str,
    seed: int,
    *,
    train_episodes: int,
) -> BenchmarkAgent:
    if condition == "dqn":
        return make_benchmark_agent(
            "dqn",
            seed,
            train_episodes=train_episodes,
        )
    if condition == "legacy_aassr":
        return make_benchmark_agent(
            "aassr_full",
            seed,
            train_episodes=train_episodes,
        )
    if condition == "controlled_legacy":
        return ControlledLegacyImaginationAgent(
            seed,
            train_episodes=train_episodes,
        )
    if condition == "neural_policy_only":
        return NeuralPolicyOnlyAgent(
            seed,
            train_episodes=train_episodes,
        )
    if condition == "neural_manual":
        return NeuralProphecyHybridAgent(
            seed,
            train_episodes=train_episodes,
            calibrated=True,
            conservative=True,
        )
    if condition == "legacy_gru_critic":
        return LegacyProphecyGRUCriticAgent(
            seed,
            train_episodes=train_episodes,
        )
    if condition == "imagination_v2":
        return ImaginationV2Agent(
            seed,
            train_episodes=train_episodes,
        )
    raise ValueError(f"unknown Imagination v2 condition: {condition}")
