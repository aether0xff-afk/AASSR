from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable

from ..autonomous_agent_core import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from ..branch_critic import CriticTransition
from ..neural_delta_prophecy import NeuralDeltaConfig
from ..semantic_control import SemanticSelfLoopASEQ
from ..skills import SKILL_VERB
from ..types import Action, Prediction, StateSnapshot, TransitionTrace
from .critic import SignedCoreGRUCritic, build_signed_core_critic
from .dqn import CoreDynamicActionDQN, CorePolicy
from .manifest import CORE_VERSION, PLUGIN_CONTRACT_VERSION
from .plugin_contract import MinimalRuntimePlugin, validate_minimal_plugin
from .prophecy_model import CoreHoldoutCalibratedProphecy, SchemaDrivenNeuralProphecy
from .public_memory import MemoryBackedRepresentation
from .representation import CoreRepresentationConfig, PluginEnvironmentAdapter
from .skills_core import CoreRelationalSkillLibrary, CoreSkillAwareProphecy


@dataclass(frozen=True, slots=True)
class CoreRuntimeConfig:
    train_transitions: int = 10_000
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_index: int = 800
    imagination_depth: int = 4
    imagination_branching_factor: int = 6
    imagination_beam_width: int = 24
    imagination_outcome_samples: int = 1
    imagination_minimum_coverage: float = 0.55
    imagination_intervention_margin: float = 0.05
    imagination_uncertainty_margin: float = 1.0
    aseq_repeat_threshold: int = 2
    preserve_knowledge_across_episodes: bool = False
    state_size: int = 256
    action_feature_size: int = 128


@dataclass(frozen=True, slots=True)
class CoreDecision:
    action: Action
    decision: ActionDecision
    semantic_state: object
    guarded_candidates: int
    all_guarded_fallback: bool


@dataclass(frozen=True, slots=True)
class CoreStep:
    decision: CoreDecision
    executed_actions: tuple[Action, ...]
    traces: tuple[TransitionTrace, ...]
    terminal: bool
    external_reward: float


@dataclass(slots=True)
class _PendingDQNTransition:
    before: StateSnapshot
    action: Action
    outcome: Any
    reward: float


class AASSRCoreRuntime:
    """Environment-neutral AASSR runtime over the minimal plugin contract."""

    def __init__(
        self,
        plugin: MinimalRuntimePlugin,
        *,
        seed: int = 0,
        device: str = "cpu",
        use_imagination: bool = True,
        config: CoreRuntimeConfig | None = None,
    ) -> None:
        validate_minimal_plugin(plugin)
        self.plugin = plugin
        self.config = config or CoreRuntimeConfig()
        self.seed = int(seed)
        self.device = str(device)
        self.requested_imagination = bool(use_imagination)

        self.representation = MemoryBackedRepresentation(
            plugin.schema,
            config=CoreRepresentationConfig(
                state_size=self.config.state_size,
                action_feature_size=self.config.action_feature_size,
            ),
        )
        self.environment = PluginEnvironmentAdapter(plugin, self.representation)

        self.dqn = CoreDynamicActionDQN(
            self.seed ^ 0xD1A6,
            representation=self.representation,
            train_transitions=self.config.train_transitions,
            device=self.device,
        )
        self.policy = CorePolicy(self.dqn)
        self.base_prophecy = SchemaDrivenNeuralProphecy(
            self.representation,
            config=NeuralDeltaConfig(
                action_feature_size=self.representation.action_feature_size,
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=self.seed ^ 0x4E455552,
            device=self.device,
        )
        self.agent = AutonomousLearningAgent(
            self.base_prophecy,
            config=AutonomousAgentConfig(
                gamma=self.config.gamma,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay_episodes=self.config.epsilon_decay_index,
                exploration_bonus=0.0,
                learn_policy=False,
                learn_prophecy=True,
                use_imagination=bool(use_imagination),
                imagination_depth=self.config.imagination_depth,
                imagination_branching_factor=self.config.imagination_branching_factor,
                imagination_beam_width=self.config.imagination_beam_width,
                imagination_outcome_samples=self.config.imagination_outcome_samples,
                imagination_minimum_coverage=self.config.imagination_minimum_coverage,
                imagination_intervention_margin=self.config.imagination_intervention_margin,
                imagination_uncertainty_margin=self.config.imagination_uncertainty_margin,
                imagination_aggregation="mean",
                validated_gain_weight=0.2,
                repeat_penalty=0.0,
                error_penalty=0.0,
                effect_novelty_weight=0.0,
                use_effect_composition=False,
            ),
            seed=self.seed ^ 0xA441,
            policy=self.policy,
        )
        self.calibrated_prophecy = CoreHoldoutCalibratedProphecy(
            self.base_prophecy,
            self.agent.holdout,
            self.representation,
        )
        self.skills = CoreRelationalSkillLibrary(self.representation)
        self.prophecy = CoreSkillAwareProphecy(
            self.calibrated_prophecy,
            self.skills,
        )
        self.agent.base_prophecy = self.prophecy
        self.agent.prophecy = self.prophecy
        self.agent.planner.prophecy = self.prophecy

        self.critic: SignedCoreGRUCritic = build_signed_core_critic(
            self.representation,
            seed=self.seed ^ 0x43524954,
            device=self.device,
        )
        self.agent.planner.scorer = self.critic

        self.aseq = SemanticSelfLoopASEQ(
            repeat_threshold=self.config.aseq_repeat_threshold
        )
        # Canonical Knowledge for the new Core is representation.public_knowledge.
        # It is Core-owned, directly affects state representation/candidate
        # construction, and follows the same episode persistence contract.  Do
        # not maintain a second legacy KnowledgeStore that no active learner
        # consumes.
        self._episode_traces: list[TransitionTrace] = []
        self._critic_trajectory: list[tuple[StateSnapshot, Action, StateSnapshot, float]] = []
        self._selected_skill_steps: list[tuple[StateSnapshot, Action, int]] = []
        self._critic_counts: Counter[str] = Counter()
        self._step_index = 0
        self._trace_index = 0
        self._episode_return = 0.0
        self._pending_dqn: _PendingDQNTransition | None = None
        self._imagination_diagnostics: Counter[str] = Counter()

    @property
    def critic_reliably_ready(self) -> bool:
        stats = self.critic.stats()
        supported_returns = sum(
            self._critic_counts[name] >= 4
            for name in ("negative", "zero", "positive")
        )
        return (
            stats.episodes >= 32
            and supported_returns >= 2
            and stats.gradient_updates > 0
        )

    def begin_episode(self, *, seed: int | None = None) -> StateSnapshot:
        self.representation.begin_episode(
            preserve=self.config.preserve_knowledge_across_episodes
        )
        self.aseq.reset_episode()
        self.agent.discard_episode()
        self._episode_traces.clear()
        self._critic_trajectory.clear()
        self._selected_skill_steps.clear()
        self._episode_return = 0.0
        self._pending_dqn = None
        return self.environment.reset(seed=seed)

    def _selection_state(
        self,
        state: StateSnapshot,
    ) -> tuple[StateSnapshot, object, int, bool]:
        semantic = self.representation.semantic_state_identity(state)
        augmented = self.skills.augment_state(state)
        filtered, guarded, fallback = self.aseq.filter_state(augmented, semantic)
        return filtered, semantic, guarded, fallback

    def _select(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> CoreDecision:
        selection, semantic, guarded, fallback = self._selection_state(state)
        original = self.agent.config
        allow_imagination = self.requested_imagination and self.critic_reliably_ready
        self.agent.config = replace(original, use_imagination=allow_imagination)
        try:
            decision = self.agent.select_action(
                selection,
                episode=episode,
                explore=training,
            )
        finally:
            self.agent.config = original

        if self.requested_imagination and not allow_imagination:
            self._imagination_diagnostics["critic_not_ready"] += 1
        if decision.imagination_opportunity:
            self._imagination_diagnostics["opportunities"] += 1
        if decision.used_imagination:
            self._imagination_diagnostics["runs"] += 1
        if decision.imagination_changed_action:
            self._imagination_diagnostics["changed_actions"] += 1
        return CoreDecision(
            action=decision.action,
            decision=decision,
            semantic_state=semantic,
            guarded_candidates=guarded,
            all_guarded_fallback=fallback,
        )

    def _flush_pending_dqn(self, *, terminal: bool) -> None:
        pending = self._pending_dqn
        if pending is None:
            return
        self.dqn.observe(
            pending.before,
            pending.action,
            pending.outcome,
            reward=pending.reward,
            terminal=terminal,
        )
        self._pending_dqn = None

    def _queue_dqn(
        self,
        *,
        before: StateSnapshot,
        action: Action,
        outcome: Any,
        reward: float,
        terminal: bool,
    ) -> None:
        self._flush_pending_dqn(terminal=False)
        self._pending_dqn = _PendingDQNTransition(
            before=before,
            action=action,
            outcome=outcome,
            reward=float(reward),
        )
        if terminal:
            self._flush_pending_dqn(terminal=True)

    def _execute_primitive(
        self,
        action: Action,
        *,
        training: bool,
    ) -> TransitionTrace:
        before = self.environment.snapshot()
        predictions: tuple[Prediction, ...] = self.prophecy.predict(
            before,
            action,
            samples=1,
        )
        confidence = (
            max(
                0.0 if "unseen" in item.source.lower() else float(item.probability)
                for item in predictions
            )
            if predictions
            else 0.0
        )

        outcome = self.environment.step(action)
        after = outcome.snapshot
        reward = float(outcome.raw.get("external_reward", 0.0))
        terminal = bool(outcome.raw.get("terminated", False)) or bool(
            outcome.raw.get("truncated", False)
        )

        self._trace_index += 1
        trace_id = f"core-real-{self._trace_index:08d}"
        trace = TransitionTrace(
            trace_id=trace_id,
            before=before,
            action=action,
            predictions=predictions,
            after=after,
            added_facts=outcome.added_facts,
            removed_facts=outcome.removed_facts,
            unlocked_actions=outcome.unlocked_actions,
            error=bool(outcome.error),
            real_reward=reward,
        )

        if training:
            observation_metrics = self.agent.observe(before, action, outcome)
            self.policy.observe_information_return(
                before,
                action,
                observation_metrics.intrinsic_value,
            )
            self._queue_dqn(
                before=before,
                action=action,
                outcome=outcome,
                reward=reward,
                terminal=terminal,
            )
            self._critic_trajectory.append((before, action, after, confidence))

        self._episode_traces.append(trace)
        self._episode_return += reward
        self._step_index += 1
        return trace

    def step(
        self,
        *,
        episode: int,
        training: bool = True,
        primitive_budget: int | None = None,
    ) -> CoreStep:
        if primitive_budget is not None and primitive_budget <= 0:
            raise ValueError("primitive_budget must be positive")
        raw_before = self.environment.snapshot()
        decision = self._select(raw_before, episode=episode, training=training)
        selected = decision.action
        executed: list[Action] = []
        traces: list[TransitionTrace] = []

        if selected.verb_name == SKILL_VERB:
            skill_id = str(selected.target)
            if training:
                self._selected_skill_steps.append(
                    (raw_before, selected, len(self._episode_traces))
                )
            primitive_indices: Iterable[int] = range(
                self.skills.template_length(skill_id)
            )
        else:
            skill_id = ""
            primitive_indices = (0,)

        for index in primitive_indices:
            if primitive_budget is not None and len(executed) >= primitive_budget:
                break
            if self.environment.terminal:
                break
            if selected.verb_name == SKILL_VERB:
                primitive = self.skills.resolve_primitive(
                    skill_id,
                    index,
                    self.environment.snapshot(),
                )
                if primitive is None:
                    if training:
                        self.skills.record_failure(skill_id)
                    break
            else:
                primitive = selected

            trace = self._execute_primitive(primitive, training=training)
            executed.append(primitive)
            traces.append(trace)
            if trace.error:
                if training and selected.verb_name == SKILL_VERB:
                    self.skills.record_failure(skill_id)
                break

        raw_after = self.environment.snapshot()
        semantic_after = self.representation.semantic_state_identity(raw_after)
        self.aseq.observe(decision.semantic_state, selected, semantic_after)

        return CoreStep(
            decision=decision,
            executed_actions=tuple(executed),
            traces=tuple(traces),
            terminal=self.environment.terminal,
            external_reward=sum(trace.real_reward for trace in traces),
        )

    def finish_episode(
        self,
        *,
        final_return: float | None = None,
        training: bool = True,
    ) -> None:
        value = self._episode_return if final_return is None else float(final_return)
        if training:
            self._flush_pending_dqn(terminal=True)
            trajectory = tuple(
                CriticTransition(before, action, after, confidence)
                for before, action, after, confidence in self._critic_trajectory
            )
            self.critic.observe_episode(trajectory, final_return=value)
            self._critic_counts["episodes"] += 1
            bucket = round(max(-1.0, min(1.0, float(value))), 3)
            self._critic_counts[f"return:{bucket:+.3f}"] += 1
            if value > 0.0:
                self._critic_counts["positive"] += 1
                self.skills.observe_goal_completion(
                    tuple(self._episode_traces),
                    achieved_goal_ids=("external:positive-return",),
                )
            elif value < 0.0:
                self._critic_counts["negative"] += 1
            else:
                self._critic_counts["zero"] += 1

            for state, action, trace_index in self._selected_skill_steps:
                remaining = max(0, len(self._episode_traces) - trace_index - 1)
                self.policy.observe_return(
                    state,
                    action,
                    value * (self.config.gamma ** remaining),
                )

        self.agent.finish_episode(final_return=value)
        self._episode_traces.clear()
        self._critic_trajectory.clear()
        self._selected_skill_steps.clear()
        self._pending_dqn = None
        self.aseq.reset_episode()

    def run_episode(
        self,
        *,
        episode: int,
        max_steps: int = 256,
        training: bool = True,
        seed: int | None = None,
    ) -> float:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.begin_episode(seed=seed)
        for _ in range(max_steps):
            if self.environment.terminal:
                break
            state = self.environment.snapshot()
            if not state.available_actions:
                break
            self.step(episode=episode, training=training)
        value = self._episode_return
        self.finish_episode(final_return=value, training=training)
        return value

    def diagnostics(self) -> dict[str, Any]:
        critic = self.critic.stats()
        knowledge = dict(self.representation.public_knowledge.diagnostics())
        return {
            "core_version": CORE_VERSION,
            "plugin_contract": PLUGIN_CONTRACT_VERSION,
            "plugin": {
                "id": self.plugin.schema.plugin_id,
                "version": self.plugin.schema.version,
                "observation_fields": len(self.plugin.schema.observations),
                "action_specs": len(self.plugin.schema.actions),
            },
            "representation": {
                "owner": "core",
                "state_size": self.representation.state_size,
                "action_feature_size": self.representation.action_feature_size,
                **dict(self.representation.experience.diagnostics()),
            },
            "knowledge": {
                "owner": "core-public-memory",
                "consumed_by": "representation+candidate-generation",
                **knowledge,
            },
            "policy": self.policy.diagnostics(),
            "prophecy": {
                "observations": self.base_prophecy.observations,
                "gradient_updates": self.base_prophecy.gradient_updates,
            },
            "calibration": self.calibrated_prophecy.diagnostics(),
            "critic": {
                "ready": self.critic_reliably_ready,
                "episodes": critic.episodes,
                "transitions": critic.transitions,
                "gradient_updates": critic.gradient_updates,
                **dict(self._critic_counts),
            },
            "imagination": {
                "requested": self.requested_imagination,
                "valid_treatment": self._imagination_diagnostics["runs"] > 0,
                **dict(self._imagination_diagnostics),
            },
            "aseq": self.aseq.diagnostics(),
            "skills": self.skills.diagnostics(),
        }


def build_aassr_core(
    plugin: MinimalRuntimePlugin,
    *,
    seed: int = 0,
    device: str = "cpu",
    use_imagination: bool = True,
    train_transitions: int = 10_000,
) -> AASSRCoreRuntime:
    """Canonical minimal-contract Core builder."""

    return AASSRCoreRuntime(
        plugin,
        seed=int(seed),
        device=device,
        use_imagination=bool(use_imagination),
        config=CoreRuntimeConfig(train_transitions=int(train_transitions)),
    )
