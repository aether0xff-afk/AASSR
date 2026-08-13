from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import log, sqrt
import random
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .action_plugins import PluginOutcome
from .autonomous_agent_core import ActionDecision
from .current_generation import (
    CurrentRelationalPolicy,
    KnowledgeBoundProphecy,
    RelationalGRUBranchCritic,
    RelationalInvariantDQN,
    RelationalSkillLibrary,
    bind_relational_dqn_representation,
)
from .current_manifest import (
    CURRENT_COMPONENTS,
    CURRENT_GENERATION_VERSION,
    LEGACY_COMPONENTS_ACTIVE,
)
from .current_performance import CurrentDepthBatchedProphecyView
from .current_runtime import (
    FrozenReplayRelationalCalibratedProphecy,
    FullyRelationalNeuralDeltaProphecy,
)
from .feature_memory import OnlineFeatureMemory
from .goals import Goal, GoalKind, GoalSet
from .imagination_tree import ImaginationConfig
from .knowledge import KnowledgeEntry, KnowledgeStore
from .learning import AdvancedEvaluation, AdvancedTransitionEvaluator
from .native_batching import DepthBatchedImaginationTree
from .neural_delta_prophecy import NeuralDeltaConfig
from .policy import PolicyMemory
from .prophecy import ProphecyStep
from .replay import ReplayBuffer
from .semantic_control import SemanticSelfLoopASEQ
from .skills import SKILL_VERB, Skill
from .types import Action, Prediction, StateSnapshot, TransitionTrace

if TYPE_CHECKING:
    from .current_plugin_api import CurrentRepresentationBinding


@dataclass(frozen=True, slots=True)
class CurrentAgentConfig:
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_index: int = 800
    imagination_depth: int = 4
    imagination_branching_factor: int = 6
    imagination_beam_width: int = 24
    imagination_outcome_samples: int = 1
    imagination_minimum_coverage: float = 0.55
    imagination_intervention_margin: float = 0.10
    imagination_uncertainty_margin: float = 1.0
    information_value_weight: float = 0.25
    aseq_repeat_threshold: int = 2
    preserve_knowledge_across_episodes: bool = False


@dataclass(frozen=True, slots=True)
class CurrentIntegratedActionDecision:
    action: Action
    core_decision: ActionDecision
    semantic_state: Any
    guarded_candidates: int
    all_guarded_fallback: bool


@dataclass(frozen=True, slots=True)
class CurrentAgentStep:
    decision: CurrentIntegratedActionDecision
    executed_actions: tuple[Action, ...]
    traces: tuple[TransitionTrace, ...]
    evaluations: tuple[AdvancedEvaluation, ...]
    newly_achieved_goals: tuple[str, ...]
    promoted_skill: Skill | None
    terminal: bool


class CurrentSkillProphecy:
    """Current relational Skill/Knowledge wrapper with no v0.4 wrapper ancestry."""

    name = "current-relational-skill-aware-prophecy"

    def __init__(
        self,
        base: KnowledgeBoundProphecy,
        library: RelationalSkillLibrary,
        knowledge: KnowledgeStore,
    ) -> None:
        self.base = base
        self.library = library
        self.knowledge = knowledge

    @property
    def training_stats(self) -> Any:
        return self.base.training_stats

    def bind_knowledge(self, knowledge: KnowledgeStore) -> None:
        self.knowledge = knowledge

    def initial_memory(self) -> Any:
        factory = getattr(self.base, "initial_memory", None)
        return factory() if callable(factory) else None

    def reset_sequence(self) -> None:
        reset = getattr(self.base, "reset_sequence", None)
        if callable(reset):
            reset()

    def reset_context(self) -> None:
        reset = getattr(self.base, "reset_context", None)
        if callable(reset):
            reset()

    def _augment_predictions(
        self,
        predictions: Sequence[Prediction],
    ) -> tuple[Prediction, ...]:
        return tuple(
            replace(
                prediction,
                next_state=self.library.augment_state(prediction.next_state),
            )
            for prediction in predictions
        )

    def _skill_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore | None,
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
                        source=f"{self.name}:skill-unavailable",
                    ),
                )
            if knowledge is None:
                predictions = self.base.predict(
                    current,
                    primitive,
                    samples=max(1, samples if index == 0 else 1),
                )
            else:
                predictions = self.base.predict_with_context(
                    current,
                    primitive,
                    knowledge=knowledge,
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
            return self._skill_predictions(
                state,
                action,
                knowledge=None,
                samples=samples,
            )
        return self._augment_predictions(
            self.base.predict(state, action, samples=samples)
        )

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if action.verb_name == SKILL_VERB:
            return self._skill_predictions(
                state,
                action,
                knowledge=knowledge,
                samples=samples,
            )
        return self._augment_predictions(
            self.base.predict_with_context(
                state,
                action,
                knowledge=knowledge,
                samples=samples,
            )
        )

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if all(action.verb_name != SKILL_VERB for action in actions):
            rows = self.base.predict_batch(states, actions, samples=samples)
            return tuple(self._augment_predictions(row) for row in rows)
        return tuple(
            self.predict(state, action, samples=samples)
            for state, action in zip(states, actions, strict=True)
        )

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        # Neural Delta has no recurrent branch memory. Keep the protocol shape so
        # the generic depth-batched tree can carry branch-local memory unchanged.
        if action.verb_name == SKILL_VERB:
            predictions = self._skill_predictions(
                state,
                action,
                knowledge=self.knowledge,
                samples=samples,
            )
        else:
            predictions = self.predict_with_context(
                state,
                action,
                knowledge=self.knowledge,
                samples=samples,
            )
        return ProphecyStep(predictions, memory)

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        if action.verb_name == SKILL_VERB:
            return
        self.base.learn(state, action, actual_next_state)

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        if action.verb_name != SKILL_VERB:
            return max(0.0, min(1.0, float(self.base.confidence(state, action))))
        skill_id = str(action.target)
        current = state
        confidence = 1.0
        for index in range(self.library.template_length(skill_id)):
            primitive = self.library.resolve_primitive(skill_id, index, current)
            if primitive is None:
                return 0.0
            confidence *= max(
                0.0,
                min(1.0, float(self.base.confidence(current, primitive))),
            )
            prediction = self.base.predict(current, primitive, samples=1)[0]
            current = prediction.next_state
        return max(0.0, min(1.0, confidence))

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(self.confidence(state, action) for action in materialized) / len(
            materialized
        )

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = getattr(self.base, "diagnostics", None)
        return dict(diagnostics()) if callable(diagnostics) else {}


class CurrentProphecyView:
    """Holdout/context view for the current model only.

    Context-free validation uses the same learned model without live Knowledge;
    planning/real decision context may use only the explicitly supplied current
    episode KnowledgeStore.
    """

    name = "current-prophecy-view"

    def __init__(self, skill_prophecy: CurrentSkillProphecy) -> None:
        self.skill_prophecy = skill_prophecy

    @property
    def training_stats(self) -> Any:
        return self.skill_prophecy.training_stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self.skill_prophecy, name)

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.skill_prophecy.predict(state, action, samples=samples)

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        return self.skill_prophecy.predict_batch(
            states,
            actions,
            samples=samples,
        )

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: KnowledgeStore,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.skill_prophecy.predict_with_context(
            state,
            action,
            knowledge=knowledge,
            samples=samples,
        )

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self.skill_prophecy.learn(state, action, actual_next_state)


@dataclass(frozen=True, slots=True)
class _PendingPolicyTransition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot


class CurrentStandalonePentestAASSRAgent:
    """Current AASSR pentest runtime with no v0.4 agent/core construction."""

    def __init__(
        self,
        *,
        seed: int,
        train_transitions: int,
        use_imagination: bool = True,
        device: str = "cpu",
        config: CurrentAgentConfig | None = None,
        representation: CurrentRepresentationBinding,
    ) -> None:
        self.config = config or CurrentAgentConfig()
        self.current_generation_version = CURRENT_GENERATION_VERSION
        self.current_components = dict(CURRENT_COMPONENTS)
        self.legacy_components_active = LEGACY_COMPONENTS_ACTIVE
        self.requested_imagination = bool(use_imagination)
        self.training_imagination = False
        self.randomizer = random.Random(int(seed) ^ 0xA441)
        self._decision_index = 0
        self._imagination_diagnostics: Counter[str] = Counter()
        self.representation = representation

        self.knowledge = KnowledgeStore()
        self.skills = RelationalSkillLibrary(representation=representation)
        self.feature_memory = OnlineFeatureMemory()
        self.goals = GoalSet(
            {
                "external:success": Goal(
                    "external:success",
                    GoalKind.GOAL_PROGRESS,
                    1.0,
                    priority=1.0,
                    threshold=1.0,
                    source="external",
                    final=True,
                )
            }
        )

        self.dqn = RelationalInvariantDQN(
            int(seed) ^ 0xD1A6,
            train_transitions=int(train_transitions),
        )
        bind_relational_dqn_representation(self.dqn, representation)
        self.policy = CurrentRelationalPolicy(
            self.dqn,
            representation=representation,
        )

        self.base_neural_prophecy = FullyRelationalNeuralDeltaProphecy(
            representation.state_codec_factory(),
            config=NeuralDeltaConfig(
                action_feature_size=representation.action_feature_size,
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=int(seed) ^ 0x4E455552,
            device=device,
            representation=representation,
        )
        replay = ReplayBuffer()
        self.calibrated_prophecy = FrozenReplayRelationalCalibratedProphecy(
            self.base_neural_prophecy,
            replay,
        )
        self.knowledge_prophecy = KnowledgeBoundProphecy(
            self.calibrated_prophecy
        )
        self.skill_prophecy = CurrentSkillProphecy(
            self.knowledge_prophecy,
            self.skills,
            self.knowledge,
        )
        self.prophecy = CurrentProphecyView(self.skill_prophecy)
        self.evaluator = AdvancedTransitionEvaluator(
            self.prophecy,
            replay=replay,
        )

        self.critic = RelationalGRUBranchCritic(
            int(seed) ^ 0x43524954,
            representation=representation,
        )
        self.current_batched_prophecy = CurrentDepthBatchedProphecyView(self)
        self.planner = DepthBatchedImaginationTree(
            self.policy,
            self.current_batched_prophecy,
            config=ImaginationConfig(
                branching_factor=self.config.imagination_branching_factor,
                maximum_depth=self.config.imagination_depth,
                beam_width=self.config.imagination_beam_width,
                outcome_samples=self.config.imagination_outcome_samples,
                minimum_path_confidence=0.1,
                uncertainty_penalty=0.2,
                aggregation="risk_adjusted",
                update_policy=False,
            ),
            scorer=self.critic,
        )
        semantic_identity = representation.semantic_state_identity
        self.planner._state_key = lambda state: repr(semantic_identity(state))
        self.current_depth_batching = True

        # Compatibility shape for diagnostics/tests that historically accessed
        # ``agent.core.planner``. This is a namespace over *current* objects, not
        # an AutonomousLearningAgent or any v0.4 runtime.
        self.core = SimpleNamespace(
            planner=self.planner,
            policy=self.policy,
            prophecy=self.prophecy,
            config=SimpleNamespace(
                gamma=self.config.gamma,
                use_effect_composition=False,
                use_imagination=bool(use_imagination),
            ),
        )

        self.aseq = SemanticSelfLoopASEQ(
            repeat_threshold=self.config.aseq_repeat_threshold
        )
        self._episode_evaluations: list[AdvancedEvaluation] = []
        self._episode_traces: list[TransitionTrace] = []
        self._previous_goal_ids: set[str] = set()
        self._selected_skill_steps: list[tuple[StateSnapshot, Action, int]] = []
        self._critic_trajectory: list[Any] = []
        self._critic_counts: Counter[str] = Counter()
        self._pending_policy_transition: _PendingPolicyTransition | None = None
        self._frozen_trace_index = 0
        self._steps = 0
        self._skill_uses = 0
        self._promoted_skills = 0

    @property
    def critic_ready(self) -> bool:
        stats = self.critic.stats()
        return (
            self._critic_counts["episodes"] >= 32
            and self._critic_counts["successes"] >= 4
            and self._critic_counts["non_successes"] >= 4
            and stats.gradient_updates > 0
        )

    def epsilon(self, index: int) -> float:
        fraction = min(
            1.0,
            max(0.0, index / max(1, self.config.epsilon_decay_index)),
        )
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _record_decision(self, decision: ActionDecision) -> ActionDecision:
        if decision.imagination_opportunity:
            self._imagination_diagnostics["opportunities"] += 1
        if decision.imagination_eligible:
            self._imagination_diagnostics["eligible"] += 1
        if decision.used_imagination:
            self._imagination_diagnostics["runs"] += 1
        if decision.imagination_switch_candidate:
            self._imagination_diagnostics["switch_candidates"] += 1
        if decision.imagination_intervention_allowed:
            self._imagination_diagnostics["interventions"] += 1
        if (
            decision.imagination_switch_candidate
            and not decision.imagination_intervention_allowed
        ):
            self._imagination_diagnostics["suppressed_switches"] += 1
        if decision.imagination_changed_action:
            self._imagination_diagnostics["changed_actions"] += 1
        self._imagination_diagnostics[
            f"gate:{decision.imagination_gate_reason}"
        ] += 1
        return decision

    def imagination_diagnostics(self) -> dict[str, int | float]:
        return dict(self._imagination_diagnostics)

    def _core_select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ) -> ActionDecision:
        epsilon = self.epsilon(episode) if explore else 0.0
        policy_action = self.policy.select(
            state,
            randomizer=self.randomizer,
            epsilon=epsilon,
            exploration_bonus=0.0,
        )
        self._decision_index += 1
        coverage = self.skill_prophecy.coverage(
            state,
            state.available_actions,
        )

        opportunity = bool(self.requested_imagination)
        if not opportunity:
            reason = "disabled"
        elif explore and not self.training_imagination:
            reason = "training_suppressed"
        elif not self.critic_ready:
            reason = "critic_not_ready"
        elif coverage < self.config.imagination_minimum_coverage:
            reason = "coverage"
        else:
            reason = "eligible"

        eligible = reason == "eligible"
        if not eligible:
            return self._record_decision(
                ActionDecision(
                    policy_action,
                    False,
                    policy_action_signature=policy_action.signature,
                    imagination_opportunity=opportunity,
                    imagination_eligible=False,
                    imagination_gate_reason=reason,
                    model_coverage=coverage,
                )
            )

        plan = self.planner.plan(state)
        if not plan.root_evaluations:
            return self._record_decision(
                ActionDecision(
                    policy_action,
                    True,
                    imagined_nodes=len(plan.nodes),
                    imagination_depth=plan.maximum_depth_reached,
                    policy_action_signature=policy_action.signature,
                    imagination_opportunity=True,
                    imagination_eligible=True,
                    imagination_gate_reason="no_root_evaluation",
                    model_coverage=coverage,
                )
            )

        best_imagined = max(
            item.aggregate_value for item in plan.root_evaluations
        )
        candidates = [
            item
            for item in plan.root_evaluations
            if abs(item.aggregate_value - best_imagined) <= 1e-12
        ]
        preferred = min(
            candidates,
            key=lambda item: (
                -self.policy.value(state, item.action),
                item.action.signature,
            ),
        )
        policy_evaluation = next(
            (
                item
                for item in plan.root_evaluations
                if item.action.signature == policy_action.signature
            ),
            None,
        )
        switch_candidate = preferred.action.signature != policy_action.signature
        policy_value = (
            policy_evaluation.aggregate_value
            if policy_evaluation is not None
            else preferred.aggregate_value
        )
        advantage = (
            preferred.aggregate_value - policy_value
            if policy_evaluation is not None
            else 0.0
        )
        required_advantage = (
            self.config.imagination_intervention_margin
            + self.config.imagination_uncertainty_margin * (1.0 - coverage)
        )
        intervention_allowed = (
            switch_candidate
            and policy_evaluation is not None
            and advantage >= required_advantage
        )
        if not switch_candidate:
            intervention_reason = "policy_agreement"
        elif policy_evaluation is None:
            intervention_reason = "policy_not_evaluated"
        elif intervention_allowed:
            intervention_reason = "intervention"
        else:
            intervention_reason = "insufficient_advantage"
        executed_action = preferred.action if intervention_allowed else policy_action
        executed_value = (
            preferred.aggregate_value if intervention_allowed else policy_value
        )
        return self._record_decision(
            ActionDecision(
                executed_action,
                True,
                imagined_nodes=len(plan.nodes),
                imagination_depth=plan.maximum_depth_reached,
                root_imagined_value=executed_value,
                policy_action_signature=policy_action.signature,
                imagination_opportunity=True,
                imagination_eligible=True,
                imagination_gate_reason=intervention_reason,
                imagination_changed_action=intervention_allowed,
                model_coverage=coverage,
                imagination_preferred_action_signature=preferred.action.signature,
                imagination_policy_value=policy_value,
                imagination_preferred_value=preferred.aggregate_value,
                imagination_advantage=advantage,
                imagination_required_advantage=required_advantage,
                imagination_switch_candidate=switch_candidate,
                imagination_intervention_allowed=intervention_allowed,
            )
        )

    def _validate_snapshot(self, state: StateSnapshot) -> None:
        self.representation.validate_observation(state)

    def begin_episode(self, *, clear_knowledge: bool | None = None) -> None:
        if clear_knowledge is None:
            clear_knowledge = not self.config.preserve_knowledge_across_episodes
        if clear_knowledge:
            self.knowledge = KnowledgeStore()
            self.skill_prophecy.bind_knowledge(self.knowledge)
        self._pending_policy_transition = None
        self._critic_trajectory.clear()
        self.aseq.reset_episode()
        self._episode_evaluations.clear()
        self._episode_traces.clear()
        self._previous_goal_ids.clear()
        self._selected_skill_steps.clear()
        recent = getattr(self.evaluator, "_recent_pairs", None)
        if isinstance(recent, list):
            recent.clear()

    def _selection_state(
        self,
        state: StateSnapshot,
    ) -> tuple[StateSnapshot, Any, int, bool]:
        self._validate_snapshot(state)
        semantic = self.representation.semantic_state_identity(state)
        augmented = self.skills.augment_state(state)
        filtered, guarded, fallback = self.aseq.filter_state(
            augmented,
            semantic,
        )
        return filtered, semantic, guarded, fallback

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool = True,
    ) -> CurrentIntegratedActionDecision:
        selection, semantic, guarded, fallback = self._selection_state(state)
        decision = self._core_select_action(
            selection,
            episode=episode,
            explore=explore,
        )
        return CurrentIntegratedActionDecision(
            decision.action,
            decision,
            semantic,
            guarded,
            fallback,
        )

    @staticmethod
    def _terminal(environment: object) -> bool:
        if bool(getattr(environment, "success", False)):
            return True
        if bool(getattr(environment, "rate_limited", False)):
            return True
        if bool(getattr(environment, "failed", False)) and bool(
            getattr(environment, "locked", False)
        ):
            return True
        terminal = getattr(environment, "terminal", None)
        if isinstance(terminal, bool):
            return terminal
        return not bool(environment.snapshot().available_actions)

    def _observe_information_features(self, trace: TransitionTrace) -> None:
        for fact in trace.added_facts:
            tokens = tuple(
                token
                for token in re.split(r"[^A-Za-z0-9_.-]+", fact)
                if token
            ) or (fact,)
            self.feature_memory.observe_information(fact, tokens)
            for component in fact.split(":")[1:]:
                if component:
                    self.feature_memory.observe_information(component, tokens)

    def _apply_frozen_knowledge(self, trace_id: str, outcome: object) -> None:
        unlocked = tuple(getattr(outcome, "unlocked_actions", ()))
        enabled = tuple(action.signature for action in unlocked)
        added = frozenset(getattr(outcome, "added_facts", frozenset()))
        removed = frozenset(getattr(outcome, "removed_facts", frozenset()))
        entries = tuple(
            KnowledgeEntry(
                fact,
                True,
                trace_id,
                enabled_action_signatures=enabled,
            )
            for fact in added
        )
        self.knowledge.apply(entries, removed)

    def _execute_frozen(
        self,
        environment: object,
        action: Action,
    ) -> TransitionTrace:
        before = environment.snapshot()
        self._validate_snapshot(before)
        predictions = self.prophecy.predict_with_context(
            before,
            action,
            knowledge=self.knowledge,
            samples=self.evaluator.samples,
        )
        outcome = environment.step(action)
        after = outcome.snapshot
        self._validate_snapshot(after)
        self._frozen_trace_index += 1
        trace_id = f"current-eval-{self._frozen_trace_index:06d}"
        self._apply_frozen_knowledge(trace_id, outcome)
        return TransitionTrace(
            trace_id,
            before,
            action,
            tuple(predictions),
            after,
            frozenset(getattr(outcome, "added_facts", after.facts - before.facts)),
            frozenset(getattr(outcome, "removed_facts", before.facts - after.facts)),
            tuple(getattr(outcome, "unlocked_actions", ())),
            bool(getattr(outcome, "error", False)),
            real_reward=float(getattr(outcome, "reward", 0.0)),
        )

    def _execute_primitive(
        self,
        environment: object,
        action: Action,
        *,
        training: bool,
    ) -> tuple[TransitionTrace, AdvancedEvaluation | None]:
        if training:
            self.calibrated_prophecy.freeze_holdout(
                tuple(self.evaluator.replay.holdout())
            )
            try:
                evaluation = self.evaluator.execute(
                    environment,
                    action,
                    self.knowledge,
                )
            finally:
                self.calibrated_prophecy.release_holdout()
            trace = evaluation.trace
            self._episode_evaluations.append(evaluation)
        else:
            evaluation = None
            trace = self._execute_frozen(environment, action)
        self._episode_traces.append(trace)
        if training:
            self._observe_information_features(trace)
        return trace, evaluation

    def _goal_update(
        self,
        state: StateSnapshot,
        *,
        training: bool,
    ) -> tuple[tuple[str, ...], Skill | None]:
        achieved = set(
            self.goals.achieved_ids(
                state,
                knowledge_keys=(entry.key for entry in self.knowledge.values()),
            )
        )
        newly_achieved = tuple(sorted(achieved - self._previous_goal_ids))
        promoted = None
        if training and newly_achieved:
            promoted = self.skills.observe_goal_completion(
                self._episode_traces,
                achieved_goal_ids=newly_achieved,
            )
            if promoted is not None:
                self._promoted_skills += 1
        self._previous_goal_ids = achieved
        return newly_achieved, promoted

    def _observe_policy_transition(
        self,
        transition: _PendingPolicyTransition,
        *,
        reward: float,
        terminal: bool,
    ) -> None:
        if terminal:
            self.dqn.mark_episode_boundary()
        self.dqn.observe(
            transition.before,
            transition.action,
            PluginOutcome(snapshot=transition.after),
            reward=float(reward),
        )

    def _flush_pending_policy(self, *, reward: float, terminal: bool) -> None:
        if self._pending_policy_transition is None:
            return
        self._observe_policy_transition(
            self._pending_policy_transition,
            reward=reward,
            terminal=terminal,
        )
        self._pending_policy_transition = None

    def _queue_policy_trace(self, trace: TransitionTrace) -> None:
        self._flush_pending_policy(reward=0.0, terminal=False)
        self._pending_policy_transition = _PendingPolicyTransition(
            trace.before,
            trace.action,
            trace.after,
        )

    @staticmethod
    def _trace_confidence(trace: TransitionTrace) -> float:
        values = []
        for prediction in trace.predictions:
            value = float(prediction.probability)
            if "unseen" in prediction.source.lower():
                value = 0.0
            values.append(value)
        return max(values, default=0.0)

    def step(
        self,
        environment: object,
        *,
        episode: int,
        training: bool = True,
        primitive_budget: int | None = None,
    ) -> CurrentAgentStep:
        if primitive_budget is not None and primitive_budget <= 0:
            raise ValueError("primitive_budget must be positive when supplied")
        if training:
            self._flush_pending_policy(reward=0.0, terminal=False)

        raw_before = environment.snapshot()
        decision = self.select_action(
            raw_before,
            episode=episode,
            explore=training,
        )
        selected = decision.action
        executed: list[Action] = []
        traces: list[TransitionTrace] = []
        evaluations: list[AdvancedEvaluation] = []
        skill_failed = False

        if selected.verb_name == SKILL_VERB:
            self._skill_uses += 1
            skill_id = str(selected.target)
            if training:
                self._selected_skill_steps.append(
                    (
                        self.skills.augment_state(raw_before),
                        selected,
                        len(self._episode_traces),
                    )
                )
            primitive_indices: Iterable[int] = range(
                self.skills.template_length(skill_id)
            )
        else:
            skill_id = ""
            primitive_indices = (0,)

        from .branch_critic import CriticTransition

        for index in primitive_indices:
            if primitive_budget is not None and len(executed) >= primitive_budget:
                break
            if self._terminal(environment):
                break
            if selected.verb_name == SKILL_VERB:
                current_state = environment.snapshot()
                primitive = self.skills.resolve_primitive(
                    skill_id,
                    index,
                    current_state,
                )
                if primitive is None:
                    skill_failed = True
                    break
            else:
                primitive = selected

            trace, evaluation = self._execute_primitive(
                environment,
                primitive,
                training=training,
            )
            executed.append(primitive)
            traces.append(trace)
            if evaluation is not None:
                evaluations.append(evaluation)
            if training:
                self._queue_policy_trace(trace)
                self._critic_trajectory.append(
                    CriticTransition(
                        trace.before,
                        trace.action,
                        trace.after,
                        self._trace_confidence(trace),
                    )
                )
            if trace.error:
                skill_failed = selected.verb_name == SKILL_VERB
                break

        if skill_failed and training and selected.verb_name == SKILL_VERB:
            self.skills.record_failure(skill_id)

        raw_after = environment.snapshot()
        semantic_after = self.representation.semantic_state_identity(raw_after)
        self.aseq.observe(
            decision.semantic_state,
            selected,
            semantic_after,
        )
        newly_achieved, promoted = self._goal_update(
            raw_after,
            training=training,
        )
        self._steps += 1
        return CurrentAgentStep(
            decision,
            tuple(executed),
            tuple(traces),
            tuple(evaluations),
            newly_achieved,
            promoted,
            self._terminal(environment),
        )

    def _learn_feature_use(self, action: Action, value: float) -> None:
        for slot, raw in action.parameters.items():
            information_id = str(raw)
            if self.feature_memory.record(information_id) is None:
                continue
            self.feature_memory.observe_use(
                information_id,
                action_id=action.verb_name,
                slot=str(slot),
                value=value,
            )

    def finish_episode(
        self,
        *,
        final_return: float,
        training: bool = True,
    ) -> None:
        if training:
            self._flush_pending_policy(
                reward=float(final_return),
                terminal=True,
            )
            if self._critic_trajectory:
                success = float(final_return) > 0.0
                self.critic.observe_episode(
                    tuple(self._critic_trajectory),
                    success=success,
                )
                self._critic_counts["episodes"] += 1
                self._critic_counts[
                    "successes" if success else "non_successes"
                ] += 1

        if training and self._episode_evaluations:
            credits = self.evaluator.finish_episode(
                self._episode_evaluations,
                final_return=float(final_return),
                policy=None,
            )
            by_id = {
                evaluation.trace.trace_id: evaluation
                for evaluation in self._episode_evaluations
            }
            for credited in credits:
                evaluation = by_id[credited.trace_id]
                learned_information_value = self.evaluator.predictor.predict(
                    evaluation.features
                )
                self.policy.observe_information_return(
                    evaluation.trace.before,
                    evaluation.trace.action,
                    self.config.information_value_weight
                    * learned_information_value,
                )
                self._learn_feature_use(
                    evaluation.trace.action,
                    credited.credit,
                )

            for state, action, trace_index in self._selected_skill_steps:
                remaining = max(0, len(self._episode_traces) - trace_index - 1)
                macro_credit = float(final_return) * (
                    self.config.gamma ** remaining
                )
                self.policy.observe_return(state, action, macro_credit)

        self._episode_evaluations.clear()
        self._episode_traces.clear()
        self._previous_goal_ids.clear()
        self._selected_skill_steps.clear()
        self._critic_trajectory.clear()
        self.aseq.reset_episode()
        recent = getattr(self.evaluator, "_recent_pairs", None)
        if isinstance(recent, list):
            recent.clear()

    def diagnostics(self) -> dict[str, Any]:
        critic = self.critic.stats()
        prophecy = self.base_neural_prophecy.diagnostics()
        return {
            "version": CURRENT_GENERATION_VERSION,
            "current_generation": True,
            "canonical_runtime": "current_standalone",
            "closed_loop": True,
            "current_components": dict(CURRENT_COMPONENTS),
            "legacy_components_active": list(LEGACY_COMPONENTS_ACTIVE),
            "calibration_same_transition_frozen": True,
            "prophecy_state_input_relational": True,
            "prophecy_action_input_relational": True,
            "identity_contracts": {
                "aseq_cycle_detection": "concrete-response-semantic-v3",
                "policy_transfer": "relational-structural-v1",
                "prophecy_transfer": "relational-state-action-v2",
                "critic_transfer": "relational-structural-v1",
                "skill_transfer": "relational-action-template-v1",
            },
            "effect_composition_active": False,
            "training_imagination": self.training_imagination,
            "knowledge_entries": len(self.knowledge.values()),
            "feature_records": len(self.feature_memory.snapshot()),
            "feature_clusters": self.feature_memory.cluster_count(),
            "goals": len(self.goals.all()),
            "skills": len(self.skills.all()),
            "skill_uses": self._skill_uses,
            "promoted_skills": self._promoted_skills,
            "steps": self._steps,
            "aseq": self.aseq.diagnostics(),
            "imagination": self.imagination_diagnostics(),
            "prophecy": prophecy,
            "observation_contract": self.representation.observation_contract,
            "preserve_knowledge_across_episodes": self.config.preserve_knowledge_across_episodes,
            "critic_ready": self.critic_ready,
            "critic": {
                "episodes": critic.episodes,
                "transitions": critic.transitions,
                "gradient_updates": critic.gradient_updates,
                "mean_loss": critic.mean_loss,
                "parameter_count": critic.parameter_count,
                **dict(self._critic_counts),
            },
            "policy": self.policy.diagnostics(),
            "prophecy_current": prophecy,
            "prophecy_calibration": self.calibrated_prophecy.diagnostics(),
            "relational_skills": self.skills.diagnostics(),
            "current_batching": self.current_batched_prophecy.runtime_diagnostics(),
        }


def build_current_standalone_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
    representation: CurrentRepresentationBinding | None = None,
) -> CurrentStandalonePentestAASSRAgent:
    if representation is None:
        # Compatibility construction still selects the pentest plugin explicitly;
        # the assembly class itself has no HTTP/environment implementation import.
        from .plugins.current_pentest import PENTEST_REPRESENTATION

        representation = PENTEST_REPRESENTATION
    return CurrentStandalonePentestAASSRAgent(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
        representation=representation,
    )
