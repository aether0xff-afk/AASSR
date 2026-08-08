from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Iterable

from .autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent
from .feature_memory import OnlineFeatureMemory
from .goals import Goal, GoalKind, GoalSet, GoalStateScorer
from .knowledge import KnowledgeEntry, KnowledgeStore
from .learning import AdvancedEvaluation, AdvancedTransitionEvaluator
from .prophecy import ProphecyStep
from .semantic_control import (
    SemanticContextualPolicy,
    SemanticSelfLoopASEQ,
    SemanticStateKeyFn,
    raw_semantic_state_key,
)
from .skills import SKILL_VERB, Skill, SkillAwareProphecy, SkillLibrary
from .types import Action, Prediction, StateSnapshot, TransitionTrace


@dataclass(frozen=True, slots=True)
class IntegratedAASSRConfig:
    """Integration-only controls for the canonical AASSR 0.4 loop."""

    use_aseq: bool = True
    aseq_repeat_threshold: int = 2
    information_value_weight: float = 0.25
    expected_observation_contract: str | None = None
    preserve_knowledge_across_episodes: bool = True

    def __post_init__(self) -> None:
        if self.aseq_repeat_threshold <= 0:
            raise ValueError("aseq_repeat_threshold must be positive")
        if self.information_value_weight < 0.0:
            raise ValueError("information_value_weight must be non-negative")


@dataclass(frozen=True, slots=True)
class IntegratedActionDecision:
    action: Action
    core_decision: Any
    semantic_state: Any
    guarded_candidates: int
    all_guarded_fallback: bool


@dataclass(frozen=True, slots=True)
class IntegratedAgentStep:
    decision: IntegratedActionDecision
    executed_actions: tuple[Action, ...]
    traces: tuple[TransitionTrace, ...]
    evaluations: tuple[AdvancedEvaluation, ...]
    newly_achieved_goal_ids: tuple[str, ...]
    promoted_skill: Skill | None
    terminal: bool


class ContextualSkillAwareProphecy(SkillAwareProphecy):
    """Skill wrapper whose normal planning API can consume live Knowledge."""

    def __init__(
        self,
        base: object,
        library: SkillLibrary,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        super().__init__(base, library)
        self.knowledge = knowledge

    def bind_knowledge(self, knowledge: KnowledgeStore | None) -> None:
        self.knowledge = knowledge

    def _base_context_predictions(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: object,
        samples: int,
    ) -> tuple[Prediction, ...]:
        contextual = getattr(self.base, "predict_with_context", None)
        if callable(contextual):
            return tuple(
                contextual(
                    state,
                    action,
                    knowledge=knowledge,
                    samples=samples,
                )
            )
        return tuple(self.base.predict(state, action, samples=samples))

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: object,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if action.verb_name != SKILL_VERB:
            return tuple(
                replace(
                    prediction,
                    next_state=self.library.augment_state(prediction.next_state),
                )
                for prediction in self._base_context_predictions(
                    state,
                    action,
                    knowledge=knowledge,
                    samples=samples,
                )
            )

        skill = self.library.get(str(action.target))
        current = state
        probability = 1.0
        for primitive in skill.primitive_actions:
            predictions = self._base_context_predictions(
                current,
                primitive,
                knowledge=knowledge,
                samples=1,
            )
            best = max(predictions, key=lambda item: item.probability)
            current = best.next_state
            probability *= best.probability
        return (
            Prediction(
                self.library.augment_state(current),
                max(0.0, min(1.0, probability)),
                source=f"{self.name}:context-skill",
            ),
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if self.knowledge is None:
            return super().predict(state, action, samples=samples)
        return self.predict_with_context(
            state,
            action,
            knowledge=self.knowledge,
            samples=samples,
        )

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        contextual = getattr(self.base, "predict_with_context", None)
        if self.knowledge is None or not callable(contextual):
            return super().predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )
        return ProphecyStep(
            self.predict_with_context(
                state,
                action,
                knowledge=self.knowledge,
                samples=samples,
            ),
            memory,
        )


class IntegratedProphecyView:
    """Align planner Knowledge, effect composition, and context-free holdout.

    Planner ``predict_step`` delegates to the normal outer model while the skill
    wrapper is bound to live Knowledge. Explicit ``predict_with_context``
    temporarily binds the caller-provided context and still traverses the outer
    effect-composed model. Plain ``predict`` deliberately removes live Knowledge
    for the duration of the call so replay holdout validation cannot use facts
    learned after the held-out transition occurred.
    """

    def __init__(
        self,
        prophecy: object,
        contextual_skill_prophecy: ContextualSkillAwareProphecy,
    ) -> None:
        self._prophecy = prophecy
        self._contextual_skill_prophecy = contextual_skill_prophecy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prophecy, name)

    @property
    def name(self) -> str:
        return f"integrated:{getattr(self._prophecy, 'name', 'prophecy')}"

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        previous = self._contextual_skill_prophecy.knowledge
        self._contextual_skill_prophecy.bind_knowledge(None)
        try:
            return tuple(
                self._prophecy.predict(
                    state,
                    action,
                    samples=samples,
                )
            )
        finally:
            self._contextual_skill_prophecy.bind_knowledge(previous)

    def predict_with_context(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        knowledge: object,
        samples: int,
    ) -> tuple[Prediction, ...]:
        previous = self._contextual_skill_prophecy.knowledge
        self._contextual_skill_prophecy.knowledge = knowledge  # type: ignore[assignment]
        try:
            return tuple(
                self._prophecy.predict(
                    state,
                    action,
                    samples=samples,
                )
            )
        finally:
            self._contextual_skill_prophecy.knowledge = previous


class IntegratedAASSRAgent:
    """Canonical AASSR 0.4 closed loop.

    One real transition is routed through a single coherent path:

    semantic state -> ASEQ -> Policy/Imagination -> real action -> Knowledge ->
    Prophecy/effect learning -> information-value learning -> delayed Policy
    credit -> GOAL completion -> Skill promotion.

    External reward is never changed by this class. Internal information value is
    learned from delayed real outcome and is kept separate from the environment's
    reward field.
    """

    def __init__(
        self,
        prophecy: object,
        *,
        goals: GoalSet | None = None,
        core_config: AutonomousAgentConfig | None = None,
        integration_config: IntegratedAASSRConfig | None = None,
        seed: int = 0,
        semantic_state_key: SemanticStateKeyFn = raw_semantic_state_key,
        knowledge: KnowledgeStore | None = None,
        skills: SkillLibrary | None = None,
        feature_memory: OnlineFeatureMemory | None = None,
        evaluator: AdvancedTransitionEvaluator | None = None,
        scorer: object | None = None,
    ) -> None:
        self.integration_config = integration_config or IntegratedAASSRConfig()
        self.semantic_state_key = semantic_state_key
        self.knowledge = knowledge or KnowledgeStore()
        self.skills = skills or SkillLibrary()
        self.feature_memory = feature_memory or OnlineFeatureMemory()
        self.goals = goals or GoalSet(
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

        policy = SemanticContextualPolicy(
            semantic_state_key,
            learning_rate=(
                core_config.policy_learning_rate
                if core_config is not None
                else 0.2
            ),
        )
        requested = core_config or AutonomousAgentConfig()
        # Prophecy and Policy updates are owned by this integration layer so one
        # real transition cannot be learned twice through two historical loops.
        resolved = replace(
            requested,
            learn_policy=False,
            learn_prophecy=False,
        )
        self.skill_prophecy = ContextualSkillAwareProphecy(
            prophecy,
            self.skills,
            self.knowledge,
        )
        self.core = AutonomousLearningAgent(
            self.skill_prophecy,
            config=resolved,
            seed=seed,
            policy=policy,
        )
        self.policy = policy
        self.effect_prophecy = self.core.prophecy
        self.prophecy = IntegratedProphecyView(
            self.effect_prophecy,
            self.skill_prophecy,
        )
        self.core.planner.prophecy = self.prophecy
        self.core.planner.scorer = scorer or GoalStateScorer(self.goals)
        # Imagination historically had its own raw snapshot identity. Override
        # only the identity hook on this instance so real ASEQ, Policy lookup and
        # imagined cycle detection share exactly one semantic contract.
        self.core.planner._state_key = (  # type: ignore[method-assign]
            lambda state: repr(self.semantic_state_key(state))
        )

        self.evaluator = evaluator or AdvancedTransitionEvaluator(self.prophecy)
        self.aseq = SemanticSelfLoopASEQ(
            repeat_threshold=self.integration_config.aseq_repeat_threshold
        )
        self._episode_evaluations: list[AdvancedEvaluation] = []
        self._episode_traces: list[TransitionTrace] = []
        self._previous_goal_ids: set[str] = set()
        self._selected_skill_steps: list[tuple[StateSnapshot, Action, int]] = []
        self._frozen_trace_index = 0
        self._steps = 0
        self._skill_uses = 0
        self._promoted_skills = 0

    def _validate_snapshot(self, state: StateSnapshot) -> None:
        expected = self.integration_config.expected_observation_contract
        if expected is None:
            return
        actual = state.metadata.get("observation_contract")
        if actual != expected:
            raise ValueError(
                "AASSR 0.4 observation contract mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )

    def begin_episode(self, *, clear_knowledge: bool | None = None) -> None:
        if clear_knowledge is None:
            clear_knowledge = not self.integration_config.preserve_knowledge_across_episodes
        if clear_knowledge:
            self.knowledge = KnowledgeStore()
            self.skill_prophecy.bind_knowledge(self.knowledge)
        self.aseq.reset_episode()
        self._episode_evaluations.clear()
        self._episode_traces.clear()
        self._previous_goal_ids.clear()
        self._selected_skill_steps.clear()
        recent = getattr(self.evaluator, "_recent_pairs", None)
        if isinstance(recent, list):
            recent.clear()

    def _selection_state(self, state: StateSnapshot) -> tuple[StateSnapshot, Any, int, bool]:
        self._validate_snapshot(state)
        semantic = self.semantic_state_key(state)
        augmented = self.skills.augment_state(state)
        if not self.integration_config.use_aseq:
            return augmented, semantic, 0, False
        filtered, guarded, fallback = self.aseq.filter_state(augmented, semantic)
        return filtered, semantic, guarded, fallback

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool = True,
    ) -> IntegratedActionDecision:
        selection, semantic, guarded, fallback = self._selection_state(state)
        decision = self.core.select_action(
            selection,
            episode=episode,
            explore=explore,
        )
        return IntegratedActionDecision(
            decision.action,
            decision,
            semantic,
            guarded,
            fallback,
        )

    @staticmethod
    def _terminal(environment: object) -> bool:
        terminal = getattr(environment, "terminal", None)
        if isinstance(terminal, bool):
            return terminal
        snapshot = environment.snapshot()
        return not bool(snapshot.available_actions)

    def _observe_information_features(self, trace: TransitionTrace) -> None:
        for fact in trace.added_facts:
            tokens = tuple(
                token
                for token in re.split(r"[^A-Za-z0-9_.-]+", fact)
                if token
            ) or (fact,)
            self.feature_memory.observe_information(fact, tokens)
            # Also retain response-derived identifier pieces so later structured
            # action parameters can match observed information without a
            # domain-specific correct-value rule.
            for component in fact.split(":")[1:]:
                if component:
                    self.feature_memory.observe_information(component, tokens)

    def _apply_frozen_knowledge(
        self,
        trace_id: str,
        outcome: object,
    ) -> None:
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
        predictions = tuple(
            self.prophecy.predict_with_context(
                before,
                action,
                knowledge=self.knowledge,
                samples=self.evaluator.samples,
            )
        )
        outcome = environment.step(action)
        after = outcome.snapshot
        self._validate_snapshot(after)
        self._frozen_trace_index += 1
        trace_id = f"v040-eval-{self._frozen_trace_index:06d}"
        self._apply_frozen_knowledge(trace_id, outcome)
        trace = TransitionTrace(
            trace_id,
            before,
            action,
            predictions,
            after,
            frozenset(getattr(outcome, "added_facts", after.facts - before.facts)),
            frozenset(getattr(outcome, "removed_facts", before.facts - after.facts)),
            tuple(getattr(outcome, "unlocked_actions", ())),
            bool(getattr(outcome, "error", False)),
            real_reward=float(getattr(outcome, "reward", 0.0)),
        )
        return trace

    def _execute_primitive(
        self,
        environment: object,
        action: Action,
        *,
        training: bool,
    ) -> tuple[TransitionTrace, AdvancedEvaluation | None]:
        if training:
            evaluation = self.evaluator.execute(
                environment,
                action,
                self.knowledge,
            )
            trace = evaluation.trace
            self._episode_evaluations.append(evaluation)
        else:
            evaluation = None
            trace = self._execute_frozen(environment, action)
        self._episode_traces.append(trace)
        self._observe_information_features(trace)
        return trace, evaluation

    def _goal_update(self, state: StateSnapshot) -> tuple[tuple[str, ...], Skill | None]:
        achieved = set(
            self.goals.achieved_ids(
                state,
                knowledge_keys=(entry.key for entry in self.knowledge.values()),
            )
        )
        newly_achieved = tuple(sorted(achieved - self._previous_goal_ids))
        promoted = None
        if newly_achieved:
            promoted = self.skills.observe_goal_completion(
                self._episode_traces,
                achieved_goal_ids=newly_achieved,
            )
            if promoted is not None:
                self._promoted_skills += 1
        self._previous_goal_ids = achieved
        return newly_achieved, promoted

    def step(
        self,
        environment: object,
        *,
        episode: int,
        training: bool = True,
    ) -> IntegratedAgentStep:
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

        if selected.verb_name == SKILL_VERB:
            self._skill_uses += 1
            self._selected_skill_steps.append(
                (self.skills.augment_state(raw_before), selected, len(self._episode_traces))
            )
            primitives: Iterable[Action] = self.skills.get(str(selected.target)).primitive_actions
        else:
            primitives = (selected,)

        skill_failed = False
        for primitive in primitives:
            if self._terminal(environment):
                break
            trace, evaluation = self._execute_primitive(
                environment,
                primitive,
                training=training,
            )
            executed.append(primitive)
            traces.append(trace)
            if evaluation is not None:
                evaluations.append(evaluation)
            if trace.error:
                skill_failed = selected.verb_name == SKILL_VERB
                break

        if skill_failed:
            self.skills.record_failure(str(selected.target))

        raw_after = environment.snapshot()
        semantic_after = self.semantic_state_key(raw_after)
        if self.integration_config.use_aseq:
            self.aseq.observe(
                decision.semantic_state,
                selected,
                semantic_after,
            )

        newly_achieved, promoted = self._goal_update(raw_after)
        self._steps += 1
        return IntegratedAgentStep(
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
                target = credited.credit + (
                    self.integration_config.information_value_weight
                    * learned_information_value
                )
                self.policy.observe_return(
                    evaluation.trace.before,
                    evaluation.trace.action,
                    target,
                )
                self._learn_feature_use(
                    evaluation.trace.action,
                    credited.credit,
                )

            # A Skill is selected as one macro action but learned through its real
            # primitive transitions. Give the macro Policy entry only terminal
            # outcome credit; never invent a synthetic environment reward.
            for state, action, trace_index in self._selected_skill_steps:
                remaining = max(0, len(self._episode_traces) - trace_index - 1)
                macro_credit = float(final_return) * (self.core.config.gamma ** remaining)
                self.policy.observe_return(state, action, macro_credit)

        self._episode_evaluations.clear()
        self._episode_traces.clear()
        self._previous_goal_ids.clear()
        self._selected_skill_steps.clear()
        self.aseq.reset_episode()
        recent = getattr(self.evaluator, "_recent_pairs", None)
        if isinstance(recent, list):
            recent.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "version": "0.4.0",
            "closed_loop": True,
            "semantic_state_contract_shared": True,
            "knowledge_bound_to_planning": True,
            "knowledge_effect_composition_aligned": True,
            "knowledge_entries": len(self.knowledge.values()),
            "feature_records": len(self.feature_memory.snapshot()),
            "feature_clusters": self.feature_memory.cluster_count(),
            "goals": len(self.goals.all()),
            "skills": len(self.skills.all()),
            "skill_uses": self._skill_uses,
            "promoted_skills": self._promoted_skills,
            "steps": self._steps,
            "aseq": self.aseq.diagnostics(),
            "imagination": self.core.imagination_diagnostics(),
            "prophecy": self.core.prophecy_diagnostics(),
            "observation_contract": self.integration_config.expected_observation_contract,
            "preserve_knowledge_across_episodes": (
                self.integration_config.preserve_knowledge_across_episodes
            ),
        }


def build_full_aassr_core(
    prophecy: object,
    *,
    goals: GoalSet | None = None,
    core_config: AutonomousAgentConfig | None = None,
    integration_config: IntegratedAASSRConfig | None = None,
    seed: int = 0,
    semantic_state_key: SemanticStateKeyFn = raw_semantic_state_key,
    scorer: object | None = None,
) -> IntegratedAASSRAgent:
    """Return the canonical integrated AASSR 0.4 agent."""

    return IntegratedAASSRAgent(
        prophecy,
        goals=goals,
        core_config=core_config,
        integration_config=integration_config,
        seed=seed,
        semantic_state_key=semantic_state_key,
        scorer=scorer,
    )


def build_pentest_aassr_core(
    prophecy: object,
    *,
    goals: GoalSet | None = None,
    core_config: AutonomousAgentConfig | None = None,
    seed: int = 0,
    scorer: object | None = None,
) -> IntegratedAASSRAgent:
    """Bind AASSR 0.4 to the audited response-causal pentest state contract."""

    from .pentest_curriculum_causal import OBSERVATION_CONTRACT
    from .pentest_curriculum_schedule import semantic_fingerprint

    return IntegratedAASSRAgent(
        prophecy,
        goals=goals,
        core_config=core_config,
        integration_config=IntegratedAASSRConfig(
            expected_observation_contract=OBSERVATION_CONTRACT,
            preserve_knowledge_across_episodes=False,
        ),
        seed=seed,
        semantic_state_key=semantic_fingerprint,
        scorer=scorer,
    )
