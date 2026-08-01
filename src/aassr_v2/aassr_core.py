from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .environment_plugin import (
    CoreEnvironmentSession,
    CoreObservationEncoder,
    EnvironmentPlugin,
)
from .feature_memory import OnlineFeatureMemory
from .goal_runtime import (
    GoalLifecycleRecord,
    ObservableGoalProgressEstimator,
    ObservableGoalRuntime,
)
from .goals import (
    Goal,
    GoalGenerator,
    GoalKind,
    GoalSet,
    GoalStateScorer,
    choose_goal,
)
from .gru_prophecy import OnlineGRUProphecy
from .imagination_tree import ImaginationConfig, ImaginationTree
from .knowledge import KnowledgeStore
from .learning import (
    ActionUnlockValueEstimator,
    AdvancedEvaluation,
    AdvancedTransitionEvaluator,
    DelayedCreditAssigner,
    InformationValuePredictor,
)
from .policy import PolicyMemory, WeightedPolicy
from .replay import PredictionValidator, ReplayBuffer
from .skills import SKILL_VERB, SkillAwareProphecy, SkillLibrary
from .tabular_prophecy import TabularProphecy
from .types import Action, StateSnapshot


CORE_MODULES = (
    "knowledge_store",
    "online_feature_memory",
    "goal",
    "policy",
    "prophecy",
    "imagination_tree",
    "advanced_transition_evaluator",
    "replay",
    "holdout",
    "information_value_predictor",
    "delayed_credit_assigner",
    "skill_library",
)

TRAINABLE_CORE_MODULES = (
    "knowledge_store",
    "online_feature_memory",
    "policy",
    "prophecy",
    "advanced_transition_evaluator",
    "replay",
    "holdout",
    "information_value_predictor",
    "skill_library",
)

FULL_CORE_EVIDENCE = (
    "goal_generator_calls",
    "internal_goals_created",
    "online_gru_predict_calls",
    "online_gru_learn_calls",
    "gru_hidden_updates",
    "gru_sequence_resets",
    "imagination_branch_hidden_clones",
    "imagination_real_hidden_unchanged",
    "knowledge_store_updates",
    "feature_memory_updates",
    "information_value_predictor_updates",
    "delayed_credit_assignments",
    "policy_updates",
    "skill_library_observations",
    "imagination_nodes",
)


@dataclass(slots=True)
class CoreCallAudit:
    """Runtime telemetry excluded from learned-state checkpoints."""

    calls: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in CORE_MODULES}
    )
    learning_updates: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in CORE_MODULES}
    )
    work_units: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in CORE_MODULES}
    )
    evidence: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in FULL_CORE_EVIDENCE}
    )

    def call(self, module: str, count: int = 1) -> None:
        self.calls[module] += int(count)

    def update(self, module: str, count: int = 1) -> None:
        self.learning_updates[module] += int(count)

    def work(self, module: str, count: int = 1) -> None:
        self.work_units[module] += int(count)

    def prove(self, item: str, count: int = 1) -> None:
        self.evidence[item] += int(count)

    def to_dict(self) -> dict[str, Mapping[str, int]]:
        return {
            "calls": dict(self.calls),
            "learning_updates": dict(self.learning_updates),
            "work_units": dict(self.work_units),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class AASSRCoreConfig:
    gamma: float = 0.97
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    prophecy_samples: int = 1
    replay_capacity: int = 2048
    holdout_stride: int = 5
    skill_promotion_successes: int = 2
    skill_maximum_length: int = 12
    feature_dimension: int = 64
    internal_goal_weight: float = 2.0
    maximum_internal_goals: int = 256
    gru_action_feature_size: int = 16
    gru_hidden_size: int = 24
    gru_learning_rate: float = 0.02
    imagination_minimum_coverage: float = 0.25
    imagination: ImaginationConfig = field(
        default_factory=lambda: ImaginationConfig(
            branching_factor=2,
            maximum_depth=3,
            beam_width=16,
            outcome_samples=1,
            minimum_path_confidence=0.1,
            uncertainty_penalty=0.2,
            aggregation="max",
            update_policy=False,
        )
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon bounds are invalid")
        if self.epsilon_decay_episodes <= 0:
            raise ValueError("epsilon_decay_episodes must be positive")
        if self.prophecy_samples <= 0:
            raise ValueError("prophecy_samples must be positive")
        if self.internal_goal_weight < 0.0:
            raise ValueError("internal_goal_weight must be non-negative")
        if self.maximum_internal_goals <= 0:
            raise ValueError("maximum_internal_goals must be positive")
        if min(self.gru_action_feature_size, self.gru_hidden_size) <= 0:
            raise ValueError("GRU sizes must be positive")
        if self.gru_learning_rate <= 0.0:
            raise ValueError("GRU learning rate must be positive")
        if not 0.0 <= self.imagination_minimum_coverage <= 1.0:
            raise ValueError("imagination_minimum_coverage must be in [0, 1]")


@dataclass(slots=True)
class AASSRCoreComponents:
    """Injected existing AASSR modules for one condition."""

    condition_name: str
    knowledge: KnowledgeStore
    feature_memory: OnlineFeatureMemory
    goals: GoalSet
    goal_generator: GoalGenerator | None
    goal_runtime: ObservableGoalRuntime | None
    policy: WeightedPolicy
    prophecy: object
    replay: ReplayBuffer
    validator: PredictionValidator
    information_value_predictor: InformationValuePredictor
    unlock_estimator: ActionUnlockValueEstimator
    delayed_credit_assigner: DelayedCreditAssigner
    skills: SkillLibrary
    skill_aware_prophecy: SkillAwareProphecy
    evaluator: AdvancedTransitionEvaluator
    imagination: ImaginationTree
    observation_encoder: CoreObservationEncoder
    use_imagination: bool

    def class_manifest(self) -> tuple[str, ...]:
        modules: tuple[object, ...] = (
            self.knowledge,
            self.feature_memory,
            self.goals,
            self.policy,
            self.prophecy,
            self.imagination,
            self.evaluator,
            self.replay,
            self.validator,
            self.information_value_predictor,
            self.delayed_credit_assigner,
            self.skills,
            self.skill_aware_prophecy,
        )
        names = [type(module).__name__ for module in modules]
        if self.goal_generator is not None:
            names.insert(3, type(self.goal_generator).__name__)
        if self.goal_runtime is not None:
            names.insert(4, type(self.goal_runtime).__name__)
        return tuple(names)


@dataclass(frozen=True, slots=True)
class CoreDecision:
    policy_action: Action
    selected_action: Action
    used_imagination: bool
    model_coverage: float
    imagined_nodes: int
    expanded_nodes: int
    selected_goal_id: str | None = None
    prophecy_hidden_before: str | None = None
    prophecy_hidden_after: str | None = None
    branch_hidden_clones: int = 0
    real_hidden_unchanged: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_action": self.policy_action.signature,
            "selected_action": self.selected_action.signature,
            "used_imagination": self.used_imagination,
            "model_coverage": self.model_coverage,
            "imagined_nodes": self.imagined_nodes,
            "expanded_nodes": self.expanded_nodes,
            "selected_goal_id": self.selected_goal_id,
            "prophecy_hidden_before": self.prophecy_hidden_before,
            "prophecy_hidden_after": self.prophecy_hidden_after,
            "branch_hidden_clones": self.branch_hidden_clones,
            "real_hidden_unchanged": self.real_hidden_unchanged,
        }


@dataclass(frozen=True, slots=True)
class CorePrimitiveStep:
    decision_action: str
    executed_action: str
    used_skill: bool
    action_succeeded: bool
    reward: float
    terminal: bool
    raw_observation_before: Mapping[str, Any]
    raw_observation_after: Mapping[str, Any]
    prediction_loss_before: float | None = None
    prediction_loss_after: float | None = None
    prophecy_hidden_before: str | None = None
    prophecy_hidden_after: str | None = None
    predicted_information_value: float = 0.0
    immediate_information_value: float = 0.0
    goal_ids_created: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["goal_ids_created"] = list(self.goal_ids_created)
        return payload


@dataclass(frozen=True, slots=True)
class CoreInformationFlow:
    trace_id: str
    predicted_information_value: float
    immediate_information_value: float
    delayed_terminal_credit: float
    feature_memory_value: float
    policy_update_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AASSREpisodeRecord:
    phase: str
    episode: int
    world_seed: int
    training: bool
    success: bool
    final_sparse_reward: float
    primitive_steps: int
    decisions: tuple[CoreDecision, ...]
    transitions: tuple[CorePrimitiveStep, ...]
    goal_lifecycle: tuple[GoalLifecycleRecord, ...] = ()
    information_flows: tuple[CoreInformationFlow, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "episode": self.episode,
            "world_seed": self.world_seed,
            "training": self.training,
            "success": self.success,
            "final_sparse_reward": self.final_sparse_reward,
            "primitive_steps": self.primitive_steps,
            "decisions": [item.to_dict() for item in self.decisions],
            "transitions": [item.to_dict() for item in self.transitions],
            "goal_lifecycle": [item.to_dict() for item in self.goal_lifecycle],
            "information_flows": [item.to_dict() for item in self.information_flows],
        }


def _external_goals() -> GoalSet:
    return GoalSet(
        {
            "terminal_success": Goal(
                "terminal_success",
                GoalKind.FACT_PRESENT,
                "terminal_success",
                priority=0.1,
                source="environment_terminal_observation",
                final=True,
            )
        }
    )


def _build_components(
    *,
    condition_name: str,
    config: AASSRCoreConfig,
    seed: int,
    prophecy: object,
    dynamic_goals: bool,
    use_imagination: bool,
) -> AASSRCoreComponents:
    knowledge = KnowledgeStore()
    feature_memory = OnlineFeatureMemory()
    goals = _external_goals()
    goal_generator = GoalGenerator() if dynamic_goals else None
    goal_runtime = (
        ObservableGoalRuntime(
            goal_generator,
            goals,
            maximum_internal_goals=config.maximum_internal_goals,
        )
        if goal_generator is not None
        else None
    )
    policy = WeightedPolicy()
    replay = ReplayBuffer(
        capacity=config.replay_capacity,
        holdout_stride=config.holdout_stride,
    )
    validator = PredictionValidator(samples=config.prophecy_samples)
    predictor = InformationValuePredictor()
    unlock_estimator = ActionUnlockValueEstimator()
    credit_assigner = DelayedCreditAssigner(discount=config.gamma)
    skills = SkillLibrary(
        promotion_successes=config.skill_promotion_successes,
        maximum_length=config.skill_maximum_length,
    )
    skill_aware = SkillAwareProphecy(prophecy, skills)
    goal_scorer = GoalStateScorer(
        goals,
        final_goal_bonus=1.0,
        internal_goal_weight=(config.internal_goal_weight if dynamic_goals else 0.0),
        step_cost=0.0,
    )
    evaluator = AdvancedTransitionEvaluator(
        skill_aware,
        replay=replay,
        validator=validator,
        predictor=predictor,
        unlock_estimator=unlock_estimator,
        credit_assigner=credit_assigner,
        goal_progress_estimator=ObservableGoalProgressEstimator(goal_scorer),
        samples=config.prophecy_samples,
    )
    imagination = ImaginationTree(
        policy,
        skill_aware,
        config=config.imagination,
        scorer=goal_scorer,
    )
    return AASSRCoreComponents(
        condition_name=condition_name,
        knowledge=knowledge,
        feature_memory=feature_memory,
        goals=goals,
        goal_generator=goal_generator,
        goal_runtime=goal_runtime,
        policy=policy,
        prophecy=prophecy,
        replay=replay,
        validator=validator,
        information_value_predictor=predictor,
        unlock_estimator=unlock_estimator,
        delayed_credit_assigner=credit_assigner,
        skills=skills,
        skill_aware_prophecy=skill_aware,
        evaluator=evaluator,
        imagination=imagination,
        observation_encoder=CoreObservationEncoder(dimension=config.feature_dimension),
        use_imagination=use_imagination,
    )


class AASSRCore:
    """Environment-neutral orchestrator over injected existing AASSR modules."""

    CHECKPOINT_SCHEMA = "aassr-core-checkpoint-v2"

    def __init__(
        self,
        *,
        config: AASSRCoreConfig | None = None,
        seed: int = 0,
        components: AASSRCoreComponents | None = None,
    ) -> None:
        self.config = config or AASSRCoreConfig()
        self.seed = int(seed)
        self.randomizer = random.Random(self.seed)
        if components is None:
            components = _build_components(
                condition_name="tabular_fixed_goal_core",
                config=self.config,
                seed=self.seed,
                prophecy=TabularProphecy(name="aassr-core-tabular"),
                dynamic_goals=False,
                use_imagination=True,
            )
        self.components = components
        self._bind_components()
        self._episode_index = 0
        self.audit = CoreCallAudit()

    def _bind_components(self) -> None:
        component = self.components
        self.condition_name = component.condition_name
        self.knowledge = component.knowledge
        self.feature_memory = component.feature_memory
        self.goals = component.goals
        self.goal_generator = component.goal_generator
        self.goal_runtime = component.goal_runtime
        self.policy = component.policy
        self.prophecy = component.prophecy
        self.replay = component.replay
        self.validator = component.validator
        self.information_value_predictor = component.information_value_predictor
        self.unlock_estimator = component.unlock_estimator
        self.delayed_credit_assigner = component.delayed_credit_assigner
        self.skills = component.skills
        self.skill_aware_prophecy = component.skill_aware_prophecy
        self.evaluator = component.evaluator
        self.imagination = component.imagination
        self.observation_encoder = component.observation_encoder
        self.use_imagination = component.use_imagination

    def component_class_manifest(self) -> tuple[str, ...]:
        return self.components.class_manifest()

    def _epsilon(self) -> float:
        fraction = min(1.0, self._episode_index / self.config.epsilon_decay_episodes)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _prophecy_hidden_fingerprint(self) -> str | None:
        method = getattr(self.prophecy, "training_memory_fingerprint", None)
        return method() if callable(method) else None

    def _feature_policy_memory(self, state: StateSnapshot) -> PolicyMemory:
        information_ids = tuple(self.feature_memory.snapshot())
        deltas: dict[str, float] = {}
        for action in state.available_actions:
            self.audit.call("online_feature_memory")
            ranked = self.feature_memory.rank_for_slot(
                action.verb_name,
                "context",
                information_ids,
                limit=1,
            )
            if ranked:
                deltas[action.signature] = self.feature_memory.estimated_value(
                    action.verb_name,
                    "context",
                    ranked[0],
                )
        return PolicyMemory(deltas)

    def select_action(
        self,
        state: StateSnapshot,
        *,
        training: bool,
    ) -> CoreDecision:
        self.audit.call("skill_library")
        augmented = self.skills.augment_state(state)
        self.audit.call("knowledge_store")
        knowledge_keys = tuple(entry.key for entry in self.knowledge.values())
        self.audit.call("goal")
        self.goals.evaluate(augmented, knowledge_keys=knowledge_keys)
        if self.goal_runtime is not None and training:
            selected_goal = self.goal_runtime.select(augmented)
        else:
            selected_goal = choose_goal(self.goals, augmented)
        memory = self._feature_policy_memory(augmented)
        self.audit.call("policy")
        ranked = self.policy.rank(
            augmented,
            limit=len(augmented.available_actions),
            memory=memory,
        )
        if not ranked:
            raise ValueError("environment has no available action")
        policy_action = ranked[0].action

        coverage_method = getattr(self.prophecy, "coverage", None)
        coverage = (
            float(coverage_method(augmented, augmented.available_actions))
            if callable(coverage_method)
            else 0.0
        )
        imagined = None
        hidden_before = self._prophecy_hidden_fingerprint()
        hidden_after = hidden_before
        branch_clones = 0
        if self.use_imagination:
            self.audit.call("imagination_tree")
            self.audit.call("prophecy")
            self.audit.prove("online_gru_predict_calls")
            imagined = self.imagination.plan(augmented)
            imagined_work = max(0, len(imagined.nodes) - 1)
            self.audit.work("imagination_tree", imagined_work)
            self.audit.prove("imagination_nodes", imagined_work)
            branch_clones = sum(
                node.depth > 0 and node.prophecy_memory is not None
                for node in imagined.nodes
            )
            self.audit.prove("imagination_branch_hidden_clones", branch_clones)
            hidden_after = self._prophecy_hidden_fingerprint()
            if hidden_before is not None and hidden_before == hidden_after:
                self.audit.prove("imagination_real_hidden_unchanged")

        explore = training and self.randomizer.random() < self._epsilon()
        if explore:
            selected = self.randomizer.choice(augmented.available_actions)
            used_imagination = False
        elif (
            imagined is not None
            and coverage >= self.config.imagination_minimum_coverage
        ):
            selected = imagined.chosen_action
            used_imagination = True
        else:
            selected = policy_action
            used_imagination = False
        return CoreDecision(
            policy_action=policy_action,
            selected_action=selected,
            used_imagination=used_imagination,
            model_coverage=coverage,
            imagined_nodes=0 if imagined is None else len(imagined.nodes),
            expanded_nodes=0 if imagined is None else imagined.expanded_nodes,
            selected_goal_id=None if selected_goal is None else selected_goal.goal_id,
            prophecy_hidden_before=hidden_before,
            prophecy_hidden_after=hidden_after,
            branch_hidden_clones=branch_clones,
            real_hidden_unchanged=(
                None if hidden_before is None else hidden_before == hidden_after
            ),
        )

    def _observe_feature_memory(self, evaluation: AdvancedEvaluation) -> None:
        trace = evaluation.trace
        tokens = (
            f"action:{trace.action.verb_name}",
            *(f"added:{fact}" for fact in sorted(trace.added_facts)),
            *(f"removed:{fact}" for fact in sorted(trace.removed_facts)),
            f"error:{str(trace.error).lower()}",
            *(f"information:{key}={value:.8f}" for key, value in evaluation.features.items()),
        )
        information_id = trace.trace_id
        self.feature_memory.observe_information(information_id, tokens)
        self.feature_memory.observe_use(
            information_id,
            action_id=trace.action.verb_name,
            slot="context",
            value=(
                evaluation.predicted_information_value
                + evaluation.immediate_information_value
            ),
        )
        self.audit.update("online_feature_memory", 2)
        self.audit.prove("feature_memory_updates", 2)

    def _execute_primitive(
        self,
        session: CoreEnvironmentSession,
        action: Action,
        *,
        training: bool,
        decision_action: Action,
        used_skill: bool,
        episode: int,
        step: int,
        episode_evaluations: list[AdvancedEvaluation],
    ) -> CorePrimitiveStep:
        raw_before = session.raw_observation()
        before = session.snapshot()
        train_before = len(self.replay.train())
        holdout_before = len(self.replay.holdout())
        stats_before = getattr(self.prophecy, "training_stats", None)
        hidden_before = self._prophecy_hidden_fingerprint()
        knowledge_before = repr(self.knowledge.values())
        self.audit.call("advanced_transition_evaluator")
        self.audit.call("knowledge_store")
        self.audit.call("prophecy", 3)
        self.audit.prove("online_gru_predict_calls", 3)
        self.audit.call("replay")
        self.audit.call("holdout", 2)
        self.audit.call("information_value_predictor")
        evaluation = self.evaluator.execute(
            session,
            action,
            self.knowledge,
            learn=training,
        )
        raw_after = session.raw_observation()
        stats_after = getattr(self.prophecy, "training_stats", None)
        hidden_after = self._prophecy_hidden_fingerprint()
        generated: tuple[Goal, ...] = ()
        if training:
            episode_evaluations.append(evaluation)
            self.audit.update("advanced_transition_evaluator")
            self.audit.update("replay")
            train_delta = len(self.replay.train()) - train_before
            holdout_delta = len(self.replay.holdout()) - holdout_before
            if train_delta:
                self.audit.update("prophecy", train_delta)
                self.audit.prove("online_gru_learn_calls", train_delta)
            if holdout_delta:
                self.audit.update("holdout", holdout_delta)
            if repr(self.knowledge.values()) != knowledge_before:
                self.audit.update("knowledge_store")
                self.audit.prove("knowledge_store_updates")
            if (
                stats_before is not None
                and stats_after is not None
                and stats_after.updates > stats_before.updates
            ):
                self.audit.prove(
                    "gru_hidden_updates",
                    int(hidden_before != hidden_after),
                )
            self._observe_feature_memory(evaluation)
            if self.goal_runtime is not None:
                calls_before = self.goal_runtime.generator_calls
                generated = self.goal_runtime.observe_transition(
                    before,
                    action,
                    evaluation.trace.after,
                    action_succeeded=not evaluation.trace.error,
                    episode=episode,
                    step=step,
                    evidence_observation=raw_after.to_dict(),
                )
                self.audit.prove(
                    "goal_generator_calls",
                    self.goal_runtime.generator_calls - calls_before,
                )
                self.audit.prove("internal_goals_created", len(generated))

        after = evaluation.trace.after
        knowledge_keys = tuple(entry.key for entry in self.knowledge.values())
        self.audit.call("goal", 2)
        before_goals = set(
            self.goals.achieved_ids(before, knowledge_keys=knowledge_keys)
        )
        if self.goal_runtime is not None and training:
            after_goals = set(
                self.goal_runtime.mark_achieved(
                    after,
                    knowledge_keys=knowledge_keys,
                )
            )
        else:
            after_goals = set(
                self.goals.achieved_ids(after, knowledge_keys=knowledge_keys)
            )
        newly_achieved = after_goals - before_goals
        if newly_achieved:
            self.audit.work("goal", len(newly_achieved))
            if training:
                self.audit.call("skill_library")
                self.skills.observe_goal_completion(
                    (item.trace for item in episode_evaluations),
                    achieved_goal_ids=newly_achieved,
                )
                self.audit.update("skill_library")
                self.audit.prove("skill_library_observations")
        return CorePrimitiveStep(
            decision_action=decision_action.signature,
            executed_action=action.signature,
            used_skill=used_skill,
            action_succeeded=not evaluation.trace.error,
            reward=evaluation.trace.real_reward,
            terminal=bool(raw_after.terminal),
            raw_observation_before=raw_before.to_dict(),
            raw_observation_after=raw_after.to_dict(),
            prediction_loss_before=(
                None if stats_before is None else stats_before.last_loss
            ),
            prediction_loss_after=(
                None if stats_after is None else stats_after.last_loss
            ),
            prophecy_hidden_before=hidden_before,
            prophecy_hidden_after=hidden_after,
            predicted_information_value=evaluation.predicted_information_value,
            immediate_information_value=evaluation.immediate_information_value,
            goal_ids_created=tuple(goal.goal_id for goal in generated),
        )

    def run_episode(
        self,
        plugin: EnvironmentPlugin,
        *,
        world_seed: int,
        episode: int,
        maximum_steps: int,
        training: bool,
        phase: str,
    ) -> AASSREpisodeRecord:
        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        plugin.reset(int(world_seed))
        reset = getattr(self.prophecy, "reset_sequence", None)
        if training and callable(reset):
            reset()
            self.audit.prove("gru_sequence_resets")
        session = CoreEnvironmentSession(plugin, self.observation_encoder)
        decisions: list[CoreDecision] = []
        transitions: list[CorePrimitiveStep] = []
        evaluations: list[AdvancedEvaluation] = []
        lifecycle_start = (
            len(self.goal_runtime.records()) if self.goal_runtime is not None else 0
        )
        while not plugin.terminal and len(transitions) < maximum_steps:
            decision = self.select_action(session.snapshot(), training=training)
            decisions.append(decision)
            action = decision.selected_action
            used_skill = action.verb_name == SKILL_VERB
            if used_skill:
                self.audit.call("skill_library")
                primitives = self.skills.get(str(action.target)).primitive_actions
            else:
                primitives = (action,)
            completed = True
            for primitive in primitives:
                if plugin.terminal or len(transitions) >= maximum_steps:
                    break
                record = self._execute_primitive(
                    session,
                    primitive,
                    training=training,
                    decision_action=action,
                    used_skill=used_skill,
                    episode=episode,
                    step=len(transitions),
                    episode_evaluations=evaluations,
                )
                transitions.append(record)
                if not record.action_succeeded:
                    completed = False
                    break
            if used_skill and training and not completed:
                self.skills.record_failure(str(action.target))
                self.audit.update("skill_library")

        final_return = plugin.final_sparse_reward()
        information_flows: list[CoreInformationFlow] = []
        if training:
            self.audit.call("delayed_credit_assigner")
            self.audit.call("information_value_predictor")
            self.audit.call("policy")
            credits = self.evaluator.finish_episode(
                evaluations,
                final_return=final_return,
                policy=self.policy,
                learn=True,
            )
            credit_by_id = {item.trace_id: item.credit for item in credits}
            for evaluation in evaluations:
                credit = credit_by_id[evaluation.trace.trace_id]
                propagated = (
                    evaluation.predicted_information_value
                    + evaluation.immediate_information_value
                    + credit
                )
                self.feature_memory.observe_use(
                    evaluation.trace.trace_id,
                    action_id=evaluation.trace.action.verb_name,
                    slot="context",
                    value=propagated,
                )
                information_flows.append(
                    CoreInformationFlow(
                        trace_id=evaluation.trace.trace_id,
                        predicted_information_value=(
                            evaluation.predicted_information_value
                        ),
                        immediate_information_value=(
                            evaluation.immediate_information_value
                        ),
                        delayed_terminal_credit=credit,
                        feature_memory_value=propagated,
                        policy_update_value=propagated,
                    )
                )
            self.audit.work("delayed_credit_assigner", len(credits))
            self.audit.update("information_value_predictor", len(credits))
            self.audit.update("policy", len(credits))
            self.audit.prove("delayed_credit_assignments", len(credits))
            self.audit.prove("information_value_predictor_updates", len(credits))
            self.audit.prove("policy_updates", len(credits))
            self.audit.prove("feature_memory_updates", len(evaluations))
            self._episode_index += 1
        lifecycle = (
            self.goal_runtime.records()[lifecycle_start:]
            if self.goal_runtime is not None
            else ()
        )
        return AASSREpisodeRecord(
            phase=phase,
            episode=int(episode),
            world_seed=int(world_seed),
            training=training,
            success=final_return == 1.0,
            final_sparse_reward=final_return,
            primitive_steps=len(transitions),
            decisions=tuple(decisions),
            transitions=tuple(transitions),
            goal_lifecycle=tuple(lifecycle),
            information_flows=tuple(information_flows),
        )

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "config": self.config,
            "seed": self.seed,
            "random_state": self.randomizer.getstate(),
            "episode_index": self._episode_index,
            "components": self.components,
        }

    def export_checkpoint(self) -> bytes:
        return pickle.dumps(self._checkpoint_payload(), protocol=5)

    def checkpoint_fingerprint(self) -> str:
        return hashlib.sha256(self.export_checkpoint()).hexdigest()

    @classmethod
    def from_checkpoint(cls, checkpoint: bytes) -> "AASSRCore":
        payload = pickle.loads(checkpoint)
        if payload.get("schema") != cls.CHECKPOINT_SCHEMA:
            raise ValueError("unsupported AASSRCore checkpoint schema")
        core = cls(
            config=payload["config"],
            seed=int(payload["seed"]),
            components=payload["components"],
        )
        core.randomizer.setstate(payload["random_state"])
        core._episode_index = int(payload["episode_index"])
        return core

    def checkpoint_contains_forbidden_environment_data(self) -> bool:
        payload = repr(pickle.loads(self.export_checkpoint())).lower()
        return any(
            forbidden in payload
            for forbidden in (
                "plate_links",
                "solver_reference",
                "minimum_actions",
                "optimal_action",
                "correct_path",
                "goal_distance",
                "goal_progress_delta",
                "block_role",
                "solution_family",
                "viability",
            )
        )


def build_tabular_fixed_goal_core(
    *,
    config: AASSRCoreConfig | None = None,
    seed: int = 0,
) -> AASSRCore:
    resolved = config or AASSRCoreConfig()
    return AASSRCore(
        config=resolved,
        seed=seed,
        components=_build_components(
            condition_name="tabular_fixed_goal_core",
            config=resolved,
            seed=seed,
            prophecy=TabularProphecy(name="aassr-core-tabular"),
            dynamic_goals=False,
            use_imagination=True,
        ),
    )


def _build_gru_core(
    *,
    condition_name: str,
    config: AASSRCoreConfig,
    seed: int,
    use_imagination: bool,
) -> AASSRCore:
    state_size = config.feature_dimension + 6
    prophecy = OnlineGRUProphecy(
        state_size,
        action_feature_size=config.gru_action_feature_size,
        hidden_size=config.gru_hidden_size,
        learning_rate=config.gru_learning_rate,
        seed=seed,
        replay_limit=config.replay_capacity,
    )
    return AASSRCore(
        config=config,
        seed=seed,
        components=_build_components(
            condition_name=condition_name,
            config=config,
            seed=seed,
            prophecy=prophecy,
            dynamic_goals=True,
            use_imagination=use_imagination,
        ),
    )


def build_full_aassr_core(
    *,
    config: AASSRCoreConfig | None = None,
    seed: int = 0,
) -> AASSRCore:
    resolved = config or AASSRCoreConfig()
    return _build_gru_core(
        condition_name="full_aassr",
        config=resolved,
        seed=seed,
        use_imagination=True,
    )


def build_no_imagination_core(
    *,
    config: AASSRCoreConfig | None = None,
    seed: int = 0,
) -> AASSRCore:
    resolved = config or AASSRCoreConfig()
    return _build_gru_core(
        condition_name="full_aassr_no_imagination",
        config=resolved,
        seed=seed,
        use_imagination=False,
    )
