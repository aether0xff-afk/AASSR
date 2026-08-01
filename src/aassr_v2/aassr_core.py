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
from .goals import Goal, GoalKind, GoalSet, GoalStateScorer
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

    def call(self, module: str, count: int = 1) -> None:
        self.calls[module] += int(count)

    def update(self, module: str, count: int = 1) -> None:
        self.learning_updates[module] += int(count)

    def work(self, module: str, count: int = 1) -> None:
        self.work_units[module] += int(count)

    def to_dict(self) -> dict[str, Mapping[str, int]]:
        return {
            "calls": dict(self.calls),
            "learning_updates": dict(self.learning_updates),
            "work_units": dict(self.work_units),
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
        if not 0.0 <= self.imagination_minimum_coverage <= 1.0:
            raise ValueError("imagination_minimum_coverage must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CoreDecision:
    policy_action: Action
    selected_action: Action
    used_imagination: bool
    model_coverage: float
    imagined_nodes: int
    expanded_nodes: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_action"] = self.policy_action.signature
        payload["selected_action"] = self.selected_action.signature
        return payload


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
        }


class AASSRCore:
    """One environment-neutral orchestration path for the complete AASSR stack."""

    CHECKPOINT_SCHEMA = "aassr-core-checkpoint-v1"

    def __init__(
        self,
        *,
        config: AASSRCoreConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or AASSRCoreConfig()
        self.seed = int(seed)
        self.randomizer = random.Random(self.seed)
        self.knowledge = KnowledgeStore()
        self.feature_memory = OnlineFeatureMemory()
        self.goals = GoalSet(
            {
                "terminal_success": Goal(
                    "terminal_success",
                    GoalKind.FACT_PRESENT,
                    "terminal_success",
                    source="environment_terminal_observation",
                    final=True,
                )
            }
        )
        self.policy = WeightedPolicy()
        self.prophecy = TabularProphecy(name="aassr-core-tabular")
        self.replay = ReplayBuffer(
            capacity=self.config.replay_capacity,
            holdout_stride=self.config.holdout_stride,
        )
        self.validator = PredictionValidator(samples=self.config.prophecy_samples)
        self.information_value_predictor = InformationValuePredictor()
        self.unlock_estimator = ActionUnlockValueEstimator()
        self.delayed_credit_assigner = DelayedCreditAssigner(
            discount=self.config.gamma
        )
        self.skills = SkillLibrary(
            promotion_successes=self.config.skill_promotion_successes,
            maximum_length=self.config.skill_maximum_length,
        )
        self._episode_index = 0
        self.audit = CoreCallAudit()
        self._wire_runtime_components()

    def _wire_runtime_components(self) -> None:
        self.skill_aware_prophecy = SkillAwareProphecy(self.prophecy, self.skills)
        self.evaluator = AdvancedTransitionEvaluator(
            self.skill_aware_prophecy,
            replay=self.replay,
            validator=self.validator,
            predictor=self.information_value_predictor,
            unlock_estimator=self.unlock_estimator,
            credit_assigner=self.delayed_credit_assigner,
            samples=self.config.prophecy_samples,
        )
        self.imagination = ImaginationTree(
            self.policy,
            self.skill_aware_prophecy,
            config=self.config.imagination,
            scorer=GoalStateScorer(
                self.goals,
                final_goal_bonus=1.0,
                internal_goal_weight=0.0,
                step_cost=0.0,
            ),
        )
        self.observation_encoder = CoreObservationEncoder(
            dimension=self.config.feature_dimension
        )

    def _epsilon(self) -> float:
        fraction = min(1.0, self._episode_index / self.config.epsilon_decay_episodes)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

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

        self.audit.call("imagination_tree")
        self.audit.call("prophecy")
        imagined = self.imagination.plan(augmented)
        self.audit.work("imagination_tree", max(0, len(imagined.nodes) - 1))
        coverage = self.prophecy.coverage(augmented, augmented.available_actions)
        explore = training and self.randomizer.random() < self._epsilon()
        if explore:
            selected = self.randomizer.choice(augmented.available_actions)
            used_imagination = False
        elif coverage >= self.config.imagination_minimum_coverage:
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
            imagined_nodes=len(imagined.nodes),
            expanded_nodes=imagined.expanded_nodes,
        )

    def _observe_feature_memory(self, evaluation: AdvancedEvaluation) -> None:
        trace = evaluation.trace
        tokens = (
            f"action:{trace.action.verb_name}",
            *(f"added:{fact}" for fact in sorted(trace.added_facts)),
            *(f"removed:{fact}" for fact in sorted(trace.removed_facts)),
            f"error:{str(trace.error).lower()}",
            f"terminal_reward:{trace.real_reward}",
        )
        information_id = trace.trace_id
        self.feature_memory.observe_information(information_id, tokens)
        self.feature_memory.observe_use(
            information_id,
            action_id=trace.action.verb_name,
            slot="context",
            value=trace.real_reward,
        )
        self.audit.update("online_feature_memory", 2)

    def _execute_primitive(
        self,
        session: CoreEnvironmentSession,
        action: Action,
        *,
        training: bool,
        decision_action: Action,
        used_skill: bool,
        episode_evaluations: list[AdvancedEvaluation],
    ) -> CorePrimitiveStep:
        raw_before = session.raw_observation()
        before = session.snapshot()
        train_before = len(self.replay.train())
        holdout_before = len(self.replay.holdout())
        self.audit.call("advanced_transition_evaluator")
        self.audit.call("knowledge_store")
        self.audit.call("prophecy", 3)
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
        if training:
            episode_evaluations.append(evaluation)
            self.audit.update("advanced_transition_evaluator")
            self.audit.update("knowledge_store")
            self.audit.update("replay")
            train_delta = len(self.replay.train()) - train_before
            holdout_delta = len(self.replay.holdout()) - holdout_before
            if train_delta:
                self.audit.update("prophecy", train_delta)
            if holdout_delta:
                self.audit.update("holdout", holdout_delta)
            self._observe_feature_memory(evaluation)

        after = evaluation.trace.after
        knowledge_keys = tuple(entry.key for entry in self.knowledge.values())
        self.audit.call("goal", 2)
        before_goals = set(
            self.goals.achieved_ids(before, knowledge_keys=knowledge_keys)
        )
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
        return CorePrimitiveStep(
            decision_action=decision_action.signature,
            executed_action=action.signature,
            used_skill=used_skill,
            action_succeeded=not evaluation.trace.error,
            reward=evaluation.trace.real_reward,
            terminal=bool(raw_after.terminal),
            raw_observation_before=raw_before.to_dict(),
            raw_observation_after=raw_after.to_dict(),
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
        session = CoreEnvironmentSession(plugin, self.observation_encoder)
        decisions: list[CoreDecision] = []
        transitions: list[CorePrimitiveStep] = []
        evaluations: list[AdvancedEvaluation] = []
        while not plugin.terminal and len(transitions) < maximum_steps:
            decision = self.select_action(session.snapshot(), training=training)
            decisions.append(decision)
            action = decision.selected_action
            primitives: tuple[Action, ...]
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
            self.audit.work("delayed_credit_assigner", len(credits))
            self.audit.update("information_value_predictor", len(credits))
            self.audit.update("policy", len(credits))
            self._episode_index += 1
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
        )

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "config": self.config,
            "seed": self.seed,
            "random_state": self.randomizer.getstate(),
            "episode_index": self._episode_index,
            "knowledge": self.knowledge,
            "feature_memory": self.feature_memory,
            "goals": self.goals,
            "policy": self.policy,
            "prophecy": self.prophecy,
            "replay": self.replay,
            "validator": self.validator,
            "information_value_predictor": self.information_value_predictor,
            "unlock_estimator": self.unlock_estimator,
            "delayed_credit_assigner": self.delayed_credit_assigner,
            "skills": self.skills,
            "evaluator_index": self.evaluator._index,
            "evaluator_recent_pairs": tuple(self.evaluator._recent_pairs),
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
        core = cls(config=payload["config"], seed=int(payload["seed"]))
        core.randomizer.setstate(payload["random_state"])
        core._episode_index = int(payload["episode_index"])
        core.knowledge = payload["knowledge"]
        core.feature_memory = payload["feature_memory"]
        core.goals = payload["goals"]
        core.policy = payload["policy"]
        core.prophecy = payload["prophecy"]
        core.replay = payload["replay"]
        core.validator = payload["validator"]
        core.information_value_predictor = payload["information_value_predictor"]
        core.unlock_estimator = payload["unlock_estimator"]
        core.delayed_credit_assigner = payload["delayed_credit_assigner"]
        core.skills = payload["skills"]
        core.audit = CoreCallAudit()
        core._wire_runtime_components()
        core.evaluator._index = int(payload["evaluator_index"])
        core.evaluator._recent_pairs = list(payload["evaluator_recent_pairs"])
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
