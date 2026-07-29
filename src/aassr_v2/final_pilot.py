from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .counterexamples import LearnableVsRandomWorld, LongDependencyWorld, UncertaintyStep
from .experiment_runner import ExperimentArtifacts, RESULT_FIELDS
from .gru_prophecy import OnlineGRUProphecy
from .imagination_tree import ImaginationConfig, ImaginationTree, StateDeltaScorer
from .knowledge import KnowledgeStore
from .metrics import expected_prediction_vector, prediction_similarity
from .policy import WeightedPolicy
from .serialization import JsonlLedgerWriter
from .skills import SkillExecutor, SkillLibrary
from .tabular_prophecy import TabularProphecy
from .types import Action, Prediction, StateSnapshot, TransitionTrace
from .validated_learning import FixedHoldoutTransitionEvaluator


@dataclass(frozen=True, slots=True)
class PilotStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    reward: float = 0.0


class VerbWeightedPolicy(WeightedPolicy):
    """WeightedPolicy with a transparent fallback by opaque action family."""

    def __init__(self, verb_weights: Mapping[str, float], **kwargs: Any) -> None:
        super().__init__({}, **kwargs)
        self.verb_weights = dict(verb_weights)

    def weight(self, action: Action) -> float:
        return self._weights.get(
            action.signature,
            self.verb_weights.get(action.verb_name, 0.0),
        )


class DeceptiveChoiceWorld:
    def __init__(self) -> None:
        self.phase = "start"

    def snapshot(self) -> StateSnapshot:
        if self.phase == "start":
            return StateSnapshot(
                (1.0, 0.0, 0.0, 0.0),
                frozenset({"phase:start"}),
                (Action("shortcut"), Action("setup")),
                0.0,
            )
        if self.phase == "prepared":
            return StateSnapshot(
                (0.0, 1.0, 0.0, 0.0),
                frozenset({"phase:prepared"}),
                (Action("finish"), Action("stall")),
                0.1,
            )
        if self.phase == "trap":
            return StateSnapshot(
                (0.0, 0.0, 1.0, 0.0),
                frozenset({"phase:trap"}),
                (Action("stall"),),
                0.6,
            )
        return StateSnapshot(
            (0.0, 0.0, 0.0, 1.0),
            frozenset({"phase:goal"}),
            (),
            1.0,
        )

    def step(self, action: Action) -> PilotStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        if self.phase == "start" and action.verb_name == "shortcut":
            self.phase = "trap"
        elif self.phase == "start" and action.verb_name == "setup":
            self.phase = "prepared"
        elif self.phase == "prepared" and action.verb_name == "finish":
            self.phase = "goal"
            reward = 1.0
        elif action.verb_name == "stall":
            pass
        else:
            error = True
        return _outcome(before, self.snapshot(), error=error, reward=reward)


class TrapDependencyWorld:
    """Long dependency chain with a tempting irreversible branch at every stage."""

    def __init__(self, length: int = 4, *, seed: int = 0) -> None:
        if length < 2:
            raise ValueError("length must be at least two")
        self.length = length
        labels = [f"node-{index:03d}" for index in range(length)]
        random.Random(seed).shuffle(labels)
        self.labels = tuple(labels)
        self.stage = 0
        self.revealed = False
        self.trapped = False

    def _stage_action(self, verb: str) -> Action:
        index = min(self.stage, self.length - 1)
        return Action(verb, parameters={"node": self.labels[index]})

    def snapshot(self) -> StateSnapshot:
        goal = self.stage >= self.length and not self.trapped
        if self.trapped:
            actions = (Action("stall"),)
            progress = min(0.95, (self.stage + 1.5) / self.length)
        elif goal:
            actions = ()
            progress = 1.0
        elif self.revealed:
            actions = (
                self._stage_action("advance_greedy"),
                self._stage_action("advance_safe"),
            )
            progress = self.stage / self.length
        else:
            actions = (self._stage_action("inspect"),)
            progress = self.stage / self.length
        one_hot = tuple(
            1.0 if index == min(self.stage, self.length - 1) else 0.0
            for index in range(self.length)
        )
        vector = (
            progress,
            float(self.revealed),
            float(self.trapped),
            float(goal),
            *one_hot,
        )
        facts = {f"stage:{self.stage}"}
        if self.revealed and not goal:
            facts.add(f"revealed:{self.labels[self.stage]}")
        if self.trapped:
            facts.add("irreversible:trap")
        return StateSnapshot(vector, frozenset(facts), actions, progress)

    def step(self, action: Action) -> PilotStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        if self.trapped and action.verb_name == "stall":
            pass
        elif (
            not self.trapped
            and self.stage < self.length
            and not self.revealed
            and action.verb_name == "inspect"
        ):
            self.revealed = True
        elif (
            self.revealed
            and action.verb_name == "advance_safe"
            and action.parameters.get("node") == self.labels[self.stage]
        ):
            self.stage += 1
            self.revealed = False
            if self.stage >= self.length:
                reward = 1.0
        elif (
            self.revealed
            and action.verb_name == "advance_greedy"
            and action.parameters.get("node") == self.labels[self.stage]
        ):
            self.trapped = True
            self.revealed = False
        else:
            error = True
        return _outcome(before, self.snapshot(), error=error, reward=reward)


class InformationChoiceWorld:
    """Choose one useful fact or many causally irrelevant facts."""

    def __init__(self, *, noise_facts: int = 8, seed: int = 0) -> None:
        if noise_facts <= 1:
            raise ValueError("noise_facts must exceed one")
        self.noise_facts = noise_facts
        self.random = random.Random(seed)
        self.phase = "start"
        self.noise: set[str] = set()

    def snapshot(self) -> StateSnapshot:
        if self.phase == "start":
            actions = (Action("noise_probe"), Action("useful_probe"))
            progress = 0.0
        elif self.phase == "useful":
            actions = (Action("finish"),)
            progress = 0.0
        else:
            actions = ()
            progress = 1.0
        facts = set(self.noise)
        if self.phase in {"useful", "goal"}:
            facts.add("causal:finish_enabled")
        vector = (
            float(self.phase == "start"),
            float(self.phase == "useful"),
            float(self.phase == "goal"),
            0.0,
        )
        return StateSnapshot(vector, frozenset(facts), actions, progress)

    def step(self, action: Action) -> PilotStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        if self.phase == "start" and action.verb_name == "useful_probe":
            self.phase = "useful"
        elif self.phase == "start" and action.verb_name == "noise_probe":
            self.noise.update(
                f"noise:{self.random.randrange(1_000_000_000)}"
                for _ in range(self.noise_facts)
            )
        elif self.phase == "useful" and action.verb_name == "finish":
            self.phase = "goal"
            reward = 1.0
        else:
            error = True
        return _outcome(before, self.snapshot(), error=error, reward=reward)


def _outcome(
    before: StateSnapshot,
    after: StateSnapshot,
    *,
    error: bool = False,
    reward: float = 0.0,
) -> PilotStep:
    before_actions = {item.signature for item in before.available_actions}
    unlocked = tuple(
        item for item in after.available_actions if item.signature not in before_actions
    )
    return PilotStep(
        after,
        after.facts - before.facts,
        before.facts - after.facts,
        unlocked,
        error,
        reward,
    )


def _empty_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(values)
    return row


def _make_prophecy(
    model: str,
    state_size: int,
    seed: int,
    options: Mapping[str, Any] | None = None,
) -> object:
    settings = dict(options or {})
    if model == "tabular":
        return TabularProphecy()
    if model == "gru":
        return OnlineGRUProphecy(
            state_size,
            action_feature_size=int(settings.get("action_feature_size", 16)),
            hidden_size=int(settings.get("hidden_size", 24)),
            learning_rate=float(settings.get("learning_rate", 0.02)),
            replay_limit=int(settings.get("replay_limit", 512)),
            seed=seed,
        )
    raise ValueError(f"unknown prophecy model: {model}")


def _learn(prophecy: object, environment: object, action: Action) -> PilotStep:
    before = environment.snapshot()
    outcome = environment.step(action)
    prophecy.learn(before, action, outcome.snapshot)
    return outcome


def _reset(prophecy: object) -> None:
    method = getattr(prophecy, "reset_sequence", None)
    if callable(method):
        method()


def _tree(
    policy: WeightedPolicy,
    prophecy: object,
    condition: Mapping[str, Any],
    *,
    maximum_depth: int | None = None,
) -> ImaginationTree:
    return ImaginationTree(
        policy,
        prophecy,
        config=ImaginationConfig(
            branching_factor=int(condition.get("branching_factor", 2)),
            maximum_depth=maximum_depth or int(condition.get("maximum_depth", 2)),
            beam_width=int(condition.get("beam_width", 8)),
            outcome_samples=int(condition.get("outcome_samples", 1)),
            discount=float(condition.get("discount", 0.95)),
            minimum_path_confidence=float(
                condition.get("minimum_path_confidence", 0.0)
            ),
            uncertainty_penalty=float(condition.get("uncertainty_penalty", 0.0)),
            aggregation=str(condition.get("aggregation", "max")),
            update_policy=False,
        ),
        scorer=StateDeltaScorer(
            goal_progress_weight=float(condition.get("goal_progress_weight", 10.0)),
            new_fact_weight=float(condition.get("new_fact_weight", 0.0)),
            unlocked_action_weight=float(
                condition.get("unlocked_action_weight", 0.0)
            ),
            step_cost=float(condition.get("step_cost", 0.01)),
        ),
    )


def _pretrain_choice(prophecy: object, episodes: int) -> None:
    for _ in range(episodes):
        _reset(prophecy)
        shortcut = DeceptiveChoiceWorld()
        _learn(prophecy, shortcut, Action("shortcut"))
        _learn(prophecy, shortcut, Action("stall"))
        _reset(prophecy)
        solution = DeceptiveChoiceWorld()
        _learn(prophecy, solution, Action("setup"))
        _learn(prophecy, solution, Action("finish"))


def _pretrain_trap(prophecy: object, length: int, seed: int, episodes: int) -> None:
    for _ in range(episodes):
        _reset(prophecy)
        safe = TrapDependencyWorld(length, seed=seed)
        while safe.snapshot().goal_progress < 1.0:
            _learn(prophecy, safe, safe.snapshot().available_actions[0])
            safe_action = next(
                item
                for item in safe.snapshot().available_actions
                if item.verb_name == "advance_safe"
            )
            _learn(prophecy, safe, safe_action)
        for target_stage in range(length):
            _reset(prophecy)
            trap = TrapDependencyWorld(length, seed=seed)
            while trap.stage < target_stage:
                _learn(prophecy, trap, trap.snapshot().available_actions[0])
                safe_action = next(
                    item
                    for item in trap.snapshot().available_actions
                    if item.verb_name == "advance_safe"
                )
                _learn(prophecy, trap, safe_action)
            _learn(prophecy, trap, trap.snapshot().available_actions[0])
            greedy = next(
                item
                for item in trap.snapshot().available_actions
                if item.verb_name == "advance_greedy"
            )
            _learn(prophecy, trap, greedy)
            _learn(prophecy, trap, Action("stall"))


def _run_prophecy(
    name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_episodes = int(suite.get("train_episodes", 60))
    eval_episodes = int(suite.get("eval_episodes", 20))
    samples = int(suite.get("samples", 4))
    options = suite.get("model_options", {})
    for seed in seeds:
        for model in suite.get("models", ("tabular", "gru")):
            prophecy = _make_prophecy(str(model), 3, seed, options.get(model, {}))
            for episode in range(train_episodes):
                environment = LearnableVsRandomWorld(seed=seed * 10_000 + episode)
                _learn(prophecy, environment, Action("probe_stable"))
                _learn(prophecy, environment, Action("probe_random"))
            for episode in range(eval_episodes):
                for family in ("stable", "random"):
                    started = time.perf_counter()
                    environment = LearnableVsRandomWorld(seed=seed * 100_000 + episode)
                    action = Action(f"probe_{family}")
                    before = environment.snapshot()
                    predictions = prophecy.predict(before, action, samples=samples)
                    outcome = environment.step(action)
                    score = prediction_similarity(
                        expected_prediction_vector(predictions), outcome.snapshot.vector
                    )
                    rows.append(
                        _empty_row(
                            experiment=name,
                            suite="prophecy",
                            condition=f"{model}_{family}",
                            environment="learnable_vs_random",
                            model=model,
                            seed=seed,
                            episode=episode,
                            phase="evaluation",
                            success=int(score >= 0.9),
                            steps=1,
                            high_level_steps=1,
                            primitive_steps=1,
                            prediction_score=score,
                            actual_return=outcome.snapshot.goal_progress,
                            action_family=family,
                            runtime_seconds=time.perf_counter() - started,
                        )
                    )
    return rows


def _run_imagination(
    name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_episodes = int(suite.get("train_episodes", 20))
    eval_episodes = int(suite.get("eval_episodes", 20))
    max_steps = int(suite.get("max_steps", 3))
    for seed in seeds:
        for condition in suite["conditions"]:
            model = str(condition.get("model", "tabular"))
            prophecy = _make_prophecy(model, 4, seed, condition.get("model_options"))
            _pretrain_choice(prophecy, train_episodes)
            policy = VerbWeightedPolicy(
                {"shortcut": 1.0, "setup": 0.0, "finish": 0.5, "stall": -0.1}
            )
            mode = str(condition.get("mode", "imagination"))
            planner = _tree(policy, prophecy, condition) if mode == "imagination" else None
            for episode in range(eval_episodes):
                started = time.perf_counter()
                environment = DeceptiveChoiceWorld()
                steps = 0
                nodes = 0
                depth = 0
                first_action = ""
                first_value: float | str = ""
                while environment.snapshot().goal_progress < 1.0 and steps < max_steps:
                    state = environment.snapshot()
                    if planner is None:
                        action = policy.rank(state, limit=1)[0].action
                    else:
                        plan = planner.plan(state)
                        action = plan.chosen_action
                        nodes += len(plan.nodes)
                        depth = max(depth, plan.maximum_depth_reached)
                        if steps == 0:
                            first_value = next(
                                item.aggregate_value
                                for item in plan.root_evaluations
                                if item.action.signature == action.signature
                            )
                    if steps == 0:
                        first_action = action.verb_name
                    environment.step(action)
                    steps += 1
                progress = environment.snapshot().goal_progress
                rows.append(
                    _empty_row(
                        experiment=name,
                        suite="imagination",
                        condition=condition["name"],
                        environment="deceptive_choice",
                        model=model,
                        seed=seed,
                        episode=episode,
                        phase="evaluation",
                        success=int(progress >= 1.0),
                        steps=steps,
                        high_level_steps=steps,
                        primitive_steps=steps,
                        reward=int(progress >= 1.0),
                        imagined_nodes=nodes,
                        imagination_depth=depth,
                        root_imagined_value=first_value,
                        actual_return=progress,
                        action_family=first_action,
                        runtime_seconds=time.perf_counter() - started,
                    )
                )
    return rows


def _run_dependency(
    name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_episodes = int(suite.get("train_episodes", 8))
    eval_episodes = int(suite.get("eval_episodes", 20))
    for seed in seeds:
        for length in (int(item) for item in suite.get("lengths", (4, 6))):
            for condition in suite["conditions"]:
                model = str(condition.get("model", "tabular"))
                prophecy = _make_prophecy(model, length + 4, seed)
                _pretrain_trap(prophecy, length, seed, train_episodes)
                policy = VerbWeightedPolicy(
                    {
                        "inspect": 0.5,
                        "advance_greedy": 1.0,
                        "advance_safe": 0.0,
                        "stall": -1.0,
                    }
                )
                mode = str(condition.get("mode", "policy"))
                depth_limit = int(
                    condition.get(
                        "maximum_depth",
                        length * int(condition.get("depth_multiplier", 2)),
                    )
                )
                planner = (
                    _tree(policy, prophecy, condition, maximum_depth=depth_limit)
                    if mode == "imagination"
                    else None
                )
                for episode in range(eval_episodes):
                    started = time.perf_counter()
                    environment = TrapDependencyWorld(length, seed=seed)
                    steps = 0
                    nodes = 0
                    reached_depth = 0
                    first_decision = ""
                    max_steps = length * 3
                    while (
                        environment.snapshot().goal_progress < 1.0
                        and not environment.trapped
                        and steps < max_steps
                    ):
                        state = environment.snapshot()
                        if planner is None:
                            action = policy.rank(state, limit=1)[0].action
                        else:
                            plan = planner.plan(state)
                            action = plan.chosen_action
                            nodes += len(plan.nodes)
                            reached_depth = max(
                                reached_depth, plan.maximum_depth_reached
                            )
                        if action.verb_name.startswith("advance") and not first_decision:
                            first_decision = action.verb_name
                        environment.step(action)
                        steps += 1
                    progress = environment.snapshot().goal_progress
                    rows.append(
                        _empty_row(
                            experiment=name,
                            suite="dependency",
                            condition=condition["name"],
                            environment=f"trap_dependency_{length}",
                            model=model,
                            seed=seed,
                            episode=episode,
                            phase="evaluation",
                            success=int(progress >= 1.0),
                            steps=steps,
                            high_level_steps=steps,
                            primitive_steps=steps,
                            reward=int(progress >= 1.0),
                            imagined_nodes=nodes,
                            imagination_depth=reached_depth,
                            actual_return=progress,
                            action_family=first_decision,
                            runtime_seconds=time.perf_counter() - started,
                        )
                    )
    return rows


def _trace(
    trace_id: str,
    before: StateSnapshot,
    action: Action,
    outcome: UncertaintyStep | PilotStep,
) -> TransitionTrace:
    return TransitionTrace(
        trace_id=trace_id,
        before=before,
        action=action,
        predictions=(Prediction(outcome.snapshot, 1.0, "observed"),),
        after=outcome.snapshot,
        added_facts=outcome.added_facts,
        removed_facts=outcome.removed_facts,
        unlocked_actions=outcome.unlocked_actions,
        error=outcome.error,
        real_reward=outcome.reward,
        goal_ids=("final",) if outcome.snapshot.goal_progress >= 1.0 else (),
    )


def _run_skills(
    name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    length = int(suite.get("length", 4))
    episodes = int(suite.get("episodes", 8))
    for seed in seeds:
        for condition in suite["conditions"]:
            enabled = bool(condition.get("use_skills", True))
            library = SkillLibrary(
                promotion_successes=int(condition.get("promotion_successes", 2)),
                maximum_length=length * 2,
            )
            executor = SkillExecutor(library)
            for episode in range(episodes):
                started = time.perf_counter()
                environment = LongDependencyWorld(length)
                traces: list[TransitionTrace] = []
                primitive_steps = 0
                high_level_steps = 0
                skill_uses = 0
                actions = library.actions_for(environment.snapshot())
                if enabled and actions:
                    result = executor.execute(environment, actions[0])
                    primitive_steps = len(result.outcomes)
                    high_level_steps = 1
                    skill_uses = 1
                else:
                    index = 0
                    while environment.snapshot().goal_progress < 1.0:
                        stage = environment.stage
                        for action in (
                            Action("inspect", parameters={"stage": stage}),
                            Action("advance", parameters={"stage": stage}),
                        ):
                            before = environment.snapshot()
                            outcome = environment.step(action)
                            index += 1
                            traces.append(
                                _trace(
                                    f"skill-{seed}-{episode}-{index}",
                                    before,
                                    action,
                                    outcome,
                                )
                            )
                            primitive_steps += 1
                            high_level_steps += 1
                    if enabled:
                        library.observe_goal_completion(
                            traces, achieved_goal_ids=("final",)
                        )
                progress = environment.snapshot().goal_progress
                rows.append(
                    _empty_row(
                        experiment=name,
                        suite="skills",
                        condition=condition["name"],
                        environment=f"long_dependency_{length}",
                        model="observed_sequence",
                        seed=seed,
                        episode=episode,
                        phase="evaluation",
                        success=int(progress >= 1.0),
                        steps=primitive_steps,
                        high_level_steps=high_level_steps,
                        primitive_steps=primitive_steps,
                        reward=int(progress >= 1.0),
                        actual_return=progress,
                        skill_count=len(library.all()),
                        skill_uses=skill_uses,
                        runtime_seconds=time.perf_counter() - started,
                    )
                )
    return rows


def _epsilon_action(
    policy: WeightedPolicy,
    state: StateSnapshot,
    randomizer: random.Random,
    epsilon: float,
) -> Action:
    if randomizer.random() < epsilon:
        return randomizer.choice(state.available_actions)
    return policy.rank(state, limit=1)[0].action


def _run_information_value(
    name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_episodes = int(suite.get("train_episodes", 80))
    eval_episodes = int(suite.get("eval_episodes", 40))
    noise_facts = int(suite.get("noise_facts", 8))
    epsilon = float(suite.get("epsilon", 0.2))
    novelty_weight = float(suite.get("novelty_weight", 0.2))
    for seed in seeds:
        for condition in suite["conditions"]:
            mode = str(condition["mode"])
            policy = WeightedPolicy(real_learning_rate=float(condition.get("lr", 0.2)))
            randomizer = random.Random(seed)
            evaluator = None
            knowledge = KnowledgeStore()
            if mode == "validated_information":
                evaluator = FixedHoldoutTransitionEvaluator(
                    TabularProphecy(),
                    logger=JsonlLedgerWriter(
                        output_dir
                        / "traces"
                        / f"information_{condition['name']}_seed{seed}.jsonl"
                    ),
                    minimum_holdout_count=int(
                        condition.get("minimum_holdout_count", 4)
                    ),
                    samples=int(condition.get("samples", 4)),
                )
            recent_intrinsic: list[float] = []
            recent_holdout: list[float] = []
            recent_novelty: list[float] = []
            for episode in range(train_episodes):
                environment = InformationChoiceWorld(
                    noise_facts=noise_facts, seed=seed * 100_000 + episode
                )
                first = _epsilon_action(
                    policy, environment.snapshot(), randomizer, epsilon
                )
                evaluations = []
                if evaluator is not None:
                    first_evaluation = evaluator.execute(environment, first, knowledge)
                    evaluations.append(first_evaluation)
                    first_novelty = len(first_evaluation.trace.added_facts)
                    recent_intrinsic.append(
                        first_evaluation.immediate_information_value
                    )
                    recent_holdout.append(first_evaluation.effect.holdout_gain)
                    recent_novelty.append(float(first_novelty))
                    if environment.snapshot().available_actions and first.verb_name == "useful_probe":
                        evaluations.append(
                            evaluator.execute(environment, Action("finish"), knowledge)
                        )
                    success = float(environment.snapshot().goal_progress >= 1.0)
                    evaluator.finish_episode(
                        evaluations, final_return=success, policy=policy
                    )
                else:
                    outcome = environment.step(first)
                    first_novelty = len(outcome.added_facts)
                    if first.verb_name == "useful_probe":
                        environment.step(Action("finish"))
                    success = float(environment.snapshot().goal_progress >= 1.0)
                    target = success
                    if mode == "novelty_count":
                        target += novelty_weight * first_novelty
                    policy.reinforce(first, target)
                    recent_intrinsic.append(target - success)
                    recent_holdout.append(0.0)
                    recent_novelty.append(float(first_novelty))
            diagnostic_intrinsic = fmean(recent_intrinsic[-20:])
            diagnostic_holdout = fmean(recent_holdout[-20:])
            diagnostic_novelty = fmean(recent_novelty[-20:])
            for episode in range(eval_episodes):
                started = time.perf_counter()
                environment = InformationChoiceWorld(
                    noise_facts=noise_facts,
                    seed=seed * 1_000_000 + episode,
                )
                first = policy.rank(environment.snapshot(), limit=1)[0].action
                outcome = environment.step(first)
                if first.verb_name == "useful_probe":
                    environment.step(Action("finish"))
                success = int(environment.snapshot().goal_progress >= 1.0)
                rows.append(
                    _empty_row(
                        experiment=name,
                        suite="information_value",
                        condition=condition["name"],
                        environment=f"information_choice_noise{noise_facts}",
                        model="tabular" if evaluator is not None else "policy",
                        seed=seed,
                        episode=episode,
                        phase="evaluation",
                        success=success,
                        steps=2 if success else 1,
                        high_level_steps=2 if success else 1,
                        primitive_steps=2 if success else 1,
                        reward=success,
                        holdout_gain=diagnostic_holdout,
                        actual_return=float(success),
                        noise_facts=sum(
                            fact.startswith("noise:") for fact in outcome.added_facts
                        ),
                        novelty_score=diagnostic_novelty,
                        intrinsic_value=diagnostic_intrinsic,
                        action_family=first.verb_name,
                        runtime_seconds=time.perf_counter() - started,
                    )
                )
    return rows


def validate_final_config(config: Mapping[str, Any]) -> None:
    if config.get("runner") != "final_pilot":
        raise ValueError("final pilot config must set runner to final_pilot")
    if not str(config.get("name", "")).strip():
        raise ValueError("config needs a name")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("config needs integer seeds")
    suites = config.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("config needs suites")
    supported = {
        "prophecy",
        "imagination",
        "dependency",
        "skills",
        "information_value",
    }
    for suite in suites:
        if suite.get("kind") not in supported:
            raise ValueError(f"unsupported suite kind: {suite.get('kind')}")
        if suite.get("kind") in {
            "imagination",
            "dependency",
            "skills",
            "information_value",
        } and not suite.get("conditions"):
            raise ValueError(f"{suite.get('kind')} needs conditions")


def load_final_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_final_config(config)
    return config


def planned_final_run_count(
    config: Mapping[str, Any],
    suite_filter: set[str] | None = None,
) -> int:
    seeds = tuple(config["seeds"])
    count = 0
    for suite in config["suites"]:
        kind = str(suite["kind"])
        if not bool(suite.get("enabled", True)):
            continue
        if suite_filter and kind not in suite_filter:
            continue
        if kind == "prophecy":
            count += (
                len(seeds)
                * len(suite.get("models", ("tabular", "gru")))
                * int(suite.get("eval_episodes", 20))
                * 2
            )
        elif kind == "imagination":
            count += len(seeds) * len(suite["conditions"]) * int(
                suite.get("eval_episodes", 20)
            )
        elif kind == "dependency":
            count += (
                len(seeds)
                * len(suite["conditions"])
                * len(suite.get("lengths", (4, 6)))
                * int(suite.get("eval_episodes", 20))
            )
        elif kind == "skills":
            count += len(seeds) * len(suite["conditions"]) * int(
                suite.get("episodes", 8)
            )
        elif kind == "information_value":
            count += len(seeds) * len(suite["conditions"]) * int(
                suite.get("eval_episodes", 40)
            )
    return count


def _output_directory(
    config: Mapping[str, Any],
    output_dir: str | Path | None,
    overwrite: bool,
) -> Path:
    target = Path(output_dir or config.get("output_dir", "runs/final_pilot"))
    if target.exists() and overwrite:
        shutil.rmtree(target)
    elif target.exists():
        target = target.with_name(
            f"{target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_final_pilot(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    suite_filter: Sequence[str] | None = None,
    seed_override: Sequence[int] | None = None,
) -> ExperimentArtifacts:
    resolved = dict(config)
    if seed_override is not None:
        resolved["seeds"] = list(seed_override)
    validate_final_config(resolved)
    target = _output_directory(resolved, output_dir, overwrite)
    (target / "traces").mkdir(exist_ok=True)
    selected = set(suite_filter or ()) or None
    seeds = tuple(int(seed) for seed in resolved["seeds"])
    rows: list[dict[str, Any]] = []
    for suite in resolved["suites"]:
        kind = str(suite["kind"])
        if not bool(suite.get("enabled", True)):
            continue
        if selected and kind not in selected:
            continue
        if kind == "prophecy":
            rows.extend(_run_prophecy(resolved["name"], suite, seeds))
        elif kind == "imagination":
            rows.extend(_run_imagination(resolved["name"], suite, seeds))
        elif kind == "dependency":
            rows.extend(_run_dependency(resolved["name"], suite, seeds))
        elif kind == "skills":
            rows.extend(_run_skills(resolved["name"], suite, seeds))
        elif kind == "information_value":
            rows.extend(
                _run_information_value(resolved["name"], suite, seeds, target)
            )
    episodes = target / "episodes.csv"
    resolved_path = target / "resolved_config.json"
    _write_rows(episodes, rows)
    resolved_path.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ExperimentArtifacts(
        target,
        episodes,
        target / "summary.csv",
        target / "report.md",
        resolved_path,
        len(rows),
    )
