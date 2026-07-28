from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from .counterexamples import (
    LearnableVsRandomWorld,
    LongDependencyWorld,
    NoisyInformationWrapper,
    UncertaintyStep,
)
from .gru_prophecy import OnlineGRUProphecy
from .imagination_tree import (
    ImaginationConfig,
    ImaginationTree,
    StateDeltaScorer,
)
from .knowledge import KnowledgeStore
from .learning import AdvancedTransitionEvaluator
from .metrics import expected_prediction_vector, prediction_similarity
from .policy import WeightedPolicy
from .serialization import JsonlLedgerWriter
from .skills import SkillExecutor, SkillLibrary
from .tabular_prophecy import TabularProphecy
from .types import Action, Prediction, StateSnapshot, TransitionTrace


RESULT_FIELDS = (
    "experiment",
    "suite",
    "condition",
    "environment",
    "model",
    "seed",
    "episode",
    "phase",
    "success",
    "steps",
    "high_level_steps",
    "primitive_steps",
    "reward",
    "errors",
    "repeats",
    "prediction_score",
    "holdout_score",
    "holdout_gain",
    "imagined_nodes",
    "imagination_depth",
    "root_imagined_value",
    "actual_return",
    "skill_count",
    "skill_uses",
    "cluster_count",
    "noise_facts",
    "novelty_score",
    "intrinsic_value",
    "action_family",
    "runtime_seconds",
)

SUMMARY_METRICS = (
    "success",
    "steps",
    "high_level_steps",
    "primitive_steps",
    "reward",
    "errors",
    "repeats",
    "prediction_score",
    "holdout_score",
    "holdout_gain",
    "imagined_nodes",
    "imagination_depth",
    "root_imagined_value",
    "actual_return",
    "skill_count",
    "skill_uses",
    "noise_facts",
    "novelty_score",
    "intrinsic_value",
    "runtime_seconds",
)


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    output_dir: Path
    episodes_csv: Path
    summary_csv: Path
    report_md: Path
    resolved_config_json: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class ChoiceStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    reward: float = 0.0


class DeceptiveChoiceWorld:
    """Pilot world where immediate progress points toward a dead end.

    ``shortcut`` looks better at depth one, but ``setup -> finish`` is the only
    route to the final goal. It is intentionally tiny so a pilot run can verify
    that deeper Imagination changes the root decision.
    """

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

    def step(self, action: Action) -> ChoiceStep:
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
        after = self.snapshot()
        before_actions = {
            candidate.signature for candidate in before.available_actions
        }
        unlocked = tuple(
            candidate
            for candidate in after.available_actions
            if candidate.signature not in before_actions
        )
        return ChoiceStep(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=unlocked,
            error=error,
            reward=reward,
        )


def _empty_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(values)
    return row


def _action_signature(name: str) -> str:
    return Action(name).signature


def _make_prophecy(
    model_name: str,
    state_size: int,
    seed: int,
    model_options: Mapping[str, Any] | None = None,
) -> object:
    options = dict(model_options or {})
    if model_name == "tabular":
        return TabularProphecy()
    if model_name == "gru":
        return OnlineGRUProphecy(
            state_size,
            action_feature_size=int(options.get("action_feature_size", 16)),
            hidden_size=int(options.get("hidden_size", 24)),
            learning_rate=float(options.get("learning_rate", 0.02)),
            replay_limit=int(options.get("replay_limit", 512)),
            seed=seed,
        )
    raise ValueError(f"unknown prophecy model: {model_name}")


def _reset_sequence(prophecy: object) -> None:
    reset = getattr(prophecy, "reset_sequence", None)
    if callable(reset):
        reset()


def _learn_step(prophecy: object, environment: object, action: Action) -> Any:
    before = environment.snapshot()
    outcome = environment.step(action)
    prophecy.learn(before, action, outcome.snapshot)
    return outcome


def _pretrain_choice_world(
    prophecy: object,
    episodes: int,
) -> None:
    for _ in range(episodes):
        _reset_sequence(prophecy)
        shortcut = DeceptiveChoiceWorld()
        _learn_step(prophecy, shortcut, Action("shortcut"))
        _learn_step(prophecy, shortcut, Action("stall"))

        _reset_sequence(prophecy)
        solution = DeceptiveChoiceWorld()
        _learn_step(prophecy, solution, Action("setup"))
        _learn_step(prophecy, solution, Action("finish"))

        _reset_sequence(prophecy)
        stalled = DeceptiveChoiceWorld()
        _learn_step(prophecy, stalled, Action("setup"))
        _learn_step(prophecy, stalled, Action("stall"))


def _dependency_actions(environment: LongDependencyWorld) -> tuple[Action, Action]:
    stage = environment.stage
    return (
        Action("inspect", parameters={"stage": stage}),
        Action("advance", parameters={"stage": stage}),
    )


def _pretrain_dependency_world(
    prophecy: object,
    length: int,
    episodes: int,
) -> None:
    for _ in range(episodes):
        _reset_sequence(prophecy)
        environment = LongDependencyWorld(length)
        while environment.snapshot().goal_progress < 1.0:
            inspect, advance = _dependency_actions(environment)
            _learn_step(prophecy, environment, inspect)
            _learn_step(prophecy, environment, advance)


def _condition_policy(
    condition: Mapping[str, Any],
) -> WeightedPolicy:
    weights = {
        _action_signature("shortcut"): float(
            condition.get("shortcut_weight", 1.0)
        ),
        _action_signature("setup"): float(
            condition.get("setup_weight", 0.0)
        ),
        _action_signature("finish"): float(
            condition.get("finish_weight", 0.5)
        ),
        _action_signature("stall"): float(
            condition.get("stall_weight", -0.1)
        ),
        _action_signature("advance"): float(
            condition.get("advance_weight", 0.5)
        ),
        _action_signature("inspect"): float(
            condition.get("inspect_weight", 0.0)
        ),
    }
    return WeightedPolicy(
        weights,
        imagination_learning_rate=float(
            condition.get("imagination_learning_rate", 0.1)
        ),
        real_learning_rate=float(
            condition.get("real_learning_rate", 0.2)
        ),
    )


def _tree_for_condition(
    condition: Mapping[str, Any],
    policy: WeightedPolicy,
    prophecy: object,
) -> ImaginationTree:
    config = ImaginationConfig(
        branching_factor=int(condition.get("branching_factor", 2)),
        maximum_depth=int(condition.get("maximum_depth", 2)),
        beam_width=int(condition.get("beam_width", 8)),
        outcome_samples=int(condition.get("outcome_samples", 1)),
        discount=float(condition.get("discount", 0.95)),
        minimum_path_confidence=float(
            condition.get("minimum_path_confidence", 0.0)
        ),
        uncertainty_penalty=float(
            condition.get("uncertainty_penalty", 0.0)
        ),
        goal_threshold=float(condition.get("goal_threshold", 1.0)),
        aggregation=str(condition.get("aggregation", "max")),
        top_mean_count=int(condition.get("top_mean_count", 2)),
        update_policy=bool(condition.get("update_policy", False)),
    )
    return ImaginationTree(
        policy,
        prophecy,
        config=config,
        scorer=StateDeltaScorer(
            goal_progress_weight=float(
                condition.get("goal_progress_weight", 10.0)
            ),
            new_fact_weight=float(condition.get("new_fact_weight", 0.0)),
            unlocked_action_weight=float(
                condition.get("unlocked_action_weight", 0.0)
            ),
            step_cost=float(condition.get("step_cost", 0.01)),
        ),
    )


def _run_prophecy_suite(
    experiment_name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = tuple(suite.get("models", ("tabular", "gru")))
    train_episodes = int(suite.get("train_episodes", 30))
    eval_episodes = int(suite.get("eval_episodes", 20))
    samples = int(suite.get("samples", 4))
    model_options = suite.get("model_options", {})

    for seed in seeds:
        for model_name in models:
            prophecy = _make_prophecy(
                str(model_name),
                3,
                seed,
                model_options.get(model_name, {}),
            )
            for episode in range(train_episodes):
                _reset_sequence(prophecy)
                environment = LearnableVsRandomWorld(seed=seed * 10_000 + episode)
                _learn_step(prophecy, environment, Action("probe_stable"))
                _learn_step(prophecy, environment, Action("probe_random"))

            for episode in range(eval_episodes):
                for action_family in ("stable", "random"):
                    started = time.perf_counter()
                    environment = LearnableVsRandomWorld(
                        seed=seed * 100_000 + episode
                    )
                    action = Action(f"probe_{action_family}")
                    before = environment.snapshot()
                    predictions = prophecy.predict(
                        before,
                        action,
                        samples=samples,
                    )
                    outcome = environment.step(action)
                    score = prediction_similarity(
                        expected_prediction_vector(predictions),
                        outcome.snapshot.vector,
                    )
                    rows.append(
                        _empty_row(
                            experiment=experiment_name,
                            suite="prophecy",
                            condition=f"{model_name}_{action_family}",
                            environment="learnable_vs_random",
                            model=model_name,
                            seed=seed,
                            episode=episode,
                            phase="evaluation",
                            success=int(score >= 0.9),
                            steps=1,
                            high_level_steps=1,
                            primitive_steps=1,
                            reward=outcome.reward,
                            errors=int(outcome.error),
                            prediction_score=score,
                            actual_return=outcome.snapshot.goal_progress,
                            action_family=action_family,
                            runtime_seconds=time.perf_counter() - started,
                        )
                    )
    return rows


def _run_imagination_suite(
    experiment_name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = tuple(suite.get("conditions", ()))
    train_episodes = int(suite.get("train_episodes", 12))
    eval_episodes = int(suite.get("eval_episodes", 20))
    max_steps = int(suite.get("max_steps", 3))

    for seed in seeds:
        for condition in conditions:
            condition_name = str(condition["name"])
            model_name = str(condition.get("model", "tabular"))
            prophecy = _make_prophecy(
                model_name,
                4,
                seed,
                condition.get("model_options", {}),
            )
            _pretrain_choice_world(prophecy, train_episodes)
            policy = _condition_policy(condition)
            mode = str(condition.get("mode", "imagination"))
            tree = (
                _tree_for_condition(condition, policy, prophecy)
                if mode == "imagination"
                else None
            )

            for episode in range(eval_episodes):
                started = time.perf_counter()
                environment = DeceptiveChoiceWorld()
                steps = 0
                errors = 0
                imagined_nodes = 0
                maximum_depth = 0
                first_root_value: float | str = ""
                first_action = ""
                while (
                    environment.snapshot().goal_progress < 1.0
                    and steps < max_steps
                ):
                    state = environment.snapshot()
                    if not state.available_actions:
                        break
                    if tree is None:
                        action = policy.rank(state, limit=1)[0].action
                    else:
                        plan = tree.plan(state)
                        action = plan.chosen_action
                        imagined_nodes += len(plan.nodes)
                        maximum_depth = max(
                            maximum_depth,
                            plan.maximum_depth_reached,
                        )
                        if steps == 0:
                            first_root_value = next(
                                item.aggregate_value
                                for item in plan.root_evaluations
                                if item.action.signature == action.signature
                            )
                    if steps == 0:
                        first_action = action.verb_name
                    outcome = environment.step(action)
                    errors += int(outcome.error)
                    steps += 1
                final_progress = environment.snapshot().goal_progress
                rows.append(
                    _empty_row(
                        experiment=experiment_name,
                        suite="imagination",
                        condition=condition_name,
                        environment="deceptive_choice",
                        model=model_name,
                        seed=seed,
                        episode=episode,
                        phase="evaluation",
                        success=int(final_progress >= 1.0),
                        steps=steps,
                        high_level_steps=steps,
                        primitive_steps=steps,
                        reward=int(final_progress >= 1.0),
                        errors=errors,
                        imagined_nodes=imagined_nodes,
                        imagination_depth=maximum_depth,
                        root_imagined_value=first_root_value,
                        actual_return=final_progress,
                        action_family=first_action,
                        runtime_seconds=time.perf_counter() - started,
                    )
                )
    return rows


def _run_dependency_suite(
    experiment_name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = tuple(suite.get("conditions", ()))
    lengths = tuple(int(value) for value in suite.get("lengths", (4, 6)))
    train_episodes = int(suite.get("train_episodes", 6))
    eval_episodes = int(suite.get("eval_episodes", 20))
    max_steps_multiplier = int(suite.get("max_steps_multiplier", 4))

    for seed in seeds:
        for length in lengths:
            for condition in conditions:
                condition_name = str(condition["name"])
                mode = str(condition.get("mode", "policy"))
                model_name = str(condition.get("model", "tabular"))
                prophecy = _make_prophecy(
                    model_name,
                    length + 1,
                    seed,
                    condition.get("model_options", {}),
                )
                _pretrain_dependency_world(prophecy, length, train_episodes)
                policy = _condition_policy(condition)
                tree = (
                    _tree_for_condition(condition, policy, prophecy)
                    if mode == "imagination"
                    else None
                )
                randomizer = random.Random(seed)

                for episode in range(eval_episodes):
                    started = time.perf_counter()
                    environment = LongDependencyWorld(length)
                    steps = 0
                    errors = 0
                    repeats = 0
                    previous_signature = None
                    imagined_nodes = 0
                    maximum_depth = 0
                    max_steps = length * max_steps_multiplier
                    while (
                        environment.snapshot().goal_progress < 1.0
                        and steps < max_steps
                    ):
                        state = environment.snapshot()
                        if mode == "random":
                            action = randomizer.choice(state.available_actions)
                        elif tree is None:
                            action = policy.rank(state, limit=1)[0].action
                        else:
                            plan = tree.plan(state)
                            action = plan.chosen_action
                            imagined_nodes += len(plan.nodes)
                            maximum_depth = max(
                                maximum_depth,
                                plan.maximum_depth_reached,
                            )
                        repeats += int(action.signature == previous_signature)
                        previous_signature = action.signature
                        outcome = environment.step(action)
                        errors += int(outcome.error)
                        steps += 1
                    progress = environment.snapshot().goal_progress
                    rows.append(
                        _empty_row(
                            experiment=experiment_name,
                            suite="dependency",
                            condition=condition_name,
                            environment=f"long_dependency_{length}",
                            model=model_name,
                            seed=seed,
                            episode=episode,
                            phase="evaluation",
                            success=int(progress >= 1.0),
                            steps=steps,
                            high_level_steps=steps,
                            primitive_steps=steps,
                            reward=int(progress >= 1.0),
                            errors=errors,
                            repeats=repeats,
                            imagined_nodes=imagined_nodes,
                            imagination_depth=maximum_depth,
                            actual_return=progress,
                            runtime_seconds=time.perf_counter() - started,
                        )
                    )
    return rows


def _trace_from_step(
    trace_id: str,
    before: StateSnapshot,
    action: Action,
    outcome: UncertaintyStep,
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


def _run_skill_suite(
    experiment_name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = tuple(suite.get("conditions", ()))
    length = int(suite.get("length", 4))
    episodes = int(suite.get("episodes", 6))

    for seed in seeds:
        del seed
        for condition in conditions:
            condition_name = str(condition["name"])
            enabled = bool(condition.get("use_skills", True))
            library = SkillLibrary(
                promotion_successes=int(
                    condition.get("promotion_successes", 2)
                ),
                maximum_length=int(
                    condition.get("maximum_length", length * 2)
                ),
            )
            executor = SkillExecutor(library)
            for episode in range(episodes):
                started = time.perf_counter()
                environment = LongDependencyWorld(length)
                traces: list[TransitionTrace] = []
                primitive_steps = 0
                high_level_steps = 0
                skill_uses = 0
                skill_actions = library.actions_for(environment.snapshot())
                if enabled and skill_actions:
                    result = executor.execute(environment, skill_actions[0])
                    primitive_steps = len(result.outcomes)
                    high_level_steps = 1
                    skill_uses = 1
                else:
                    trace_index = 0
                    while environment.snapshot().goal_progress < 1.0:
                        for action in _dependency_actions(environment):
                            before = environment.snapshot()
                            outcome = environment.step(action)
                            trace_index += 1
                            traces.append(
                                _trace_from_step(
                                    f"skill-{episode}-{trace_index}",
                                    before,
                                    action,
                                    outcome,
                                )
                            )
                            primitive_steps += 1
                            high_level_steps += 1
                    if enabled:
                        library.observe_goal_completion(
                            traces,
                            achieved_goal_ids=("final",),
                        )
                progress = environment.snapshot().goal_progress
                rows.append(
                    _empty_row(
                        experiment=experiment_name,
                        suite="skills",
                        condition=condition_name,
                        environment=f"long_dependency_{length}",
                        model="observed_sequence",
                        seed=0,
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


def _run_information_value_suite(
    experiment_name: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    noise_levels = tuple(int(value) for value in suite.get("noise_levels", (0, 8)))
    steps = int(suite.get("steps", 30))
    samples = int(suite.get("samples", 4))

    for seed in seeds:
        for noise in noise_levels:
            prophecy = TabularProphecy()
            knowledge = KnowledgeStore()
            logger = JsonlLedgerWriter(
                output_dir / "traces" / f"information_seed{seed}_noise{noise}.jsonl"
            )
            evaluator = AdvancedTransitionEvaluator(
                prophecy,
                logger=logger,
                samples=samples,
            )
            environment: object = LearnableVsRandomWorld(seed=seed)
            if noise > 0:
                environment = NoisyInformationWrapper(
                    environment,
                    facts_per_step=noise,
                    seed=seed,
                )
            previous_signature = None
            for episode in range(steps):
                started = time.perf_counter()
                action_family = "stable" if episode % 2 == 0 else "random"
                action = Action(f"probe_{action_family}")
                evaluation = evaluator.execute(environment, action, knowledge)
                features = evaluation.features
                noise_facts = sum(
                    fact.startswith("noise:")
                    for fact in evaluation.trace.added_facts
                )
                repeats = int(action.signature == previous_signature)
                previous_signature = action.signature
                rows.append(
                    _empty_row(
                        experiment=experiment_name,
                        suite="information_value",
                        condition=f"noise_{noise}",
                        environment="learnable_vs_random",
                        model="tabular",
                        seed=seed,
                        episode=episode,
                        phase="training",
                        success=int(
                            evaluation.trace.after.goal_progress >= 1.0
                        ),
                        steps=1,
                        high_level_steps=1,
                        primitive_steps=1,
                        reward=evaluation.trace.real_reward,
                        errors=int(evaluation.trace.error),
                        repeats=repeats,
                        prediction_score=(
                            evaluation.effect.latest_prediction_before
                        ),
                        holdout_score=evaluation.effect.holdout_after,
                        holdout_gain=evaluation.effect.holdout_gain,
                        actual_return=evaluation.trace.after.goal_progress,
                        noise_facts=noise_facts,
                        novelty_score=len(evaluation.trace.added_facts),
                        intrinsic_value=evaluation.immediate_information_value,
                        action_family=action_family,
                        runtime_seconds=time.perf_counter() - started,
                    )
                )
    return rows


def validate_config(config: Mapping[str, Any]) -> None:
    if not str(config.get("name", "")).strip():
        raise ValueError("config needs a non-empty name")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("config needs a non-empty seeds list")
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("every seed must be an integer")
    suites = config.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("config needs a non-empty suites list")
    supported = {
        "prophecy",
        "imagination",
        "dependency",
        "skills",
        "information_value",
    }
    for suite in suites:
        if not isinstance(suite, dict):
            raise ValueError("each suite must be an object")
        kind = suite.get("kind")
        if kind not in supported:
            raise ValueError(f"unsupported suite kind: {kind}")
        if not bool(suite.get("enabled", True)):
            continue
        if kind in {"imagination", "dependency", "skills"}:
            conditions = suite.get("conditions")
            if not isinstance(conditions, list) or not conditions:
                raise ValueError(f"{kind} suite needs conditions")
            if any("name" not in item for item in conditions):
                raise ValueError(f"every {kind} condition needs a name")


def planned_run_count(
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
            count += (
                len(seeds)
                * len(suite["conditions"])
                * int(suite.get("eval_episodes", 20))
            )
        elif kind == "dependency":
            count += (
                len(seeds)
                * len(suite["conditions"])
                * len(suite.get("lengths", (4, 6)))
                * int(suite.get("eval_episodes", 20))
            )
        elif kind == "skills":
            count += (
                len(seeds)
                * len(suite["conditions"])
                * int(suite.get("episodes", 6))
            )
        elif kind == "information_value":
            count += (
                len(seeds)
                * len(suite.get("noise_levels", (0, 8)))
                * int(suite.get("steps", 30))
            )
    return count


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("suite", "")),
            str(row.get("condition", "")),
            str(row.get("environment", "")),
            str(row.get("model", "")),
            str(row.get("action_family", "")),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        record: dict[str, Any] = {
            "suite": key[0],
            "condition": key[1],
            "environment": key[2],
            "model": key[3],
            "action_family": key[4],
            "rows": len(items),
            "seed_count": len({str(item.get("seed", "")) for item in items}),
        }
        for metric in SUMMARY_METRICS:
            values = [
                numeric
                for item in items
                if (numeric := _to_float(item.get(metric))) is not None
            ]
            if not values:
                record[f"{metric}_mean"] = ""
                record[f"{metric}_sd"] = ""
                record[f"{metric}_ci95"] = ""
                continue
            mean = fmean(values)
            deviation = stdev(values) if len(values) > 1 else 0.0
            ci95 = 1.96 * deviation / (len(values) ** 0.5) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = deviation
            record[f"{metric}_ci95"] = ci95
        summary.append(record)
    return summary


def _summary_fields() -> tuple[str, ...]:
    fields = [
        "suite",
        "condition",
        "environment",
        "model",
        "action_family",
        "rows",
        "seed_count",
    ]
    for metric in SUMMARY_METRICS:
        fields.extend(
            (
                f"{metric}_mean",
                f"{metric}_sd",
                f"{metric}_ci95",
            )
        )
    return tuple(fields)


def _format_number(value: Any) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric:.4f}"


def write_report(
    path: Path,
    config: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        f"# {config['name']} experiment report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Suite | Condition | Environment | Model | Action | Success | Steps | Prediction | Imagined nodes | Skill uses |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["suite"]),
                    str(row["condition"]),
                    str(row["environment"]),
                    str(row["model"]),
                    str(row["action_family"] or "-"),
                    _format_number(row.get("success_mean")),
                    _format_number(row.get("steps_mean")),
                    _format_number(row.get("prediction_score_mean")),
                    _format_number(row.get("imagined_nodes_mean")),
                    _format_number(row.get("skill_uses_mean")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Interpretation guardrail",
            "",
            "This report verifies execution and measurement wiring. A pilot run is not evidence of general performance. Use the larger configs and paired seed-level analysis before making research claims.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def regenerate_summary(output_dir: str | Path) -> ExperimentArtifacts:
    directory = Path(output_dir)
    episodes_csv = directory / "episodes.csv"
    config_path = directory / "resolved_config.json"
    rows = read_rows(episodes_csv)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = summarize_rows(rows)
    summary_csv = directory / "summary.csv"
    report_md = directory / "report.md"
    _write_csv(summary_csv, summary, _summary_fields())
    write_report(report_md, config, summary)
    return ExperimentArtifacts(
        directory,
        episodes_csv,
        summary_csv,
        report_md,
        config_path,
        len(rows),
    )


def run_experiment(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    suite_filter: Iterable[str] | None = None,
    seed_override: Sequence[int] | None = None,
) -> ExperimentArtifacts:
    resolved = json.loads(json.dumps(config))
    if seed_override is not None:
        resolved["seeds"] = list(seed_override)
    validate_config(resolved)
    filters = set(suite_filter or ()) or None
    destination = Path(
        output_dir
        or resolved.get("output_dir")
        or f"runs/{resolved['name']}"
    )
    if destination.exists():
        if not overwrite:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = destination.with_name(f"{destination.name}_{stamp}")
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "traces").mkdir(exist_ok=True)
    resolved_config = destination / "resolved_config.json"
    resolved_config.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    seeds = tuple(int(seed) for seed in resolved["seeds"])
    rows: list[dict[str, Any]] = []
    for suite in resolved["suites"]:
        if not bool(suite.get("enabled", True)):
            continue
        kind = str(suite["kind"])
        if filters and kind not in filters:
            continue
        if kind == "prophecy":
            rows.extend(_run_prophecy_suite(resolved["name"], suite, seeds))
        elif kind == "imagination":
            rows.extend(_run_imagination_suite(resolved["name"], suite, seeds))
        elif kind == "dependency":
            rows.extend(_run_dependency_suite(resolved["name"], suite, seeds))
        elif kind == "skills":
            rows.extend(_run_skill_suite(resolved["name"], suite, seeds))
        elif kind == "information_value":
            rows.extend(
                _run_information_value_suite(
                    resolved["name"],
                    suite,
                    seeds,
                    destination,
                )
            )

    episodes_csv = destination / "episodes.csv"
    _write_csv(episodes_csv, rows, RESULT_FIELDS)
    summary = summarize_rows(rows)
    summary_csv = destination / "summary.csv"
    _write_csv(summary_csv, summary, _summary_fields())
    report_md = destination / "report.md"
    write_report(report_md, resolved, summary)
    return ExperimentArtifacts(
        destination,
        episodes_csv,
        summary_csv,
        report_md,
        resolved_config,
        len(rows),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config
