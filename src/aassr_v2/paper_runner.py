from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent
from .autonomous_experiment import (
    _learning_fingerprint,
    _make_agent,
    _run_episode,
    run_autonomous_experiment,
)
from .creativity import (
    MultiSolutionDependencyWorld,
    novelty_against_references,
    strategy_record_from_trace,
)
from .experiment_runner import RESULT_FIELDS
from .paper_artifacts import (
    make_paper_figures,
    make_paper_tables,
    validate_paper_artifacts,
    write_paper_report,
)
from .paper_protocol import (
    DEFAULT_ADAPTATION_BUDGETS,
    PaperPaths,
    assert_frozen,
    build_manifest,
    capture_checkpoint_parts,
    checkpoint_fingerprint,
    expand_suite_conditions,
    planned_paper_run_count,
    restore_checkpoint_parts,
    sha256_json,
    utc_now,
    validate_paper_config,
    write_final_manifest,
)
from .paper_statistics import analyze_paper_results, write_csv_rows
from .paper_types import (
    BudgetLedger,
    EffectProfile,
    ExperimentPhase,
    StrategyRecord,
)
from .safe_application import (
    SafeLocalApplicationWorld,
    safe_world_manifest,
    validate_compose_safety,
)
from .tabular_prophecy import TabularProphecy
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class PaperArtifacts:
    output_dir: Path
    episodes_csv: Path
    transitions_jsonl: Path
    strategies_jsonl: Path
    manifest_json: Path
    report_md: Path
    row_count: int


def _empty_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(values)
    return row


def _condition(
    name: str, raw: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result = dict(raw or {})
    result["name"] = name
    defaults: dict[str, dict[str, Any]] = {
        "random": {
            "algorithm": "random",
            "random_policy": True,
            "learn_policy": False,
            "learn_prophecy": False,
            "use_imagination": False,
            "validated_gain_weight": 0.0,
        },
        "contextual_policy": {
            "algorithm": "contextual_policy",
            "learn_policy": True,
            "learn_prophecy": False,
            "use_imagination": False,
            "validated_gain_weight": 0.0,
        },
        "q_learning": {"algorithm": "q_learning"},
        "dqn": {"algorithm": "dqn"},
        "prophecy_no_imagination": {
            "algorithm": "aassr",
            "learn_policy": True,
            "learn_prophecy": True,
            "use_imagination": False,
        },
        "full_aassr": {
            "algorithm": "aassr",
            "learn_policy": True,
            "learn_prophecy": True,
            "use_imagination": True,
            "effect_novelty_weight": 0.1,
        },
        "novelty_search": {
            "algorithm": "aassr",
            "learn_policy": True,
            "learn_prophecy": False,
            "use_imagination": False,
            "validated_gain_weight": 0.0,
            "effect_novelty_weight": 0.5,
            "extrinsic_reward_weight": 0.0,
        },
        "oracle_upper_bound": {"algorithm": "oracle"},
        "aassr_no_novelty": {
            "algorithm": "aassr",
            "learn_policy": True,
            "learn_prophecy": True,
            "use_imagination": True,
            "validated_gain_weight": 0.0,
            "effect_novelty_weight": 0.0,
        },
        "aassr_no_imagination": {
            "algorithm": "aassr",
            "learn_policy": True,
            "learn_prophecy": True,
            "use_imagination": False,
            "effect_novelty_weight": 0.1,
        },
    }
    return {**defaults.get(name, {}), **result}


def _suite_conditions(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = expand_suite_conditions(suite)
    if raw:
        return [
            _condition(str(item["name"]), item)
            for item in raw
            if isinstance(item, Mapping)
        ]
    return [
        _condition(name)
        for name in (
            "random",
            "contextual_policy",
            "q_learning",
            "prophecy_no_imagination",
            "full_aassr",
            "oracle_upper_bound",
        )
    ]


def _worlds(config: Mapping[str, Any]) -> Mapping[str, list[int]]:
    return config["world_seeds"]


def _autonomy_environment_specs(
    config: Mapping[str, Any], suite: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw = suite.get("environments")
    if isinstance(raw, list) and raw:
        environments = [dict(item) for item in raw]
    else:
        environments = [
            {"name": f"opaque_dependency_l{length}", "length": length}
            for length in suite.get("lengths", [4, 6, 8])
        ]
    worlds = _worlds(config)
    for environment in environments:
        environment.update(
            {
                "train_world_seeds": list(worlds["train"]),
                "seen_world_seeds": list(worlds["seen"]),
                "unseen_world_seeds": list(worlds["unseen"]),
                "train_worlds_per_seed": len(worlds["train"]),
                "eval_worlds_per_seed": len(worlds["unseen"]),
            }
        )
    return environments


def _run_autonomy(
    config: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    output_dir: Path,
) -> tuple[
    Iterable[dict[str, Any]],
    Iterable[dict[str, Any]],
    list[StrategyRecord],
]:
    budgets = config["budgets"]
    kind = str(suite["kind"])
    autonomous_config = {
        "name": f"{config['name']}:{kind}",
        "runner": "autonomous_main",
        "seeds": list(config["research_seeds"]),
        "train_episodes": int(budgets["train_episodes"]),
        "eval_episodes": int(budgets["eval_episodes"]),
        "evaluation_modes": [
            "evaluation_seen",
            "evaluation_unseen_zero_shot",
        ],
        "execution": dict(config.get("execution", {})),
        "progress": {"every_episodes": 100, "every_seconds": 10.0},
        "environments": _autonomy_environment_specs(config, suite),
        "conditions": _suite_conditions(suite),
    }
    artifacts = run_autonomous_experiment(
        autonomous_config,
        output_dir=output_dir,
        overwrite=True,
        progress_console=False,
    )
    def iter_rows() -> Iterable[dict[str, Any]]:
        with artifacts.episodes_csv.open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                yield {
                    **row,
                    "suite": kind,
                    "experiment": str(config["name"]),
                }

    transitions_path = artifacts.output_dir / "transitions.jsonl"

    def iter_transitions() -> Iterable[dict[str, Any]]:
        if not transitions_path.exists():
            return
        with transitions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield {
                    **json.loads(line),
                    "suite": kind,
                    "experiment": str(config["name"]),
                }

    return iter_rows(), iter_transitions(), []


def _transition_effect_profile(
    transition: Mapping[str, Any]
) -> EffectProfile:
    before = transition.get("before", {})
    after = transition.get("after", {})
    before_vector = [float(item) for item in before.get("vector", ())]
    after_vector = [float(item) for item in after.get("vector", ())]
    distance = math.sqrt(
        sum(
            (right - left) ** 2
            for left, right in zip(
                before_vector, after_vector, strict=False
            )
        )
    )
    before_facts = set(before.get("facts", ()))
    after_facts = set(after.get("facts", ()))
    before_actions = set(before.get("available_actions", ()))
    after_actions = set(after.get("available_actions", ()))
    risk_change = (
        after_vector[-2] - before_vector[-2]
        if len(before_vector) >= 2 and len(after_vector) >= 2
        else 0.0
    )
    return EffectProfile.from_observations(
        [
            {
                "error": bool(transition.get("error", False)),
                "state_change": distance,
                "facts_added": len(after_facts - before_facts),
                "facts_removed": len(before_facts - after_facts),
                "unlocked": bool(after_actions - before_actions),
                "risk_change": risk_change,
                "goal_change": float(after.get("goal_progress", 0.0))
                - float(before.get("goal_progress", 0.0)),
                "prediction_uncertainty": 0.0,
                "information_gain": float(bool(after_facts - before_facts)),
            }
        ]
    )


def _profile_representation(
    transitions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    materialized = list(transitions)
    episode_returns: dict[tuple[Any, Any], float] = defaultdict(float)
    for transition in materialized:
        key = (transition.get("world_seed"), transition.get("episode"))
        episode_returns[key] = max(
            episode_returns[key], float(transition.get("reward", 0.0))
        )
    prototypes: dict[tuple[float, ...], list[float]] = defaultdict(list)
    for transition in materialized:
        profile = _transition_effect_profile(transition)
        vector = tuple(round(value, 4) for value in profile.vector())
        key = (transition.get("world_seed"), transition.get("episode"))
        value = episode_returns[key]
        value -= 0.5 * max(0.0, profile.mean_risk_change)
        value += 0.1 * max(0.0, profile.information_gain)
        prototypes[vector].append(value)
    return {
        "schema_version": 1,
        "feature_order": [
            "error_rate",
            "mean_state_change",
            "mean_facts_added",
            "mean_facts_removed",
            "unlock_rate",
            "mean_risk_change",
            "mean_goal_change",
            "prediction_uncertainty",
            "information_gain",
        ],
        "profiles": [
            {
                "vector": list(vector),
                "value": fmean(values),
                "observations": len(values),
            }
            for vector, values in sorted(prototypes.items())
        ],
    }


def _apply_effect_transfer(
    agent: AutonomousLearningAgent,
    transitions: Sequence[Mapping[str, Any]],
    representation: Mapping[str, Any],
) -> None:
    prototypes = [
        item
        for item in representation.get("profiles", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("vector"), list)
    ]
    if not prototypes:
        return
    for transition in transitions:
        profile = _transition_effect_profile(transition)
        vector = profile.vector()
        nearest = min(
            prototypes,
            key=lambda item: math.sqrt(
                sum(
                    (left - float(right)) ** 2
                    for left, right in zip(
                        vector, item["vector"], strict=True
                    )
                )
            ),
        )
        before = transition.get("before", {})
        action_signature = str(transition.get("action", ""))
        if "|_|_|_" not in action_signature:
            continue
        action = Action(action_signature.split("|", 1)[0])
        state = StateSnapshot(
            vector=tuple(float(item) for item in before.get("vector", ())),
            facts=frozenset(str(item) for item in before.get("facts", ())),
            available_actions=tuple(
                Action(str(item).split("|", 1)[0])
                for item in before.get("available_actions", ())
                if "|_|_|_" in str(item)
            ),
            goal_progress=float(before.get("goal_progress", 0.0)),
        )
        agent.policy.observe_return(
            state, action, float(nearest.get("value", 0.0))
        )


def _transfer_config(name: str) -> AutonomousAgentConfig:
    if name == "from_scratch_contextual_policy":
        return AutonomousAgentConfig(
            learn_prophecy=False,
            use_imagination=False,
            validated_gain_weight=0.0,
        )
    return AutonomousAgentConfig(
        minimum_holdout_count=2,
        validation_interval=4,
        imagination_interval=2,
    )


def _run_transfer(
    config: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[StrategyRecord]]:
    budgets = config["budgets"]
    length = int(suite.get("length", 6))
    train_episodes = int(budgets["train_episodes"])
    eval_episodes = int(budgets["eval_episodes"])
    adaptation_budgets = [
        int(item)
        for item in budgets.get(
            "adaptation_episodes", DEFAULT_ADAPTATION_BUDGETS
        )
    ]
    condition_names = [
        str(item["name"]) for item in suite.get("conditions", ())
    ] or [
        "from_scratch_contextual_policy",
        "from_scratch_full_aassr",
        "policy_reset_prophecy_retained",
        "policy_reset_effect_retained",
        "full_transfer",
    ]
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for research_seed in config["research_seeds"]:
        base_agent = AutonomousLearningAgent(
            TabularProphecy(name="transfer_pretraining"),
            config=_transfer_config("full_transfer"),
            seed=int(research_seed) * 7919 + length,
        )
        base_transitions: list[dict[str, Any]] = []
        for episode in range(train_episodes):
            world_seed = int(
                config["world_seeds"]["train"][
                    episode % len(config["world_seeds"]["train"])
                ]
            )
            result = _run_episode(
                base_agent,
                length=length,
                world_seed=world_seed,
                episode=episode,
                phase=ExperimentPhase.TRAINING.value,
                learn=True,
            )
            base_transitions.extend(result.pop("_transitions"))
            rows.append(
                _empty_row(
                    experiment=config["name"],
                    suite="transfer",
                    condition="transfer_pretraining",
                    environment=f"opaque_dependency_l{length}",
                    model="tabular_online",
                    seed=research_seed,
                    research_seed=research_seed,
                    episode=episode,
                    **result,
                )
            )
        effect_representation = _profile_representation(base_transitions)
        base_parts = capture_checkpoint_parts(
            base_agent,
            episode=train_episodes,
            effect_representation=effect_representation,
        )
        transitions.extend(
            {
                **item,
                "suite": "transfer",
                "condition": "transfer_pretraining",
                "research_seed": research_seed,
            }
            for item in base_transitions
        )

        for condition_name in condition_names:
            retain_policy = condition_name == "full_transfer"
            retain_prophecy = condition_name in {
                "policy_reset_prophecy_retained",
                "full_transfer",
            }
            retain_holdout = condition_name == "full_transfer"
            retain_effect = condition_name in {
                "policy_reset_effect_retained",
                "full_transfer",
            }
            for world_seed in config["world_seeds"]["unseen"]:
                for adaptation_budget in adaptation_budgets:
                    agent = AutonomousLearningAgent(
                        TabularProphecy(name=f"transfer:{condition_name}"),
                        config=_transfer_config(condition_name),
                        seed=(
                            int(research_seed) * 1_000_003
                            + int(world_seed)
                        ),
                    )
                    retained_representation: dict[str, Any] = {}
                    if any(
                        (
                            retain_policy,
                            retain_prophecy,
                            retain_holdout,
                            retain_effect,
                        )
                    ):
                        retained_representation = restore_checkpoint_parts(
                            agent,
                            base_parts.selected(
                                policy=retain_policy,
                                prophecy=retain_prophecy,
                                holdout=retain_holdout,
                                effect_representation=retain_effect,
                            ),
                            retain_policy=retain_policy,
                            retain_prophecy=retain_prophecy,
                            retain_holdout=retain_holdout,
                            retain_effect_representation=retain_effect,
                        )
                    branch_start = checkpoint_fingerprint(
                        agent,
                        effect_representation=retained_representation,
                    )
                    adaptation_transitions: list[dict[str, Any]] = []
                    for episode in range(adaptation_budget):
                        result = _run_episode(
                            agent,
                            length=length,
                            world_seed=int(world_seed),
                            episode=train_episodes + episode,
                            phase=ExperimentPhase.ADAPTATION.value,
                            learn=True,
                        )
                        episode_transitions = result.pop("_transitions")
                        adaptation_transitions.extend(episode_transitions)
                        if retain_effect:
                            _apply_effect_transfer(
                                agent,
                                episode_transitions,
                                retained_representation,
                            )
                        rows.append(
                            _empty_row(
                                experiment=config["name"],
                                suite="transfer",
                                condition=condition_name,
                                environment=f"opaque_dependency_l{length}",
                                model="tabular_online",
                                seed=research_seed,
                                research_seed=research_seed,
                                episode=episode,
                                adaptation_budget=adaptation_budget,
                                branch_start_fingerprint=branch_start,
                                checkpoint_fingerprint_before=branch_start,
                                **result,
                            )
                        )
                    if adaptation_transitions:
                        retained_representation = {
                            **retained_representation,
                            "adaptation_profile": _profile_representation(
                                adaptation_transitions
                            ),
                        }
                    evaluation_before = checkpoint_fingerprint(
                        agent,
                        effect_representation=retained_representation,
                    )
                    evaluation_rows: list[dict[str, Any]] = []
                    evaluation_transitions: list[dict[str, Any]] = []
                    for episode in range(eval_episodes):
                        result = _run_episode(
                            agent,
                            length=length,
                            world_seed=int(world_seed),
                            episode=train_episodes
                            + adaptation_budget
                            + episode,
                            phase=ExperimentPhase.EVALUATION_UNSEEN_ADAPTATION.value,
                            learn=False,
                        )
                        evaluation_transitions.extend(
                            result.pop("_transitions")
                        )
                        evaluation_rows.append(
                            _empty_row(
                                experiment=config["name"],
                                suite="transfer",
                                condition=condition_name,
                                environment=f"opaque_dependency_l{length}",
                                model="tabular_online",
                                seed=research_seed,
                                research_seed=research_seed,
                                episode=episode,
                                adaptation_budget=adaptation_budget,
                                branch_start_fingerprint=branch_start,
                                checkpoint_fingerprint_before=evaluation_before,
                                **result,
                            )
                        )
                    evaluation_after = checkpoint_fingerprint(
                        agent,
                        effect_representation=retained_representation,
                    )
                    assert_frozen(
                        evaluation_before,
                        evaluation_after,
                        ExperimentPhase.EVALUATION_UNSEEN_ADAPTATION,
                    )
                    for row in evaluation_rows:
                        row["checkpoint_fingerprint_after"] = evaluation_after
                    rows.extend(evaluation_rows)
                    transitions.extend(
                        {
                            **item,
                            "suite": "transfer",
                            "condition": condition_name,
                            "research_seed": research_seed,
                            "adaptation_budget": adaptation_budget,
                        }
                        for item in (
                            adaptation_transitions + evaluation_transitions
                        )
                    )
    return rows, transitions, []


def _run_creative_episode(
    agent: object,
    *,
    research_seed: int,
    world_seed: int,
    episode: int,
    phase: ExperimentPhase,
    learn: bool,
    condition_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], StrategyRecord]:
    world = MultiSolutionDependencyWorld(
        seed=world_seed, variant=episode % 7
    )
    events: list[Mapping[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    errors = 0
    repeats = 0
    imagined = 0
    risk_entries = 0
    started = time.perf_counter()
    while not world.terminal and len(transitions) < 16:
        before = world.snapshot()
        decision = agent.select_action(
            before, episode=episode, explore=learn
        )
        outcome = world.step(decision.action)
        events.extend(outcome.effect_events)
        imagined += int(decision.imagined_nodes)
        risk_entries += int(outcome.risk_delta > 0.0)
        if learn:
            metrics = agent.observe(before, decision.action, outcome)
            errors += int(metrics.error)
            repeats += int(metrics.repeated)
        transitions.append(
            {
                "transition_index": len(transitions),
                "phase": phase.value,
                "episode": episode,
                "world_seed": world_seed,
                "before": {
                    "vector": list(before.vector),
                    "facts": sorted(before.facts),
                    "available_actions": [
                        item.signature for item in before.available_actions
                    ],
                    "goal_progress": before.goal_progress,
                },
                "action": decision.action.signature,
                "after": {
                    "vector": list(outcome.snapshot.vector),
                    "facts": sorted(outcome.snapshot.facts),
                    "available_actions": [
                        item.signature
                        for item in outcome.snapshot.available_actions
                    ],
                    "goal_progress": outcome.snapshot.goal_progress,
                },
                "reward": outcome.reward,
                "error": outcome.error,
                "learning_enabled": learn,
                "effect_events": list(outcome.effect_events),
            }
        )
    final_return = 1.0 if world.terminal else 0.0
    if learn:
        agent.finish_episode(final_return=final_return)
    else:
        agent.discard_episode()
    strategy_id = (
        f"{condition_name}_{research_seed}_{world_seed}_{episode}"
    )
    source_kind = (
        "aassr" if condition_name == "full_aassr" else "baseline"
    )
    record = strategy_record_from_trace(
        strategy_id=strategy_id,
        source_kind=source_kind,
        research_seed=research_seed,
        world_seed=world_seed,
        success=world.terminal,
        primitive_steps=len(transitions),
        errors=errors,
        resources_used=world.analysis_resource_total,
        risk_entries=risk_entries,
        events=events,
        solution_family=world.analysis_solution_family,
        trace=transitions,
    )
    row = _empty_row(
        experiment="",
        suite="creativity",
        condition=condition_name,
        environment="multi_solution_dependency",
        model="online",
        seed=research_seed,
        research_seed=research_seed,
        world_seed=world_seed,
        episode=episode,
        phase=phase.value,
        success=int(world.terminal),
        steps=len(transitions),
        high_level_steps=len(transitions),
        primitive_steps=len(transitions),
        reward=final_return,
        actual_return=final_return,
        errors=errors,
        repeats=repeats,
        imagined_nodes=imagined,
        imagined_transitions=imagined,
        real_transitions=len(transitions),
        action_proposals=len(transitions),
        runtime_seconds=time.perf_counter() - started,
        action_family="causal_effect_graph",
        solution_family=world.analysis_solution_family,
        strategy_id=strategy_id,
        novelty_score=record.novelty_score,
    )
    return row, transitions, record


def _run_creativity(
    config: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[StrategyRecord]]:
    episodes = int(
        suite.get("episodes", config["budgets"]["train_episodes"])
    )
    conditions = _suite_conditions(suite)
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    strategies: list[StrategyRecord] = []
    worlds = (
        list(config["world_seeds"]["train"])
        + list(config["world_seeds"]["seen"])
    )
    execution = config.get("execution", {})
    for research_seed in config["research_seeds"]:
        for condition in conditions:
            name = str(condition["name"])
            agent, _, _ = _make_agent(
                condition,
                length=9,
                seed=int(research_seed) * 7919 + 9,
                execution=execution if isinstance(execution, Mapping) else {},
            )
            for episode in range(episodes):
                world_seed = int(worlds[episode % len(worlds)])
                row, episode_transitions, record = _run_creative_episode(
                    agent,
                    research_seed=int(research_seed),
                    world_seed=world_seed,
                    episode=episode,
                    phase=ExperimentPhase.TRAINING,
                    learn=True,
                    condition_name=name,
                )
                row["experiment"] = config["name"]
                rows.append(row)
                strategies.append(record)
                transitions.extend(
                    {
                        **item,
                        "suite": "creativity",
                        "condition": name,
                        "research_seed": research_seed,
                    }
                    for item in episode_transitions
                )
            frozen_before = _learning_fingerprint(agent)
            reuse_successes = []
            for offset in range(3):
                row, episode_transitions, _ = _run_creative_episode(
                    agent,
                    research_seed=int(research_seed),
                    world_seed=int(config["world_seeds"]["unseen"][offset % len(config["world_seeds"]["unseen"])]),
                    episode=episodes + offset,
                    phase=ExperimentPhase.EVALUATION_UNSEEN_ADAPTATION,
                    learn=False,
                    condition_name=name,
                )
                row["experiment"] = config["name"]
                row["checkpoint_fingerprint_before"] = frozen_before
                rows.append(row)
                reuse_successes.append(float(row["success"]))
                transitions.extend(
                    {
                        **item,
                        "suite": "creativity",
                        "condition": name,
                        "research_seed": research_seed,
                    }
                    for item in episode_transitions
                )
            frozen_after = _learning_fingerprint(agent)
            if frozen_before != frozen_after:
                raise RuntimeError("creativity reuse evaluation mutated agent")
            for row in rows[-3:]:
                row["checkpoint_fingerprint_after"] = frozen_after
            reuse_rate = fmean(reuse_successes)
            start = len(strategies) - episodes
            for index in range(start, len(strategies)):
                strategies[index] = replace(
                    strategies[index],
                    reusable_success_rate=reuse_rate,
                )

    reference_items = [
        item.graph for item in strategies if item.source_kind != "aassr"
    ] + [item.graph for item in _human_reference_records(config)]
    reference_counts: dict[Any, int] = defaultdict(int)
    for graph in reference_items:
        reference_counts[graph] += 1
    reference_graphs = tuple(dict.fromkeys(reference_items))
    novelty_cache: dict[tuple[Any, bool], dict[str, float]] = {}
    rescored: list[StrategyRecord] = []
    for item in strategies:
        exclude_only_self = bool(
            item.source_kind != "aassr"
            and reference_counts[item.graph] == 1
        )
        cache_key = (item.graph, exclude_only_self)
        novelty = novelty_cache.get(cache_key)
        if novelty is None:
            novelty = novelty_against_references(
                item.graph,
                (
                    tuple(
                        graph
                        for graph in reference_graphs
                        if graph != item.graph
                    )
                    if exclude_only_self
                    else reference_graphs
                ),
            )
            novelty_cache[cache_key] = novelty
        rescored.append(
            replace(
                item,
                novelty_score=novelty["aggregate"],
                novelty_components={
                    key: value
                    for key, value in novelty.items()
                    if key != "aggregate"
                },
            )
        )
    strategies = rescored
    scores = {item.strategy_id: item.novelty_score for item in strategies}
    for row in rows:
        if row.get("strategy_id") in scores:
            row["novelty_score"] = scores[str(row["strategy_id"])]
    return rows, transitions, strategies


def _run_safe_application(
    config: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[StrategyRecord]]:
    settings = config["safe_application"]
    compose_path = Path(str(settings["compose_file"]))
    if not compose_path.is_absolute():
        compose_path = Path.cwd() / compose_path
    validate_compose_safety(compose_path)
    episodes = int(suite.get("episodes", config["budgets"]["eval_episodes"]))
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for research_seed in config["research_seeds"]:
        for episode in range(episodes):
            world_seed = int(
                config["world_seeds"]["unseen"][
                    episode % len(config["world_seeds"]["unseen"])
                ]
            )
            world = SafeLocalApplicationWorld(
                seed=world_seed,
                allowed_hosts=settings["allowed_hosts"],
            )
            step_records = []
            route = (int(research_seed) + episode) % 3
            route_steps = (
                ("observe_local_services", "request_local_flag"),
                ("inspect_local_config", "use_local_configuration"),
                (
                    "read_local_note",
                    "inspect_local_config",
                    "combine_local_evidence",
                ),
            )[route]
            while not world.terminal and len(step_records) < 8:
                before = world.snapshot()
                desired = route_steps[
                    min(len(step_records), len(route_steps) - 1)
                ]
                action = next(
                    candidate
                    for candidate in before.available_actions
                    if candidate.verb_name == desired
                )
                outcome = world.step(action)
                step_records.append(
                    {
                        "transition_index": len(step_records),
                        "phase": ExperimentPhase.EVALUATION_UNSEEN_ZERO_SHOT.value,
                        "episode": episode,
                        "world_seed": world_seed,
                        "action": action.signature,
                        "before": {
                            "vector": list(before.vector),
                            "facts": sorted(before.facts),
                        },
                        "after": {
                            "vector": list(outcome.snapshot.vector),
                            "facts": sorted(outcome.snapshot.facts),
                        },
                        "reward": outcome.reward,
                        "error": outcome.error,
                        "effect_events": list(outcome.effect_events),
                        "learning_enabled": False,
                    }
                )
            transitions.extend(
                {
                    **item,
                    "suite": "safe_application",
                    "condition": "safe_rule_agent",
                    "research_seed": research_seed,
                }
                for item in step_records
            )
            rows.append(
                _empty_row(
                    experiment=config["name"],
                    suite="safe_application",
                    condition="safe_rule_agent",
                    environment="docker_local_assessment",
                    model="deterministic",
                    seed=research_seed,
                    research_seed=research_seed,
                    world_seed=world_seed,
                    episode=episode,
                    phase=ExperimentPhase.EVALUATION_UNSEEN_ZERO_SHOT.value,
                    success=int(world.terminal),
                    steps=len(step_records),
                    high_level_steps=len(step_records),
                    primitive_steps=len(step_records),
                    reward=float(world.terminal),
                    actual_return=float(world.terminal),
                    real_transitions=len(step_records),
                    imagined_transitions=0,
                    action_proposals=len(step_records),
                    solution_family=world.analysis_solution_family,
                    action_family="safe_local",
                )
            )
    manifest_value = str(settings.get("world_manifest", "")).strip()
    if manifest_value:
        manifest_path = Path(manifest_value)
        sample = SafeLocalApplicationWorld(
            seed=int(config["world_seeds"]["unseen"][0]),
            allowed_hosts=settings["allowed_hosts"],
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                safe_world_manifest(sample), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    return rows, transitions, []


def _require_frozen_creativity_rules(config: Mapping[str, Any]) -> None:
    if config.get("study_stage") != "final":
        return
    if "creativity" not in {
        str(item["kind"]) for item in config.get("suites", ())
    }:
        return
    source = Path(str(config["frozen_creativity_rules"]))
    if not source.is_absolute():
        source = Path.cwd() / source
    if not source.exists():
        raise FileNotFoundError(
            f"frozen creativity rules do not exist: {source}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("creativity rules file is not marked frozen")
    if not all(
        key in payload
        for key in ("novelty_threshold", "utility_criteria", "frozen_at_utc")
    ):
        raise ValueError("frozen creativity rules file is incomplete")
    threshold = float(payload["novelty_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen novelty threshold must be in [0, 1]")
    utility = payload["utility_criteria"]
    if not isinstance(utility, Mapping) or not bool(
        utility.get("valid", False)
    ):
        raise ValueError("frozen utility criteria are invalid")
    if not str(payload.get("reviewer", "")).strip():
        raise ValueError("frozen creativity rules need a reviewer")


def _require_acceptance_gate_manifest(config: Mapping[str, Any]) -> None:
    if config.get("study_stage") != "final":
        return
    source = Path(str(config["acceptance_gate_manifest"]))
    if not source.is_absolute():
        source = Path.cwd() / source
    if not source.exists():
        raise FileNotFoundError(
            f"Final acceptance-gate manifest does not exist: {source}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "accepted" or not all(
        bool(payload.get(name, False)) for name in ("p0", "p1", "p2", "p3")
    ):
        raise ValueError("Final acceptance-gate manifest is not accepted")
    if not str(payload.get("frozen_at_utc", "")).strip():
        raise ValueError("Final acceptance-gate manifest is not frozen")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Final acceptance-gate evidence is missing")
    for gate in ("p0", "p1", "p2", "p3"):
        item = evidence.get(gate)
        if not isinstance(item, Mapping):
            raise ValueError(f"Final gate {gate} has no evidence")
        root = Path(str(item.get("path", "")))
        if not root.is_absolute():
            root = Path.cwd() / root
        manifest = root / "manifests" / "protocol_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Final gate {gate} evidence is unavailable: {manifest}"
            )
        actual = __import__("hashlib").sha256(
            manifest.read_bytes()
        ).hexdigest()
        if actual != str(item.get("manifest_sha256", "")):
            raise ValueError(
                f"Final gate {gate} evidence manifest changed"
            )
        gate_issues = validate_paper_artifacts(root)
        if gate_issues:
            raise ValueError(
                f"Final gate {gate} evidence is invalid: "
                + "; ".join(gate_issues)
            )


def _write_creativity_threshold_candidate(
    paths: PaperPaths, strategies: Sequence[Mapping[str, Any]]
) -> Path | None:
    baseline_records = [
        StrategyRecord.from_dict(item)
        for item in strategies
        if item.get("source_kind") != "aassr"
    ]
    unique_graphs: dict[str, Any] = {}
    for record in baseline_records:
        unique_graphs.setdefault(
            sha256_json(record.graph.to_dict()), record.graph
        )
    graphs = list(unique_graphs.values())
    values = sorted(
        novelty_against_references(
            graph,
            [
                other
                for other in graphs
                if other is not graph
            ],
        )["aggregate"]
        for graph in graphs
        if len(graphs) > 1
    )
    if not values:
        return None
    index = min(len(values) - 1, int(0.75 * len(values)))
    path = paths.manifests / "creativity_threshold_candidate.json"
    path.write_text(
        json.dumps(
            {
                "status": "candidate",
                "novelty_threshold": values[index],
                "utility_criteria": {
                    "valid": True,
                    "minimum_reusable_success_rate": 0.5,
                    "must_not_be_dominated_on_all_utility_metrics": True,
                },
                "derived_from": "pilot baseline strategy distances",
                "baseline_strategy_count": len(baseline_records),
                "unique_reference_graph_count": len(graphs),
                "distance_count": len(values),
                "generated_at_utc": utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _human_reference_records(
    config: Mapping[str, Any],
) -> list[StrategyRecord]:
    human = config.get("human_study", {})
    if not isinstance(human, Mapping) or not human.get("merge_enabled", False):
        return []
    dataset_dir = Path(str(human["dataset_dir"]))
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    path = dataset_dir / "human_paths.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"human reference paths not found: {path}")
    return [
        StrategyRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _merge_human_dataset(
    config: Mapping[str, Any], paths: PaperPaths
) -> None:
    human = config.get("human_study", {})
    if not isinstance(human, Mapping) or not human.get("merge_enabled", False):
        return
    source = Path(str(human["dataset_dir"]))
    if not source.is_absolute():
        source = Path.cwd() / source
    metadata_path = source / "human_dataset.json"
    paths_path = source / "human_paths.jsonl"
    ratings_path = source / "human_ratings.csv"
    for path in (metadata_path, paths_path, ratings_path):
        if not path.exists():
            raise FileNotFoundError(f"human dataset artifact missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("dataset_version", "")) != str(
        human["dataset_version"]
    ):
        raise ValueError("human dataset version does not match config")
    if str(metadata.get("approval_id", "")).strip() != str(
        human["approval_id"]
    ).strip():
        raise ValueError("human dataset approval ID does not match config")
    with ratings_path.open(newline="", encoding="utf-8-sig") as handle:
        evaluator_count = len(
            {
                str(row["evaluator_id"])
                for row in csv.DictReader(handle)
                if row.get("evaluator_id")
            }
        )
    if evaluator_count < int(human.get("minimum_raters", 2)):
        raise ValueError("human dataset has fewer raters than required")
    shutil.copy2(paths_path, paths.raw / "human_paths.jsonl")
    shutil.copy2(ratings_path, paths.raw / "human_ratings.csv")
    shutil.copy2(metadata_path, paths.manifests / "human_dataset.json")


def _copy_protocol_locks(
    config: Mapping[str, Any], paths: PaperPaths
) -> None:
    if config.get("study_stage") != "final":
        return
    mapping = {
        "acceptance_gate_manifest": "acceptance_gate_manifest.json",
        "frozen_creativity_rules": "frozen_creativity_rules.json",
    }
    for field, filename in mapping.items():
        value = str(config.get(field, "")).strip()
        if not value:
            continue
        source = Path(value)
        if not source.is_absolute():
            source = Path.cwd() / source
        shutil.copy2(source, paths.manifests / filename)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


def _enforce_episode_budgets(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    limit = int(config["budgets"]["real_transitions_per_episode"])
    for index, row in enumerate(rows):
        actual = row.get("real_transitions", row.get("steps", 0))
        if actual in (None, ""):
            actual = 0
        ledger = BudgetLedger(
            limit,
            action_proposal_limit=limit,
            wall_clock_limit_seconds=(
                float(config["budgets"]["wall_clock_limit_seconds"])
                if config["budgets"].get("wall_clock_limit_seconds")
                is not None
                else None
            ),
        )
        try:
            ledger.consume_real(int(float(actual)))
            proposals = row.get("action_proposals", actual)
            ledger.record_proposals(int(float(proposals or 0)))
            imagined = row.get("imagined_transitions", 0)
            ledger.record_imagined(int(float(imagined or 0)))
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"episode row {index} violates the declared budget"
            ) from error


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_suite_cache_rows(
    paths: PaperPaths,
    suites: Sequence[Mapping[str, Any]],
    cache_index: int,
) -> Iterable[dict[str, Any]]:
    for suite in suites:
        cache_path = _suite_cache_paths(
            paths, str(suite["kind"])
        )[cache_index]
        yield from _iter_jsonl(cache_path)


def _write_episode_csv_from_caches(
    paths: PaperPaths,
    suites: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[Path, int]:
    fields: list[str] = []
    for row in _iter_suite_cache_rows(paths, suites, 0):
        for key in row:
            if key not in fields:
                fields.append(key)
    destination = paths.raw / "episodes.csv"
    row_count = 0
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fields:
            return destination, 0
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        for row in _iter_suite_cache_rows(paths, suites, 0):
            _enforce_episode_budgets((row,), config)
            writer.writerow(row)
            row_count += 1
    return destination, row_count


def _concatenate_suite_caches(
    destination: Path,
    paths: PaperPaths,
    suites: Sequence[Mapping[str, Any]],
    cache_index: int,
) -> Path:
    with destination.open("wb") as output:
        for suite in suites:
            source = _suite_cache_paths(
                paths, str(suite["kind"])
            )[cache_index]
            if not source.exists():
                continue
            with source.open("rb") as input_handle:
                shutil.copyfileobj(
                    input_handle, output, length=16 * 1024 * 1024
                )
    return destination


def _suite_cache_paths(
    paths: PaperPaths, kind: str
) -> tuple[Path, Path, Path]:
    cache = paths.raw / "suite_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return (
        cache / f"{kind}_episodes.jsonl",
        cache / f"{kind}_transitions.jsonl",
        cache / f"{kind}_strategies.jsonl",
    )


def run_paper_suite(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> PaperArtifacts:
    validate_paper_config(config)
    _require_acceptance_gate_manifest(config)
    _require_frozen_creativity_rules(config)
    resolved = json.loads(json.dumps(config))
    target = Path(
        output_dir
        or resolved.get(
            "output_dir",
            f"paper_results/{resolved['protocol_version']}",
        )
    )
    if target.exists() and overwrite:
        shutil.rmtree(target)
    elif target.exists() and not resume:
        raise FileExistsError(
            f"paper output already exists: {target}; use resume or overwrite"
        )
    paths = PaperPaths.create(target)
    config_hash = sha256_json(resolved)
    state_path = paths.manifests / "run_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if resume and state_path.exists()
        else {
            "config_sha256": config_hash,
            "completed_suites": [],
            "failed_runs": [],
            "started_at_utc": utc_now(),
        }
    )
    if state["config_sha256"] != config_hash:
        raise ValueError("resume config does not match original config SHA256")
    (paths.manifests / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    completed = set(state.get("completed_suites", ()))
    failed_runs = list(state.get("failed_runs", ()))
    handlers = {
        "autonomy": lambda suite: _run_autonomy(
            resolved,
            suite,
            output_dir=paths.raw / "suite_runs" / "autonomy",
        ),
        "ablation": lambda suite: _run_autonomy(
            resolved,
            suite,
            output_dir=paths.raw / "suite_runs" / "ablation",
        ),
        "transfer": lambda suite: _run_transfer(resolved, suite),
        "creativity": lambda suite: _run_creativity(resolved, suite),
        "safe_application": lambda suite: _run_safe_application(
            resolved, suite
        ),
    }
    try:
        for suite in resolved["suites"]:
            kind = str(suite["kind"])
            if kind in completed:
                continue
            try:
                rows, transitions, strategies = handlers[kind](suite)
                episode_cache, transition_cache, strategy_cache = (
                    _suite_cache_paths(paths, kind)
                )
                _write_jsonl(episode_cache, rows)
                _write_jsonl(transition_cache, transitions)
                _write_jsonl(
                    strategy_cache,
                    (item.to_dict() for item in strategies),
                )
                del rows, transitions, strategies
                completed.add(kind)
                state["completed_suites"] = sorted(completed)
                state["last_completed_at_utc"] = utc_now()
                state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except BaseException as error:
                failure = {
                    "suite": kind,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "failed_at_utc": utc_now(),
                }
                failed_runs.append(failure)
                state["failed_runs"] = failed_runs
                state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise
    except BaseException:
        manifest = build_manifest(
            resolved,
            started_at_utc=str(state["started_at_utc"]),
            completed_at_utc=utc_now(),
            failed_runs=failed_runs,
            workdir=Path.cwd(),
        )
        write_final_manifest(
            paths.manifests / "protocol_manifest.json", manifest
        )
        raise

    episodes_csv, row_count = _write_episode_csv_from_caches(
        paths, resolved["suites"], resolved
    )
    transitions_jsonl = _concatenate_suite_caches(
        paths.raw / "transitions.jsonl",
        paths,
        resolved["suites"],
        1,
    )
    strategies_jsonl = _concatenate_suite_caches(
        paths.raw / "strategies.jsonl",
        paths,
        resolved["suites"],
        2,
    )
    _merge_human_dataset(resolved, paths)
    _copy_protocol_locks(resolved, paths)
    if any(
        str(item.get("kind", "")) == "safe_application"
        for item in resolved.get("suites", ())
    ):
        safe_settings = resolved["safe_application"]
        sample = SafeLocalApplicationWorld(
            seed=int(resolved["world_seeds"]["unseen"][0]),
            allowed_hosts=safe_settings["allowed_hosts"],
        )
        compose_source = Path(str(safe_settings["compose_file"]))
        if not compose_source.is_absolute():
            compose_source = Path.cwd() / compose_source
        safe_manifest = {
            **safe_world_manifest(sample),
            "compose_sha256": __import__("hashlib").sha256(
                compose_source.read_bytes()
            ).hexdigest(),
            "compose_file": str(compose_source.resolve()),
            "docker_execution_opt_in": True,
        }
        (paths.manifests / "safe_application_world.json").write_text(
            json.dumps(
                safe_manifest, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    if (
        resolved["study_stage"] == "pilot"
        and strategies_jsonl.stat().st_size > 0
    ):
        _write_creativity_threshold_candidate(
            paths, list(_iter_jsonl(strategies_jsonl))
        )
    manifest = build_manifest(
        resolved,
        started_at_utc=str(state["started_at_utc"]),
        completed_at_utc=utc_now(),
        failed_runs=failed_runs,
        workdir=Path.cwd(),
    )
    manifest_json, _ = write_final_manifest(
        paths.manifests / "protocol_manifest.json", manifest
    )
    analyze_paper_results(paths.root)
    make_paper_tables(paths.root)
    make_paper_figures(paths.root)
    report = write_paper_report(paths.root)
    human = resolved.get("human_study", {})
    require_human = bool(
        isinstance(human, Mapping) and human.get("merge_enabled", False)
    )
    issues = validate_paper_artifacts(
        paths.root, require_human_merge=require_human
    )
    if issues:
        raise RuntimeError(
            "paper artifact validation failed: " + "; ".join(issues)
        )
    state["completed_at_utc"] = utc_now()
    state["row_count"] = row_count
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return PaperArtifacts(
        paths.root,
        episodes_csv,
        transitions_jsonl,
        strategies_jsonl,
        manifest_json,
        report,
        row_count,
    )
