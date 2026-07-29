from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent
from .autonomous_benchmarks import OpaqueDependencyWorld
from .experiment_runner import ExperimentArtifacts, RESULT_FIELDS
from .tabular_prophecy import TabularProphecy


def _empty_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(values)
    return row


def validate_autonomous_config(config: Mapping[str, Any]) -> None:
    if config.get("runner") != "autonomous_main":
        raise ValueError("autonomous config must set runner to autonomous_main")
    if not str(config.get("name", "")).strip():
        raise ValueError("config needs a name")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("config needs integer seeds")
    if int(config.get("train_episodes", 0)) <= 0:
        raise ValueError("train_episodes must be positive")
    if int(config.get("eval_episodes", 0)) <= 0:
        raise ValueError("eval_episodes must be positive")
    environments = config.get("environments")
    conditions = config.get("conditions")
    if not isinstance(environments, list) or not environments:
        raise ValueError("config needs environments")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("config needs conditions")
    for environment in environments:
        if int(environment.get("length", 0)) < 2:
            raise ValueError("environment length must be at least two")
    for condition in conditions:
        if not str(condition.get("name", "")).strip():
            raise ValueError("every condition needs a name")


def load_autonomous_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_autonomous_config(config)
    return config


def planned_autonomous_run_count(
    config: Mapping[str, Any],
    suite_filter: set[str] | None = None,
) -> int:
    if suite_filter and "autonomous_discovery" not in suite_filter:
        return 0
    return (
        len(config["seeds"])
        * len(config["environments"])
        * len(config["conditions"])
        * (int(config["train_episodes"]) + int(config["eval_episodes"]))
    )


def _agent_config(condition: Mapping[str, Any], length: int) -> AutonomousAgentConfig:
    return AutonomousAgentConfig(
        gamma=float(condition.get("gamma", 0.97)),
        epsilon_start=float(condition.get("epsilon_start", 0.8)),
        epsilon_end=float(condition.get("epsilon_end", 0.05)),
        epsilon_decay_episodes=int(condition.get("epsilon_decay_episodes", 1000)),
        exploration_bonus=float(condition.get("exploration_bonus", 0.3)),
        policy_learning_rate=float(condition.get("policy_learning_rate", 0.2)),
        learn_policy=bool(condition.get("learn_policy", True)),
        learn_prophecy=bool(condition.get("learn_prophecy", True)),
        random_policy=bool(condition.get("random_policy", False)),
        use_imagination=bool(condition.get("use_imagination", True)),
        imagination_depth=int(condition.get("imagination_depth", length)),
        imagination_branching_factor=int(
            condition.get("imagination_branching_factor", 2)
        ),
        imagination_beam_width=int(condition.get("imagination_beam_width", 32)),
        imagination_minimum_coverage=float(
            condition.get("imagination_minimum_coverage", 0.75)
        ),
        validated_gain_weight=float(condition.get("validated_gain_weight", 0.2)),
        repeat_penalty=float(condition.get("repeat_penalty", 0.05)),
        error_penalty=float(condition.get("error_penalty", 0.2)),
        holdout_stride=int(condition.get("holdout_stride", 5)),
        minimum_holdout_count=int(condition.get("minimum_holdout_count", 4)),
        holdout_evaluation_limit=int(
            condition.get("holdout_evaluation_limit", 32)
        ),
        validation_interval=int(condition.get("validation_interval", 8)),
        imagination_interval=int(condition.get("imagination_interval", 1)),
    )


def _run_episode(
    agent: AutonomousLearningAgent,
    *,
    length: int,
    world_seed: int,
    episode: int,
    phase: str,
    learn: bool,
) -> dict[str, Any]:
    environment = OpaqueDependencyWorld(length, seed=world_seed)
    prediction_scores: list[float] = []
    holdout_scores: list[float] = []
    holdout_gains: list[float] = []
    intrinsic_values: list[float] = []
    imagined_nodes = 0
    imagination_depth = 0
    root_values: list[float] = []
    errors = 0
    repeats = 0
    used_imagination = 0
    steps = 0
    started = time.perf_counter()
    final_return = 0.0

    while not environment.terminal:
        state = environment.snapshot()
        decision = agent.select_action(state, episode=episode, explore=learn)
        imagined_nodes += decision.imagined_nodes
        imagination_depth = max(imagination_depth, decision.imagination_depth)
        if decision.used_imagination:
            used_imagination += 1
            root_values.append(decision.root_imagined_value)
        outcome = environment.step(decision.action)
        final_return = outcome.reward
        if learn:
            metrics = agent.observe(state, decision.action, outcome)
            prediction_scores.append(metrics.prediction_score)
            holdout_scores.append(metrics.holdout_after)
            holdout_gains.append(metrics.holdout_gain)
            intrinsic_values.append(metrics.intrinsic_value)
            errors += int(metrics.error)
            repeats += int(metrics.repeated)
        steps += 1

    if learn:
        agent.finish_episode(final_return=final_return)
    else:
        agent.discard_episode()

    return {
        "phase": phase,
        "success": int(final_return > 0.0),
        "steps": steps,
        "reward": final_return,
        "errors": errors,
        "repeats": repeats,
        "prediction_score": fmean(prediction_scores) if prediction_scores else "",
        "holdout_score": fmean(holdout_scores) if holdout_scores else "",
        "holdout_gain": fmean(holdout_gains) if holdout_gains else "",
        "imagined_nodes": imagined_nodes,
        "imagination_depth": imagination_depth,
        "root_imagined_value": fmean(root_values) if root_values else "",
        "actual_return": final_return,
        "intrinsic_value": fmean(intrinsic_values) if intrinsic_values else "",
        "skill_uses": used_imagination,
        "action_family": "opaque",
        "runtime_seconds": time.perf_counter() - started,
    }


def _output_directory(
    config: Mapping[str, Any],
    output_dir: str | Path | None,
    overwrite: bool,
) -> Path:
    target = Path(output_dir or config.get("output_dir", "runs/autonomous_main"))
    if target.exists() and overwrite:
        shutil.rmtree(target)
    elif target.exists():
        target = target.with_name(
            f"{target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_autonomous_experiment(
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
    validate_autonomous_config(resolved)
    if suite_filter and "autonomous_discovery" not in set(suite_filter):
        raise ValueError("autonomous_main supports only autonomous_discovery")

    target = _output_directory(resolved, output_dir, overwrite)
    rows: list[dict[str, Any]] = []
    train_episodes = int(resolved["train_episodes"])
    eval_episodes = int(resolved["eval_episodes"])

    for seed in resolved["seeds"]:
        for environment_spec in resolved["environments"]:
            length = int(environment_spec["length"])
            environment_name = str(
                environment_spec.get("name", f"opaque_dependency_{length}")
            )
            world_seed = seed * 1009 + int(environment_spec.get("seed_offset", 0))
            for condition in resolved["conditions"]:
                condition_name = str(condition["name"])
                prophecy = TabularProphecy(name=f"online:{condition_name}")
                agent_config = _agent_config(condition, length)
                agent = AutonomousLearningAgent(
                    prophecy,
                    config=agent_config,
                    seed=seed * 7919 + length,
                )
                for episode in range(train_episodes):
                    result = _run_episode(
                        agent,
                        length=length,
                        world_seed=world_seed,
                        episode=episode,
                        phase="training",
                        learn=True,
                    )
                    rows.append(
                        _empty_row(
                            experiment=resolved["name"],
                            suite="autonomous_discovery",
                            condition=condition_name,
                            environment=environment_name,
                            model=(
                                "tabular_online"
                                if agent_config.learn_prophecy
                                else "none"
                            ),
                            seed=seed,
                            episode=episode,
                            high_level_steps=result["steps"],
                            primitive_steps=result["steps"],
                            **result,
                        )
                    )
                for eval_episode in range(eval_episodes):
                    result = _run_episode(
                        agent,
                        length=length,
                        world_seed=world_seed,
                        episode=train_episodes + eval_episode,
                        phase="evaluation",
                        learn=False,
                    )
                    rows.append(
                        _empty_row(
                            experiment=resolved["name"],
                            suite="autonomous_discovery",
                            condition=condition_name,
                            environment=environment_name,
                            model=(
                                "tabular_online"
                                if agent_config.learn_prophecy
                                else "none"
                            ),
                            seed=seed,
                            episode=eval_episode,
                            high_level_steps=result["steps"],
                            primitive_steps=result["steps"],
                            **result,
                        )
                    )

    episodes_csv = target / "episodes.csv"
    with episodes_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    resolved_path = target / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    first_config = _agent_config(
        resolved["conditions"][0], int(resolved["environments"][0]["length"])
    )
    manifest = {
        "no_oracle_pretraining": True,
        "opaque_action_names": True,
        "terminal_reward_only": True,
        "agent_config_fields": list(asdict(first_config).keys()),
    }
    (target / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ExperimentArtifacts(
        target,
        episodes_csv,
        target / "summary.csv",
        target / "report.md",
        resolved_path,
        len(rows),
    )
