from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .autonomous_agent import AutonomousAgentConfig, AutonomousLearningAgent
from .autonomous_benchmarks import OpaqueDependencyWorld
from .baseline_agents import DQNAgent, OracleAgent, TabularQLearningAgent
from .escape_reporting import serialize_agent_checkpoint
from .experiment_runner import ExperimentArtifacts, RESULT_FIELDS
from .gru_prophecy import OnlineGRUProphecy
from .metrics import expected_prediction_vector, prediction_similarity
from .progress import ProgressReporter
from .tabular_prophecy import TabularProphecy


_ALLOWED_EVALUATION_MODES = {
    "evaluation",
    "seen",
    "unseen",
    "evaluation_seen",
    "evaluation_unseen_zero_shot",
}


@dataclass(frozen=True, slots=True)
class _JobSpec:
    experiment_name: str
    seed: int
    environment: dict[str, Any]
    condition: dict[str, Any]
    train_episodes: int
    eval_episodes: int
    evaluation_modes: tuple[str, ...]
    execution: dict[str, Any]


def _empty_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in RESULT_FIELDS}
    row.update(values)
    return row


def _evaluation_modes(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("evaluation_modes", ("evaluation",))
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("evaluation_modes must be a non-empty list")
    modes = tuple(str(item) for item in raw)
    unknown = set(modes) - _ALLOWED_EVALUATION_MODES
    if unknown:
        raise ValueError(f"unknown evaluation modes: {sorted(unknown)}")
    return modes


def _execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("execution", {})
    if not isinstance(raw, Mapping):
        raise ValueError("execution must be an object")
    workers = int(raw.get("workers", 1))
    cuda_workers = int(raw.get("cuda_workers", 1))
    if workers < 0:
        raise ValueError("execution.workers must be zero or positive")
    if cuda_workers <= 0:
        raise ValueError("execution.cuda_workers must be positive")
    return {
        "workers": workers,
        "cuda_workers": cuda_workers,
        "device": str(raw.get("device", "auto")),
        "allow_cpu_fallback": bool(raw.get("allow_cpu_fallback", True)),
    }


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
        if int(environment.get("train_worlds_per_seed", 1)) <= 0:
            raise ValueError("train_worlds_per_seed must be positive")
        if int(environment.get("eval_worlds_per_seed", 1)) <= 0:
            raise ValueError("eval_worlds_per_seed must be positive")
    supported_models = {"tabular", "gru", "torch_gru"}
    supported_algorithms = {
        "aassr",
        "contextual_policy",
        "random",
        "q_learning",
        "dqn",
        "oracle",
    }
    for condition in conditions:
        if not str(condition.get("name", "")).strip():
            raise ValueError("every condition needs a name")
        model = str(condition.get("model", "tabular"))
        if model not in supported_models:
            raise ValueError(f"unsupported autonomous model: {model}")
        algorithm = str(condition.get("algorithm", "aassr"))
        if algorithm not in supported_algorithms:
            raise ValueError(f"unsupported autonomous algorithm: {algorithm}")
    progress = config.get("progress", {})
    if not isinstance(progress, Mapping):
        raise ValueError("progress must be an object")
    if int(progress.get("every_episodes", 100)) <= 0:
        raise ValueError("progress.every_episodes must be positive")
    if float(progress.get("every_seconds", 10.0)) <= 0.0:
        raise ValueError("progress.every_seconds must be positive")
    _evaluation_modes(config)
    _execution_settings(config)


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
    evaluation_rows = int(config["eval_episodes"]) * len(
        _evaluation_modes(config)
    )
    return (
        len(config["seeds"])
        * len(config["environments"])
        * len(config["conditions"])
        * (int(config["train_episodes"]) + evaluation_rows)
    )


def _agent_config(
    condition: Mapping[str, Any], length: int
) -> AutonomousAgentConfig:
    return AutonomousAgentConfig(
        gamma=float(condition.get("gamma", 0.97)),
        epsilon_start=float(condition.get("epsilon_start", 0.8)),
        epsilon_end=float(condition.get("epsilon_end", 0.05)),
        epsilon_decay_episodes=int(
            condition.get("epsilon_decay_episodes", 1000)
        ),
        exploration_bonus=float(condition.get("exploration_bonus", 0.3)),
        policy_learning_rate=float(
            condition.get("policy_learning_rate", 0.2)
        ),
        learn_policy=bool(condition.get("learn_policy", True)),
        learn_prophecy=bool(condition.get("learn_prophecy", True)),
        random_policy=bool(condition.get("random_policy", False)),
        use_imagination=bool(condition.get("use_imagination", True)),
        imagination_depth=int(condition.get("imagination_depth", length)),
        imagination_branching_factor=int(
            condition.get("imagination_branching_factor", 2)
        ),
        imagination_beam_width=int(
            condition.get("imagination_beam_width", 32)
        ),
        imagination_minimum_coverage=float(
            condition.get("imagination_minimum_coverage", 0.35)
        ),
        validated_gain_weight=float(
            condition.get("validated_gain_weight", 0.2)
        ),
        repeat_penalty=float(condition.get("repeat_penalty", 0.05)),
        error_penalty=float(condition.get("error_penalty", 0.2)),
        holdout_stride=int(condition.get("holdout_stride", 5)),
        minimum_holdout_count=int(
            condition.get("minimum_holdout_count", 4)
        ),
        holdout_evaluation_limit=int(
            condition.get("holdout_evaluation_limit", 32)
        ),
        validation_interval=int(
            condition.get("validation_interval", 8)
        ),
        imagination_interval=int(
            condition.get("imagination_interval", 1)
        ),
        imagination_aggregation=str(
            condition.get("imagination_aggregation", "max")
        ),
        effect_novelty_weight=float(
            condition.get("effect_novelty_weight", 0.0)
        ),
        extrinsic_reward_weight=float(
            condition.get("extrinsic_reward_weight", 1.0)
        ),
        use_effect_composition=bool(
            condition.get("use_effect_composition", True)
        ),
        effect_minimum_samples=int(
            condition.get("effect_minimum_samples", 2)
        ),
    )


def _make_prophecy(
    condition: Mapping[str, Any],
    *,
    length: int,
    seed: int,
    execution: Mapping[str, Any],
):
    model = str(condition.get("model", "tabular"))
    name = str(condition["name"])
    options = condition.get("model_options", {})
    if not isinstance(options, Mapping):
        raise ValueError("condition.model_options must be an object")
    if model == "tabular":
        return TabularProphecy(name=f"online:{name}"), "tabular_online"
    state_size = length + 3
    if model == "gru":
        return (
            OnlineGRUProphecy(
                state_size,
                action_feature_size=int(
                    options.get("action_feature_size", 16)
                ),
                hidden_size=int(options.get("hidden_size", 24)),
                learning_rate=float(options.get("learning_rate", 0.02)),
                replay_limit=int(options.get("replay_limit", 512)),
                seed=seed,
            ),
            "gru_online_cpu",
        )
    from .torch_gru_prophecy import TorchGRUProphecy

    device = str(condition.get("device", execution.get("device", "auto")))
    prophecy = TorchGRUProphecy(
        state_size,
        action_feature_size=int(options.get("action_feature_size", 32)),
        hidden_size=int(options.get("hidden_size", 64)),
        learning_rate=float(options.get("learning_rate", 1e-3)),
        replay_limit=int(options.get("replay_limit", 2048)),
        seed=seed,
        device=device,
        allow_cpu_fallback=bool(execution.get("allow_cpu_fallback", True)),
    )
    return prophecy, f"torch_gru_online:{prophecy.device}"


def _world_seed(
    seed: int,
    environment: Mapping[str, Any],
    *,
    mode: str,
    episode: int,
) -> int:
    explicit_key = {
        "training": "train_world_seeds",
        "evaluation": "seen_world_seeds",
        "seen": "seen_world_seeds",
        "evaluation_seen": "seen_world_seeds",
        "unseen": "unseen_world_seeds",
        "evaluation_unseen_zero_shot": "unseen_world_seeds",
    }.get(mode)
    explicit = environment.get(explicit_key, ()) if explicit_key else ()
    if isinstance(explicit, (list, tuple)) and explicit:
        return int(explicit[episode % len(explicit)])
    base_offset = int(environment.get("seed_offset", 0))
    if mode in {
        "training",
        "seen",
        "evaluation",
        "evaluation_seen",
    }:
        count = int(environment.get("train_worlds_per_seed", 1))
        return seed * 1009 + base_offset + (episode % count) * 104729
    count = int(environment.get("eval_worlds_per_seed", 1))
    eval_offset = int(environment.get("eval_seed_offset", base_offset + 10_000_000))
    return seed * 1_000_003 + eval_offset + (episode % count) * 130363


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
    imagination_opportunities = 0
    imagination_eligible = 0
    imagination_runs = 0
    imagination_changed_actions = 0
    imagination_coverages: list[float] = []
    imagination_gate_reasons: Counter[str] = Counter()
    policy_oracle_agreements = 0
    executed_oracle_agreements = 0
    imagination_corrections = 0
    imagination_harms = 0
    imagination_neutral_changes = 0
    errors = 0
    repeats = 0
    steps = 0
    started = time.perf_counter()
    final_return = 0.0
    transition_records: list[dict[str, Any]] = []

    while not environment.terminal:
        state = environment.snapshot()
        if hasattr(agent, "select_action_for_environment"):
            decision = agent.select_action_for_environment(
                environment,
                state,
                episode=episode,
                explore=learn,
            )
        else:
            decision = agent.select_action(
                state, episode=episode, explore=learn
            )
        opportunity = bool(
            getattr(decision, "imagination_opportunity", False)
        )
        eligible = bool(
            getattr(decision, "imagination_eligible", False)
        )
        changed_action = bool(
            getattr(decision, "imagination_changed_action", False)
        )
        gate_reason = str(
            getattr(decision, "imagination_gate_reason", "not_supported")
        )
        coverage = float(getattr(decision, "model_coverage", 0.0))
        policy_action_signature = str(
            getattr(
                decision,
                "policy_action_signature",
                decision.action.signature,
            )
        )
        oracle_action_signature = environment.oracle_action().signature
        policy_oracle_agreement = (
            policy_action_signature == oracle_action_signature
        )
        executed_oracle_agreement = (
            decision.action.signature == oracle_action_signature
        )
        imagination_oracle_delta = (
            int(executed_oracle_agreement)
            - int(policy_oracle_agreement)
        )
        policy_oracle_agreements += int(policy_oracle_agreement)
        executed_oracle_agreements += int(executed_oracle_agreement)
        if changed_action:
            if imagination_oracle_delta > 0:
                imagination_corrections += 1
            elif imagination_oracle_delta < 0:
                imagination_harms += 1
            else:
                imagination_neutral_changes += 1
        imagination_opportunities += int(opportunity)
        imagination_eligible += int(eligible)
        imagination_runs += int(decision.used_imagination)
        imagination_changed_actions += int(changed_action)
        imagination_gate_reasons[gate_reason] += 1
        if opportunity:
            imagination_coverages.append(coverage)
        imagined_nodes += decision.imagined_nodes
        imagination_depth = max(
            imagination_depth, decision.imagination_depth
        )
        if decision.used_imagination:
            root_values.append(decision.root_imagined_value)
        outcome = environment.step(decision.action)
        transition_records.append(
            {
                "transition_index": steps,
                "phase": phase,
                "episode": episode,
                "world_seed": world_seed,
                "before": {
                    "vector": list(state.vector),
                    "facts": sorted(state.facts),
                    "available_actions": [
                        item.signature for item in state.available_actions
                    ],
                    "goal_progress": state.goal_progress,
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
                "imagined_nodes": decision.imagined_nodes,
                "policy_action_signature": policy_action_signature,
                "imagination_opportunity": opportunity,
                "imagination_eligible": eligible,
                "imagination_gate_reason": gate_reason,
                "imagination_changed_action": changed_action,
                "model_coverage": coverage,
                "privileged_analysis": {
                    "oracle_action_signature": oracle_action_signature,
                    "policy_oracle_agreement": policy_oracle_agreement,
                    "executed_oracle_agreement": executed_oracle_agreement,
                    "imagination_oracle_delta": imagination_oracle_delta,
                },
            }
        )
        final_return = outcome.reward
        if learn:
            metrics = agent.observe(state, decision.action, outcome)
            prediction_scores.append(metrics.prediction_score)
            holdout_scores.append(metrics.holdout_after)
            holdout_gains.append(metrics.holdout_gain)
            intrinsic_values.append(metrics.intrinsic_value)
            errors += int(metrics.error)
            repeats += int(metrics.repeated)
        elif hasattr(agent, "prophecy"):
            predictions = agent.prophecy.predict(
                state, decision.action, samples=1
            )
            prediction_scores.append(
                prediction_similarity(
                    expected_prediction_vector(predictions),
                    outcome.snapshot.vector,
                )
            )
            holdout = getattr(agent, "holdout", None)
            if holdout is not None:
                holdout_scores.append(
                    holdout.score(
                        agent.prophecy,
                        limit=getattr(
                            getattr(agent, "config", object()),
                            "holdout_evaluation_limit",
                            32,
                        ),
                    )
                )
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
        "prediction_score": (
            fmean(prediction_scores) if prediction_scores else ""
        ),
        "holdout_score": fmean(holdout_scores) if holdout_scores else "",
        "holdout_gain": fmean(holdout_gains) if holdout_gains else "",
        "imagined_nodes": imagined_nodes,
        "imagination_opportunities": imagination_opportunities,
        "imagination_eligible": imagination_eligible,
        "imagination_runs": imagination_runs,
        "imagination_changed_actions": imagination_changed_actions,
        "imagination_change_rate": (
            imagination_changed_actions / imagination_runs
            if imagination_runs
            else 0.0
        ),
        "imagination_eligibility_rate": (
            imagination_eligible / imagination_opportunities
            if imagination_opportunities
            else 0.0
        ),
        "imagination_coverage_mean": (
            fmean(imagination_coverages)
            if imagination_coverages
            else 0.0
        ),
        "imagination_gate_reasons": json.dumps(
            dict(sorted(imagination_gate_reasons.items())),
            sort_keys=True,
        ),
        "policy_oracle_agreements": policy_oracle_agreements,
        "executed_oracle_agreements": executed_oracle_agreements,
        "policy_oracle_agreement_rate": (
            policy_oracle_agreements / steps if steps else 0.0
        ),
        "executed_oracle_agreement_rate": (
            executed_oracle_agreements / steps if steps else 0.0
        ),
        "imagination_corrections": imagination_corrections,
        "imagination_harms": imagination_harms,
        "imagination_neutral_changes": imagination_neutral_changes,
        "imagination_net_corrections": (
            imagination_corrections - imagination_harms
        ),
        "imagination_oracle_gain": (
            executed_oracle_agreements - policy_oracle_agreements
        ),
        "imagination_correction_rate": (
            imagination_corrections / imagination_changed_actions
            if imagination_changed_actions
            else 0.0
        ),
        "imagination_harm_rate": (
            imagination_harms / imagination_changed_actions
            if imagination_changed_actions
            else 0.0
        ),
        "imagination_depth": imagination_depth,
        "root_imagined_value": fmean(root_values) if root_values else "",
        "actual_return": final_return,
        "intrinsic_value": (
            fmean(intrinsic_values) if intrinsic_values else ""
        ),
        "skill_uses": 0,
        "action_family": "opaque",
        "runtime_seconds": time.perf_counter() - started,
        "real_transitions": steps,
        "imagined_transitions": imagined_nodes,
        "action_proposals": steps,
        "world_seed": world_seed,
        "_transitions": transition_records,
    }


def _learning_fingerprint(agent: object) -> str:
    if hasattr(agent, "learning_fingerprint"):
        return str(agent.learning_fingerprint())
    payload = serialize_agent_checkpoint(agent, episode=0)
    learned = {
        "policy": payload.get("policy", {}),
        "prophecy": payload.get("prophecy", {}),
        "holdout": payload.get("holdout", {}),
        "transition_index": payload.get("transition_index", 0),
        "decision_index": payload.get("decision_index", 0),
        "random_state": payload.get("random_state"),
        "effect_novelty_motifs": payload.get(
            "effect_novelty_motifs", ()
        ),
    }
    encoded = json.dumps(
        learned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_agent(
    condition: Mapping[str, Any],
    *,
    length: int,
    seed: int,
    execution: Mapping[str, Any],
) -> tuple[object, str, AutonomousAgentConfig | None]:
    algorithm = str(condition.get("algorithm", "aassr"))
    common = {
        "seed": seed,
        "gamma": float(condition.get("gamma", 0.97)),
        "epsilon_start": float(condition.get("epsilon_start", 0.8)),
        "epsilon_end": float(condition.get("epsilon_end", 0.05)),
        "epsilon_decay_episodes": int(
            condition.get("epsilon_decay_episodes", 1000)
        ),
    }
    if algorithm == "q_learning":
        return (
            TabularQLearningAgent(
                **common,
                learning_rate=float(
                    condition.get("policy_learning_rate", 0.2)
                ),
            ),
            "tabular_q_learning",
            None,
        )
    if algorithm == "dqn":
        options = condition.get("model_options", {})
        options = options if isinstance(options, Mapping) else {}
        agent = DQNAgent(
            length + 3,
            **common,
            action_feature_size=int(options.get("action_feature_size", 16)),
            hidden_size=int(options.get("hidden_size", 32)),
            learning_rate=float(options.get("learning_rate", 1e-3)),
            device=str(condition.get("device", execution.get("device", "auto"))),
            allow_cpu_fallback=bool(
                execution.get("allow_cpu_fallback", True)
            ),
        )
        return agent, f"dqn:{agent.device}", None
    if algorithm == "oracle":
        return OracleAgent(), "oracle_privileged", None

    prophecy, model_label = _make_prophecy(
        condition,
        length=length,
        seed=seed,
        execution=execution,
    )
    agent_config = _agent_config(condition, length)
    agent = AutonomousLearningAgent(
        prophecy,
        config=agent_config,
        seed=seed,
    )
    return agent, model_label, agent_config


def _run_job(spec: _JobSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = spec.seed
    environment_spec = spec.environment
    condition = spec.condition
    length = int(environment_spec["length"])
    environment_name = str(
        environment_spec.get("name", f"opaque_dependency_{length}")
    )
    condition_name = str(condition["name"])
    agent, model_label, agent_config = _make_agent(
        condition,
        length=length,
        seed=seed * 7919 + length,
        execution=spec.execution,
    )
    rows: list[dict[str, Any]] = []
    training_success = deque(maxlen=100)
    for episode in range(spec.train_episodes):
        result = _run_episode(
            agent,
            length=length,
            world_seed=_world_seed(
                seed, environment_spec, mode="training", episode=episode
            ),
            episode=episode,
            phase="training",
            learn=True,
        )
        training_success.append(int(result["success"]))
        rows.append(
            _empty_row(
                experiment=spec.experiment_name,
                suite="autonomous_discovery",
                condition=condition_name,
                environment=environment_name,
                model=(
                    model_label
                    if agent_config is None or agent_config.learn_prophecy
                    else "none"
                ),
                seed=seed,
                research_seed=seed,
                episode=episode,
                high_level_steps=result["steps"],
                primitive_steps=result["steps"],
                **result,
            )
        )

    evaluation_summary: dict[str, float] = {}
    for mode in spec.evaluation_modes:
        fingerprint_before = _learning_fingerprint(agent)
        successes = deque(maxlen=max(1, spec.eval_episodes))
        for eval_episode in range(spec.eval_episodes):
            result = _run_episode(
                agent,
                length=length,
                world_seed=_world_seed(
                    seed,
                    environment_spec,
                    mode=mode,
                    episode=eval_episode,
                ),
                episode=spec.train_episodes + eval_episode,
                phase=(
                    "evaluation"
                    if mode == "evaluation"
                    else (
                        mode
                        if mode.startswith("evaluation_")
                        else f"evaluation_{mode}"
                    )
                ),
                learn=False,
            )
            successes.append(int(result["success"]))
            rows.append(
                _empty_row(
                    experiment=spec.experiment_name,
                    suite="autonomous_discovery",
                    condition=condition_name,
                    environment=environment_name,
                    model=(
                        model_label
                        if agent_config is None or agent_config.learn_prophecy
                        else "none"
                    ),
                    seed=seed,
                    research_seed=seed,
                    episode=eval_episode,
                    checkpoint_fingerprint_before=fingerprint_before,
                    high_level_steps=result["steps"],
                    primitive_steps=result["steps"],
                    **result,
                )
            )
        fingerprint_after = _learning_fingerprint(agent)
        if fingerprint_before != fingerprint_after:
            raise RuntimeError(
                f"learning state mutated during frozen evaluation mode {mode}"
            )
        for row in rows[-spec.eval_episodes :]:
            row["checkpoint_fingerprint_after"] = fingerprint_after
        evaluation_summary[mode] = fmean(successes) if successes else 0.0

    context = {
        "seed": seed,
        "environment": environment_name,
        "condition": condition_name,
        "model": model_label,
        "recent_training_success": (
            fmean(training_success) if training_success else 0.0
        ),
        "evaluation": evaluation_summary,
    }
    return context, rows


def _output_directory(
    config: Mapping[str, Any],
    output_dir: str | Path | None,
    overwrite: bool,
) -> Path:
    target = Path(
        output_dir or config.get("output_dir", "runs/autonomous_main")
    )
    if target.exists() and overwrite:
        shutil.rmtree(target)
    elif target.exists():
        target = target.with_name(
            f"{target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _progress_settings(
    config: Mapping[str, Any],
    *,
    every_episodes: int | None,
    every_seconds: float | None,
) -> tuple[int, float]:
    raw = config.get("progress", {})
    settings = raw if isinstance(raw, Mapping) else {}
    item_interval = int(
        every_episodes
        if every_episodes is not None
        else settings.get("every_episodes", 100)
    )
    time_interval = float(
        every_seconds
        if every_seconds is not None
        else settings.get("every_seconds", 10.0)
    )
    if item_interval <= 0 or time_interval <= 0.0:
        raise ValueError("progress intervals must be positive")
    return item_interval, time_interval


def _uses_cuda(spec: _JobSpec) -> bool:
    algorithm = str(spec.condition.get("algorithm", "aassr"))
    if (
        algorithm != "dqn"
        and str(spec.condition.get("model", "tabular")) != "torch_gru"
    ):
        return False
    requested = str(
        spec.condition.get("device", spec.execution.get("device", "auto"))
    ).lower()
    return requested != "cpu"


def _resolved_worker_count(value: int) -> int:
    if value > 0:
        return value
    return max(1, (os.cpu_count() or 2) - 1)


def run_autonomous_experiment(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    suite_filter: Sequence[str] | None = None,
    seed_override: Sequence[int] | None = None,
    progress_every: int | None = None,
    progress_seconds: float | None = None,
    progress_console: bool = True,
) -> ExperimentArtifacts:
    resolved = dict(config)
    if seed_override is not None:
        resolved["seeds"] = list(seed_override)
    validate_autonomous_config(resolved)
    if suite_filter and "autonomous_discovery" not in set(suite_filter):
        raise ValueError(
            "autonomous_main supports only autonomous_discovery"
        )

    target = _output_directory(resolved, output_dir, overwrite)
    train_episodes = int(resolved["train_episodes"])
    eval_episodes = int(resolved["eval_episodes"])
    modes = _evaluation_modes(resolved)
    execution = _execution_settings(resolved)
    workers = _resolved_worker_count(int(execution["workers"]))
    cuda_workers = int(execution["cuda_workers"])
    total_rows = planned_autonomous_run_count(resolved)
    progress_items, progress_time = _progress_settings(
        resolved,
        every_episodes=progress_every,
        every_seconds=progress_seconds,
    )
    reporter = ProgressReporter(
        total_rows,
        target,
        every_items=progress_items,
        every_seconds=progress_time,
        console=progress_console,
    )

    resolved["execution"] = {
        **execution,
        "resolved_workers": workers,
    }
    resolved_path = target / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    first_config = _agent_config(
        resolved["conditions"][0],
        int(resolved["environments"][0]["length"]),
    )
    manifest = {
        "no_oracle_pretraining": True,
        "opaque_action_names": True,
        "terminal_reward_only": True,
        "privileged_oracle_analysis_only": True,
        "oracle_labels_agent_visible": False,
        "oracle_labels_used_for_learning": False,
        "train_test_world_separation": any(
            mode in {"unseen", "evaluation_unseen_zero_shot"}
            for mode in modes
        ),
        "evaluation_modes": list(modes),
        "parallel_process_workers": workers,
        "cuda_workers": cuda_workers,
        "requested_device": execution["device"],
        "streaming_episode_csv": True,
        "persistent_progress": True,
        "progress_files": [
            "progress.log",
            "progress.jsonl",
            "progress.json",
        ],
        "agent_config_fields": list(asdict(first_config).keys()),
    }
    (target / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    specs = [
        _JobSpec(
            str(resolved["name"]),
            int(seed),
            dict(environment),
            dict(condition),
            train_episodes,
            eval_episodes,
            modes,
            dict(execution),
        )
        for seed in resolved["seeds"]
        for environment in resolved["environments"]
        for condition in resolved["conditions"]
    ]
    cpu_specs = [spec for spec in specs if not _uses_cuda(spec)]
    cuda_specs = [spec for spec in specs if _uses_cuda(spec)]

    episodes_csv = target / "episodes.csv"
    transitions_jsonl = target / "transitions.jsonl"
    row_count = 0
    current_context: dict[str, Any] = {}
    reporter.start(
        {
            "jobs": len(specs),
            "cpu_jobs": len(cpu_specs),
            "cuda_jobs": len(cuda_specs),
            "workers": workers,
            "cuda_workers": cuda_workers,
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "evaluation_modes": ",".join(modes),
            "output": str(target),
        }
    )

    def consume(
        result: tuple[dict[str, Any], list[dict[str, Any]]],
        writer: csv.DictWriter,
        handle: Any,
        transition_handle: Any,
        job_number: int,
    ) -> None:
        nonlocal row_count, current_context
        context, rows = result
        current_context = {
            **context,
            "job": f"{job_number}/{len(specs)}",
        }
        reporter.stage("job_complete", current_context)
        for row in rows:
            for transition in row.get("_transitions", ()):
                transition_handle.write(
                    json.dumps(
                        {
                            "experiment": row.get("experiment", ""),
                            "suite": row.get("suite", ""),
                            "condition": row.get("condition", ""),
                            "environment": row.get("environment", ""),
                            "research_seed": row.get("research_seed", row.get("seed", "")),
                            **transition,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            writer.writerow(row)
            row_count += 1
            reporter.advance(
                {
                    **current_context,
                    "phase": row.get("phase", ""),
                    "episode": row.get("episode", ""),
                }
            )
            if row_count % progress_items == 0:
                handle.flush()
                transition_handle.flush()

    try:
        with (
            episodes_csv.open(
                "w", newline="", encoding="utf-8-sig"
            ) as handle,
            transitions_jsonl.open("w", encoding="utf-8") as transition_handle,
        ):
            writer = csv.DictWriter(
                handle,
                fieldnames=RESULT_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            handle.flush()

            if workers == 1 and not cuda_specs:
                for job_number, spec in enumerate(specs, start=1):
                    consume(
                        _run_job(spec),
                        writer,
                        handle,
                        transition_handle,
                        job_number,
                    )
            else:
                context = multiprocessing.get_context("spawn")
                executors: list[ProcessPoolExecutor] = []
                futures = []
                try:
                    if cpu_specs:
                        cpu_executor = ProcessPoolExecutor(
                            max_workers=workers,
                            mp_context=context,
                        )
                        executors.append(cpu_executor)
                        futures.extend(
                            cpu_executor.submit(_run_job, spec)
                            for spec in cpu_specs
                        )
                    if cuda_specs:
                        cuda_executor = ProcessPoolExecutor(
                            max_workers=cuda_workers,
                            mp_context=context,
                        )
                        executors.append(cuda_executor)
                        futures.extend(
                            cuda_executor.submit(_run_job, spec)
                            for spec in cuda_specs
                        )
                    for job_number, future in enumerate(
                        as_completed(futures), start=1
                    ):
                        consume(
                            future.result(),
                            writer,
                            handle,
                            transition_handle,
                            job_number,
                        )
                finally:
                    for executor in executors:
                        executor.shutdown(wait=True, cancel_futures=True)
            handle.flush()
            transition_handle.flush()
    except BaseException as error:
        reporter.fail(error, current_context)
        raise

    reporter.finish(
        {
            "jobs": len(specs),
            "rows": row_count,
            "output": str(target),
        }
    )
    return ExperimentArtifacts(
        target,
        episodes_csv,
        target / "summary.csv",
        target / "report.md",
        resolved_path,
        row_count,
    )
