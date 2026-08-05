from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from . import baseline_efficiency_benchmark as benchmark
from .imagination_v2 import make_imagination_v2_agent


FINAL_COMPLEXITY_CONDITIONS: tuple[str, ...] = (
    "dqn",
    "legacy_aassr",
    "neural_policy_only",
    "imagination_v2",
)
COMPLEXITY_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class MapComplexityProfile:
    map_seed: int
    oracle_shortest_steps: int
    reachable_nonterminal_states: int
    max_graph_depth: int
    irreversible_failure_ratio: float
    mean_nonfailure_actions: float
    success_edge_ratio: float
    phase_state_counts: Mapping[int, int]

    @property
    def order_key(self) -> tuple[float, ...]:
        """Predeclared, reward-independent structural ordering.

        Shortest successful dependency length is primary. Immediate irreversible
        failure risk, reachable-state burden, and graph depth only break ties.
        Agent performance never enters this ordering.
        """

        return (
            float(self.oracle_shortest_steps),
            float(self.irreversible_failure_ratio),
            float(self.reachable_nonterminal_states),
            float(self.max_graph_depth),
            float(self.map_seed),
        )

    def row(self, *, split: str, level: int) -> dict[str, Any]:
        return {
            "split": split,
            "level": int(level),
            "map_seed": self.map_seed,
            "oracle_shortest_steps": self.oracle_shortest_steps,
            "reachable_nonterminal_states": self.reachable_nonterminal_states,
            "max_graph_depth": self.max_graph_depth,
            "irreversible_failure_ratio": self.irreversible_failure_ratio,
            "mean_nonfailure_actions": self.mean_nonfailure_actions,
            "success_edge_ratio": self.success_edge_ratio,
            "phase_state_counts": json.dumps(
                {str(key): value for key, value in self.phase_state_counts.items()},
                sort_keys=True,
            ),
        }


def profile_map_complexity(map_seed: int) -> MapComplexityProfile | None:
    """Enumerate the frozen strict GridPush state graph for one map.

    This profiler uses only environment transitions and terminal outcomes. It is
    never exposed to an agent and is used solely to assign maps to predeclared
    complexity strata and to calculate the exact shortest successful path.
    """

    root = benchmark.BenchmarkGridPushWorld(int(map_seed))
    queue: deque[tuple[Any, int]] = deque([(root, 0)])
    visited = {benchmark._world_key(root)}
    shortest: int | None = None
    max_depth = 0
    total_actions = 0
    failure_actions = 0
    nonfailure_actions = 0
    success_edges = 0
    phase_counts: Counter[int] = Counter()

    while queue:
        world, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        phase_counts[int(world.phase)] += 1
        for action in benchmark.CHOICE_ACTIONS:
            candidate = deepcopy(world)
            candidate.step(action)
            total_actions += 1
            if candidate.failed:
                failure_actions += 1
                continue
            nonfailure_actions += 1
            if candidate.success:
                success_edges += 1
                if shortest is None:
                    shortest = depth + 1
                continue
            key = benchmark._world_key(candidate)
            if key in visited:
                continue
            visited.add(key)
            queue.append((candidate, depth + 1))

    if shortest is None:
        benchmark._ORACLE_STEPS[int(map_seed)] = None
        return None

    benchmark._ORACLE_STEPS[int(map_seed)] = shortest
    state_count = len(visited)
    return MapComplexityProfile(
        map_seed=int(map_seed),
        oracle_shortest_steps=int(shortest),
        reachable_nonterminal_states=state_count,
        max_graph_depth=max_depth,
        irreversible_failure_ratio=(
            failure_actions / total_actions if total_actions else 0.0
        ),
        mean_nonfailure_actions=(
            nonfailure_actions / state_count if state_count else 0.0
        ),
        success_edge_ratio=(success_edges / total_actions if total_actions else 0.0),
        phase_state_counts=dict(sorted(phase_counts.items())),
    )


def _quantile_bins(
    ordered: Sequence[MapComplexityProfile],
) -> dict[int, tuple[MapComplexityProfile, ...]]:
    if len(ordered) < len(COMPLEXITY_LEVELS):
        raise ValueError("not enough maps for five complexity levels")
    bins: dict[int, tuple[MapComplexityProfile, ...]] = {}
    total = len(ordered)
    for index, level in enumerate(COMPLEXITY_LEVELS):
        start = (index * total) // len(COMPLEXITY_LEVELS)
        end = ((index + 1) * total) // len(COMPLEXITY_LEVELS)
        bins[level] = tuple(ordered[start:end])
    return bins


def build_stratified_map_pool(
    *,
    start_seed: int,
    maps_per_level: int,
    selection_seed: int,
    oversample: int = 3,
) -> dict[int, tuple[MapComplexityProfile, ...]]:
    """Create balanced map strata without consulting agent performance."""

    if maps_per_level <= 0:
        raise ValueError("maps_per_level must be positive")
    if oversample <= 0:
        raise ValueError("oversample must be positive")

    target = maps_per_level * len(COMPLEXITY_LEVELS) * oversample
    profiles: list[MapComplexityProfile] = []
    candidate = int(start_seed)
    while len(profiles) < target:
        profile = profile_map_complexity(candidate)
        if profile is not None:
            profiles.append(profile)
        candidate += 1

    ordered = tuple(sorted(profiles, key=lambda item: item.order_key))
    bins = _quantile_bins(ordered)
    randomizer = random.Random(int(selection_seed))
    selected: dict[int, tuple[MapComplexityProfile, ...]] = {}
    for level in COMPLEXITY_LEVELS:
        candidates = list(bins[level])
        randomizer.shuffle(candidates)
        selected[level] = tuple(
            sorted(candidates[:maps_per_level], key=lambda item: item.map_seed)
        )
    return selected


def _flatten_pool(
    pool: Mapping[int, Sequence[MapComplexityProfile]],
) -> list[tuple[int, MapComplexityProfile]]:
    return [
        (level, profile)
        for level in COMPLEXITY_LEVELS
        for profile in pool[level]
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_success(rows: Sequence[Mapping[str, Any]]) -> float:
    return fmean(float(row["success"]) for row in rows) if rows else 0.0


def _mean_success_field(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    successful = [row for row in rows if int(row["success"]) == 1]
    return (
        fmean(float(row[field]) for row in successful)
        if successful
        else 0.0
    )


def _run_episode(
    agent: Any,
    *,
    condition: str,
    research_seed: int,
    phase: str,
    level: int,
    profile: MapComplexityProfile,
    checkpoint_transition_target: int,
    episode_index: int,
    training: bool,
    environment_steps_total: int,
    schedule_horizon: int,
) -> tuple[dict[str, Any], int]:
    world = benchmark.BenchmarkGridPushWorld(profile.map_seed)
    world.optimal_steps = profile.oracle_shortest_steps
    agent.begin_episode(training=training)
    steps = 0
    select_seconds = 0.0
    update_seconds = 0.0
    imagined_nodes = 0
    imagination_runs = 0

    while world.snapshot().available_actions:
        before = world.snapshot()
        schedule_position = min(schedule_horizon, environment_steps_total)
        started = time.perf_counter()
        decision = agent.select_action(
            before,
            episode=schedule_position,
            training=training,
        )
        select_seconds += time.perf_counter() - started

        outcome = world.step(decision.action)
        steps += 1
        imagined_nodes += int(decision.imagined_nodes)
        imagination_runs += int(decision.used_imagination)

        if training:
            environment_steps_total += 1
            started = time.perf_counter()
            agent.observe(before, decision.action, outcome)
            update_seconds += time.perf_counter() - started
        if world.success or world.failed:
            break

    agent.end_episode(success=world.success, training=training)
    row = {
        "condition": condition,
        "seed": int(research_seed),
        "phase": phase,
        "level": int(level),
        "checkpoint_transition_target": int(checkpoint_transition_target),
        "episode": int(episode_index),
        "map_seed": profile.map_seed,
        "success": int(world.success),
        "reward": 1.0 if world.success else 0.0,
        "steps": steps,
        "optimal_steps": profile.oracle_shortest_steps,
        "path_efficiency": (
            profile.oracle_shortest_steps / steps
            if world.success and steps
            else 0.0
        ),
        "environment_steps_total": int(environment_steps_total),
        "select_seconds": select_seconds,
        "update_seconds": update_seconds,
        "imagined_nodes": imagined_nodes,
        "imagination_runs": imagination_runs,
        "oracle_shortest_steps": profile.oracle_shortest_steps,
        "reachable_nonterminal_states": profile.reachable_nonterminal_states,
        "max_graph_depth": profile.max_graph_depth,
        "irreversible_failure_ratio": profile.irreversible_failure_ratio,
        "mean_nonfailure_actions": profile.mean_nonfailure_actions,
        "success_edge_ratio": profile.success_edge_ratio,
        "termination": "success" if world.success else "environment_failure",
    }
    return row, environment_steps_total


def _evaluate_level(
    agent: Any,
    *,
    condition: str,
    research_seed: int,
    phase: str,
    level: int,
    profiles: Sequence[MapComplexityProfile],
    episode_count: int,
    checkpoint_transition_target: int,
    environment_steps_total: int,
    schedule_horizon: int,
) -> tuple[list[dict[str, Any]], float]:
    if not profiles:
        raise ValueError("evaluation profile list cannot be empty")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(episode_count):
        profile = profiles[index % len(profiles)]
        row, _ = _run_episode(
            agent,
            condition=condition,
            research_seed=research_seed,
            phase=phase,
            level=level,
            profile=profile,
            checkpoint_transition_target=checkpoint_transition_target,
            episode_index=index,
            training=False,
            environment_steps_total=environment_steps_total,
            schedule_horizon=schedule_horizon,
        )
        rows.append(row)
    return rows, time.perf_counter() - started


def _normalize_checkpoints(
    checkpoints: Iterable[int],
    transition_budget: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(item)
                for item in checkpoints
                if 0 <= int(item) <= transition_budget
            }
            | {0, int(transition_budget)}
        )
    )


def run_final_complexity_scaling(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    transition_budget: int = 20_000,
    train_maps_per_level: int = 32,
    evaluation_maps_per_level: int = 100,
    checkpoints: Sequence[int] = (0, 2_500, 5_000, 10_000, 20_000),
    pool_oversample: int = 3,
) -> dict[str, Any]:
    """Run one condition/seed cell of the frozen final scaling experiment."""

    if condition not in FINAL_COMPLEXITY_CONDITIONS:
        raise ValueError(f"unknown final complexity condition: {condition}")
    if transition_budget <= 0:
        raise ValueError("transition_budget must be positive")
    if train_maps_per_level <= 0 or evaluation_maps_per_level <= 0:
        raise ValueError("map counts must be positive")

    normalized_checkpoints = _normalize_checkpoints(
        checkpoints,
        transition_budget,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Disjoint deterministic ranges guarantee identical maps across conditions
    # while preventing train/unseen overlap for every research seed.
    range_base = int(seed) * 10_000_000
    training_pool = build_stratified_map_pool(
        start_seed=range_base + 1_000_000,
        maps_per_level=train_maps_per_level,
        selection_seed=int(seed) ^ 0x54524149,
        oversample=pool_oversample,
    )
    unseen_pool = build_stratified_map_pool(
        start_seed=range_base + 6_000_000,
        maps_per_level=evaluation_maps_per_level,
        selection_seed=int(seed) ^ 0x554E5345,
        oversample=pool_oversample,
    )

    manifest_rows = [
        profile.row(split="training", level=level)
        for level in COMPLEXITY_LEVELS
        for profile in training_pool[level]
    ] + [
        profile.row(split="unseen", level=level)
        for level in COMPLEXITY_LEVELS
        for profile in unseen_pool[level]
    ]
    _write_csv(output / "map_manifest.csv", manifest_rows)

    training_sequence = _flatten_pool(training_pool)
    random.Random(int(seed) ^ 0x53434844).shuffle(training_sequence)

    # train_episodes is intentionally used as a generic schedule horizon here.
    # The episode value passed to every policy is the completed real-transition
    # count, making exploration decay comparable across conditions.
    agent = make_imagination_v2_agent(
        condition,
        int(seed),
        train_episodes=transition_budget,
    )
    if condition == "legacy_aassr":
        # Preserve the legacy policy itself while expressing its decay horizon
        # in the same real-transition unit used by the other conditions.
        agent.agent.config = replace(
            agent.agent.config,
            epsilon_decay_episodes=max(1, int(transition_budget * 0.80)),
        )

    training_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    environment_steps_total = 0
    training_episode_count = 0
    selection_seconds_total = 0.0
    update_seconds_total = 0.0
    training_started = time.perf_counter()

    for checkpoint_target in normalized_checkpoints:
        while environment_steps_total < checkpoint_target:
            level, profile = training_sequence[
                training_episode_count % len(training_sequence)
            ]
            row, environment_steps_total = _run_episode(
                agent,
                condition=condition,
                research_seed=seed,
                phase="training",
                level=level,
                profile=profile,
                checkpoint_transition_target=checkpoint_target,
                episode_index=training_episode_count,
                training=True,
                environment_steps_total=environment_steps_total,
                schedule_horizon=transition_budget,
            )
            training_rows.append(row)
            training_episode_count += 1
            selection_seconds_total += float(row["select_seconds"])
            update_seconds_total += float(row["update_seconds"])

        recent = training_rows[-min(200, len(training_rows)) :]
        stats = dict(agent.model_stats())
        for level in COMPLEXITY_LEVELS:
            seen_rows, seen_seconds = _evaluate_level(
                agent,
                condition=condition,
                research_seed=seed,
                phase="evaluation_seen",
                level=level,
                profiles=training_pool[level],
                episode_count=evaluation_maps_per_level,
                checkpoint_transition_target=checkpoint_target,
                environment_steps_total=environment_steps_total,
                schedule_horizon=transition_budget,
            )
            unseen_rows, unseen_seconds = _evaluate_level(
                agent,
                condition=condition,
                research_seed=seed,
                phase="evaluation_unseen",
                level=level,
                profiles=unseen_pool[level],
                episode_count=evaluation_maps_per_level,
                checkpoint_transition_target=checkpoint_target,
                environment_steps_total=environment_steps_total,
                schedule_horizon=transition_budget,
            )
            evaluation_rows.extend(seen_rows)
            evaluation_rows.extend(unseen_rows)
            for phase, rows, elapsed in (
                ("evaluation_seen", seen_rows, seen_seconds),
                ("evaluation_unseen", unseen_rows, unseen_seconds),
            ):
                checkpoint_rows.append(
                    {
                        "condition": condition,
                        "seed": int(seed),
                        "checkpoint_transition_target": checkpoint_target,
                        "environment_steps_total": environment_steps_total,
                        "training_episode_count": training_episode_count,
                        "phase": phase,
                        "level": level,
                        "success_rate": _mean_success(rows),
                        "mean_steps_on_success": _mean_success_field(rows, "steps"),
                        "mean_path_efficiency": _mean_success_field(
                            rows,
                            "path_efficiency",
                        ),
                        "mean_imagined_nodes": fmean(
                            float(row["imagined_nodes"]) for row in rows
                        ),
                        "imagination_use_rate": fmean(
                            float(int(row["imagination_runs"]) > 0)
                            for row in rows
                        ),
                        "evaluation_seconds": elapsed,
                        "training_recent_success_rate": _mean_success(recent),
                        "training_wall_seconds": time.perf_counter()
                        - training_started,
                        "selection_seconds_total": selection_seconds_total,
                        "update_seconds_total": update_seconds_total,
                        "model_units": int(stats.get("model_units", 0)),
                        "model_bytes": int(stats.get("model_bytes", 0)),
                        "gradient_updates": int(stats.get("gradient_updates", 0)),
                    }
                )

    diagnostics_fn = getattr(agent, "diagnostics", None)
    diagnostics = dict(diagnostics_fn()) if callable(diagnostics_fn) else {}
    final_actual_steps = environment_steps_total
    payload = {
        "config": {
            "condition": condition,
            "seed": int(seed),
            "transition_budget": transition_budget,
            "actual_training_transitions": final_actual_steps,
            "transition_budget_overshoot": final_actual_steps - transition_budget,
            "train_maps_per_level": train_maps_per_level,
            "evaluation_maps_per_level": evaluation_maps_per_level,
            "checkpoints": list(normalized_checkpoints),
            "pool_oversample": pool_oversample,
            "levels": list(COMPLEXITY_LEVELS),
            "fixed_environment": "strict BenchmarkGridPushWorld",
            "complexity_order": [
                "oracle_shortest_steps",
                "irreversible_failure_ratio",
                "reachable_nonterminal_states",
                "max_graph_depth",
            ],
            "complexity_partition": "within-seed structural quintiles",
            "external_reward": "final_success_only",
            "fixed_action_space": True,
            "artificial_step_limit": False,
            "abandonment_enabled": False,
            "episode_termination": "success_or_environment_failure_only",
            "oracle_used_for_agent_training_or_action": False,
            "policy_exploration_schedule_unit": "real_environment_transition",
        },
        "final": {
            "condition": condition,
            "seed": int(seed),
            "actual_training_transitions": final_actual_steps,
            "training_episode_count": training_episode_count,
            "training_wall_seconds": time.perf_counter() - training_started,
            "selection_seconds_total": selection_seconds_total,
            "update_seconds_total": update_seconds_total,
            "model_stats": dict(agent.model_stats()),
        },
        "agent_diagnostics": diagnostics,
    }

    _write_csv(output / "training_episodes.csv", training_rows)
    _write_csv(output / "evaluation_episodes.csv", evaluation_rows)
    _write_csv(output / "checkpoint_level_metrics.csv", checkpoint_rows)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr),
        encoding="utf-8",
    )
    return payload
