from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .baseline_efficiency_benchmark import (
    BenchmarkGridPushWorld,
    _run_episode,
    solvable_map_seeds,
)
from .bottleneck_sota_diagnostic import remaining_oracle_steps
from .branch_critic import CriticTransition
from .imagination_v2 import ImaginationV2Agent
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class AbandonmentSignal:
    step: int
    success_probability: float | None
    low_probability_streak: int
    ready: bool
    should_abandon: bool
    reason: str


class CriticAbandonmentProbe:
    """Read a frozen Imagination-v2 critic as a voluntary-stop signal.

    The probe never trains the critic. It only advances a separate real-trajectory
    GRU memory with transitions that actually occurred. A declaration requires a
    low predicted success probability for several consecutive states, preventing
    one noisy estimate from ending an episode.
    """

    def __init__(
        self,
        agent: ImaginationV2Agent,
        *,
        threshold: float,
        minimum_steps: int = 2,
        patience: int = 2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        if minimum_steps < 0 or patience <= 0:
            raise ValueError("minimum_steps must be non-negative and patience positive")
        self.agent = agent
        self.threshold = float(threshold)
        self.minimum_steps = int(minimum_steps)
        self.patience = int(patience)
        self.reset()

    def reset(self) -> None:
        self.memory = self.agent.critic.initial_memory()
        self.steps = 0
        self.low_probability_streak = 0
        self.success_probability: float | None = None
        self.declared = False

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
    ) -> None:
        confidence_fn = getattr(self.agent.agent.prophecy, "confidence", None)
        confidence = (
            float(confidence_fn(before, action))
            if callable(confidence_fn)
            else 1.0
        )
        scored = self.agent.critic.score_step(
            before,
            action,
            after,
            memory=self.memory,
            prophecy_confidence=max(0.0, min(1.0, confidence)),
        )
        self.memory = scored.memory
        self.success_probability = float(scored.value)
        self.steps += 1
        if self.success_probability <= self.threshold:
            self.low_probability_streak += 1
        else:
            self.low_probability_streak = 0

    def signal(self) -> AbandonmentSignal:
        if self.declared:
            return AbandonmentSignal(
                self.steps,
                self.success_probability,
                self.low_probability_streak,
                bool(self.agent.critic_ready),
                False,
                "already_declared",
            )
        if not self.agent.critic_ready:
            return AbandonmentSignal(
                self.steps,
                self.success_probability,
                self.low_probability_streak,
                False,
                False,
                "critic_not_ready",
            )
        if self.steps < self.minimum_steps:
            return AbandonmentSignal(
                self.steps,
                self.success_probability,
                self.low_probability_streak,
                True,
                False,
                "minimum_steps",
            )
        if self.success_probability is None:
            return AbandonmentSignal(
                self.steps,
                None,
                self.low_probability_streak,
                True,
                False,
                "no_real_transition_yet",
            )
        should_abandon = self.low_probability_streak >= self.patience
        if should_abandon:
            self.declared = True
        return AbandonmentSignal(
            self.steps,
            self.success_probability,
            self.low_probability_streak,
            True,
            should_abandon,
            "critic_low_success_probability" if should_abandon else "continue",
        )


@dataclass(frozen=True, slots=True)
class AbandonmentEpisode:
    mode: str
    split: str
    threshold: float
    seed: int
    episode: int
    map_seed: int
    success: int
    abandoned: int
    declaration_step: int | None
    declaration_probability: float | None
    declaration_phase: int | None
    declaration_solvable: int | None
    declaration_remaining_oracle_steps: int | None
    encountered_dead_end: int
    appropriate_declaration: int
    premature_declaration: int
    safety_stop: int
    environment_steps: int
    natural_terminal: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AbandonmentEvent:
    mode: str
    split: str
    threshold: float
    seed: int
    episode: int
    map_seed: int
    step: int
    success_probability: float
    low_probability_streak: int
    phase: int
    used_cell_count: int
    bridge_built: int
    key_held: int
    door_open: int
    oracle_solvable: int
    oracle_remaining_steps: int | None
    facts: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


def _phase(state: StateSnapshot) -> int:
    for fact in state.facts:
        if fact.startswith("phase:"):
            return int(fact.split(":", 1)[1])
    return -1


def _used_cell_count(state: StateSnapshot) -> int:
    return sum(fact.startswith("used:") for fact in state.facts)


def _run_frozen_episode(
    agent: ImaginationV2Agent,
    *,
    map_seed: int,
    seed: int,
    episode: int,
    split: str,
    mode: str,
    threshold: float,
    minimum_steps: int,
    patience: int,
    safety_cap: int,
) -> tuple[AbandonmentEpisode, AbandonmentEvent | None]:
    if mode not in {"shadow", "active"}:
        raise ValueError("mode must be shadow or active")
    world = BenchmarkGridPushWorld(map_seed)
    agent.begin_episode(training=False)
    probe = CriticAbandonmentProbe(
        agent,
        threshold=threshold,
        minimum_steps=minimum_steps,
        patience=patience,
    )
    declaration: AbandonmentEvent | None = None
    encountered_dead_end = False
    abandoned = False
    safety_stop = False

    while world.snapshot().available_actions:
        before = world.snapshot()
        remaining = remaining_oracle_steps(before)
        if remaining is None:
            encountered_dead_end = True

        signal = probe.signal()
        if signal.should_abandon and declaration is None:
            declaration = AbandonmentEvent(
                mode=mode,
                split=split,
                threshold=threshold,
                seed=seed,
                episode=episode,
                map_seed=map_seed,
                step=signal.step,
                success_probability=float(signal.success_probability),
                low_probability_streak=signal.low_probability_streak,
                phase=_phase(before),
                used_cell_count=_used_cell_count(before),
                bridge_built=int("bridge_built" in before.facts),
                key_held=int("key_held" in before.facts),
                door_open=int("door_open" in before.facts),
                oracle_solvable=int(remaining is not None),
                oracle_remaining_steps=remaining,
                facts="|".join(sorted(before.facts)),
            )
            if mode == "active":
                abandoned = True
                break

        decision = agent.select_action(
            before,
            episode=episode,
            training=False,
        )
        outcome = world.step(decision.action)
        probe.observe(before, decision.action, outcome.snapshot)
        if world.success or world.failed:
            break
        if world.steps >= safety_cap:
            safety_stop = True
            break

    agent.end_episode(success=world.success, training=False)
    if declaration is None:
        declaration_solvable = None
        declaration_remaining = None
    else:
        declaration_solvable = declaration.oracle_solvable
        declaration_remaining = declaration.oracle_remaining_steps
    appropriate = int(declaration is not None and not declaration.oracle_solvable)
    premature = int(declaration is not None and declaration.oracle_solvable)
    if world.success:
        terminal = "success"
    elif abandoned:
        terminal = "abandoned"
    elif safety_stop:
        terminal = "safety_stop"
    else:
        terminal = "environment_failure"
    row = AbandonmentEpisode(
        mode=mode,
        split=split,
        threshold=threshold,
        seed=seed,
        episode=episode,
        map_seed=map_seed,
        success=int(world.success),
        abandoned=int(abandoned),
        declaration_step=None if declaration is None else declaration.step,
        declaration_probability=(
            None if declaration is None else declaration.success_probability
        ),
        declaration_phase=None if declaration is None else declaration.phase,
        declaration_solvable=declaration_solvable,
        declaration_remaining_oracle_steps=declaration_remaining,
        encountered_dead_end=int(encountered_dead_end),
        appropriate_declaration=appropriate,
        premature_declaration=premature,
        safety_stop=int(safety_stop),
        environment_steps=world.steps,
        natural_terminal=terminal,
    )
    return row, declaration


def _summarize(rows: Sequence[AbandonmentEpisode]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[AbandonmentEpisode]] = {}
    for row in rows:
        grouped.setdefault((row.mode, row.split, row.threshold), []).append(row)
    result: list[dict[str, Any]] = []
    for (mode, split, threshold), items in sorted(grouped.items()):
        declarations = [item for item in items if item.declaration_step is not None]
        dead_ends = [item for item in items if item.encountered_dead_end]
        result.append(
            {
                "mode": mode,
                "split": split,
                "threshold": threshold,
                "episodes": len(items),
                "success_rate": fmean(item.success for item in items),
                "abandonment_rate": fmean(item.abandoned for item in items),
                "declaration_rate": len(declarations) / len(items),
                "appropriate_declarations": sum(
                    item.appropriate_declaration for item in items
                ),
                "premature_declarations": sum(
                    item.premature_declaration for item in items
                ),
                "declaration_precision": (
                    sum(item.appropriate_declaration for item in items)
                    / len(declarations)
                    if declarations
                    else 0.0
                ),
                "dead_end_episode_count": len(dead_ends),
                "dead_end_detection_rate": (
                    sum(item.appropriate_declaration for item in dead_ends)
                    / len(dead_ends)
                    if dead_ends
                    else 0.0
                ),
                "mean_declaration_step": (
                    fmean(float(item.declaration_step) for item in declarations)
                    if declarations
                    else 0.0
                ),
                "mean_declaration_probability": (
                    fmean(
                        float(item.declaration_probability)
                        for item in declarations
                        if item.declaration_probability is not None
                    )
                    if declarations
                    else 0.0
                ),
                "mean_environment_steps": fmean(
                    item.environment_steps for item in items
                ),
                "safety_stop_rate": fmean(item.safety_stop for item in items),
            }
        )
    return result


def _pair_active_shadow(rows: Sequence[AbandonmentEpisode]) -> list[dict[str, Any]]:
    indexed = {
        (row.split, row.threshold, row.map_seed, row.mode): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    keys = sorted({(row.split, row.threshold, row.map_seed) for row in rows})
    for split, threshold, map_seed in keys:
        active = indexed.get((split, threshold, map_seed, "active"))
        shadow = indexed.get((split, threshold, map_seed, "shadow"))
        if active is None or shadow is None:
            continue
        pairs.append(
            {
                "split": split,
                "threshold": threshold,
                "map_seed": map_seed,
                "active_abandoned": active.abandoned,
                "active_success": active.success,
                "shadow_success": shadow.success,
                "prevented_success": int(active.abandoned and shadow.success),
                "saved_steps_on_shadow_failure": (
                    max(0, shadow.environment_steps - active.environment_steps)
                    if active.abandoned and not shadow.success
                    else 0
                ),
            }
        )
    return pairs


def _write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        return
    normalized = [row.row() if hasattr(row, "row") else dict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def run_abandonment_smoke(
    output_dir: str | Path,
    *,
    seed: int = 7,
    train_episodes: int = 300,
    train_map_count: int = 32,
    evaluation_episodes: int = 30,
    thresholds: Sequence[float] = (0.05, 0.15, 0.30),
    minimum_steps: int = 2,
    patience: int = 2,
    safety_cap: int = 128,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agent = ImaginationV2Agent(seed, train_episodes=train_episodes)
    training_maps = solvable_map_seeds(seed * 1_000_000, train_map_count)
    environment_steps = 0
    training_successes = 0
    for episode in range(train_episodes):
        metric, environment_steps = _run_episode(
            agent,
            condition="imagination_v2_abandonment_training",
            seed=seed,
            phase="training",
            checkpoint_episode=train_episodes,
            episode=episode,
            map_seed=training_maps[episode % len(training_maps)],
            training=True,
            environment_steps_total=environment_steps,
        )
        training_successes += metric.success

    frozen = deepcopy(agent)
    seen_maps = tuple(
        training_maps[index % len(training_maps)]
        for index in range(evaluation_episodes)
    )
    unseen_maps = solvable_map_seeds(
        seed * 1_000_000 + 500_000,
        evaluation_episodes,
    )
    rows: list[AbandonmentEpisode] = []
    events: list[AbandonmentEvent] = []
    for threshold in thresholds:
        for mode in ("shadow", "active"):
            evaluation_agent = deepcopy(frozen)
            for split, map_seeds in (("seen", seen_maps), ("unseen", unseen_maps)):
                for episode, map_seed in enumerate(map_seeds):
                    row, event = _run_frozen_episode(
                        evaluation_agent,
                        map_seed=map_seed,
                        seed=seed,
                        episode=episode,
                        split=split,
                        mode=mode,
                        threshold=float(threshold),
                        minimum_steps=minimum_steps,
                        patience=patience,
                        safety_cap=safety_cap,
                    )
                    rows.append(row)
                    if event is not None:
                        events.append(event)

    summary_rows = _summarize(rows)
    paired_rows = _pair_active_shadow(rows)
    critic_stats = asdict(agent.critic.stats())
    payload = {
        "config": {
            "seed": seed,
            "train_episodes": train_episodes,
            "train_map_count": train_map_count,
            "evaluation_episodes_per_split": evaluation_episodes,
            "thresholds": [float(value) for value in thresholds],
            "minimum_steps": minimum_steps,
            "patience": patience,
            "safety_cap": safety_cap,
            "environment": "strict_gridpush_final",
            "abandonment_training": "disabled; frozen post-training critic only",
        },
        "training": {
            "success_rate": training_successes / train_episodes,
            "environment_steps": environment_steps,
            "critic_ready": bool(agent.critic_ready),
            "critic_stats": critic_stats,
        },
        "summary": summary_rows,
        "paired": {
            "prevented_successes": sum(
                item["prevented_success"] for item in paired_rows
            ),
            "saved_steps_on_shadow_failures": sum(
                item["saved_steps_on_shadow_failure"] for item in paired_rows
            ),
            "rows": paired_rows,
        },
        "interpretation_guardrails": {
            "oracle_used_for_training": False,
            "oracle_used_only_for_posthoc_abandonment_audit": True,
            "abandoned_episodes_used_to_train_critic": False,
            "fixed_episode_step_limit": False,
            "safety_cap_is_nontermination_guard_only": True,
        },
    }
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "abandonment_events.csv", events)
    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "paired_active_shadow.csv", paired_rows)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
