from __future__ import annotations

import csv
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .pentest_curriculum_env import STALL_PATIENCE
from .pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from . import pentest_curriculum_schedule as schedule


DEFAULT_CURRENT_TRANSITION_BUDGET = 10_000
DEFAULT_CURRENT_BLOCK_TARGET = 512


@dataclass(frozen=True, slots=True)
class CurrentEpisodeRow:
    phase: str
    condition: str
    research_seed: int
    block: int
    episode: int
    scenario_seed: int
    curriculum_level: int
    curriculum_stage: str
    focus_level: int
    status: str
    success: int
    failure: int
    stalled: int
    truncation: int
    primitive_transitions: int
    transition_total: int
    reward: float
    budget_truncated: int
    aseq_guard_events: int
    aseq_all_guarded_fallbacks: int
    imagination_runs: int
    imagination_interventions: int
    imagination_changed_actions: int
    skill_uses: int
    skill_promotions: int


def write_current_csv(path: Path, rows: Sequence[CurrentEpisodeRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _status(world: TransferDiagnosticWorld, *, stalled: bool, exhausted: bool) -> str:
    if world.success and world.proof_acquired:
        return "success"
    if world.failed and world.locked:
        return "failure"
    if stalled:
        return "stalled"
    if world.rate_limited or exhausted:
        return "truncation"
    return "truncation"


def _reward_for_status(status: str) -> float:
    if status == "success":
        return 1.0
    if status == "failure":
        return -1.0
    return 0.0


def _delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _counter_snapshot(agent: object) -> dict[str, int]:
    diagnostics = agent.diagnostics()
    aseq = diagnostics.get("aseq", {})
    imagination = diagnostics.get("imagination", {})
    return {
        "aseq_guard_events": int(aseq.get("guard_events", 0)),
        "aseq_all_guarded_fallbacks": int(aseq.get("all_guarded_fallbacks", 0)),
        "imagination_runs": int(imagination.get("runs", 0)),
        "imagination_interventions": int(imagination.get("interventions", 0)),
        "imagination_changed_actions": int(imagination.get("changed_actions", 0)),
        "skill_uses": int(diagnostics.get("skill_uses", 0)),
        "skill_promotions": int(diagnostics.get("promoted_skills", 0)),
    }


def exploration_index(transition_total: int, transition_budget: int) -> int:
    if transition_budget <= 0:
        return 1000
    return min(
        1000,
        max(0, int(1000 * transition_total / transition_budget)),
    )


def run_current_episode(
    agent: object,
    *,
    condition: str,
    research_seed: int,
    stage_index: int,
    scenario_seed: int,
    phase: str,
    block: int,
    episode: int,
    focus_level: int,
    transition_start: int,
    transition_cap: int,
    transition_budget: int,
    training: bool,
    budget_cap: bool = False,
) -> tuple[CurrentEpisodeRow, int]:
    """Execute one episode under the current sparse-reward protocol.

    During training, exploration is indexed by the fraction of the *global real
    transition budget* already consumed. It never resets at a curriculum block
    boundary. The supplied `episode` remains the recorded episode label and is
    used directly only in frozen evaluation, where exploration is disabled.
    """

    stage = TRANSFER_STAGES[int(stage_index)]
    plugin = getattr(agent, "runtime_plugin", None)
    factory = getattr(plugin, "environment_factory", None)
    world = (
        factory(scenario_seed=int(scenario_seed), stage=stage)
        if callable(factory)
        else TransferDiagnosticWorld(int(scenario_seed), stage=stage)
    )
    agent.begin_episode()
    counters_before = _counter_snapshot(agent)

    recent_pairs: deque[tuple[tuple[Any, ...], str]] = deque(
        maxlen=STALL_PATIENCE
    )
    transitions = 0
    unchanged = 0
    stalled = False

    while (
        transitions < transition_cap
        and not world.success
        and not (world.failed and world.locked)
        and not world.rate_limited
    ):
        remaining = transition_cap - transitions
        decision_index = (
            exploration_index(
                transition_start + transitions,
                transition_budget,
            )
            if training
            else int(episode)
        )
        step = agent.step(
            world,
            episode=decision_index,
            training=training,
            primitive_budget=remaining,
        )
        if not step.traces:
            break

        for trace in step.traces:
            semantic_before = schedule.semantic_fingerprint(trace.before)
            semantic_after = schedule.semantic_fingerprint(trace.after)
            unchanged = unchanged + 1 if semantic_after == semantic_before else 0
            recent_pairs.append((semantic_before, trace.action.signature))
            transitions += 1
            if unchanged >= STALL_PATIENCE:
                counts = Counter(recent_pairs)
                stalled = (
                    len(counts) <= 3
                    or max(counts.values(), default=0) >= 4
                )
            if stalled or transitions >= transition_cap:
                break
        if stalled:
            break

    exhausted = transitions >= transition_cap
    status = _status(world, stalled=stalled, exhausted=exhausted)
    reward = _reward_for_status(status)
    agent.finish_episode(final_return=reward, training=training)
    counters_after = _counter_snapshot(agent)

    return (
        CurrentEpisodeRow(
            phase=phase,
            condition=condition,
            research_seed=int(research_seed),
            block=int(block),
            episode=int(episode),
            scenario_seed=int(scenario_seed),
            curriculum_level=stage.level,
            curriculum_stage=stage.name,
            focus_level=int(focus_level),
            status=status,
            success=int(status == "success"),
            failure=int(status == "failure"),
            stalled=int(status == "stalled"),
            truncation=int(status == "truncation"),
            primitive_transitions=transitions,
            transition_total=transition_start + transitions,
            reward=reward,
            budget_truncated=int(
                budget_cap and exhausted and status == "truncation"
            ),
            aseq_guard_events=_delta(
                counters_after,
                counters_before,
                "aseq_guard_events",
            ),
            aseq_all_guarded_fallbacks=_delta(
                counters_after,
                counters_before,
                "aseq_all_guarded_fallbacks",
            ),
            imagination_runs=_delta(
                counters_after,
                counters_before,
                "imagination_runs",
            ),
            imagination_interventions=_delta(
                counters_after,
                counters_before,
                "imagination_interventions",
            ),
            imagination_changed_actions=_delta(
                counters_after,
                counters_before,
                "imagination_changed_actions",
            ),
            skill_uses=_delta(
                counters_after,
                counters_before,
                "skill_uses",
            ),
            skill_promotions=_delta(
                counters_after,
                counters_before,
                "skill_promotions",
            ),
        ),
        transitions,
    )


def aggregate_current_diagnostic(
    rows: Sequence[CurrentEpisodeRow],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stage in TRANSFER_STAGES:
        items = [row for row in rows if row.curriculum_level == stage.level]
        output.append(
            {
                "level": stage.level,
                "stage": stage.name,
                "episodes": len(items),
                "successes": sum(row.success for row in items),
                "success_rate": (
                    fmean(row.success for row in items) if items else 0.0
                ),
                "stalled": sum(row.stalled for row in items),
                "truncations": sum(row.truncation for row in items),
                "mean_primitive_steps": (
                    fmean(row.primitive_transitions for row in items)
                    if items
                    else 0.0
                ),
                "aseq_guard_events": sum(row.aseq_guard_events for row in items),
                "imagination_runs": sum(row.imagination_runs for row in items),
                "imagination_interventions": sum(
                    row.imagination_interventions for row in items
                ),
            }
        )
    return output


def current_frontier(
    summary: Sequence[Mapping[str, Any]],
) -> dict[str, int | None]:
    highest: int | None = None
    first_zero: int | None = None
    for item in summary:
        level = int(item["level"])
        successes = int(item["successes"])
        if successes > 0 and first_zero is None:
            highest = level
        elif successes == 0 and first_zero is None:
            first_zero = level
    return {
        "highest_contiguous_nonzero_level": highest,
        "first_zero_success_level": first_zero,
    }
