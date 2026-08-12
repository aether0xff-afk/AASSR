from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Sequence

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_protocol import CurrentEpisodeRow, run_current_episode, write_current_csv
from aassr_v2.pentest_curriculum_env import STALL_PATIENCE
from aassr_v2.pentest_current_generation_main import (
    CURRENT_CURRICULUM_VALIDATION_SEEDS,
    _learning_counters,
    _run_aassr_frozen_eval,
)
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_STAGES,
    TRANSFER_TRAIN_SEEDS,
    TransferAdaptiveCurriculum,
)
from aassr_v2 import pentest_curriculum_schedule as schedule

import run_imagination_intervention_trace as detail


VERSION = "imagination-only-fast-trace-v1"
DEFAULT_STAGES = (1, 2, 3, 4)
DEFAULT_SCENARIO_SEEDS = (94_001, 94_003)


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def _train_once(
    agent: object,
    *,
    research_seed: int,
    transition_budget: int,
    block_target: int,
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
) -> tuple[list[CurrentEpisodeRow], list[CurrentEpisodeRow], list[dict], int]:
    curriculum = TransferAdaptiveCurriculum()
    training_rows: list[CurrentEpisodeRow] = []
    validation_rows: list[CurrentEpisodeRow] = []
    curriculum_trace: list[dict] = []
    transition_total = 0
    block = 0

    while transition_total < transition_budget:
        block_used = 0
        episode = 0
        weights = curriculum.weights()
        rng = random.Random(int(research_seed) ^ (block * 0x9E3779B1))
        focus_before = curriculum.focus_level

        while block_used < block_target and transition_total < transition_budget:
            level = schedule.weighted_level(rng, weights)
            stage = TRANSFER_STAGES[level]
            scenario_seed = int(train_seeds[(block * 97 + episode) % len(train_seeds)])
            natural_cap = max(24, stage.rate_limit + STALL_PATIENCE)
            hard_left = transition_budget - transition_total
            cap = min(natural_cap, hard_left)
            row, consumed = run_current_episode(
                agent,
                condition="imagination_only_train",
                research_seed=int(research_seed),
                stage_index=level,
                scenario_seed=scenario_seed,
                phase="train",
                block=block,
                episode=episode,
                focus_level=focus_before,
                transition_start=transition_total,
                transition_cap=cap,
                transition_budget=transition_budget,
                training=True,
                budget_cap=hard_left < natural_cap,
            )
            if consumed <= 0:
                raise RuntimeError("Imagination-only training consumed no transitions")
            training_rows.append(row)
            transition_total += consumed
            block_used += consumed
            episode += 1

        block_validation = _run_aassr_frozen_eval(
            agent,
            research_seed=research_seed,
            stage_index=curriculum.focus_level,
            scenario_seeds=validation_seeds,
            phase="curriculum_validation",
            block=block,
            focus_level=curriculum.focus_level,
            use_imagination=False,
            transition_budget=transition_budget,
        )
        validation_rows.extend(block_validation)
        validation_success = fmean(row.success for row in block_validation)
        movement = curriculum.observe_block(validation_success)
        curriculum_trace.append(
            {
                "block": block,
                "transition_total": transition_total,
                "block_transitions": block_used,
                "focus_before": focus_before,
                "validation_success_rate": validation_success,
                "movement": movement,
                "focus_after": curriculum.focus_level,
                "train_weights": weights,
            }
        )
        block += 1

    return training_rows, validation_rows, curriculum_trace, curriculum.focus_level


def _evaluate_imagination_only(
    agent: object,
    *,
    research_seed: int,
    scenario_seeds: Sequence[int],
    stage_indices: Sequence[int],
    transition_budget: int,
    margin: float,
    output: Path,
) -> tuple[list[CurrentEpisodeRow], dict]:
    original_config = agent.config
    agent.config = replace(
        original_config,
        imagination_intervention_margin=float(margin),
        imagination_uncertainty_margin=0.0,
    )

    before_learning = _learning_counters(agent)
    capture = detail.DetailedTraceCapture(agent)
    capture.install()
    rows: list[CurrentEpisodeRow] = []
    started = time.perf_counter()
    try:
        for stage_index in stage_indices:
            stage = TRANSFER_STAGES[int(stage_index)]
            stage_rows = _run_aassr_frozen_eval(
                agent,
                research_seed=research_seed,
                stage_index=int(stage_index),
                scenario_seeds=scenario_seeds,
                phase="imagination_trace",
                block=-1,
                focus_level=stage.level,
                use_imagination=True,
                transition_budget=transition_budget,
            )
            rows.extend(
                replace(row, condition=f"aassr_imagination_margin_{margin:g}")
                for row in stage_rows
            )
    finally:
        capture.uninstall()
        agent.config = original_config

    if _learning_counters(agent) != before_learning:
        raise AssertionError("Imagination-only frozen evaluation mutated learned state")

    capture.annotate_rows(rows)
    condition = f"aassr_imagination_margin_{margin:g}"
    detail._write_trace_files(output, condition=condition, capture=capture)

    switch_events = [event for event in capture.events if event["switch_candidate"]]
    intervention_events = [event for event in capture.events if event["intervention_allowed"]]
    return rows, {
        "condition": condition,
        "wall_seconds": time.perf_counter() - started,
        "decisions": len(capture.events),
        "episodes": len(capture.episodes),
        "imagination_runs": sum(bool(event["used_imagination"]) for event in capture.events),
        "switch_candidates": len(switch_events),
        "interventions": len(intervention_events),
        "changed_actions": sum(bool(event["changed_action"]) for event in capture.events),
        "successes": sum(row.success for row in rows),
        "failures": sum(row.failure for row in rows),
        "stalls": sum(row.stalled for row in rows),
        "truncations": sum(row.truncation for row in rows),
        "primitive_transitions": sum(row.primitive_transitions for row in rows),
        "switch_event_indices": [int(event["event_index"]) for event in switch_events],
        "intervention_event_indices": [int(event["event_index"]) for event in intervention_events],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train current AASSR once, then run only one frozen Imagination condition "
            "with full decision/Prophecy/planner/real-transition traces."
        )
    )
    parser.add_argument("--output-dir", default="runs/imagination_only_fast_trace")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transitions", type=int, default=2048)
    parser.add_argument("--block-target", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--stages", type=_parse_ints, default=DEFAULT_STAGES)
    parser.add_argument("--scenario-seeds", type=_parse_ints, default=DEFAULT_SCENARIO_SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--allow-critic-not-ready", action="store_true")
    args = parser.parse_args()

    if args.transitions <= 0 or args.block_target <= 0:
        parser.error("--transitions and --block-target must be positive")
    if args.margin < 0.0:
        parser.error("--margin must be non-negative")
    if any(index < 0 or index >= len(TRANSFER_STAGES) for index in args.stages):
        parser.error("--stages contains an out-of-range stage index")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    agent = build_current_pentest_aassr_core(
        seed=int(args.seed),
        train_transitions=int(args.transitions),
        use_imagination=True,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )

    training_rows, validation_rows, curriculum_trace, final_focus_level = _train_once(
        agent,
        research_seed=int(args.seed),
        transition_budget=int(args.transitions),
        block_target=int(args.block_target),
        train_seeds=TRANSFER_TRAIN_SEEDS,
        validation_seeds=CURRENT_CURRICULUM_VALIDATION_SEEDS,
    )

    critic_ready = bool(agent.critic_ready)
    if not critic_ready and not args.allow_critic_not_ready:
        raise RuntimeError(
            "Critic is not ready; increase --transitions or use "
            "--allow-critic-not-ready only for smoke debugging"
        )

    checkpoint = list(_learning_counters(agent))
    eval_rows, imagination = _evaluate_imagination_only(
        agent,
        research_seed=int(args.seed),
        scenario_seeds=tuple(args.scenario_seeds),
        stage_indices=tuple(args.stages),
        transition_budget=int(args.transitions),
        margin=float(args.margin),
        output=output,
    )

    write_current_csv(output / "training_aassr.csv", training_rows)
    write_current_csv(output / "curriculum_validation_aassr.csv", validation_rows)
    write_current_csv(output / "imagination_only_eval.csv", eval_rows)
    (output / "curriculum_trace_aassr.json").write_text(
        json.dumps(curriculum_trace, indent=2, sort_keys=True), encoding="utf-8"
    )

    result = {
        "version": VERSION,
        "research_seed": int(args.seed),
        "device": args.device,
        "transition_budget": int(args.transitions),
        "exact_budget": sum(row.primitive_transitions for row in training_rows) == int(args.transitions),
        "critic_ready": critic_ready,
        "shared_checkpoint": checkpoint,
        "final_focus_level": final_focus_level,
        "margin": float(args.margin),
        "coverage_uncertainty_lambda": 0.0,
        "stages": list(map(int, args.stages)),
        "scenario_seeds": list(map(int, args.scenario_seeds)),
        "no_imagination_baseline_evaluated": False,
        "training_successes": sum(row.success for row in training_rows),
        "training_stalls": sum(row.stalled for row in training_rows),
        "imagination": imagination,
        "elapsed_seconds": time.perf_counter() - started,
        "trace_files": {
            "all_decisions": f"decision_trace_aassr_imagination_margin_{args.margin:g}.jsonl",
            "episodes": f"episode_trace_aassr_imagination_margin_{args.margin:g}.jsonl",
            "switches": f"switch_trace_aassr_imagination_margin_{args.margin:g}.jsonl",
            "interventions": f"intervention_trace_aassr_imagination_margin_{args.margin:g}.jsonl",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
