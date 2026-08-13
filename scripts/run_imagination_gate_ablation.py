from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from statistics import fmean, median
from types import MethodType
from typing import Any, Iterable, Sequence

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_protocol import (
    DEFAULT_CURRENT_BLOCK_TARGET,
    DEFAULT_CURRENT_TRANSITION_BUDGET,
    CurrentEpisodeRow,
    run_current_episode,
    write_current_csv,
)
from aassr_v2.pentest_curriculum_env import STALL_PATIENCE
from aassr_v2.pentest_current_generation_main import (
    ALL_DIAGNOSTIC_STAGE_INDICES,
    CURRENT_CURRICULUM_VALIDATION_SEEDS,
    _learning_counters,
    _run_aassr_frozen_eval,
    _summarize_diagnostic,
    _validate_diagnostic_stages,
    _validate_seed_pools,
)
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_DIAGNOSTIC_SEEDS,
    TRANSFER_STAGES,
    TRANSFER_TRAIN_SEEDS,
    TransferAdaptiveCurriculum,
)
from aassr_v2 import pentest_curriculum_schedule as schedule


ABLATION_VERSION = "imagination-gate-coverage-penalty-v1"
DEFAULT_LAMBDAS = (1.0, 0.5, 0.25, 0.0)


def _gate_contract_formula(intervention_margin: float) -> str:
    """Describe the configured gate without changing its executable values."""

    return (
        f"required_advantage = {float(intervention_margin):g} + "
        "lambda * (1 - coverage); coverage eligibility remains unchanged"
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    left = int(position)
    right = min(len(ordered) - 1, left + 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    return {
        "count": len(materialized),
        "mean": fmean(materialized) if materialized else None,
        "median": median(materialized) if materialized else None,
        "p10": _percentile(materialized, 0.10),
        "p25": _percentile(materialized, 0.25),
        "p75": _percentile(materialized, 0.75),
        "p90": _percentile(materialized, 0.90),
        "min": min(materialized) if materialized else None,
        "max": max(materialized) if materialized else None,
    }


def _install_decision_capture(agent: object) -> list[Any]:
    if hasattr(agent, "_gate_ablation_original_record_decision"):
        return agent._gate_ablation_captured_decisions

    captured: list[Any] = []
    original = agent._record_decision

    def recording(self: object, decision: Any) -> Any:
        captured.append(decision)
        return original(decision)

    agent._gate_ablation_original_record_decision = original
    agent._gate_ablation_captured_decisions = captured
    agent._record_decision = MethodType(recording, agent)
    return captured


def _decision_summary(decisions: Sequence[Any]) -> dict[str, Any]:
    imagined = [item for item in decisions if bool(getattr(item, "used_imagination", False))]
    switch = [
        item
        for item in imagined
        if bool(getattr(item, "imagination_switch_candidate", False))
    ]
    eligible = [
        item
        for item in decisions
        if bool(getattr(item, "imagination_eligible", False))
    ]
    interventions = [
        item
        for item in imagined
        if bool(getattr(item, "imagination_intervention_allowed", False))
    ]
    insufficient = [
        item
        for item in switch
        if getattr(item, "imagination_gate_reason", None) == "insufficient_advantage"
    ]

    def numbers(items: Sequence[Any], name: str) -> list[float]:
        output = []
        for item in items:
            value = getattr(item, name, None)
            if value is not None:
                output.append(float(value))
        return output

    advantages = numbers(switch, "imagination_advantage")
    required = numbers(switch, "imagination_required_advantage")
    gaps = [right - left for left, right in zip(advantages, required, strict=True)]
    return {
        "decisions": len(decisions),
        "eligible": len(eligible),
        "imagination_runs": len(imagined),
        "switch_candidates": len(switch),
        "interventions": len(interventions),
        "insufficient_advantage": len(insufficient),
        "coverage_all_imagined": _distribution(numbers(imagined, "model_coverage")),
        "coverage_switch_candidates": _distribution(numbers(switch, "model_coverage")),
        "advantage_switch_candidates": _distribution(advantages),
        "required_advantage_switch_candidates": _distribution(required),
        "required_minus_observed_switch_candidates": _distribution(gaps),
    }


def _evaluate_variant(
    agent: object,
    *,
    research_seed: int,
    diagnostic_seeds: Sequence[int],
    diagnostic_stage_indices: Sequence[int],
    transition_budget: int,
    condition: str,
    use_imagination: bool,
    uncertainty_margin: float,
) -> dict[str, Any]:
    original_config = agent.config
    agent.config = replace(
        original_config,
        imagination_uncertainty_margin=float(uncertainty_margin),
    )
    captured = _install_decision_capture(agent)
    captured.clear()
    before = _learning_counters(agent)
    rows: list[CurrentEpisodeRow] = []
    started = time.perf_counter()
    try:
        for stage_index in diagnostic_stage_indices:
            stage = TRANSFER_STAGES[int(stage_index)]
            variant_rows = _run_aassr_frozen_eval(
                agent,
                research_seed=research_seed,
                stage_index=int(stage_index),
                scenario_seeds=diagnostic_seeds,
                phase="diagnostic",
                block=-1,
                focus_level=stage.level,
                use_imagination=use_imagination,
                transition_budget=transition_budget,
            )
            rows.extend(replace(row, condition=condition) for row in variant_rows)
    finally:
        agent.config = original_config
    if _learning_counters(agent) != before:
        raise AssertionError(f"{condition} mutated the shared frozen checkpoint")

    summary = _summarize_diagnostic(rows, diagnostic_stage_indices)
    decision_summary = _decision_summary(tuple(captured))
    return {
        "condition": condition,
        "use_imagination": bool(use_imagination),
        "imagination_intervention_margin": float(original_config.imagination_intervention_margin),
        "imagination_uncertainty_margin": float(uncertainty_margin),
        "imagination_minimum_coverage": float(original_config.imagination_minimum_coverage),
        "gate_formula": (
            "required_advantage = intervention_margin + "
            "uncertainty_margin * (1 - coverage)"
        ),
        "episodes": len(rows),
        "successes": sum(row.success for row in rows),
        "success_rate": fmean(row.success for row in rows) if rows else 0.0,
        "failures": sum(row.failure for row in rows),
        "stalls": sum(row.stalled for row in rows),
        "truncations": sum(row.truncation for row in rows),
        "primitive_transitions": sum(row.primitive_transitions for row in rows),
        "imagination_interventions_from_rows": sum(row.imagination_interventions for row in rows),
        "imagination_changed_actions_from_rows": sum(row.imagination_changed_actions for row in rows),
        "diagnostic": summary,
        "decision_diagnostics": decision_summary,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def run_gate_ablation(
    output_dir: str | Path,
    *,
    research_seed: int,
    transition_budget: int = DEFAULT_CURRENT_TRANSITION_BUDGET,
    block_target: int = DEFAULT_CURRENT_BLOCK_TARGET,
    train_seeds: Sequence[int] = TRANSFER_TRAIN_SEEDS,
    validation_seeds: Sequence[int] = CURRENT_CURRICULUM_VALIDATION_SEEDS,
    diagnostic_seeds: Sequence[int] = TRANSFER_DIAGNOSTIC_SEEDS,
    diagnostic_stage_indices: Sequence[int] = ALL_DIAGNOSTIC_STAGE_INDICES,
    uncertainty_margins: Sequence[float] = DEFAULT_LAMBDAS,
    device: str = "cpu",
    allow_tf32: bool = True,
    require_critic_ready: bool = True,
) -> dict[str, Any]:
    if transition_budget <= 0 or block_target <= 0:
        raise ValueError("transition budget and block target must be positive")
    if not uncertainty_margins:
        raise ValueError("at least one uncertainty margin is required")
    if any(float(value) < 0.0 for value in uncertainty_margins):
        raise ValueError("uncertainty margins must be non-negative")
    _validate_seed_pools(train_seeds, validation_seeds, diagnostic_seeds)
    diagnostic_stage_indices = _validate_diagnostic_stages(diagnostic_stage_indices)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    agent = build_current_pentest_aassr_core(
        seed=int(research_seed),
        train_transitions=int(transition_budget),
        use_imagination=True,
        device=device,
        allow_tf32=bool(allow_tf32),
    )
    curriculum = TransferAdaptiveCurriculum()
    training_rows: list[CurrentEpisodeRow] = []
    validation_rows: list[CurrentEpisodeRow] = []
    curriculum_trace: list[dict[str, Any]] = []
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
            scenario_seed = int(
                train_seeds[(block * 97 + episode) % len(train_seeds)]
            )
            natural_cap = max(24, stage.rate_limit + STALL_PATIENCE)
            hard_left = transition_budget - transition_total
            cap = min(natural_cap, hard_left)
            row, consumed = run_current_episode(
                agent,
                condition="aassr_gate_ablation_train",
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
                raise RuntimeError("gate ablation training consumed no transitions")
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

    shared_checkpoint = _learning_counters(agent)
    critic_ready = bool(agent.critic_ready)
    if require_critic_ready and not critic_ready:
        raise RuntimeError(
            "gate ablation is invalid because the learned critic is not ready; "
            "increase the real-transition budget or pass --allow-critic-not-ready "
            "only for runner smoke validation"
        )

    variants: list[dict[str, Any]] = []
    variants.append(
        _evaluate_variant(
            agent,
            research_seed=research_seed,
            diagnostic_seeds=diagnostic_seeds,
            diagnostic_stage_indices=diagnostic_stage_indices,
            transition_budget=transition_budget,
            condition="aassr_no_imagination",
            use_imagination=False,
            uncertainty_margin=float(agent.config.imagination_uncertainty_margin),
        )
    )
    for value in uncertainty_margins:
        label = str(float(value)).replace(".", "p")
        variants.append(
            _evaluate_variant(
                agent,
                research_seed=research_seed,
                diagnostic_seeds=diagnostic_seeds,
                diagnostic_stage_indices=diagnostic_stage_indices,
                transition_budget=transition_budget,
                condition=f"aassr_gate_lambda_{label}",
                use_imagination=True,
                uncertainty_margin=float(value),
            )
        )

    if _learning_counters(agent) != shared_checkpoint:
        raise AssertionError("gate ablation did not preserve one shared checkpoint")

    serializable_variants = []
    for variant in variants:
        rows = variant.pop("rows")
        write_current_csv(output / f"diagnostic_{variant['condition']}.csv", rows)
        serializable_variants.append(variant)

    write_current_csv(output / "training_aassr.csv", training_rows)
    write_current_csv(output / "curriculum_validation_aassr.csv", validation_rows)
    (output / "curriculum_trace_aassr.json").write_text(
        json.dumps(curriculum_trace, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = {
        "version": ABLATION_VERSION,
        "research_seed": int(research_seed),
        "transition_budget": int(transition_budget),
        "exact_budget": transition_total == transition_budget,
        "shared_checkpoint": list(shared_checkpoint),
        "critic_ready": critic_ready,
        "final_focus_level": curriculum.focus_level,
        "training_successes": sum(row.success for row in training_rows),
        "training_stalls": sum(row.stalled for row in training_rows),
        "gate_contract": {
            "minimum_coverage": float(agent.config.imagination_minimum_coverage),
            "intervention_margin": float(agent.config.imagination_intervention_margin),
            "ablated_parameter": "imagination_uncertainty_margin",
            "values": [float(value) for value in uncertainty_margins],
            "formula": _gate_contract_formula(
                agent.config.imagination_intervention_margin
            ),
        },
        "diagnostic_stage_indices": list(map(int, diagnostic_stage_indices)),
        "diagnostic_seeds": list(map(int, diagnostic_seeds)),
        "variants": serializable_variants,
        "learning_frozen_for_all_variants": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Same-checkpoint ablation of the current Imagination coverage penalty."
    )
    parser.add_argument("--output-dir", default="runs/imagination_gate_ablation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transitions", type=int, default=DEFAULT_CURRENT_TRANSITION_BUDGET)
    parser.add_argument("--block-target", type=int, default=DEFAULT_CURRENT_BLOCK_TARGET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambdas", type=_parse_float_list, default=DEFAULT_LAMBDAS)
    parser.add_argument(
        "--diagnostic-max-level",
        type=int,
        default=len(TRANSFER_STAGES) - 1,
    )
    parser.add_argument(
        "--diagnostic-seed-count",
        type=int,
        default=len(TRANSFER_DIAGNOSTIC_SEEDS),
    )
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--allow-critic-not-ready", action="store_true")
    args = parser.parse_args()

    max_level = max(0, min(int(args.diagnostic_max_level), len(TRANSFER_STAGES) - 1))
    seed_count = max(1, min(int(args.diagnostic_seed_count), len(TRANSFER_DIAGNOSTIC_SEEDS)))
    result = run_gate_ablation(
        args.output_dir,
        research_seed=int(args.seed),
        transition_budget=int(args.transitions),
        block_target=int(args.block_target),
        diagnostic_seeds=TRANSFER_DIAGNOSTIC_SEEDS[:seed_count],
        diagnostic_stage_indices=tuple(range(max_level + 1)),
        uncertainty_margins=tuple(args.lambdas),
        device=args.device,
        allow_tf32=not args.no_tf32,
        require_critic_ready=not args.allow_critic_not_ready,
    )

    print(
        "gate ablation complete:",
        f"seed={result['research_seed']}",
        f"transitions={result['transition_budget']}",
        f"critic_ready={result['critic_ready']}",
    )
    for row in result["variants"]:
        decision = row["decision_diagnostics"]
        print(
            row["condition"],
            f"successes={row['successes']}/{row['episodes']}",
            f"switches={decision['switch_candidates']}",
            f"interventions={decision['interventions']}",
            f"insufficient={decision['insufficient_advantage']}",
        )
    print("artifact:", str(Path(args.output_dir) / "summary.json"))


if __name__ == "__main__":
    main()
