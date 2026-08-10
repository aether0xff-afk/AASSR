from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import run_imagination_gate_ablation as gate
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES


CALIBRATION_VERSION = "imagination-gate-base-margin-calibration-v1"
DEFAULT_MARGINS = (0.10, 0.075, 0.05, 0.025, 0.0)
# Deliberately disjoint from training, curriculum-validation, and final diagnostic
# pools. These seeds are used only to select the gate threshold.
GATE_CALIBRATION_SEEDS = tuple(range(94_001, 94_009))


_ORIGINAL_EVALUATE_VARIANT = gate._evaluate_variant


def _margin_variant(
    agent: object,
    *,
    research_seed: int,
    diagnostic_seeds,
    diagnostic_stage_indices,
    transition_budget: int,
    condition: str,
    use_imagination: bool,
    uncertainty_margin: float,
) -> dict[str, Any]:
    """Interpret the ablation value as the fixed base margin, with lambda=0."""

    if not use_imagination:
        return _ORIGINAL_EVALUATE_VARIANT(
            agent,
            research_seed=research_seed,
            diagnostic_seeds=diagnostic_seeds,
            diagnostic_stage_indices=diagnostic_stage_indices,
            transition_budget=transition_budget,
            condition="aassr_no_imagination",
            use_imagination=False,
            uncertainty_margin=0.0,
        )

    margin = float(uncertainty_margin)
    original_config = agent.config
    label = str(margin).replace(".", "p")
    calibrated_condition = f"aassr_gate_margin_{label}"
    agent.config = replace(
        original_config,
        imagination_intervention_margin=margin,
        imagination_uncertainty_margin=0.0,
    )
    try:
        result = _ORIGINAL_EVALUATE_VARIANT(
            agent,
            research_seed=research_seed,
            diagnostic_seeds=diagnostic_seeds,
            diagnostic_stage_indices=diagnostic_stage_indices,
            transition_budget=transition_budget,
            condition=calibrated_condition,
            use_imagination=True,
            uncertainty_margin=0.0,
        )
    finally:
        agent.config = original_config

    result["calibration_margin"] = margin
    result["gate_formula"] = "required_advantage = calibration_margin"
    return result


def _margin_of(variant: dict[str, Any]) -> float:
    return float(variant["calibration_margin"])


def _selection_key(variant: dict[str, Any]) -> tuple[float, ...]:
    """Predeclared lexicographic selector; final tie prefers conservatism."""

    return (
        float(variant["successes"]),
        -float(variant["failures"]),
        -float(variant["stalls"]),
        -float(variant["truncations"]),
        _margin_of(variant),
    )


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated margin list")
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("margins must be non-negative")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the hard-coded Imagination intervention margin on a "
            "dedicated seed split while keeping lambda=0 and one frozen checkpoint."
        )
    )
    parser.add_argument("--output-dir", default="runs/imagination_gate_margin_calibration")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transitions", type=int, default=10_000)
    parser.add_argument("--block-target", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margins", type=_parse_float_list, default=DEFAULT_MARGINS)
    parser.add_argument(
        "--calibration-max-level",
        type=int,
        default=len(TRANSFER_STAGES) - 1,
    )
    parser.add_argument(
        "--calibration-seed-count",
        type=int,
        default=len(GATE_CALIBRATION_SEEDS),
    )
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--allow-critic-not-ready", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.calibration_max_level < len(TRANSFER_STAGES):
        parser.error("--calibration-max-level is outside the transfer-stage range")
    if not 1 <= args.calibration_seed_count <= len(GATE_CALIBRATION_SEEDS):
        parser.error("--calibration-seed-count is outside the dedicated seed pool")

    # Reuse the audited same-checkpoint runner, but reinterpret its swept value
    # as the fixed base margin. No final diagnostic seed is consumed here.
    gate._evaluate_variant = _margin_variant
    result = gate.run_gate_ablation(
        args.output_dir,
        research_seed=int(args.seed),
        transition_budget=int(args.transitions),
        block_target=int(args.block_target),
        diagnostic_seeds=GATE_CALIBRATION_SEEDS[: args.calibration_seed_count],
        diagnostic_stage_indices=tuple(range(args.calibration_max_level + 1)),
        uncertainty_margins=tuple(float(value) for value in args.margins),
        device=args.device,
        allow_tf32=not args.no_tf32,
        require_critic_ready=not args.allow_critic_not_ready,
    )

    candidates = [
        variant
        for variant in result["variants"]
        if variant.get("use_imagination") and "calibration_margin" in variant
    ]
    if not candidates:
        raise RuntimeError("margin calibration produced no Imagination candidates")
    selected = max(candidates, key=_selection_key)

    result["version"] = CALIBRATION_VERSION
    result["calibration_contract"] = {
        "purpose": "select_hard_coded_imagination_intervention_margin",
        "formula": "required_advantage = margin",
        "coverage_uncertainty_lambda": 0.0,
        "minimum_coverage_unchanged": True,
        "margins": [float(value) for value in args.margins],
        "calibration_seeds": list(
            GATE_CALIBRATION_SEEDS[: args.calibration_seed_count]
        ),
        "final_diagnostic_seeds_consumed": False,
        "selection_rule": (
            "maximize successes; then minimize failures; then stalls; then "
            "truncations; if still tied choose the larger margin"
        ),
    }
    result["selected_margin"] = _margin_of(selected)
    result["selected_condition"] = selected["condition"]
    result.pop("gate_contract", None)
    result.pop("diagnostic_seeds", None)
    result["calibration_stage_indices"] = result.pop("diagnostic_stage_indices")

    output = Path(args.output_dir)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
