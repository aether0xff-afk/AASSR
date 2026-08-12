from __future__ import annotations

import json
import sys
from pathlib import Path

import analyze_repaired_imagination_trace as analyzer
import run_imagination_gate_ablation as gate
import run_imagination_intervention_trace as detail
import run_imagination_intervention_trace_verbose as verbose

from aassr_v2.current_entrypoint import CURRENT_INTERVENTION_MARGIN
from aassr_v2.current_mixture_entrypoint import (
    MIXTURE_CURRENT_COMPONENTS,
    build_current_mixture_pentest_aassr_core,
)
from aassr_v2.current_status_models import STATUS_OBJECTIVE


RUN_VERSION = "repaired-imagination-one-shot-validation-v5-pre10k-audited"


def _arg_value(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def _install_frozen_builder() -> None:
    # The detailed runner delegates all training/evaluation construction to the
    # gate module. Patch that single constructor before any checkpoint is built so
    # no condition can silently fall back to a historical world model.
    gate.build_current_pentest_aassr_core = build_current_mixture_pentest_aassr_core
    detail.gate = gate

    # Historical trace serialization predates stochastic outcome mass and records
    # only Prediction.probability (reliability). Preserve both fields so an
    # expensive run can verify the complete chance distribution separately from
    # calibrated reliability.
    original_prediction = detail._prediction

    def probability_aware_prediction(prediction: object, before: object) -> dict[str, object]:
        row = original_prediction(prediction, before)
        row["outcome_probability"] = float(
            getattr(prediction, "outcome_probability", 1.0)
        )
        return row

    detail._prediction = probability_aware_prediction


def _patch_summary(output: Path, diagnostics: dict[str, object], margin: float) -> None:
    path = output / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"validation summary was not produced: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    old_gate = dict(summary.get("gate_contract") or {})
    summary["version"] = RUN_VERSION
    summary["current_components"] = dict(MIXTURE_CURRENT_COMPONENTS)
    summary["frozen_builder"] = "build_current_mixture_pentest_aassr_core"
    summary["prophecy_contract"] = {
        "input": "relational-public-state-v3+latest-http-status+relational-action",
        "output": (
            "conditional-mixture(relational-next-state-v3+categorical-latest-http-status,"
            "legal-mask,active-success-failure-truncation)"
        ),
        "mixture_components": 3,
        "status_objective": STATUS_OBJECTIVE,
        "status_inference": "softmax-categorical-realized-one-hot",
        "status_balancing": "inverse-sqrt-frequency-capped-normalized",
        "status_task_specific_rules": "none",
        "reliability": (
            "ensemble-mode-set-disagreement*state-local-semantic-holdout-calibration"
        ),
        "outcome_probability": "complete-probability-weighted-chance-backup",
        "mode_identity": "status+legal-action-surface+terminal-preserving",
        "decision_backup": "max-over-agent-actions",
        "chance_backup": "expected-external-sparse-return",
        "exact_multimodal_replay": "complete-empirical-frequency-override-when-available",
        "hidden_audit_session_pressure": "masked-by-response-causal-contract",
    }
    summary["gate_contract"] = {
        "minimum_coverage": old_gate.get("minimum_coverage"),
        "coverage_domain": "unique-structural-actions",
        "confidence_role": "reliability_gate_only",
        "planner_uncertainty_penalty": 0.0,
        "critic_confidence_input": "constant",
        "critic_readiness": "positive-and-negative-signed-return-support",
        "intervention_margin": float(margin),
        "canonical_intervention_margin": CURRENT_INTERVENTION_MARGIN,
        "uncertainty_margin": 0.0,
        "formula": "required_advantage = intervention_margin",
        "comparison": "same_frozen_checkpoint_no_imagination_vs_full",
    }
    summary["repair_diagnostics_file"] = "repair_diagnostics.json"
    summary["repair_diagnostics"] = diagnostics
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    output = Path(
        _arg_value(
            "--output-dir",
            "runs/repaired_imagination_validation_v2",
        )
    )
    margin = float(_arg_value("--margin", str(CURRENT_INTERVENTION_MARGIN)))
    _install_frozen_builder()
    verbose.main()
    diagnostics = analyzer.analyze(output)
    _patch_summary(output, diagnostics, margin)
    print("\n[REPAIRED V5 ANALYSIS]")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"[REPAIRED V5 SUMMARY] {output / 'summary.json'}")


if __name__ == "__main__":
    main()
