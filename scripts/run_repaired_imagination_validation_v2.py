from __future__ import annotations

import json
import sys
from pathlib import Path

import analyze_repaired_imagination_trace as analyzer
import run_imagination_gate_ablation as gate
import run_imagination_intervention_trace as detail
import run_imagination_intervention_trace_verbose as verbose

from aassr_v2.current_mixture_entrypoint import (
    MIXTURE_CURRENT_COMPONENTS,
    build_current_mixture_pentest_aassr_core,
)


RUN_VERSION = "repaired-imagination-one-shot-validation-v2-mixture"


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
    # no condition can silently fall back to the deterministic-v1 world model.
    gate.build_current_pentest_aassr_core = build_current_mixture_pentest_aassr_core
    detail.gate = gate


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
        "input": "relational-state-v2+relational-action",
        "output": "conditional-mixture(relational-next-state-v2,legal-mask,terminal)",
        "mixture_components": 3,
        "reliability": "ensemble-mode-set-disagreement*semantic-holdout-calibration",
        "outcome_probability": "separate-never-value-bonus",
        "exact_multimodal_replay": "empirical-frequency-override-when-available",
    }
    summary["gate_contract"] = {
        "minimum_coverage": old_gate.get("minimum_coverage"),
        "confidence_role": "reliability_gate_only",
        "planner_uncertainty_penalty": 0.0,
        "critic_confidence_input": "constant",
        "intervention_margin": float(margin),
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
    margin = float(_arg_value("--margin", "0.05"))
    _install_frozen_builder()
    verbose.main()
    diagnostics = analyzer.analyze(output)
    _patch_summary(output, diagnostics, margin)
    print("\n[REPAIRED V2 ANALYSIS]")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"[REPAIRED V2 SUMMARY] {output / 'summary.json'}")


if __name__ == "__main__":
    main()
