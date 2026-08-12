from __future__ import annotations

import json
import sys
from pathlib import Path

import analyze_repaired_imagination_trace as analyzer
import run_imagination_intervention_trace_verbose as verbose


RUN_VERSION = "repaired-imagination-one-shot-validation-v1"


def _arg_value(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def _patch_summary(output: Path, diagnostics: dict[str, object], margin: float) -> None:
    path = output / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"validation summary was not produced: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    old_gate = dict(summary.get("gate_contract") or {})
    summary["version"] = RUN_VERSION
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
    output = Path(_arg_value("--output-dir", "runs/repaired_imagination_validation"))
    margin = float(_arg_value("--margin", "0.05"))
    verbose.main()
    diagnostics = analyzer.analyze(output)
    _patch_summary(output, diagnostics, margin)
    print("\n[REPAIR ANALYSIS]")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"[REPAIR SUMMARY] {output / 'summary.json'}")


if __name__ == "__main__":
    main()
