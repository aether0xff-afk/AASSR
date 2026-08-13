from __future__ import annotations

import importlib.util
from pathlib import Path

from aassr_v2.current_entrypoint import CURRENT_INTERVENTION_MARGIN


def _load_gate_ablation_script() -> object:
    path = Path(__file__).parents[1] / "scripts" / "run_imagination_gate_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "run_imagination_gate_ablation_metadata_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_ablation_formula_metadata_uses_the_configured_margin() -> None:
    module = _load_gate_ablation_script()

    assert module._gate_contract_formula(CURRENT_INTERVENTION_MARGIN) == (
        "required_advantage = 0.05 + lambda * (1 - coverage); "
        "coverage eligibility remains unchanged"
    )
    assert "0.125" in module._gate_contract_formula(0.125)
