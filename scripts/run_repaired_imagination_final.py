from __future__ import annotations

import subprocess
import sys

import run_repaired_imagination_validation_v2 as validation


PREFLIGHT_TESTS = (
    "tests/test_current_generation.py",
    "tests/test_current_repair.py",
    "tests/test_current_repair_action_surface.py",
    "tests/test_current_repair_multimodal.py",
    "tests/test_current_outcome_probability.py",
    "tests/test_current_semantic_evaluator.py",
    "tests/test_current_relational_state_v2.py",
    "tests/test_current_relational_state_v3.py",
    "tests/test_current_relational_import_order.py",
    "tests/test_current_status_supervision.py",
    "tests/test_current_planner_structural_branching.py",
    "tests/test_current_planner_probability_backup.py",
    "tests/test_current_imagination_state_identity.py",
    "tests/test_current_mixture_generation.py",
    "tests/test_current_confidence_gate.py",
    "tests/test_current_critic_support.py",
    "tests/test_current_root_dedup.py",
)


def _run_preflight() -> None:
    compile_targets = (
        "scripts/run_repaired_imagination_validation_v2.py",
        "scripts/analyze_repaired_imagination_trace.py",
        "src/aassr_v2/current_relational_state.py",
        "src/aassr_v2/current_relational_state_v3.py",
        "src/aassr_v2/current_relational_decode_v2.py",
        "src/aassr_v2/current_relational_mixture_model.py",
        "src/aassr_v2/current_status_models.py",
        "src/aassr_v2/current_mixture_entrypoint.py",
        "src/aassr_v2/current_semantic_calibration.py",
        "src/aassr_v2/current_semantic_evaluator.py",
        "src/aassr_v2/current_return_critic.py",
        "src/aassr_v2/current_critic_support.py",
        "src/aassr_v2/current_root_dedup.py",
        "src/aassr_v2/current_planner.py",
        "src/aassr_v2/current_repair.py",
    )
    print("[PREFLIGHT] compile repaired current-generation stack")
    subprocess.run(
        [sys.executable, "-m", "py_compile", *compile_targets],
        check=True,
    )

    print("[PREFLIGHT] run repaired contract tests")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *PREFLIGHT_TESTS],
        check=True,
    )
    print("[PREFLIGHT] PASS — starting the single 2k validation")


def main() -> None:
    _run_preflight()
    validation.main()


if __name__ == "__main__":
    main()
