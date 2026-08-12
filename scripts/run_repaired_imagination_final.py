from __future__ import annotations

import subprocess
import sys

import run_repaired_imagination_validation_v2 as validation


PREFLIGHT_TESTS = (
    "tests/test_current_generation.py",
    "tests/test_current_fresh_process_contract.py",
    "tests/test_current_probability_contract.py",
    "tests/test_current_latent_contracts.py",
    "tests/test_current_checkpoint.py",
    "tests/test_current_core_plugin_boundary.py",
    "tests/test_current_architecture_layers.py",
    "tests/test_current_runtime_performance.py",
    "tests/test_current_runtime_performance_policy.py",
    "tests/test_current_runtime_performance_v2.py",
    "tests/test_current_replay_performance.py",
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
        "scripts/profile_current_runtime_performance.py",
        "scripts/benchmark_current_performance_v2.py",
        "src/aassr_v2/current_core_manifest.py",
        "src/aassr_v2/current_architecture_layers.py",
        "src/aassr_v2/current_plugin_api.py",
        "src/aassr_v2/plugins/current_pentest.py",
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
        "src/aassr_v2/current_decision_optimization.py",
        "src/aassr_v2/current_runtime_performance.py",
        "src/aassr_v2/current_runtime_performance_policy.py",
        "src/aassr_v2/current_runtime_performance_v2.py",
        "src/aassr_v2/current_replay_performance.py",
        "src/aassr_v2/current_checkpoint.py",
        "src/aassr_v2/current_run_artifacts.py",
        "src/aassr_v2/current_planner.py",
        "src/aassr_v2/current_repair.py",
    )
    print("[PREFLIGHT] compile audited core + plugin + performance stack")
    subprocess.run(
        [sys.executable, "-m", "py_compile", *compile_targets],
        check=True,
    )

    print("[PREFLIGHT] run audited current-generation + boundary + performance contracts")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *PREFLIGHT_TESTS],
        check=True,
    )
    print("[PREFLIGHT] PASS — starting the explicitly requested validation run")


def main() -> None:
    _run_preflight()
    validation.main()


if __name__ == "__main__":
    main()
