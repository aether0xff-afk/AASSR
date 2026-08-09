from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .dreamerv3_baseline import (
    DREAMERV3_CONDITION,
    DREAMERV3_UPSTREAM_COMMIT,
)


CURRENT_FINAL_SUITE_VERSION = "aassr-current-generation-suite-v1-dreamerv3"
CURRENT_FINAL_EXPERIMENT_CONDITIONS: tuple[str, ...] = (
    "dqn_raw",
    "dqn_relational",
    DREAMERV3_CONDITION,
    "aassr_current_no_imagination",
    "aassr_current_full",
)
CANONICAL_DREAMERV3_PRESET = "dmc_proprio+size1m"
CANONICAL_DREAMERV3_TRAIN_RATIO = 1024.0
CANONICAL_DREAMERV3_COMPUTE_DTYPE = "bfloat16"
CANONICAL_DREAMERV3_JAX_PLATFORM = "cuda"
CANONICAL_DREAMERV3_ACTION_SPACE = "categorical-relational-slot-240"
CANONICAL_DREAMERV3_ACTION_ADAPTER = "nearest-current-relational-slot"


def _require_equal(label: str, left: Any, right: Any) -> None:
    if left != right:
        raise ValueError(
            f"five-condition suite mismatch for {label}: {left!r} != {right!r}"
        )


def _require_true(label: str, value: Any) -> None:
    if value is not True:
        raise ValueError(f"five-condition suite requires {label}=true")


def _validate_current_summary(current_summary: Mapping[str, Any]) -> None:
    """Require every local checkpoint/evaluation contract before final assembly."""

    if tuple(current_summary.get("experiment_conditions", ())) != (
        "dqn_raw",
        "dqn_relational",
        "aassr_current_no_imagination",
        "aassr_current_full",
    ):
        raise ValueError("current artifact does not contain the canonical four local rows")
    _require_equal(
        "current_training_checkpoint_count",
        int(current_summary.get("training_checkpoint_count", -1)),
        3,
    )
    budget = int(current_summary.get("transition_budget_per_training_condition", -1))
    if budget <= 0:
        raise ValueError("current artifact has an invalid training transition budget")
    _require_equal(
        "current_nominal_total_training_transitions",
        int(current_summary.get("nominal_total_training_transitions", -1)),
        budget * 3,
    )

    for condition in ("dqn_raw", "dqn_relational"):
        row = current_summary.get(condition, {})
        _require_true(f"{condition}.exact_budget", row.get("exact_budget"))
        _require_equal(
            f"{condition}.transitions_used",
            int(row.get("transitions_used", -1)),
            budget,
        )
        _require_true(
            f"{condition}.learning_frozen_in_evaluation",
            row.get("learning_frozen_in_evaluation"),
        )
        _require_true(f"{condition}.dqn_only", row.get("dqn_only"))

    aassr = current_summary.get("aassr", {})
    _require_true("aassr.exact_budget", aassr.get("exact_budget"))
    _require_equal(
        "aassr.transitions_used",
        int(aassr.get("transitions_used", -1)),
        budget,
    )
    _require_true("aassr.same_checkpoint_comparison", aassr.get("same_checkpoint_comparison"))
    _require_equal("aassr.training_imagination", aassr.get("training_imagination"), False)
    _require_true("current.validation_learning_frozen", current_summary.get("validation_learning_frozen"))
    _require_true("current.diagnostic_learning_frozen", current_summary.get("diagnostic_learning_frozen"))
    _require_true("current.diagnostic_full_stage_sweep", current_summary.get("diagnostic_full_stage_sweep"))
    _require_equal("current.final_blind_consumed", current_summary.get("final_blind_consumed"), False)

    control = current_summary.get("control_contract", {})
    for key in (
        "same_initial_network_seed",
        "same_network_shape",
        "same_sparse_reward",
        "same_environment_and_action_surface",
        "same_curriculum_rule",
        "same_seed_pools",
        "independent_adaptive_curriculum_per_training_checkpoint",
    ):
        _require_true(f"current.control_contract.{key}", control.get(key))


def _validate_canonical_dreamer(dreamer_summary: Mapping[str, Any]) -> None:
    if dreamer_summary.get("condition") != DREAMERV3_CONDITION:
        raise ValueError(
            "Dreamer artifact is not the current relational DreamerV3 condition"
        )
    _require_true("dreamerv3.exact_budget", dreamer_summary.get("exact_budget"))

    upstream = dreamer_summary.get("official_upstream", {})
    if upstream.get("actual_commit") != DREAMERV3_UPSTREAM_COMMIT:
        raise ValueError(
            "canonical suite requires the pinned official DreamerV3 commit"
        )
    _require_true("dreamerv3.commit_matches_pin", upstream.get("commit_matches_pin"))
    _require_equal("dreamerv3.upstream_agent_modified", upstream.get("upstream_agent_modified"), False)
    _require_equal("dreamerv3.oracle_information", upstream.get("oracle_information"), False)
    _require_equal(
        "dreamerv3_actor_action_space",
        upstream.get("actor_action_space"),
        CANONICAL_DREAMERV3_ACTION_SPACE,
    )
    _require_equal(
        "dreamerv3_legal_action_adapter",
        upstream.get("legal_action_adapter"),
        CANONICAL_DREAMERV3_ACTION_ADAPTER,
    )
    _require_equal("dreamerv3_slot_count", int(upstream.get("slot_count", -1)), 240)

    config = dreamer_summary.get("official_config", {})
    _require_equal("dreamerv3_preset", config.get("preset"), CANONICAL_DREAMERV3_PRESET)
    _require_equal("dreamerv3_model_size", config.get("model_size"), "size1m")
    _require_equal(
        "dreamerv3_train_ratio",
        float(config.get("train_ratio", float("nan"))),
        CANONICAL_DREAMERV3_TRAIN_RATIO,
    )
    _require_equal(
        "dreamerv3_compute_dtype",
        str(config.get("compute_dtype")),
        CANONICAL_DREAMERV3_COMPUTE_DTYPE,
    )
    _require_equal(
        "dreamerv3_jax_platform",
        str(config.get("jax_platform")),
        CANONICAL_DREAMERV3_JAX_PLATFORM,
    )

    hardware = dreamer_summary.get("jax_hardware", {})
    _require_true("dreamerv3.jax_hardware.accelerator_required", hardware.get("accelerator_required"))
    _require_true("dreamerv3.jax_hardware.accelerator_present", hardware.get("accelerator_present"))
    actual_platforms = tuple(str(item) for item in hardware.get("actual_platforms", ()))
    if "gpu" not in actual_platforms:
        raise ValueError(
            f"canonical DreamerV3 artifact did not actually run on a JAX GPU: {actual_platforms}"
        )

    sparse = dreamer_summary.get("sparse_reward", {})
    _require_equal("dreamerv3_success_reward", float(sparse.get("success", 99)), 1.0)
    _require_equal("dreamerv3_failure_reward", float(sparse.get("failure", 99)), -1.0)
    _require_equal("dreamerv3_other_reward", float(sparse.get("otherwise", 99)), 0.0)
    _require_true(
        "dreamerv3.bootstrap_cut_on_episode_boundary",
        dreamer_summary.get("bootstrap_cut_on_episode_boundary"),
    )
    _require_true("dreamerv3.validation_learning_frozen", dreamer_summary.get("validation_learning_frozen"))
    _require_true("dreamerv3.diagnostic_learning_frozen", dreamer_summary.get("diagnostic_learning_frozen"))
    _require_equal(
        "dreamerv3_train_ratio_step_semantics",
        dreamer_summary.get("official_train_ratio_step_semantics"),
        "embodied-driver-step-including-is_first",
    )
    if int(dreamer_summary.get("dreamer_driver_steps", -1)) < int(
        dreamer_summary.get("transitions_used", 0)
    ):
        raise ValueError("DreamerV3 driver-step clock is below its real-transition count")


def assemble_current_generation_suite(
    current_summary: Mapping[str, Any],
    dreamer_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Join independent PyTorch/AASSR and official DreamerV3 checkpoints.

    Heavy runtimes intentionally execute in separate processes. Final assembly is
    the hard scientific boundary: debug/CPU/retuned Dreamer results, partial current
    sweeps, budget drift, evaluation mutation, and mismatched seed/environment
    contracts are rejected rather than merely annotated.
    """

    _validate_current_summary(current_summary)
    _validate_canonical_dreamer(dreamer_summary)

    _require_equal(
        "research_seed",
        int(current_summary["research_seed"]),
        int(dreamer_summary["research_seed"]),
    )
    budget = int(current_summary["transition_budget_per_training_condition"])
    _require_equal("real_transition_budget", budget, int(dreamer_summary["transitions_used"]))
    _require_equal("train_seeds", current_summary["train_seeds"], dreamer_summary["train_seeds"])
    _require_equal(
        "validation_seeds", current_summary["validation_seeds"], dreamer_summary["validation_seeds"]
    )
    _require_equal(
        "diagnostic_seeds", current_summary["diagnostic_seeds"], dreamer_summary["diagnostic_seeds"]
    )
    _require_equal("stage_manifest", current_summary["stage_manifest"], dreamer_summary["stage_manifest"])
    _require_equal("dreamerv3.final_blind_consumed", dreamer_summary.get("final_blind_consumed"), False)

    current_successes = dict(current_summary["diagnostic_successes"])
    current_frontiers = dict(current_summary["frontier"])
    successes = {
        "dqn_raw": int(current_successes["dqn_raw"]),
        "dqn_relational": int(current_successes["dqn_relational"]),
        DREAMERV3_CONDITION: int(dreamer_summary["diagnostic_successes"]),
        "aassr_current_no_imagination": int(current_successes["aassr_current_no_imagination"]),
        "aassr_current_full": int(current_successes["aassr_current_full"]),
    }
    frontiers = {
        "dqn_raw": current_frontiers["dqn_raw"],
        "dqn_relational": current_frontiers["dqn_relational"],
        DREAMERV3_CONDITION: dreamer_summary["frontier"],
        "aassr_current_no_imagination": current_frontiers["aassr_current_no_imagination"],
        "aassr_current_full": current_frontiers["aassr_current_full"],
    }

    return {
        "suite_version": CURRENT_FINAL_SUITE_VERSION,
        "architecture_version": current_summary["architecture_version"],
        "experiment_conditions": list(CURRENT_FINAL_EXPERIMENT_CONDITIONS),
        "research_seed": int(current_summary["research_seed"]),
        "transition_budget_per_training_checkpoint": budget,
        "training_checkpoint_count": 4,
        "nominal_total_training_transitions": budget * 4,
        "diagnostic_successes": successes,
        "frontier": frontiers,
        "comparison_contract": {
            "dqn_raw_to_relational": "relational-representation-effect",
            "dqn_relational_to_dreamerv3": "matched-sample-dreamerv3-algorithm-comparison",
            "dqn_relational_to_aassr_no_imagination": "aassr-non-imagination-stack-comparison",
            "aassr_no_imagination_to_full": "same-checkpoint-aassr-imagination-marginal-effect",
            "dreamerv3_to_aassr_full": "model-based-imagination-family-comparison",
            "same_sparse_reward": True,
            "same_real_transition_budget": True,
            "same_seed_pools": True,
            "same_stage_manifest": True,
            "same_response_causal_environment": True,
            "dreamerv3_upstream_algorithm_modified": False,
            "dreamerv3_dynamic_action_adapter": (
                "relational-categorical-nearest-legal-slot-projection"
            ),
            "dreamerv3_actor_action_space": CANONICAL_DREAMERV3_ACTION_SPACE,
            "dreamerv3_canonical_preset": CANONICAL_DREAMERV3_PRESET,
            "dreamerv3_canonical_train_ratio": CANONICAL_DREAMERV3_TRAIN_RATIO,
            "dreamerv3_canonical_compute_dtype": CANONICAL_DREAMERV3_COMPUTE_DTYPE,
            "dreamerv3_canonical_jax_platform": CANONICAL_DREAMERV3_JAX_PLATFORM,
            "dreamerv3_actual_gpu_required": True,
        },
        "current": dict(current_summary),
        "dreamerv3": dict(dreamer_summary),
        "final_blind_consumed": False,
    }


def assemble_current_generation_suite_files(
    current_summary_path: str | Path,
    dreamer_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    current_summary = json.loads(Path(current_summary_path).read_text(encoding="utf-8"))
    dreamer_summary = json.loads(Path(dreamer_summary_path).read_text(encoding="utf-8"))
    result = assemble_current_generation_suite(current_summary, dreamer_summary)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
