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


def _validate_canonical_dreamer(dreamer_summary: Mapping[str, Any]) -> None:
    if dreamer_summary.get("condition") != DREAMERV3_CONDITION:
        raise ValueError(
            "Dreamer artifact is not the current relational DreamerV3 condition"
        )
    if not bool(dreamer_summary.get("exact_budget")):
        raise ValueError(
            "Dreamer artifact did not consume the exact real-transition budget"
        )

    upstream = dreamer_summary.get("official_upstream", {})
    if upstream.get("actual_commit") != DREAMERV3_UPSTREAM_COMMIT:
        raise ValueError(
            "canonical suite requires the pinned official DreamerV3 commit"
        )
    if not bool(upstream.get("commit_matches_pin")):
        raise ValueError("Dreamer artifact reports an upstream pin mismatch")
    if bool(upstream.get("upstream_agent_modified", True)):
        raise ValueError("canonical suite requires an unmodified DreamerV3 agent")
    if bool(upstream.get("oracle_information", True)):
        raise ValueError("canonical DreamerV3 adapter must not use oracle information")
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
    _require_equal(
        "dreamerv3_preset",
        config.get("preset"),
        CANONICAL_DREAMERV3_PRESET,
    )
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

    sparse = dreamer_summary.get("sparse_reward", {})
    _require_equal("dreamerv3_success_reward", float(sparse.get("success", 99)), 1.0)
    _require_equal("dreamerv3_failure_reward", float(sparse.get("failure", 99)), -1.0)
    _require_equal("dreamerv3_other_reward", float(sparse.get("otherwise", 99)), 0.0)
    if not bool(dreamer_summary.get("bootstrap_cut_on_episode_boundary")):
        raise ValueError("DreamerV3 did not use the corrected episode-boundary contract")
    if not bool(dreamer_summary.get("validation_learning_frozen")):
        raise ValueError("DreamerV3 validation was not frozen")
    if not bool(dreamer_summary.get("diagnostic_learning_frozen")):
        raise ValueError("DreamerV3 diagnostic was not frozen")
    _require_equal(
        "dreamerv3_train_ratio_step_semantics",
        dreamer_summary.get("official_train_ratio_step_semantics"),
        "embodied-driver-step-including-is_first",
    )


def assemble_current_generation_suite(
    current_summary: Mapping[str, Any],
    dreamer_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Join independent PyTorch/AASSR and official DreamerV3 checkpoints.

    The two heavy runtimes intentionally execute in separate processes. This
    assembler refuses to combine artifacts unless their scientific contracts are
    identical on seed pools, environment stages, sparse-reward budget, and final
    blind status. It also rejects debug/CPU/retuned Dreamer artifacts so a smoke
    run cannot accidentally become the canonical five-condition result.
    """

    _validate_canonical_dreamer(dreamer_summary)

    _require_equal(
        "research_seed",
        int(current_summary["research_seed"]),
        int(dreamer_summary["research_seed"]),
    )
    budget = int(current_summary["transition_budget_per_training_condition"])
    _require_equal(
        "real_transition_budget",
        budget,
        int(dreamer_summary["transitions_used"]),
    )
    _require_equal(
        "train_seeds",
        current_summary["train_seeds"],
        dreamer_summary["train_seeds"],
    )
    _require_equal(
        "validation_seeds",
        current_summary["validation_seeds"],
        dreamer_summary["validation_seeds"],
    )
    _require_equal(
        "diagnostic_seeds",
        current_summary["diagnostic_seeds"],
        dreamer_summary["diagnostic_seeds"],
    )
    _require_equal(
        "stage_manifest",
        current_summary["stage_manifest"],
        dreamer_summary["stage_manifest"],
    )
    if bool(current_summary.get("final_blind_consumed")) or bool(
        dreamer_summary.get("final_blind_consumed")
    ):
        raise ValueError("final blind seeds must remain unconsumed")
    if not bool(current_summary.get("diagnostic_full_stage_sweep")):
        raise ValueError(
            "canonical five-condition suite requires the full diagnostic stage sweep"
        )

    current_successes = dict(current_summary["diagnostic_successes"])
    current_frontiers = dict(current_summary["frontier"])
    successes = {
        "dqn_raw": int(current_successes["dqn_raw"]),
        "dqn_relational": int(current_successes["dqn_relational"]),
        DREAMERV3_CONDITION: int(dreamer_summary["diagnostic_successes"]),
        "aassr_current_no_imagination": int(
            current_successes["aassr_current_no_imagination"]
        ),
        "aassr_current_full": int(current_successes["aassr_current_full"]),
    }
    frontiers = {
        "dqn_raw": current_frontiers["dqn_raw"],
        "dqn_relational": current_frontiers["dqn_relational"],
        DREAMERV3_CONDITION: dreamer_summary["frontier"],
        "aassr_current_no_imagination": current_frontiers[
            "aassr_current_no_imagination"
        ],
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
            "dqn_relational_to_dreamerv3": (
                "official-world-model-imagined-actor-critic-effect"
            ),
            "dqn_relational_to_aassr_no_imagination": (
                "aassr-non-imagination-stack-effect"
            ),
            "aassr_no_imagination_to_full": (
                "same-checkpoint-aassr-imagination-marginal-effect"
            ),
            "dreamerv3_to_aassr_full": (
                "model-based-imagination-family-comparison"
            ),
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
    current_summary = json.loads(
        Path(current_summary_path).read_text(encoding="utf-8")
    )
    dreamer_summary = json.loads(
        Path(dreamer_summary_path).read_text(encoding="utf-8")
    )
    result = assemble_current_generation_suite(current_summary, dreamer_summary)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
