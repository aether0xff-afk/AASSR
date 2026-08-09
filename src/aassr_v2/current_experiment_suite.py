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


def _require_equal(label: str, left: Any, right: Any) -> None:
    if left != right:
        raise ValueError(f"five-condition suite mismatch for {label}: {left!r} != {right!r}")


def assemble_current_generation_suite(
    current_summary: Mapping[str, Any],
    dreamer_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Join independent PyTorch/AASSR and official DreamerV3 checkpoints.

    The two heavy runtimes intentionally execute in separate processes. This
    assembler refuses to combine artifacts unless their scientific contracts are
    identical on seed pools, environment stages, sparse-reward budget, and final
    blind status. A non-pinned Dreamer checkout is never accepted as a canonical
    five-condition result.
    """

    if dreamer_summary.get("condition") != DREAMERV3_CONDITION:
        raise ValueError("Dreamer artifact is not the current relational DreamerV3 condition")
    if not bool(dreamer_summary.get("exact_budget")):
        raise ValueError("Dreamer artifact did not consume the exact real-transition budget")
    upstream = dreamer_summary.get("official_upstream", {})
    if upstream.get("actual_commit") != DREAMERV3_UPSTREAM_COMMIT:
        raise ValueError("canonical suite requires the pinned official DreamerV3 commit")
    if not bool(upstream.get("commit_matches_pin")):
        raise ValueError("Dreamer artifact reports an upstream pin mismatch")

    _require_equal(
        "research_seed",
        int(current_summary["research_seed"]),
        int(dreamer_summary["research_seed"]),
    )
    budget = int(current_summary["transition_budget_per_training_condition"])
    _require_equal("real_transition_budget", budget, int(dreamer_summary["transitions_used"]))
    _require_equal("train_seeds", current_summary["train_seeds"], dreamer_summary["train_seeds"])
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
        raise ValueError("canonical five-condition suite requires the full diagnostic stage sweep")

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
            "dqn_relational_to_dreamerv3": "official-world-model-imagined-actor-critic-effect",
            "dqn_relational_to_aassr_no_imagination": "aassr-non-imagination-stack-effect",
            "aassr_no_imagination_to_full": "same-checkpoint-aassr-imagination-marginal-effect",
            "dreamerv3_to_aassr_full": "model-based-imagination-family-comparison",
            "same_sparse_reward": True,
            "same_real_transition_budget": True,
            "same_seed_pools": True,
            "same_stage_manifest": True,
            "same_response_causal_environment": True,
            "dreamerv3_upstream_algorithm_modified": False,
            "dreamerv3_dynamic_action_adapter": "relational-continuous-nearest-legal-projection",
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
