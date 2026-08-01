from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .causal_dependency_world import CausalDependencyWorldV2, certify_world
from .causal_diagnostic import diagnostic_one_gates, run_diagnostic_one
from .paper_v2_protocol import (
    V2ArtifactWriter,
    build_run_identity,
    replay_gzip_trace,
    reserve_confirmation_once,
    reserve_run,
    sha256_json,
    v2_run_directory,
)
from .representation_diagnostic import (
    run_diagnostic_two_a,
    run_diagnostic_two_b,
)
from .causal_imagination import ImaginationGateConfig
from .imagination_diagnostic_v2 import run_diagnostic_four
from .open_creativity_v2 import (
    creativity_environment_adequacy,
    freeze_baseline_reference,
    load_frozen_reference,
    run_open_creativity_diagnostic,
)
from .transfer_diagnostic_v2 import run_transfer_diagnostic


@dataclass(frozen=True, slots=True)
class V2RunArtifacts:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    episode_rows: int
    trace_rows: int


def run_paper_v2_suite(
    config: Mapping[str, Any],
    *,
    run_id: str,
    repository_root: str | Path | None = None,
    resume: bool = False,
) -> V2RunArtifacts:
    root = Path(repository_root or Path.cwd()).resolve()
    identity = build_run_identity(config, run_id=run_id, repository_root=root)
    output = v2_run_directory(identity, repository_root=root)
    reserve_confirmation_once(root / "paper_results_v2", identity, resume=resume)
    reserve_run(output, identity, resume=resume)
    raw = output / "raw"
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if resume and (manifests / "protocol_manifest.json").is_file():
        payload = json.loads(
            (manifests / "protocol_manifest.json").read_text(encoding="utf-8")
        )
        return V2RunArtifacts(
            output,
            manifests / "protocol_manifest.json",
            output / "report.md",
            int(payload["episode_rows"]),
            int(payload["trace_rows"]),
        )
    if raw.exists():
        raise FileExistsError("partial v2 raw output requires a future explicit recovery")
    started = datetime.now(timezone.utc).isoformat()
    certifications: dict[str, list[dict[str, Any]]] = {}
    for world_set, seeds in config["world_seeds"].items():
        certifications[str(world_set)] = []
        for seed in seeds:
            certification = certify_world(
                CausalDependencyWorldV2(
                    world_seed=int(seed), reward_mode=str(config["reward_mode"])
                )
            )
            certifications[str(world_set)].append(certification.to_dict())
    settings = config.get("diagnostics", {})
    diagnostic_rows = run_diagnostic_one(
        research_seeds=config["research_seeds"],
        train_world_seeds=config["world_seeds"]["train"],
        training_episodes=int(settings.get("training_episodes", 500)),
        evaluation_episodes=int(settings.get("evaluation_episodes", 100)),
    )
    gates = diagnostic_one_gates(diagnostic_rows)
    diagnostic_two_a = run_diagnostic_two_a(
        research_seeds=config["research_seeds"],
        train_world_seeds=config["world_seeds"]["train"],
        isomorphic_world_seeds=config["world_seeds"]["isomorphic"],
        training_episodes=int(settings.get("training_episodes", 500)),
        evaluation_episodes=int(settings.get("evaluation_episodes", 100)),
    )
    diagnostic_two_b = run_diagnostic_two_b(
        research_seeds=config["research_seeds"],
        train_world_seeds=config["world_seeds"]["train"],
        adaptation_world_seeds=config["world_seeds"]["adaptation"],
        training_episodes=int(settings.get("training_episodes", 500)),
        evaluation_episodes=int(settings.get("evaluation_episodes", 100)),
        budgets=tuple(settings.get("action_remap_probe_budgets", (1, 4, 16))),
    )
    threshold_config = config.get("thresholds", {})
    diagnostic_four, imagination_decisions = run_diagnostic_four(
        research_seeds=config["research_seeds"],
        world_seeds=config["world_seeds"]["unseen_composition"],
        training_episodes=int(settings.get("training_episodes", 500)),
        evaluation_episodes=int(settings.get("evaluation_episodes", 100)),
        gate=ImaginationGateConfig(
            calibration_confidence_minimum=float(
                threshold_config.get("calibration_confidence_minimum", 0.8)
            ),
            uncertainty_maximum=float(
                threshold_config.get("uncertainty_maximum", 0.25)
            ),
            ood_maximum=float(threshold_config.get("ood_maximum", 0.25)),
            minimum_return_margin=float(
                threshold_config.get("imagined_return_margin", 0.05)
            ),
        ),
    )
    transfer_rows, transfer_metrics = run_transfer_diagnostic(
        research_seeds=config["research_seeds"],
        train_world_seeds=config["world_seeds"]["train"],
        unseen_world_seeds=config["world_seeds"]["unseen_composition"],
        pretraining_episodes=int(settings.get("training_episodes", 500)),
        evaluation_episodes=int(settings.get("evaluation_episodes", 100)),
        budgets=tuple(settings.get("adaptation_budgets", (0, 1, 4, 16, 64))),
    )
    creativity_settings = config.get("creativity", {})
    creativity_budget = int(creativity_settings.get("interaction_budget", 500))
    if identity.stage.value == "development_diagnostic":
        reference_path = manifests / "development_baseline_reference.json"
        freeze_baseline_reference(
            reference_path,
            world_seeds=config["world_seeds"]["unseen_composition"],
            interaction_budget=creativity_budget,
        )
    else:
        reference_path = Path(str(creativity_settings["reference_manifest"]))
        if not reference_path.is_absolute():
            reference_path = root / reference_path
    references = load_frozen_reference(reference_path)
    creativity_adequacy = [
        creativity_environment_adequacy(
            world_seed=int(seed), references=references
        )
        for seed in config["world_seeds"]["unseen_composition"]
    ]
    creative_candidates, creativity_metrics = run_open_creativity_diagnostic(
        research_seeds=config["research_seeds"],
        world_seeds=config["world_seeds"]["unseen_composition"],
        references=references,
        interaction_budget=creativity_budget,
    )
    rows: list[dict[str, Any]] = []
    for item in diagnostic_rows:
        source = item.to_dict()
        rows.append(
            {
                "diagnostic": "diagnostic_1",
                "condition": source["condition"],
                "representation": "identity_representation",
                "research_seed": source["research_seed"],
                "adaptation_budget": 0,
                "training_final_tail": source["training_final_tail"],
                "frozen_success": source["frozen_success"],
                "random_success": source["random_success"],
                "success_rate": source["frozen_success"],
                "start_checkpoint_fingerprint": "",
                "evaluation_checkpoint_fingerprint": "",
                "final_checkpoint_fingerprint": "",
                "effect_updates": 0,
            }
        )
    rows.extend(
        {"condition": "encoder_comparison", "training_final_tail": "", "frozen_success": "", "random_success": "", **item.to_dict()}
        for item in (*diagnostic_two_a, *diagnostic_two_b)
    )
    rows.extend(
        {
            "diagnostic": "diagnostic_4",
            "condition": item["condition"],
            "representation": "relational_effect_representation",
            "research_seed": item["research_seed"],
            "adaptation_budget": 0,
            "training_final_tail": "",
            "frozen_success": "",
            "random_success": "",
            "success_rate": item["success_rate"],
            "start_checkpoint_fingerprint": "",
            "evaluation_checkpoint_fingerprint": "",
            "final_checkpoint_fingerprint": "",
            "effect_updates": 0,
            "intervention_rate": item["intervention_rate"],
            "mean_decision_regret": item["mean_decision_regret"],
            "dead_end_entry_rate": item["dead_end_entry_rate"],
            "root_action_optimality": item["root_action_optimality"],
        }
        for item in diagnostic_four["summaries"]
    )
    rows.extend(
        {
            "diagnostic": "diagnostic_3",
            "condition": item.condition,
            "representation": "relational_effect_representation",
            "research_seed": item.research_seed,
            "adaptation_budget": item.adaptation_budget,
            "training_final_tail": "",
            "frozen_success": "",
            "random_success": "",
            "success_rate": item.success_rate,
            "start_checkpoint_fingerprint": item.branch_start_fingerprint,
            "evaluation_checkpoint_fingerprint": item.evaluation_fingerprint_before,
            "final_checkpoint_fingerprint": item.evaluation_fingerprint_after,
            "effect_updates": 0,
        }
        for item in transfer_rows
    )
    rows.extend(
        {
            "diagnostic": "diagnostic_5",
            "condition": "full_aassr",
            "representation": "relational_effect_representation",
            "research_seed": "",
            "adaptation_budget": 0,
            "training_final_tail": "",
            "frozen_success": "",
            "random_success": "",
            "success_rate": "",
            "start_checkpoint_fingerprint": "",
            "evaluation_checkpoint_fingerprint": "",
            "final_checkpoint_fingerprint": "",
            "effect_updates": 0,
            "graph_sha256": item.graph_sha256,
            "novelty_score": item.novelty_score,
            "reusable_success_rate": item.reusable_success_rate,
            "creative_candidate": item.candidate,
        }
        for item in creative_candidates
    )
    columns = (
        "diagnostic",
        "condition",
        "representation",
        "research_seed",
        "adaptation_budget",
        "training_final_tail",
        "frozen_success",
        "random_success",
        "success_rate",
        "start_checkpoint_fingerprint",
        "evaluation_checkpoint_fingerprint",
        "final_checkpoint_fingerprint",
        "effect_updates",
        "intervention_rate",
        "mean_decision_regret",
        "dead_end_entry_rate",
        "root_action_optimality",
        "graph_sha256",
        "novelty_score",
        "reusable_success_rate",
        "creative_candidate",
    )
    with V2ArtifactWriter(raw, columns) as writer:
        for payload in rows:
            writer.write_episode(payload)
            writer.write_trace(
                {
                    "record_type": "seed_summary",
                    **payload,
                    "private_state_included": False,
                    "learning_enabled": False,
                }
            )
        for decision in imagination_decisions:
            writer.write_trace(
                {
                    "record_type": "imagination_decision",
                    **decision,
                    "private_state_included": False,
                    "learning_enabled": False,
                }
            )
        for candidate in creative_candidates:
            writer.write_trace(
                {
                    "record_type": "creativity_graph_summary",
                    **candidate.to_dict(),
                    "private_state_included": False,
                    "learning_enabled": False,
                }
            )
        episode_rows = writer.episode_rows
        trace_rows = writer.trace_rows
    replayed = replay_gzip_trace(raw / "trace.jsonl.gz")
    certification_ok = all(
        bool(item["adequate"])
        for values in certifications.values()
        for item in values
    )
    engineering = {
        "frozen_checkpoint_immutable": bool(gates["checkpoint_immutable"]),
        "evaluation_learning_calls_zero": bool(
            gates["evaluation_learning_calls_zero"]
        ),
        "private_state_leaks_zero": all(
            int(item["private_state_leak_count"]) == 0
            for values in certifications.values()
            for item in values
        ),
        "gzip_trace_replay": len(replayed) == trace_rows,
        "adaptation_branches_share_checkpoint": all(
            len(
                {
                    row.start_checkpoint_fingerprint
                    for row in diagnostic_two_b
                    if row.research_seed == seed and row.representation == representation
                }
            )
            == 1
            for seed in config["research_seeds"]
            for representation in ("identity_representation", "relational_effect_representation")
        ),
        "frozen_representation_evaluation_immutable": all(
            row.evaluation_checkpoint_fingerprint == row.final_checkpoint_fingerprint
            for row in (*diagnostic_two_a, *diagnostic_two_b)
        ),
        "oracle_transition_accuracy_100_percent": abs(
            float(diagnostic_four["engineering"]["oracle_transition_accuracy"])
            - 1.0
        )
        <= 1e-12,
        "oracle_root_action_optimality_at_least_95_percent": float(
            diagnostic_four["engineering"]["oracle_root_action_optimality"]
        )
        >= 0.95,
        "oracle_regret_below_policy": bool(
            diagnostic_four["engineering"]["oracle_regret_below_policy"]
        ),
        "oracle_dead_end_not_above_policy": bool(
            diagnostic_four["engineering"]["oracle_dead_end_not_above_policy"]
        ),
        "random_model_low_confidence_interventions_zero": float(
            diagnostic_four["engineering"]["random_model_intervention_rate"]
        )
        == 0.0,
        "transfer_branches_share_checkpoint": all(
            len(
                {
                    row.branch_start_fingerprint
                    for row in transfer_rows
                    if row.condition == condition and row.research_seed == seed
                }
            )
            == 1
            for condition in {row.condition for row in transfer_rows}
            for seed in config["research_seeds"]
        ),
        "transfer_frozen_evaluation_immutable": all(
            row.evaluation_fingerprint_before == row.evaluation_fingerprint_after
            for row in transfer_rows
        ),
    }
    def mean_success(items: list[Any], representation: str, budget: int) -> float:
        selected = [
            row.success_rate
            for row in items
            if row.representation == representation and row.adaptation_budget == budget
        ]
        return sum(selected) / len(selected) if selected else 0.0

    empirical = {
        "diagnostic_2a_relational_minus_identity": (
            mean_success(diagnostic_two_a, "relational_effect_representation", 0)
            - mean_success(diagnostic_two_a, "identity_representation", 0)
        ),
        "imagination_success_by_condition": {
            condition: sum(
                row["success_rate"]
                for row in diagnostic_four["summaries"]
                if row["condition"] == condition
            )
            / len(config["research_seeds"])
            for condition in {
                row["condition"] for row in diagnostic_four["summaries"]
            }
        },
        "diagnostic_2b_budget16_relational_minus_identity": (
            mean_success(diagnostic_two_b, "relational_effect_representation", 16)
            - mean_success(diagnostic_two_b, "identity_representation", 16)
        ),
        "transfer": transfer_metrics,
        "creativity": creativity_metrics,
    }
    adequacy = {
        "world_certification": certification_ok,
        "contextual_training_above_random": bool(
            gates["contextual_training_above_random"]
        ),
        "contextual_frozen_above_random": bool(
            gates["contextual_frozen_above_random"]
        ),
        "contextual_replay_gap": bool(gates["contextual_replay_gap_within_0_10"]),
        "full_replay_gap": bool(gates["full_replay_gap_within_0_10"]),
        "open_creativity_world_adequate": all(
            bool(item["adequate"]) for item in creativity_adequacy
        ),
    }
    completed = datetime.now(timezone.utc).isoformat()
    manifest = {
        **identity.to_dict(),
        "schema_version": 2,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "episode_rows": episode_rows,
        "trace_rows": trace_rows,
        "trace_replay_count": len(replayed),
        "engineering_integrity_gates": engineering,
        "benchmark_adequacy_gates": adequacy,
        "empirical_hypotheses_are_gates": False,
        "empirical_hypotheses": empirical,
        "diagnostic_1_metrics": gates["metrics"],
        "world_certifications": certifications,
        "creativity_environment_adequacy": creativity_adequacy,
        "creativity_reference_sha256": json.loads(
            reference_path.read_text(encoding="utf-8")
        )["reference_sha256"],
        "config_sha256_runtime": sha256_json(config),
        "final_executed": False,
    }
    manifest_path = manifests / "protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = output / "report.md"
    report_path.write_text(
        "\n".join(
            (
                f"# {identity.protocol_version} — {identity.stage.value}",
                "",
                f"{identity.stage.value} results are not Final performance evidence.",
                "",
                "## Engineering integrity",
                *[f"- {key}: {value}" for key, value in engineering.items()],
                "",
                "## Benchmark adequacy",
                *[f"- {key}: {value}" for key, value in adequacy.items()],
                "",
                "## Diagnostic 1 metrics",
                *[
                    f"- {key}: {value:.6f}"
                    for key, value in gates["metrics"].items()
                ],
                "",
                "Empirical hypotheses are reported, not used as progression gates.",
            )
        ),
        encoding="utf-8",
    )
    return V2RunArtifacts(output, manifest_path, report_path, episode_rows, trace_rows)
