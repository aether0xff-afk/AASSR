from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .aassr_core import (
    CORE_MODULES,
    TRAINABLE_CORE_MODULES,
    AASSRCore,
    AASSRCoreConfig,
)
from .grid_push_creativity import freeze_solver_reference
from .grid_push_plugin import GridPushEnvironmentPlugin
from .grid_push_solver import ProceduralGridPushGenerator
from .grid_push_world import GRID_PUSH_LAW_SHA256, GridPushSpec, GridPushWorld
from .imagination_tree import ImaginationConfig
from .paper_v2_protocol import (
    V2ArtifactWriter,
    build_run_identity,
    implementation_tree_sha256,
    replay_gzip_trace,
    reserve_run,
    v2_run_directory,
)


@dataclass(frozen=True, slots=True)
class GridCoreDevelopmentArtifacts:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    certified_worlds: int
    episode_rows: int


def _module_probe_spec() -> GridPushSpec:
    width, height = 5, 4
    boundary = frozenset(
        (x, y)
        for x in range(width)
        for y in range(height)
        if x in {0, width - 1} or y in {0, height - 1}
    )
    return GridPushSpec(
        width=width,
        height=height,
        walls=boundary,
        start=(1, 1),
        goal=(2, 1),
        blocks=frozenset({(2, 2)}),
        pits=frozenset({(3, 2)}),
        plates=frozenset({(1, 2)}),
        doors=frozenset({(3, 1)}),
        plate_links={(1, 2): ((3, 1),)},
        generator_seed=88001,
    )


def _core_config(settings: Mapping[str, Any], *, probe: bool = False) -> AASSRCoreConfig:
    core = settings["core"]
    return AASSRCoreConfig(
        gamma=float(core["gamma"]),
        epsilon_start=0.0 if probe else float(core["epsilon_start"]),
        epsilon_end=0.0 if probe else float(core["epsilon_end"]),
        epsilon_decay_episodes=int(core["epsilon_decay_episodes"]),
        prophecy_samples=int(core["prophecy_samples"]),
        replay_capacity=int(core["replay_capacity"]),
        holdout_stride=int(core["holdout_stride"]),
        skill_promotion_successes=(
            1 if probe else int(core["skill_promotion_successes"])
        ),
        skill_maximum_length=int(core["skill_maximum_length"]),
        feature_dimension=int(core["feature_dimension"]),
        imagination_minimum_coverage=(
            0.0 if probe else float(core["imagination_minimum_coverage"])
        ),
        imagination=ImaginationConfig(
            branching_factor=int(core["imagination_branching_factor"]),
            maximum_depth=int(core["imagination_depth"]),
            beam_width=int(core["imagination_beam_width"]),
            outcome_samples=int(core["prophecy_samples"]),
            minimum_path_confidence=float(core["minimum_path_confidence"]),
            uncertainty_penalty=float(core["uncertainty_penalty"]),
            aggregation=str(core["imagination_aggregation"]),
            update_policy=False,
        ),
    )


def _fresh_runtime_import_excludes_solver(repository_root: Path) -> bool:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import aassr_v2.grid_push_plugin; "
                "print('aassr_v2.grid_push_solver' in sys.modules)"
            ),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() == "False"


def _plugin_for(specs: Mapping[int, GridPushSpec]) -> GridPushEnvironmentPlugin:
    return GridPushEnvironmentPlugin(lambda seed: specs[int(seed)])


def _run_module_probe(
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = _module_probe_spec()
    specs = {88001: spec}
    core = AASSRCore(config=_core_config(settings, probe=True), seed=88101)
    records = [
        core.run_episode(
            _plugin_for(specs),
            world_seed=88001,
            episode=episode,
            maximum_steps=4,
            training=True,
            phase="development_module_call_probe",
        )
        for episode in range(int(settings["module_probe_episodes"]))
    ]
    checkpoint = core.export_checkpoint()
    clone = AASSRCore.from_checkpoint(checkpoint)
    before = clone.checkpoint_fingerprint()
    frozen = clone.run_episode(
        _plugin_for(specs),
        world_seed=88001,
        episode=0,
        maximum_steps=4,
        training=False,
        phase="evaluation_train_world_frozen",
    )
    after = clone.checkpoint_fingerprint()
    training_audit = core.audit.to_dict()
    frozen_audit = clone.audit.to_dict()
    all_calls = all(training_audit["calls"][name] > 0 for name in CORE_MODULES)
    trainable_updates = all(
        training_audit["learning_updates"][name] > 0
        for name in TRAINABLE_CORE_MODULES
    )
    structural_work = all(
        training_audit["work_units"][name] > 0
        for name in ("goal", "imagination_tree", "delayed_credit_assigner")
    )
    sparse_reward = all(
        step.reward == 0.0 or step.terminal
        for record in (*records, frozen)
        for step in record.transitions
    )
    result = {
        "fixture_role": "engineering_call_probe_not_performance_evidence",
        "episodes": len(records),
        "successes": sum(record.success for record in records),
        "training_audit": training_audit,
        "frozen_audit": frozen_audit,
        "all_core_module_calls_positive": all_calls,
        "all_trainable_module_updates_positive": trainable_updates,
        "structural_module_work_positive": structural_work,
        "holdout_items": len(core.replay.holdout()),
        "skills_promoted": len(core.skills.all()),
        "checkpoint_before_frozen": before,
        "checkpoint_after_frozen": after,
        "frozen_checkpoint_immutable": before == after,
        "frozen_learning_updates_zero": all(
            value == 0 for value in frozen_audit["learning_updates"].values()
        ),
        "checkpoint_private_solver_leak": (
            core.checkpoint_contains_forbidden_environment_data()
        ),
        "strict_sparse_environment_reward": sparse_reward,
    }
    return result, [record.to_dict() for record in (*records, frozen)]


def run_grid_push_core_development(
    config: Mapping[str, Any],
    *,
    run_id: str,
    repository_root: str | Path | None = None,
) -> GridCoreDevelopmentArtifacts:
    root = Path(repository_root or Path.cwd()).resolve()
    identity = build_run_identity(config, run_id=run_id, repository_root=root)
    output = v2_run_directory(identity, repository_root=root)
    reserve_run(output, identity, resume=False)
    settings = config["grid_push_core"]
    started = datetime.now(timezone.utc).isoformat()

    # Analysis-only phase. Solver objects and paths are discarded before an
    # AASSRCore or plugin instance is created.
    generator = ProceduralGridPushGenerator(
        maximum_attempts=int(settings["maximum_generation_attempts"])
    )
    reference_dir = output / "manifests" / "solver_references"
    specs: dict[int, GridPushSpec] = {}
    certifications: list[dict[str, Any]] = []
    for world_seed in settings["certification_world_seeds"]:
        spec, certification, solver_result = generator.generate(
            int(world_seed),
            maximum_actions=int(settings["solver_maximum_actions"]),
            random_rollouts=int(settings["certification_random_rollouts"]),
        )
        specs[int(world_seed)] = spec
        reference = freeze_solver_reference(
            reference_dir / f"world_{int(world_seed)}.json",
            world=GridPushWorld(spec),
            solver_result=solver_result,
            maximum_actions=int(settings["solver_maximum_actions"]),
        )
        row = certification.to_dict()
        row["frozen_reference_sha256"] = reference["reference_sha256"]
        certifications.append(row)
        del solver_result
    references_frozen_at = datetime.now(timezone.utc).isoformat()
    del generator

    module_probe, probe_traces = _run_module_probe(settings)
    agent_started = datetime.now(timezone.utc).isoformat()
    train_worlds = tuple(int(seed) for seed in config["world_seeds"]["train"])[
        : int(settings["training_world_count"])
    ]
    research_summaries: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    for research_seed in config["research_seeds"]:
        core = AASSRCore(
            config=_core_config(settings),
            seed=int(research_seed),
        )
        training = []
        for episode in range(int(settings["training_episodes"])):
            record = core.run_episode(
                _plugin_for(specs),
                world_seed=train_worlds[episode % len(train_worlds)],
                episode=episode,
                maximum_steps=int(settings["episode_maximum_steps"]),
                training=True,
                phase="training",
            )
            training.append(record)
            episode_records.append(
                {"condition": "full_aassr", "research_seed": int(research_seed), **record.to_dict()}
            )
        checkpoint = core.export_checkpoint()
        clone = AASSRCore.from_checkpoint(checkpoint)
        checkpoint_before = clone.checkpoint_fingerprint()
        frozen = []
        for episode in range(int(settings["evaluation_episodes"])):
            record = clone.run_episode(
                _plugin_for(specs),
                world_seed=train_worlds[episode % len(train_worlds)],
                episode=episode,
                maximum_steps=int(settings["episode_maximum_steps"]),
                training=False,
                phase="evaluation_train_world_frozen",
            )
            frozen.append(record)
            episode_records.append(
                {"condition": "full_aassr", "research_seed": int(research_seed), **record.to_dict()}
            )
        checkpoint_after = clone.checkpoint_fingerprint()
        tail = training[-min(20, len(training)) :]
        research_summaries.append(
            {
                "condition": "full_aassr",
                "research_seed": int(research_seed),
                "training_episodes": len(training),
                "training_final_tail_success": fmean(item.success for item in tail),
                "frozen_episodes": len(frozen),
                "frozen_success_rate": fmean(item.success for item in frozen),
                "training_mean_steps": fmean(item.primitive_steps for item in training),
                "frozen_mean_steps": fmean(item.primitive_steps for item in frozen),
                "training_imagination_use_rate": (
                    sum(decision.used_imagination for item in training for decision in item.decisions)
                    / max(1, sum(len(item.decisions) for item in training))
                ),
                "frozen_imagination_use_rate": (
                    sum(decision.used_imagination for item in frozen for decision in item.decisions)
                    / max(1, sum(len(item.decisions) for item in frozen))
                ),
                "checkpoint_before_evaluation": checkpoint_before,
                "checkpoint_after_evaluation": checkpoint_after,
                "checkpoint_immutable": checkpoint_before == checkpoint_after,
                "checkpoint_private_solver_leak": (
                    clone.checkpoint_contains_forbidden_environment_data()
                ),
                "training_audit": core.audit.to_dict(),
                "frozen_audit": clone.audit.to_dict(),
            }
        )

    raw_dir = output / "raw"
    columns = (
        "condition", "research_seed", "phase", "episode", "world_seed",
        "training", "success", "final_sparse_reward", "primitive_steps",
        "decision_count", "imagination_decisions", "imagined_nodes",
    )
    with V2ArtifactWriter(raw_dir, columns) as writer:
        for row in episode_records:
            decisions = row["decisions"]
            writer.write_episode(
                {
                    "condition": row["condition"],
                    "research_seed": row["research_seed"],
                    "phase": row["phase"],
                    "episode": row["episode"],
                    "world_seed": row["world_seed"],
                    "training": row["training"],
                    "success": row["success"],
                    "final_sparse_reward": row["final_sparse_reward"],
                    "primitive_steps": row["primitive_steps"],
                    "decision_count": len(decisions),
                    "imagination_decisions": sum(
                        item["used_imagination"] for item in decisions
                    ),
                    "imagined_nodes": sum(item["imagined_nodes"] for item in decisions),
                }
            )
            writer.write_trace({"record_type": "core_episode", **row})
        for row in probe_traces:
            writer.write_trace({"record_type": "module_call_probe", **row})
        episode_rows = writer.episode_rows
        trace_rows = writer.trace_rows

    (output / "module_call_audit.json").write_text(
        json.dumps(module_probe, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "research_seed_summary.json").write_text(
        json.dumps(research_summaries, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trace_replayed = len(replay_gzip_trace(raw_dir / "trace.jsonl.gz")) == trace_rows
    runtime_isolated = _fresh_runtime_import_excludes_solver(root)
    frozen_updates_zero = all(
        all(value == 0 for value in row["frozen_audit"]["learning_updates"].values())
        for row in research_summaries
    )
    checkpoints_immutable = all(row["checkpoint_immutable"] for row in research_summaries)
    checkpoints_private_free = all(
        not row["checkpoint_private_solver_leak"] for row in research_summaries
    )
    engineering_gates = {
        "all_20_worlds_certified": len(certifications) == 20
        and all(row["adequate"] for row in certifications),
        "solver_references_frozen_before_core": references_frozen_at <= agent_started,
        "runtime_fresh_import_excludes_solver": runtime_isolated,
        "all_core_module_calls_positive": module_probe["all_core_module_calls_positive"],
        "all_trainable_module_updates_positive": module_probe[
            "all_trainable_module_updates_positive"
        ],
        "structural_module_work_positive": module_probe[
            "structural_module_work_positive"
        ],
        "module_probe_frozen_checkpoint_immutable": module_probe[
            "frozen_checkpoint_immutable"
        ],
        "module_probe_frozen_learning_updates_zero": module_probe[
            "frozen_learning_updates_zero"
        ],
        "generated_world_frozen_checkpoints_immutable": checkpoints_immutable,
        "generated_world_frozen_learning_updates_zero": frozen_updates_zero,
        "checkpoint_private_solver_leaks_zero": checkpoints_private_free
        and not module_probe["checkpoint_private_solver_leak"],
        "strict_sparse_environment_reward": module_probe[
            "strict_sparse_environment_reward"
        ],
        "gzip_trace_replays": trace_replayed,
        "only_plugin_core_condition_named_full_aassr": all(
            row["condition"] == "full_aassr" for row in episode_records
        ),
    }
    completed = datetime.now(timezone.utc).isoformat()
    artifacts = {
        "episodes_csv": hashlib.sha256((raw_dir / "episodes.csv").read_bytes()).hexdigest(),
        "trace_jsonl_gz": hashlib.sha256((raw_dir / "trace.jsonl.gz").read_bytes()).hexdigest(),
        "module_call_audit": hashlib.sha256((output / "module_call_audit.json").read_bytes()).hexdigest(),
        "research_seed_summary": hashlib.sha256((output / "research_seed_summary.json").read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": 2,
        **identity.to_dict(),
        "suite": "grid_push_full_core_development",
        "development_only_not_research_evidence": True,
        "locked_confirmation_created": False,
        "pilot_executed": False,
        "final_executed": False,
        "started_at_utc": started,
        "solver_references_frozen_at_utc": references_frozen_at,
        "core_started_at_utc": agent_started,
        "completed_at_utc": completed,
        "implementation_tree_sha256": implementation_tree_sha256(root),
        "causal_law_sha256": GRID_PUSH_LAW_SHA256,
        "world_certifications": certifications,
        "module_call_probe": module_probe,
        "research_seed_summaries": research_summaries,
        "engineering_gates": engineering_gates,
        "episode_rows": episode_rows,
        "trace_rows": trace_rows,
        "artifact_sha256": artifacts,
    }
    manifest_path = output / "manifests" / "protocol_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    mean_training = fmean(
        row["training_final_tail_success"] for row in research_summaries
    )
    mean_frozen = fmean(row["frozen_success_rate"] for row in research_summaries)
    report_path = output / "report.md"
    report_path.write_text(
        "# GridPush full AASSRCore Development Diagnostic\n\n"
        "This is engineering/development evidence only. No Confirmation, Pilot, or Final was run.\n\n"
        f"- Certified worlds: {len(certifications)}/20\n"
        f"- Engineering gates: {'PASS' if all(engineering_gates.values()) else 'FAIL'}\n"
        f"- Full AASSR training final-tail success: {mean_training:.4f}\n"
        f"- Full AASSR frozen success: {mean_frozen:.4f}\n"
        f"- Episode rows: {episode_rows}\n"
        f"- Trace rows: {trace_rows}\n",
        encoding="utf-8",
    )
    return GridCoreDevelopmentArtifacts(
        output,
        manifest_path,
        report_path,
        len(certifications),
        episode_rows,
    )
