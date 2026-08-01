from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .minecraft_causal_world import (
    MINECRAFT_CAUSAL_LAW_SHA256,
    MinecraftCausalWorld,
    MinecraftSkillTrack,
    MockMinecraftAdapter,
    certify_minecraft_world,
)
from .paper_v2_protocol import (
    V2ArtifactWriter,
    acquire_run_execution_lock,
    build_run_identity,
    preserve_partial_directory,
    replay_gzip_trace,
    release_run_execution_lock,
    reserve_confirmation_once,
    reserve_run,
    sha256_json,
    v2_run_directory,
)


@dataclass(frozen=True, slots=True)
class MinecraftRunArtifacts:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    row_count: int


def run_minecraft_diagnostic(
    config: Mapping[str, Any],
    *,
    run_id: str,
    repository_root: str | Path | None = None,
    resume: bool = False,
) -> MinecraftRunArtifacts:
    root = Path(repository_root or Path.cwd()).resolve()
    identity = build_run_identity(config, run_id=run_id, repository_root=root)
    if str(config["causal_law_sha256"]) != MINECRAFT_CAUSAL_LAW_SHA256:
        raise ValueError("config does not identify the Minecraft causal law")
    output = v2_run_directory(identity, repository_root=root)
    reserve_confirmation_once(root / "paper_results_v2", identity, resume=resume)
    reserve_run(output, identity, resume=resume)
    manifest_path = output / "manifests" / "protocol_manifest.json"
    if resume and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return MinecraftRunArtifacts(output, manifest_path, output / "report.md", int(payload["episode_rows"]))
    execution_lock, execution_nonce = acquire_run_execution_lock(
        output, resume=resume
    )
    raw = output / "raw"
    if raw.exists():
        if not resume:
            raise FileExistsError("partial Minecraft diagnostic output requires --resume")
        preserve_partial_directory(raw, label="raw")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    minecraft_settings = config.get("minecraft", {})
    if minecraft_settings.get("backend") != "mock":
        raise ValueError("Protocol v2.0 only permits the mock Minecraft backend")
    if minecraft_settings.get("real_runtime_enabled") is not False:
        raise ValueError("real Minecraft runtime is outside Protocol v2.0")
    track = MinecraftSkillTrack(str(minecraft_settings.get("track", "")))
    rows: list[dict[str, Any]] = []
    certifications: list[dict[str, Any]] = []
    token_separation: dict[int, bool] = {}
    with V2ArtifactWriter(
        raw,
        (
            "research_seed", "world_seed", "track", "affordance_ablation",
            "solvable", "minimum_plan_length", "valid_solution_count",
            "causal_family_count", "dead_end_count", "random_success",
            "adequate",
        ),
    ) as writer:
        for research_seed in config["research_seeds"]:
            for world_seed in config["world_seeds"]["train"]:
                expose = bool(
                    track is MinecraftSkillTrack.OPAQUE
                    and minecraft_settings.get("opaque_affordance_ablation", False)
                )
                world = MinecraftCausalWorld(
                    world_seed=int(world_seed), track=track,
                    expose_opaque_affordances=expose,
                )
                alternate = MinecraftCausalWorld(
                    world_seed=int(world_seed),
                    track=(MinecraftSkillTrack.OPAQUE if track is MinecraftSkillTrack.SEMANTIC else MinecraftSkillTrack.SEMANTIC),
                )
                token_separation[int(world_seed)] = world.action_token_sha256 != alternate.action_token_sha256
                certification = certify_minecraft_world(
                    world,
                    random_rollouts=int(minecraft_settings.get("random_rollouts", 1000)),
                    random_budget=int(minecraft_settings.get("interaction_budget", 8)),
                )
                record = certification.to_dict()
                record.update({
                    "research_seed": int(research_seed),
                    "world_seed": int(world_seed),
                    "track": track.value,
                    "affordance_ablation": expose,
                })
                certifications.append(record)
                row = {
                    "research_seed": research_seed,
                    "world_seed": world_seed,
                    "track": track.value,
                    "affordance_ablation": int(expose),
                    "solvable": int(certification.solvable),
                    "minimum_plan_length": certification.minimum_plan_length,
                    "valid_solution_count": certification.valid_solution_count,
                    "causal_family_count": certification.causal_family_count,
                    "dead_end_count": certification.dead_end_count,
                    "random_success": certification.random_policy_success_estimate,
                    "adequate": int(certification.adequate),
                }
                rows.append(row)
                writer.write_episode(row)
                writer.write_trace({
                    "record_type": "minecraft_world_certification",
                    "track": track.value,
                    "research_seed": research_seed,
                    "world_seed": world_seed,
                    "certification": record,
                })
        row_count = writer.episode_rows
        trace_count = writer.trace_rows
    replayed = replay_gzip_trace(raw / "trace.jsonl.gz")
    separate_tracks = all(token_separation.values())
    adapter = MockMinecraftAdapter(track=track)
    adapter.reset(seed=int(config["world_seeds"]["train"][0]))
    adapter_contract = adapter.observe() == adapter.world.observe()
    adapter.close()
    engineering_gates = {
        "same_causal_law_across_tracks": all(
            row["causal_law_sha256"] == MINECRAFT_CAUSAL_LAW_SHA256
            for row in certifications
        ),
        "semantic_and_opaque_tokens_separate": separate_tracks,
        "private_information_leak_zero": all(
            row["private_state_leak_count"] == 0 for row in certifications
        ),
        "strict_sparse_terminal_reward_only": str(config.get("reward_mode")) == "strict_sparse",
        "mock_adapter_contract": adapter_contract,
        "gzip_trace_replays": len(replayed) == trace_count,
        "single_track_result_contract": {row["track"] for row in rows}
        == {track.value},
    }
    adequacy_gates = {
        "all_worlds_solver_certified": all(row["adequate"] for row in certifications),
        "multiple_causal_families": all(row["causal_family_count"] >= 3 for row in certifications),
        "random_success_below_ceiling": all(row["random_policy_success_estimate"] <= 0.10 for row in certifications),
    }
    completed = datetime.now(timezone.utc).isoformat()
    manifest = {
        **identity.to_dict(),
        "schema_version": 2,
        "suite": "minecraft_like_mock",
        "started_at_utc": started,
        "completed_at_utc": completed,
        "config_sha256": identity.config_sha256,
        "causal_law_sha256": MINECRAFT_CAUSAL_LAW_SHA256,
        "episode_rows": row_count,
        "trace_rows": trace_count,
        "engineering_integrity_gates": engineering_gates,
        "benchmark_adequacy_gates": adequacy_gates,
        "empirical_hypotheses": {
            "semantic_track_improves_long_horizon_planning": "not_tested_by_environment_diagnostic",
            "opaque_track_learns_action_effects": "not_tested_by_environment_diagnostic",
        },
        "track_claims_must_remain_separate": True,
        "real_minecraft_runtime_executed": False,
        "certifications_sha256": sha256_json(certifications),
        "artifacts": {
            "episodes_csv_sha256": hashlib.sha256((raw / "episodes.csv").read_bytes()).hexdigest(),
            "trace_gzip_sha256": hashlib.sha256((raw / "trace.jsonl.gz").read_bytes()).hexdigest(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output / "report.md"
    report_path.write_text(
        "\n".join((
            "# Minecraft-like Protocol v2 diagnostic",
            "",
            f"- Stage: `{identity.stage.value}`",
            f"- Track: `{track.value}`",
            f"- Rows: `{row_count}`",
            f"- Engineering gates: `{'PASS' if all(engineering_gates.values()) else 'FAIL'}`",
            f"- Adequacy gates: `{'PASS' if all(adequacy_gates.values()) else 'FAIL'}`",
            "- Real MineStudio/MineRL/Malmo runtime: `NOT RUN`",
            "- Empirical performance claims: `NOT TESTED`",
            "",
        )),
        encoding="utf-8",
    )
    release_run_execution_lock(execution_lock, execution_nonce)
    return MinecraftRunArtifacts(output, manifest_path, report_path, row_count)
