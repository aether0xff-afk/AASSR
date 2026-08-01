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
    columns = tuple(diagnostic_rows[0].to_dict())
    with V2ArtifactWriter(raw, columns) as writer:
        for row in diagnostic_rows:
            payload = row.to_dict()
            writer.write_episode(payload)
            writer.write_trace(
                {
                    "record_type": "diagnostic_1_seed_summary",
                    **payload,
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
        "diagnostic_1_metrics": gates["metrics"],
        "world_certifications": certifications,
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
                "Development Diagnostic results are not paper performance evidence.",
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
