from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .grid_push_agents import run_small_grid_diagnostic
from .grid_push_creativity import (
    freeze_solver_reference,
    load_solver_reference,
    normalize_grid_strategy,
    score_grid_creativity,
)
from .grid_push_world import (
    GRID_PUSH_LAW_SHA256,
    GridPushSpec,
    GridPushWorld,
    ProceduralGridPushGenerator,
    SolverResult,
)
from .paper_v2_protocol import (
    V2ArtifactWriter,
    build_run_identity,
    replay_gzip_trace,
    reserve_run,
    sha256_json,
    v2_run_directory,
)


@dataclass(frozen=True, slots=True)
class GridDevelopmentArtifacts:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    certified_worlds: int
    episode_rows: int


def _replay_trace(spec: GridPushSpec, actions: Sequence[str]) -> dict[str, Any]:
    world = GridPushWorld(spec)
    steps = []
    for index, action in enumerate(actions):
        if world.terminal:
            break
        before_ascii = world.render_ascii()
        before_observation = world.observe().to_dict()
        outcome = world.step(action)
        steps.append(
            {
                "step": index,
                "action": action,
                "succeeded": outcome.action_succeeded,
                "reward": outcome.reward,
                "before_ascii": before_ascii,
                "after_ascii": world.render_ascii(),
                "agent_observation_before": before_observation,
                "agent_observation_after": outcome.observation.to_dict(),
                "analysis_events": [
                    event.to_dict() for event in world.analysis_last_events
                ],
            }
        )
    return {
        "world_seed": spec.generator_seed,
        "success": world.analysis_private_state.success,
        "actions": list(actions),
        "steps": steps,
    }


def _find_trace(
    specs: Sequence[GridPushSpec],
    results: Sequence[SolverResult],
    event_kind: str,
) -> dict[str, Any] | None:
    for spec, result in zip(specs, results):
        for solution in result.solutions:
            if any(
                event.kind == event_kind
                for events in solution.event_steps
                for event in events
            ):
                return _replay_trace(spec, solution.actions)
    return None


def _boundary(width: int, height: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (x, y)
        for x in range(width)
        for y in range(height)
        if x in {0, width - 1} or y in {0, height - 1}
    )


def _pressure_plate_rule_trace() -> dict[str, Any]:
    plate, door = (3, 2), (4, 1)
    spec = GridPushSpec(
        width=6,
        height=5,
        walls=_boundary(6, 5),
        start=(1, 2),
        goal=(4, 3),
        blocks=frozenset({(2, 2)}),
        plates=frozenset({plate}),
        doors=frozenset({door}),
        plate_links={plate: (door,)},
    )
    trace = _replay_trace(
        spec, ("MOVE_EAST", "MOVE_EAST", "MOVE_SOUTH")
    )
    trace["evidence_scope"] = "physics_rule_probe_not_agent_performance"
    return trace


def _pit_fill_rule_trace() -> dict[str, Any]:
    spec = GridPushSpec(
        width=6,
        height=5,
        walls=_boundary(6, 5),
        start=(1, 2),
        goal=(4, 3),
        blocks=frozenset({(2, 2)}),
        pits=frozenset({(3, 2)}),
    )
    trace = _replay_trace(spec, ("MOVE_EAST", "MOVE_EAST"))
    trace["evidence_scope"] = "physics_rule_probe_not_agent_performance"
    return trace


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty summary")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_grid_push_development(
    config: Mapping[str, Any],
    *,
    run_id: str,
    repository_root: str | Path | None = None,
) -> GridDevelopmentArtifacts:
    root = Path(repository_root or Path.cwd()).resolve()
    identity = build_run_identity(config, run_id=run_id, repository_root=root)
    if identity.stage.value != "development_diagnostic":
        raise ValueError("grid push runner only permits Development Diagnostic")
    if str(config["causal_law_sha256"]) != GRID_PUSH_LAW_SHA256:
        raise ValueError("grid push causal-law hash mismatch")
    output = v2_run_directory(identity, repository_root=root)
    reserve_run(output, identity, resume=False)
    raw = output / "raw"
    manifests = output / "manifests"
    references_dir = manifests / "solver_references"
    manifests.mkdir(parents=True, exist_ok=True)
    settings = dict(config.get("grid_push", {}))
    world_seeds = [int(seed) for seed in settings["certification_world_seeds"]]
    if len(world_seeds) != 20 or len(set(world_seeds)) != 20:
        raise ValueError("Development grid certification requires 20 unique seeds")
    started = datetime.now(timezone.utc).isoformat()
    generator = ProceduralGridPushGenerator(
        maximum_attempts=int(settings.get("maximum_generation_attempts", 300))
    )
    specs: list[GridPushSpec] = []
    certifications = []
    solver_results: list[SolverResult] = []
    reference_payloads = []
    maximum_actions = int(settings.get("solver_maximum_actions", 35))
    for seed in world_seeds:
        spec, certification, solver_result = generator.generate(
            seed,
            maximum_actions=maximum_actions,
            random_rollouts=int(settings.get("certification_random_rollouts", 100)),
        )
        specs.append(spec)
        certifications.append(certification)
        solver_results.append(solver_result)
        # This occurs before any agent object is created.
        reference_payloads.append(
            freeze_solver_reference(
                references_dir / f"world_{seed}.json",
                world=GridPushWorld(spec),
                solver_result=solver_result,
                maximum_actions=maximum_actions,
            )
        )
    references_frozen_at = datetime.now(timezone.utc).isoformat()
    loaded_references = {
        int(spec.generator_seed or 0): load_solver_reference(
            references_dir / f"world_{spec.generator_seed}.json",
            expected_world_sha256=GridPushWorld(spec).world_sha256,
        )
        for spec in specs
    }
    example_worlds = [
        {
            "world_seed": spec.generator_seed,
            "world_sha256": certification.world_sha256,
            "ascii": GridPushWorld(spec).render_ascii(),
            "minimum_actions": certification.minimum_actions,
            "solution_count": certification.solution_count,
            "structural_solution_count": certification.structural_solution_count,
            "bounded_dead_end_count": certification.bounded_dead_end_count,
            "random_success_estimate": certification.random_success_estimate,
            "solver_reference_sha256": reference_payload["reference_sha256"],
        }
        for spec, certification, reference_payload in zip(
            specs[:5], certifications[:5], reference_payloads[:5]
        )
    ]
    (output / "world_examples.json").write_text(
        json.dumps(example_worlds, indent=2), encoding="utf-8"
    )
    traces = {
        "block_push": _find_trace(specs, solver_results, "block_moved"),
        "pit_fill": _pit_fill_rule_trace(),
        "plate_and_door": _pressure_plate_rule_trace(),
    }
    (output / "physics_traces.json").write_text(
        json.dumps(traces, indent=2), encoding="utf-8"
    )

    agent_started = datetime.now(timezone.utc).isoformat()
    train_count = int(settings.get("training_world_count", 5))
    summaries, episode_records, decision_records = run_small_grid_diagnostic(
        specs=specs[:train_count],
        research_seeds=config["research_seeds"],
        training_episodes=int(settings.get("training_episodes", 150)),
        evaluation_episodes=int(settings.get("evaluation_episodes", 20)),
        maximum_steps=int(settings.get("episode_maximum_steps", 35)),
    )
    episode_columns = (
        "condition", "research_seed", "phase", "episode", "world_seed",
        "success", "steps", "failed_actions", "block_moves",
        "imagination_interventions",
    )
    with V2ArtifactWriter(raw, episode_columns) as writer:
        for record in episode_records:
            payload = record.to_dict()
            writer.write_episode({key: payload[key] for key in episode_columns})
            writer.write_trace({"record_type": "agent_episode", **payload})
        for decision in decision_records:
            writer.write_trace({"record_type": "imagination_decision", **decision})
        episode_rows = writer.episode_rows
        trace_rows = writer.trace_rows
    replayed = replay_gzip_trace(raw / "trace.jsonl.gz")
    summary_rows = [summary.to_dict() for summary in summaries]
    _write_summary_csv(output / "condition_summary.csv", summary_rows)

    spec_by_seed = {int(spec.generator_seed or 0): spec for spec in specs}
    successful = [record for record in episode_records if record.success]
    normalized_by_episode: list[tuple[Any, Any]] = []
    graph_worlds: dict[str, set[int]] = defaultdict(set)
    for record in successful:
        normalized = normalize_grid_strategy(
            spec_by_seed[record.world_seed], record.actions
        )
        graph_hash = sha256_json(normalized.graph.to_dict())
        graph_worlds[graph_hash].add(record.world_seed)
        normalized_by_episode.append((record, normalized))
    creativity_rows = []
    for record, normalized in normalized_by_episode:
        graph_hash = sha256_json(normalized.graph.to_dict())
        certification = next(
            item for item in certifications if item.world_sha256 == GridPushWorld(spec_by_seed[record.world_seed]).world_sha256
        )
        reproduction = len(graph_worlds[graph_hash]) / max(1, train_count)
        score = score_grid_creativity(
            normalized,
            reference=loaded_references[record.world_seed],
            minimum_actions=int(certification.minimum_actions or 1),
            reproducibility=reproduction,
            novelty_threshold=float(settings.get("novelty_threshold", 0.20)),
            utility_threshold=float(settings.get("utility_threshold", 0.80)),
            reproduction_threshold=float(settings.get("reproduction_threshold", 0.50)),
        )
        creativity_rows.append(
            {
                "source": "agent_episode",
                "condition": record.condition,
                "research_seed": record.research_seed,
                "world_seed": record.world_seed,
                "graph_sha256": graph_hash,
                "normalized": normalized.to_dict(),
                "score": score.to_dict(),
            }
        )
    # Always record a transparent calculation example. It is selected from a
    # solver alternative but evaluated after the already-frozen reference.
    example_index = 0
    example_solution = solver_results[0].solutions[0]
    for index, (result, reference_payload) in enumerate(
        zip(solver_results, reference_payloads)
    ):
        reference_actions = {
            tuple(entry["actions_analysis_only"])
            for entry in reference_payload["references"]
        }
        alternative = next(
            (
                solution
                for solution in result.solutions
                if tuple(solution.actions) not in reference_actions
            ),
            None,
        )
        if alternative is not None:
            example_index = index
            example_solution = alternative
            break
    example_normalized = normalize_grid_strategy(
        specs[example_index], example_solution.actions
    )
    example_score = score_grid_creativity(
        example_normalized,
        reference=loaded_references[int(specs[example_index].generator_seed or 0)],
        minimum_actions=int(certifications[example_index].minimum_actions or 1),
        reproducibility=0.0,
        novelty_threshold=float(settings.get("novelty_threshold", 0.20)),
        utility_threshold=float(settings.get("utility_threshold", 0.80)),
        reproduction_threshold=float(settings.get("reproduction_threshold", 0.50)),
    )
    creativity_example = {
        "source": "posthoc_solver_alternative_example_not_agent_evidence",
        "world_seed": specs[example_index].generator_seed,
        "actions_differ_from_first_reference": (
            tuple(example_solution.actions)
            not in {
                tuple(entry["actions_analysis_only"])
                for entry in reference_payloads[example_index]["references"]
            }
        ),
        "normalized": example_normalized.to_dict(),
        "score": example_score.to_dict(),
    }
    (output / "creativity_evaluation.json").write_text(
        json.dumps(
            {"agent_strategies": creativity_rows, "calculation_example": creativity_example},
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = datetime.now(timezone.utc).isoformat()
    engineering = {
        "all_20_worlds_certified": len(certifications) == 20 and all(item.adequate for item in certifications),
        "structurally_distinct_solutions_present": sum(
            item.structural_solution_count >= 2 for item in certifications
        ) >= 10,
        "combined_pit_and_plate_worlds_present": any(
            item.all_solutions_fill_pit and item.all_solutions_open_door
            for item in certifications
        ),
        "private_observation_leaks_zero": all(item.private_observation_leaks == 0 for item in certifications),
        "only_general_movement_actions": all(
            set(GridPushWorld(spec).observe().available_actions)
            == {"MOVE_NORTH", "MOVE_SOUTH", "MOVE_WEST", "MOVE_EAST"}
            for spec in specs
        ),
        "solver_references_frozen_before_agents": references_frozen_at <= agent_started,
        "solver_reference_hashes_valid": len(loaded_references) == 20,
        "frozen_checkpoints_immutable": all(
            not summary.checkpoint_before_evaluation
            or summary.checkpoint_before_evaluation == summary.checkpoint_after_evaluation
            for summary in summaries
        ),
        "evaluation_learning_calls_zero": all(summary.evaluation_learning_calls == 0 for summary in summaries),
        "gzip_trace_replays": len(replayed) == trace_rows,
        "strict_sparse_reward": str(config.get("reward_mode")) == "strict_sparse",
    }
    manifest = {
        **identity.to_dict(),
        "schema_version": 2,
        "suite": "grid_push_development",
        "started_at_utc": started,
        "solver_references_frozen_at_utc": references_frozen_at,
        "agent_started_at_utc": agent_started,
        "completed_at_utc": completed,
        "development_only_not_research_evidence": True,
        "locked_confirmation_created": False,
        "pilot_executed": False,
        "final_executed": False,
        "episode_rows": episode_rows,
        "trace_rows": trace_rows,
        "engineering_gates": engineering,
        "world_certifications": [item.to_dict() for item in certifications],
        "condition_summaries": summary_rows,
        "creativity": {
            "successful_agent_episodes": len(successful),
            "scored_agent_strategies": len(creativity_rows),
            "final_candidates": sum(row["score"]["final_candidate"] for row in creativity_rows),
            "calculation_example": creativity_example,
        },
        "artifact_sha256": {
            "episodes_csv": hashlib.sha256((raw / "episodes.csv").read_bytes()).hexdigest(),
            "trace_jsonl_gz": hashlib.sha256((raw / "trace.jsonl.gz").read_bytes()).hexdigest(),
            "world_examples": hashlib.sha256((output / "world_examples.json").read_bytes()).hexdigest(),
            "physics_traces": hashlib.sha256((output / "physics_traces.json").read_bytes()).hexdigest(),
        },
        "world_set_sha256": sha256_json([item.world_sha256 for item in certifications]),
    }
    manifest_path = manifests / "protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    by_condition = {
        condition: {
            "seed_count": len([row for row in summaries if row.condition == condition]),
            "frozen_success": fmean(row.frozen_success_rate for row in summaries if row.condition == condition),
            "training_tail": fmean(row.training_final_tail_success for row in summaries if row.condition == condition),
        }
        for condition in ("random", "contextual_policy", "full_aassr")
    }
    report_path = output / "report.md"
    report_path.write_text(
        "\n".join(
            (
                "# Grid push Development Diagnostic",
                "",
                "This is development evidence only; no Confirmation, Pilot, or Final was run.",
                "",
                f"- Certified worlds: {len(certifications)}/20",
                f"- Engineering gates: {'PASS' if all(engineering.values()) else 'FAIL'}",
                f"- Random frozen success: {by_condition['random']['frozen_success']:.4f}",
                f"- Contextual training tail / frozen: {by_condition['contextual_policy']['training_tail']:.4f} / {by_condition['contextual_policy']['frozen_success']:.4f}",
                f"- Full AASSR training tail / frozen: {by_condition['full_aassr']['training_tail']:.4f} / {by_condition['full_aassr']['frozen_success']:.4f}",
                f"- Successful agent episodes scored for creativity: {len(creativity_rows)}",
                f"- Final creative candidates: {manifest['creativity']['final_candidates']}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return GridDevelopmentArtifacts(
        output, manifest_path, report_path, len(certifications), episode_rows
    )
