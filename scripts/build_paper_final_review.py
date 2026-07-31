from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sqlite3
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aassr_v2.creativity import strategy_distance_components
from aassr_v2.paper_artifacts import validate_paper_artifacts
from aassr_v2.paper_protocol import planned_paper_run_count, sha256_json
from aassr_v2.paper_statistics import (
    bootstrap_confidence_interval,
    holm_correction,
    paired_permutation_test,
)
from aassr_v2.paper_types import CausalEffectGraph


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "paper_results"
CONFIG_ROOT = ROOT / "configs"

EXPERIMENTS: dict[str, dict[str, str]] = {
    "autonomy": {
        "protocol": "paper-autonomy-final-v1",
        "config": "paper_autonomy_final_v1.json",
        "pilot_config": "paper_autonomy_pilot_v1.json",
    },
    "ablation": {
        "protocol": "paper-ablation-final-v1",
        "config": "paper_ablation_final_v1.json",
        "pilot_config": "paper_ablation_pilot_v1.json",
    },
    "transfer": {
        "protocol": "paper-transfer-final-v1",
        "config": "paper_transfer_final_v1.json",
        "pilot_config": "paper_transfer_pilot_v1.json",
    },
    "creativity": {
        "protocol": "paper-creativity-final-v1",
        "config": "paper_creativity_final_v1.json",
        "pilot_config": "paper_creativity_pilot_v1.json",
    },
    "safe_application": {
        "protocol": "paper-safe-application-final-v1",
        "config": "paper_safe_application_final_v1.json",
        "pilot_config": "paper_safe_application_pilot_v1.json",
    },
}

NUMERIC_FIELDS = {
    "action_proposals",
    "actual_return",
    "adaptation_budget",
    "cluster_count",
    "episode",
    "errors",
    "high_level_steps",
    "holdout_gain",
    "holdout_score",
    "imagination_depth",
    "imagined_nodes",
    "imagined_transitions",
    "intrinsic_value",
    "noise_facts",
    "novelty_score",
    "prediction_score",
    "primitive_steps",
    "real_transitions",
    "repeats",
    "research_seed",
    "reward",
    "root_imagined_value",
    "runtime_seconds",
    "seed",
    "skill_count",
    "skill_uses",
    "steps",
    "success",
    "world_seed",
}

NONNEGATIVE_FIELDS = {
    "action_proposals",
    "adaptation_budget",
    "cluster_count",
    "episode",
    "errors",
    "high_level_steps",
    "imagination_depth",
    "imagined_nodes",
    "imagined_transitions",
    "noise_facts",
    "primitive_steps",
    "real_transitions",
    "repeats",
    "runtime_seconds",
    "skill_count",
    "skill_uses",
    "steps",
}

REPEATED_VALUE_FIELDS = {
    "action_family",
    "branch_start_fingerprint",
    "checkpoint_fingerprint_after",
    "checkpoint_fingerprint_before",
    "condition",
    "environment",
    "experiment",
    "model",
    "phase",
    "solution_family",
    "suite",
}

SEED_METRICS = (
    "success",
    "steps",
    "high_level_steps",
    "primitive_steps",
    "reward",
    "errors",
    "repeats",
    "prediction_score",
    "holdout_score",
    "holdout_gain",
    "imagined_nodes",
    "imagination_depth",
    "actual_return",
    "runtime_seconds",
    "real_transitions",
    "imagined_transitions",
    "action_proposals",
    "novelty_score",
    "repeat_rate",
    "error_rate",
    "learning_auc",
    "final_tail_success",
    "first_success_real_transitions",
    "adaptation_auc",
    "episodes_to_50",
    "episodes_to_80",
    "transfer_gain_vs_from_scratch",
    "sample_saving_to_50",
    "sample_saving_to_80",
    "unseen_prediction_calibration_error",
)

PRIVATE_VISIBLE_TOKENS = {
    "information_route",
    "resource_route",
    "bypass_route",
    "tool_route",
    "emergent_combination",
    "viable_branch",
    "solution_family",
    "solution_label",
    "correct_action",
    "target_sequence",
    "hidden_solution",
    "oracle_answer",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    number = numeric(value)
    return int(number) if number is not None else None


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def row_identifier(row: Mapping[str, Any]) -> str:
    fields = (
        "suite",
        "condition",
        "environment",
        "phase",
        "seed",
        "world_seed",
        "adaptation_budget",
        "episode",
        "strategy_id",
    )
    return "|".join(str(row.get(field, "")) for field in fields)


def digest_values(values: Iterable[Any]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for value in values:
        digest.update(str(value).encode("utf-8", errors="replace"))
        digest.update(b"\x1f")
    return digest.digest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_gzip_size(path: Path) -> int:
    compressor = zlib.compressobj(level=6, wbits=31)
    total = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            total += len(compressor.compress(block))
    total += len(compressor.flush())
    return total


def bootstrap(values: Sequence[float], key: str, samples: int = 5000) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return bootstrap_confidence_interval(values, samples=samples, seed=seed)


def effect_size(differences: Sequence[float]) -> float:
    if len(differences) < 2:
        return 0.0
    deviation = statistics.stdev(differences)
    return statistics.fmean(differences) / deviation if deviation > 0 else 0.0


def flatten_counts(counter: Counter[tuple[str, str, str]]) -> list[dict[str, Any]]:
    return [
        {"condition": key[0], "environment": key[1], "phase": key[2], "rows": count}
        for key, count in sorted(counter.items())
    ]


def make_context_sample(
    *,
    experiment: str,
    anomaly_type: str,
    severity: str,
    row: Mapping[str, Any],
    previous_identifier: str,
    note: str,
) -> dict[str, Any]:
    metrics = {
        name: row.get(name, "")
        for name in (
            "success",
            "prediction_score",
            "holdout_gain",
            "steps",
            "errors",
            "repeats",
            "real_transitions",
        )
    }
    return {
        "severity": severity,
        "experiment": experiment,
        "anomaly_type": anomaly_type,
        "identifier": row_identifier(row),
        "previous_identifier": previous_identifier,
        "next_identifier": "",
        "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        "context_note": note,
    }


def profile_episode_csv(
    experiment: str,
    csv_path: Path,
    config: Mapping[str, Any],
    temp_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    duplicate_db = temp_dir / f"{experiment}_duplicates.sqlite"
    if duplicate_db.exists():
        duplicate_db.unlink()
    connection = sqlite3.connect(duplicate_db)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE row_hash (hash BLOB PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE grain_hash (hash BLOB PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
    )
    upsert = (
        "INSERT INTO {table}(hash,count) VALUES(?,1) "
        "ON CONFLICT(hash) DO UPDATE SET count=count+1"
    )

    row_count = 0
    fields: list[str] = []
    column_payload_bytes: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    invalid_numeric: Counter[str] = Counter()
    nan_or_infinite: Counter[str] = Counter()
    abnormal_counts: Counter[str] = Counter()
    distinct_values: dict[str, set[str]] = defaultdict(set)
    research_seeds: set[int] = set()
    world_seeds: set[int] = set()
    condition_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    combination_counts: Counter[tuple[str, str, str]] = Counter()
    group_stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "success_sum": 0.0,
            "prediction_sum": 0.0,
            "prediction_count": 0,
        }
    )
    group_first_context: dict[tuple[str, str, str], dict[str, Any]] = {}
    budget_totals: Counter[tuple[str, str, str, str]] = Counter()
    branch_origins: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    branch_budgets: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    anomaly_samples: list[dict[str, Any]] = []
    pending_samples: list[dict[str, Any]] = []
    previous_identifier = ""
    high_prediction_failures = 0
    high_prediction_limit = 10
    row_hash_batch: list[tuple[bytes]] = []
    grain_hash_batch: list[tuple[bytes]] = []
    budget_limit = int(config["budgets"]["real_transitions_per_episode"])

    grain_fields = (
        "suite",
        "condition",
        "environment",
        "model",
        "seed",
        "research_seed",
        "world_seed",
        "phase",
        "adaptation_budget",
        "episode",
        "strategy_id",
        "action_family",
    )

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        for row in reader:
            row_count += 1
            current_identifier = row_identifier(row)
            for sample in pending_samples:
                sample["next_identifier"] = current_identifier
            pending_samples = []
            for field in fields:
                value = row.get(field, "")
                encoded = str(value).encode("utf-8", errors="replace")
                column_payload_bytes[field] += len(encoded)
                if value == "":
                    empty_counts[field] += 1
                if field in REPEATED_VALUE_FIELDS:
                    distinct_values[field].add(str(value))
                if field in NUMERIC_FIELDS and value != "":
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        invalid_numeric[field] += 1
                    else:
                        if not math.isfinite(parsed):
                            nan_or_infinite[field] += 1
                        if field in NONNEGATIVE_FIELDS and parsed < 0:
                            abnormal_counts[f"negative_{field}"] += 1
            success = numeric(row.get("success"))
            if success is None or success not in {0.0, 1.0}:
                abnormal_counts["invalid_success"] += 1
            real_transitions = numeric(row.get("real_transitions")) or 0.0
            if real_transitions > budget_limit:
                abnormal_counts["transition_budget_exceeded"] += 1
            seed = integer(row.get("research_seed", row.get("seed")))
            world_seed = integer(row.get("world_seed"))
            if seed is not None:
                research_seeds.add(seed)
            if world_seed is not None:
                world_seeds.add(world_seed)
            if row.get("seed") and row.get("research_seed") and row["seed"] != row["research_seed"]:
                abnormal_counts["seed_research_seed_mismatch"] += 1
            condition = str(row.get("condition", ""))
            environment = str(row.get("environment", ""))
            phase = str(row.get("phase", ""))
            group_key = (condition, environment, phase)
            condition_counts[condition] += 1
            environment_counts[environment] += 1
            phase_counts[phase] += 1
            combination_counts[group_key] += 1
            group = group_stats[group_key]
            group["rows"] += 1
            group["success_sum"] += success or 0.0
            prediction = numeric(row.get("prediction_score"))
            if prediction is not None:
                group["prediction_sum"] += prediction
                group["prediction_count"] += 1
            if group_key not in group_first_context:
                context = make_context_sample(
                    experiment=experiment,
                    anomaly_type="group_extreme_success_rate",
                    severity="high",
                    row=row,
                    previous_identifier=previous_identifier,
                    note="Representative row for a condition/environment/phase group later flagged as all-zero or all-one.",
                )
                group_first_context[group_key] = context
                pending_samples.append(context)
            if prediction is not None and prediction >= 0.8 and (success or 0.0) == 0.0:
                high_prediction_failures += 1
                if high_prediction_failures <= high_prediction_limit:
                    sample = make_context_sample(
                        experiment=experiment,
                        anomaly_type="high_prediction_failed_episode",
                        severity="high",
                        row=row,
                        previous_identifier=previous_identifier,
                        note="Prediction score >= 0.8 while the episode failed.",
                    )
                    anomaly_samples.append(sample)
                    pending_samples.append(sample)
            evaluating = phase.startswith("evaluation")
            before = str(row.get("checkpoint_fingerprint_before", ""))
            after = str(row.get("checkpoint_fingerprint_after", ""))
            if evaluating and before and after and before != after:
                abnormal_counts["evaluation_checkpoint_mutation"] += 1
                if abnormal_counts["evaluation_checkpoint_mutation"] <= 5:
                    sample = make_context_sample(
                        experiment=experiment,
                        anomaly_type="evaluation_checkpoint_mutation",
                        severity="critical",
                        row=row,
                        previous_identifier=previous_identifier,
                        note="Checkpoint fingerprint changed during evaluation.",
                    )
                    anomaly_samples.append(sample)
                    pending_samples.append(sample)
            if (
                evaluating
                and experiment in {"autonomy", "ablation", "transfer", "creativity"}
                and (not before or not after)
            ):
                abnormal_counts["evaluation_fingerprint_missing"] += 1
            budget_key = (environment, phase, str(seed or ""), condition)
            budget_totals[budget_key] += real_transitions
            if experiment == "transfer":
                origin = str(row.get("branch_start_fingerprint", ""))
                branch_key = (condition, str(seed or ""), str(world_seed or ""))
                if origin:
                    branch_origins[branch_key].add(origin)
                adaptation_budget = integer(row.get("adaptation_budget"))
                if adaptation_budget is not None:
                    branch_budgets[branch_key].add(adaptation_budget)
            row_hash_batch.append(
                (digest_values(row.get(field, "") for field in fields),)
            )
            grain_hash_batch.append(
                (digest_values(row.get(field, "") for field in grain_fields),)
            )
            if len(row_hash_batch) >= 10_000:
                connection.executemany(upsert.format(table="row_hash"), row_hash_batch)
                connection.executemany(upsert.format(table="grain_hash"), grain_hash_batch)
                row_hash_batch.clear()
                grain_hash_batch.clear()
            previous_identifier = current_identifier
    if row_hash_batch:
        connection.executemany(upsert.format(table="row_hash"), row_hash_batch)
        connection.executemany(upsert.format(table="grain_hash"), grain_hash_batch)
    connection.commit()
    exact_duplicates = int(
        connection.execute("SELECT COALESCE(SUM(count-1),0) FROM row_hash").fetchone()[0]
    )
    grain_duplicates = int(
        connection.execute("SELECT COALESCE(SUM(count-1),0) FROM grain_hash").fetchone()[0]
    )
    connection.close()
    duplicate_db.unlink(missing_ok=True)

    for key, group in group_stats.items():
        rate = group["success_sum"] / group["rows"] if group["rows"] else 0.0
        group["success_rate"] = rate
        group["prediction_mean"] = (
            group["prediction_sum"] / group["prediction_count"]
            if group["prediction_count"]
            else None
        )
        if rate in {0.0, 1.0}:
            context = group_first_context[key]
            context["context_note"] = (
                f"All {group['rows']} rows in this group have success={rate:.0f}."
            )
            anomaly_samples.append(context)

    fairness_groups: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for (environment, phase, seed_value, condition), total in budget_totals.items():
        fairness_groups[(environment, phase, seed_value)][condition] = total
    unequal_budget_groups = [
        {
            "environment": key[0],
            "phase": key[1],
            "seed": key[2],
            "minimum": min(values.values()),
            "maximum": max(values.values()),
        }
        for key, values in fairness_groups.items()
        if experiment in {"autonomy", "ablation"}
        and len(values) > 1
        and min(values.values()) != max(values.values())
    ]
    expected_budgets = {int(item) for item in config["budgets"]["adaptation_episodes"]}
    branch_origin_mismatches = sum(
        1 for origins in branch_origins.values() if len(origins) != 1
    )
    branch_budget_mismatches = sum(
        1 for budgets in branch_budgets.values() if budgets != expected_budgets
    )

    file_size = csv_path.stat().st_size
    gzip_size = exact_gzip_size(csv_path)
    payload_total = sum(column_payload_bytes.values())
    repeated_bytes = {}
    for field, values in distinct_values.items():
        unique_payload = sum(len(value.encode("utf-8")) for value in values)
        repeated_bytes[field] = max(0, column_payload_bytes[field] - unique_payload)
    profile = {
        "path": str(csv_path.resolve()),
        "file_size_bytes": file_size,
        "gzip_size_bytes_exact": gzip_size,
        "gzip_ratio": gzip_size / file_size if file_size else 0.0,
        "parquet_size_estimate_bytes": {
            "low": int(file_size * 0.08),
            "high": int(file_size * 0.22),
            "method": "Estimated range for typed columnar storage with dictionary encoding; pyarrow/duckdb were unavailable, so no Parquet file was materialized.",
        },
        "row_count": row_count,
        "column_count": len(fields),
        "columns": fields,
        "average_serialized_row_bytes": file_size / row_count if row_count else 0.0,
        "column_payload_bytes": dict(column_payload_bytes),
        "column_payload_share": {
            field: count / payload_total if payload_total else 0.0
            for field, count in column_payload_bytes.items()
        },
        "csv_syntax_overhead_bytes": max(0, file_size - payload_total),
        "repeated_value_estimated_bytes": repeated_bytes,
        "distinct_counts_for_repeated_fields": {
            field: len(values) for field, values in distinct_values.items()
        },
        "empty_counts": dict(empty_counts),
        "invalid_numeric_counts": dict(invalid_numeric),
        "nan_or_infinite_counts": dict(nan_or_infinite),
        "abnormal_counts": dict(abnormal_counts),
        "research_seeds": sorted(research_seeds),
        "world_seeds": sorted(world_seeds),
        "condition_counts": dict(sorted(condition_counts.items())),
        "environment_counts": dict(sorted(environment_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "condition_environment_phase_counts": flatten_counts(combination_counts),
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_episode_grain_rows": grain_duplicates,
        "duplicate_method": "BLAKE2b-128 fingerprints stored and counted in a temporary SQLite table; raw rows were never retained in memory.",
        "high_prediction_failed_rows": high_prediction_failures,
        "unequal_transition_budget_groups": unequal_budget_groups,
        "transfer_branch_origin_mismatches": branch_origin_mismatches,
        "transfer_branch_budget_mismatches": branch_budget_mismatches,
    }
    return profile, anomaly_samples, group_stats


def scan_transitions(experiment: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {
            "path": str(path.resolve()),
            "exists": False,
            "line_count": 0,
        }, []
    line_count = 0
    json_errors = 0
    evaluation_learning_enabled = 0
    private_visible_leaks = 0
    samples: list[dict[str, Any]] = []
    previous_identifier = ""
    pending: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            identifier = "|".join(
                str(payload.get(field, ""))
                for field in (
                    "suite",
                    "condition",
                    "research_seed",
                    "world_seed",
                    "phase",
                    "episode",
                    "transition_index",
                )
            )
            for sample in pending:
                sample["next_identifier"] = identifier
            pending = []
            phase = str(payload.get("phase", ""))
            if phase.startswith("evaluation") and payload.get("learning_enabled") is not False:
                evaluation_learning_enabled += 1
                if len(samples) < 20:
                    sample = {
                        "severity": "critical",
                        "experiment": experiment,
                        "anomaly_type": "evaluation_transition_learning_enabled",
                        "identifier": identifier,
                        "previous_identifier": previous_identifier,
                        "next_identifier": "",
                        "metrics_json": "{}",
                        "context_note": "Evaluation transition was not explicitly frozen.",
                    }
                    samples.append(sample)
                    pending.append(sample)
            visible = json.dumps(
                {
                    "before": payload.get("before"),
                    "action": payload.get("action"),
                    "after": payload.get("after"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).lower()
            leaked = sorted(token for token in PRIVATE_VISIBLE_TOKENS if token in visible)
            if leaked:
                private_visible_leaks += 1
                if len(samples) < 20:
                    sample = {
                        "severity": "critical",
                        "experiment": experiment,
                        "anomaly_type": "private_label_agent_visible",
                        "identifier": identifier,
                        "previous_identifier": previous_identifier,
                        "next_identifier": "",
                        "metrics_json": json.dumps({"tokens": leaked}),
                        "context_note": "Private/oracle token appeared in before/action/after.",
                    }
                    samples.append(sample)
                    pending.append(sample)
            previous_identifier = identifier
    return {
        "path": str(path.resolve()),
        "exists": True,
        "file_size_bytes": path.stat().st_size,
        "line_count": line_count,
        "json_parse_errors": json_errors,
        "evaluation_learning_enabled_rows": evaluation_learning_enabled,
        "private_agent_visible_leak_rows": private_visible_leaks,
        "private_tokens_checked": sorted(PRIVATE_VISIBLE_TOKENS),
    }, samples


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_seed_rows(experiment: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learning_rows = load_csv(root / "statistics" / "learning_summary.csv")
    learning_index = {
        (
            row.get("suite", ""),
            row.get("condition", ""),
            row.get("environment", ""),
            row.get("seed", ""),
        ): row
        for row in learning_rows
    }
    for source in load_csv(root / "seed_level" / "seed_summary.csv"):
        row: dict[str, Any] = {"experiment": experiment, **source}
        real = numeric(row.get("real_transitions"))
        row["repeat_rate"] = (
            (numeric(row.get("repeats")) or 0.0) / real if real and real > 0 else ""
        )
        row["error_rate"] = (
            (numeric(row.get("errors")) or 0.0) / real if real and real > 0 else ""
        )
        if row.get("phase") == "training":
            learning = learning_index.get(
                (
                    row.get("suite", ""),
                    row.get("condition", ""),
                    row.get("environment", ""),
                    row.get("seed", ""),
                )
            )
            if learning:
                row["learning_auc"] = learning.get("learning_auc", "")
                row["final_tail_success"] = learning.get(
                    "final_10_percent_success", ""
                )
                row["first_success_real_transitions"] = learning.get(
                    "first_success_real_transitions", ""
                )
        rows.append(row)
    adaptation_rows = load_csv(root / "statistics" / "adaptation_summary.csv")
    for source in adaptation_rows:
        rows.append(
            {
                "experiment": experiment,
                "suite": "transfer",
                "condition": source.get("condition", ""),
                "environment": source.get("environment", ""),
                "model": "checkpoint_transfer",
                "phase": "adaptation_curve",
                "action_family": "effect_profile",
                "seed": source.get("seed", ""),
                "episode_rows": "",
                "adaptation_auc": source.get("adaptation_auc", ""),
                "episodes_to_50": source.get("episodes_to_50", ""),
                "episodes_to_80": source.get("episodes_to_80", ""),
                "transfer_gain_vs_from_scratch": source.get(
                    "transfer_gain_vs_from_scratch", ""
                ),
                "sample_saving_to_50": source.get("sample_saving_to_50", ""),
                "sample_saving_to_80": source.get("sample_saving_to_80", ""),
                "unseen_prediction_calibration_error": source.get(
                    "unseen_prediction_calibration_error", ""
                ),
            }
        )
    return rows


MATRIX_PATTERN = re.compile(
    r"^full_depth(?P<depth>\d+)_branch(?P<branch>\d+)_(?P<aggregation>max|mean|risk-adjusted)$"
)


def add_ablation_axis_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        if row.get("experiment") != "ablation" or row.get("phase") != "training":
            continue
        match = MATRIX_PATTERN.match(str(row.get("condition", "")))
        if not match:
            continue
        for axis in ("depth", "branch", "aggregation"):
            grouped[(axis, match.group(axis), str(row.get("seed", "")))].append(row)
    synthetic: list[dict[str, Any]] = []
    for (axis, level, seed), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "experiment": "ablation",
            "suite": "ablation",
            "condition": f"axis_{axis}={level}",
            "environment": rows[0].get("environment", ""),
            "model": "axis_marginal_mean",
            "phase": "training",
            "action_family": "ablation_axis",
            "seed": seed,
            "episode_rows": sum(integer(row.get("episode_rows")) or 0 for row in rows),
        }
        for metric in (
            "learning_auc",
            "final_tail_success",
            "first_success_real_transitions",
            "prediction_score",
            "holdout_gain",
            "success",
        ):
            values = [value for row in rows if (value := numeric(row.get(metric))) is not None]
            item[metric] = statistics.fmean(values) if values else ""
        synthetic.append(item)
    return synthetic


def cross_seed_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        "experiment",
        "suite",
        "condition",
        "environment",
        "model",
        "phase",
        "action_family",
    )
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = tuple(str(row.get(field, "")) for field in dimensions)
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        for metric in SEED_METRICS:
            values = [value for row in rows if (value := numeric(row.get(metric))) is not None]
            if not values:
                continue
            low, high = bootstrap(values, "|".join((*key, metric)))
            output.append(
                {
                    **{field: key[index] for index, field in enumerate(dimensions)},
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": statistics.fmean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    return output


def collapse_seed_metric(
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in seed_rows:
        key = (
            str(row.get("experiment", "")),
            str(row.get("suite", "")),
            str(row.get("environment", "")),
            str(row.get("phase", "")),
            str(row.get("condition", "")),
            str(row.get("seed", "")),
        )
        for metric in SEED_METRICS:
            number = numeric(row.get(metric))
            if number is not None:
                values[key][metric].append(number)
    return {
        key: {metric: statistics.fmean(items) for metric, items in metrics.items()}
        for key, metrics in values.items()
    }


def condition_comparison_rows(
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    collapsed = collapse_seed_metric(seed_rows)
    by_group: dict[
        tuple[str, str, str, str], dict[str, dict[str, dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for (experiment, suite, environment, phase, condition, seed), metrics in collapsed.items():
        by_group[(experiment, suite, environment, phase)][condition][seed] = metrics

    output: list[dict[str, Any]] = []
    for group, conditions in sorted(by_group.items()):
        experiment, suite, environment, phase = group
        targets: list[tuple[str, list[str]]] = []
        if experiment in {"autonomy", "ablation", "creativity"} and "full_aassr" in conditions:
            targets.append(("full_aassr", sorted(name for name in conditions if name != "full_aassr" and not name.startswith("oracle"))))
        if experiment == "transfer" and "full_transfer" in conditions:
            targets.append(("full_transfer", sorted(name for name in conditions if name != "full_transfer")))
        axis_conditions = [name for name in conditions if name.startswith("axis_")]
        for prefix, reference in (
            ("axis_depth=", "axis_depth=1"),
            ("axis_branch=", "axis_branch=1"),
            ("axis_aggregation=", "axis_aggregation=max"),
        ):
            if reference in conditions:
                targets.extend(
                    (name, [reference])
                    for name in axis_conditions
                    if name.startswith(prefix) and name != reference
                )
        for target, baselines in targets:
            for baseline in baselines:
                common = sorted(set(conditions[target]) & set(conditions[baseline]))
                for metric in SEED_METRICS:
                    paired = [
                        (
                            conditions[target][seed].get(metric),
                            conditions[baseline][seed].get(metric),
                        )
                        for seed in common
                    ]
                    paired = [
                        (left, right)
                        for left, right in paired
                        if left is not None and right is not None
                    ]
                    if not paired:
                        continue
                    left = [item[0] for item in paired]
                    right = [item[1] for item in paired]
                    differences = [a - b for a, b in paired]
                    key = "|".join((*group, target, baseline, metric))
                    low, high = bootstrap(differences, key)
                    permutation_seed = int.from_bytes(
                        hashlib.sha256(key.encode("utf-8")).digest()[:4], "big"
                    )
                    output.append(
                        {
                            "experiment": experiment,
                            "suite": suite,
                            "environment": environment,
                            "phase": phase,
                            "metric": metric,
                            "target": target,
                            "baseline": baseline,
                            "paired_seed_count": len(paired),
                            "target_mean": statistics.fmean(left),
                            "baseline_mean": statistics.fmean(right),
                            "paired_mean_difference": statistics.fmean(differences),
                            "ci95_low": low,
                            "ci95_high": high,
                            "effect_size_dz": effect_size(differences),
                            "p_value": paired_permutation_test(
                                left,
                                right,
                                samples=20_000,
                                seed=permutation_seed,
                            ),
                            "oracle_excluded": True,
                        }
                    )
    holm_groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        holm_groups[
            (
                str(row["experiment"]),
                str(row["environment"]),
                str(row["phase"]),
                str(row["metric"]),
            )
        ].append(index)
    for indexes in holm_groups.values():
        adjusted = holm_correction([float(output[index]["p_value"]) for index in indexes])
        for index, value in zip(indexes, adjusted, strict=True):
            output[index]["p_value_holm"] = value
    return output


def adaptation_curve_rows(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in load_csv(root / "statistics" / "adaptation_summary.csv"):
        try:
            curve = json.loads(row.get("curve_json", "[]"))
        except json.JSONDecodeError:
            curve = []
        for budget, success_rate in curve:
            output.append(
                {
                    "experiment": "transfer",
                    "condition": row.get("condition", ""),
                    "environment": row.get("environment", ""),
                    "seed": row.get("seed", ""),
                    "adaptation_budget": budget,
                    "success_rate": success_rate,
                    "adaptation_auc": row.get("adaptation_auc", ""),
                    "episodes_to_50": row.get("episodes_to_50", ""),
                    "episodes_to_80": row.get("episodes_to_80", ""),
                    "sample_saving_to_50": row.get("sample_saving_to_50", ""),
                    "sample_saving_to_80": row.get("sample_saving_to_80", ""),
                    "transfer_gain_vs_from_scratch": row.get(
                        "transfer_gain_vs_from_scratch", ""
                    ),
                    "unseen_prediction_calibration_error": row.get(
                        "unseen_prediction_calibration_error", ""
                    ),
                }
            )
    return output


def strategy_condition(strategy_id: str, conditions: Sequence[str]) -> str:
    matches = [name for name in conditions if strategy_id.startswith(name + "_")]
    return max(matches, key=len) if matches else "unknown"


def graph_key(payload: Mapping[str, Any]) -> str:
    graph = CausalEffectGraph.from_dict(payload)
    return hashlib.sha256(
        json.dumps(graph.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def analyze_creativity(
    root: Path,
    config: Mapping[str, Any],
    frozen_rules: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conditions = [str(item["name"]) for item in config["suites"][0]["conditions"]]
    path = root / "raw" / "strategies.jsonl"
    qualification_index = {
        str(row.get("strategy_id", "")): row
        for row in load_csv(root / "statistics" / "creativity_summary.csv")
    }
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    by_family: Counter[str] = Counter()
    unique_by_condition: dict[str, set[str]] = defaultdict(set)
    unique_global: set[str] = set()
    graph_seeds: dict[str, set[int]] = defaultdict(set)
    graph_worlds: dict[str, set[int]] = defaultdict(set)
    baseline_references: dict[str, tuple[str, CausalEffectGraph]] = {}
    candidate_records: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            qualification = qualification_index.get(
                str(record.get("strategy_id", "")), {}
            )
            total += 1
            condition = strategy_condition(str(record.get("strategy_id", "")), conditions)
            family = str(record.get("graph", {}).get("solution_family", ""))
            key = graph_key(record.get("graph", {}))
            unique_global.add(key)
            unique_by_condition[condition].add(key)
            graph_seeds[key].add(int(record.get("research_seed", 0)))
            graph_worlds[key].add(int(record.get("world_seed", 0)))
            by_family[family] += 1
            bucket = by_condition[condition]
            bucket["strategies"] += 1
            bucket["successful"] += int(truthy(record.get("valid", record.get("success"))))
            novelty = numeric(record.get("novelty_score")) or 0.0
            bucket["novelty_score_scaled"] += int(round(novelty * 1_000_000_000))
            bucket["novel_above_threshold"] += int(
                truthy(qualification.get("novel_above_frozen_threshold"))
            )
            bucket["utility_qualified"] += int(
                truthy(qualification.get("utility_qualified"))
            )
            bucket["reproducible"] += int(
                truthy(qualification.get("reproducible"))
            )
            bucket["creative_candidate"] += int(
                truthy(qualification.get("creative_candidate"))
            )
            if str(record.get("source_kind", "")) != "aassr":
                baseline_references.setdefault(
                    key,
                    (
                        str(record.get("strategy_id", "")),
                        CausalEffectGraph.from_dict(record.get("graph", {})),
                    ),
                )
            if truthy(qualification.get("creative_candidate")):
                candidate_records.append({**record, **qualification})
    candidates: list[dict[str, Any]] = []
    reference_values = list(baseline_references.values())
    for record in candidate_records:
        graph = CausalEffectGraph.from_dict(record.get("graph", {}))
        nearest_id = ""
        nearest_components: dict[str, float] = {}
        nearest_distance = math.inf
        for reference_id, reference_graph in reference_values:
            components = strategy_distance_components(graph, reference_graph)
            distance = statistics.fmean(components.values())
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_id = reference_id
                nearest_components = components
        key = graph_key(record.get("graph", {}))
        candidates.append(
            {
                "strategy_id": record.get("strategy_id", ""),
                "causal_effect_graph": record.get("graph", {}),
                "nearest_reference_kind": "baseline-generated",
                "nearest_reference_strategy_id": nearest_id,
                "novelty_components": nearest_components,
                "novelty_score": record.get("novelty_score", ""),
                "success": record.get("success", record.get("valid", "")),
                "primitive_steps": record.get("primitive_steps", ""),
                "risk_entries": record.get("risk_entries", ""),
                "resources_used": record.get("resources_used", ""),
                "reusable_success_rate": record.get("reusable_success_rate", ""),
                "appearance_seed_count": len(graph_seeds[key]),
                "appearance_world_count": len(graph_worlds[key]),
                "classification_reason": "successful + frozen novelty threshold + utility non-dominated + reproduced",
            }
        )
    summary_rows = []
    for condition, counts in sorted(by_condition.items()):
        total_condition = counts["strategies"]
        summary_rows.append(
            {
                "condition": condition,
                **dict(counts),
                "unique_causal_graphs": len(unique_by_condition[condition]),
                "mean_novelty_score": (
                    counts["novelty_score_scaled"] / 1_000_000_000 / total_condition
                    if total_condition
                    else 0.0
                ),
            }
        )
    return {
        "strategy_rows": total,
        "unique_causal_graphs_global": len(unique_global),
        "baseline_reference_unique_graphs": len(baseline_references),
        "human_reference_rows": 0,
        "frozen_novelty_threshold": (
            frozen_rules.get("novelty_threshold") if frozen_rules else None
        ),
        "by_condition": summary_rows,
        "by_solution_family": dict(sorted(by_family.items())),
        "creative_candidate_count": len(candidates),
        "deduplication_rule": "CausalEffectGraph canonical nodes, prerequisite/enablement edges, effect sequence, and solution family; raw action strings are excluded.",
        "reference_label": "baseline-generated reference (no human participant data)",
    }, candidates


def human_data_status(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    human_paths = root / "raw" / "human_paths.jsonl"
    human_ratings = root / "raw" / "human_ratings.csv"
    metadata = root / "manifests" / "human_dataset.json"
    participant_ids: set[str] = set()
    valid_strategies = 0
    rating_rows: list[dict[str, str]] = []
    if human_paths.exists():
        with human_paths.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("participant_id"):
                    participant_ids.add(str(row["participant_id"]))
                valid_strategies += int(truthy(row.get("valid", row.get("success"))))
    if human_ratings.exists():
        rating_rows = load_csv(human_ratings)
        participant_ids.update(
            str(row.get("participant_id", row.get("rater_id", "")))
            for row in rating_rows
            if row.get("participant_id") or row.get("rater_id")
        )
    metadata_payload = read_json(metadata) if metadata.exists() else {}
    settings = config.get("human_study", {})
    ratings_per_strategy = Counter(
        str(row.get("strategy_id", "")) for row in rating_rows if row.get("strategy_id")
    )
    return {
        "human_paths_present": human_paths.exists(),
        "human_ratings_present": human_ratings.exists(),
        "human_dataset_manifest_present": metadata.exists(),
        "participant_count": len(participant_ids),
        "valid_human_strategy_count": valid_strategies,
        "rating_row_count": len(rating_rows),
        "minimum_raters_per_strategy": min(ratings_per_strategy.values()) if ratings_per_strategy else None,
        "maximum_raters_per_strategy": max(ratings_per_strategy.values()) if ratings_per_strategy else None,
        "self_rating_removed": None if not rating_rows else "not derivable from available schema",
        "duplicate_rating_removed": None if not rating_rows else "not derivable from exported rows alone",
        "inter_rater_agreement": None,
        "automatic_human_novelty_correlation": None,
        "configured_approval_id": settings.get("approval_id"),
        "manifest_approval_id": metadata_payload.get("approval_id"),
        "approval_id_matches": bool(metadata.exists()) and metadata_payload.get("approval_id") == settings.get("approval_id"),
        "configured_dataset_version": settings.get("dataset_version"),
        "manifest_dataset_version": metadata_payload.get("dataset_version"),
        "dataset_version_matches": bool(metadata.exists()) and metadata_payload.get("dataset_version") == settings.get("dataset_version"),
        "merge_enabled": bool(settings.get("merge_enabled", False)),
        "interpretation": "No human participant dataset is present; all non-AASSR references are baseline-generated." if not human_paths.exists() else "Human dataset present.",
    }


def file_inventory(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        output.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
        )
    return output


def source_lock_status(
    config_path: Path,
    result_root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_path = result_root / "manifests" / "resolved_config.json"
    resolved = read_json(resolved_path)
    acceptance_source = ROOT / str(config.get("acceptance_gate_manifest", ""))
    acceptance_copy = result_root / "manifests" / "acceptance_gate_manifest.json"
    frozen_value = str(config.get("frozen_creativity_rules", ""))
    frozen_source = ROOT / frozen_value if frozen_value else None
    frozen_copy = result_root / "manifests" / "frozen_creativity_rules.json"
    return {
        "config_path": str(config_path.resolve()),
        "config_canonical_sha256": sha256_json(config),
        "resolved_config_canonical_sha256": sha256_json(resolved),
        "manifest_config_sha256": manifest.get("config_sha256"),
        "config_hash_matches_manifest": sha256_json(config) == manifest.get("config_sha256"),
        "resolved_config_hash_matches_manifest": sha256_json(resolved) == manifest.get("config_sha256"),
        "acceptance_gate_declared": bool(config.get("acceptance_gate_manifest")),
        "acceptance_gate_copy_present": acceptance_copy.exists(),
        "acceptance_gate_hash_matches": (
            acceptance_source.exists()
            and acceptance_copy.exists()
            and file_sha256(acceptance_source) == file_sha256(acceptance_copy)
        ),
        "frozen_creativity_rule_declared": bool(frozen_value),
        "frozen_creativity_rule_copy_present": frozen_copy.exists(),
        "frozen_creativity_rule_hash_matches": (
            frozen_source is not None
            and frozen_source.exists()
            and frozen_copy.exists()
            and file_sha256(frozen_source) == file_sha256(frozen_copy)
        ),
    }


def seed_overlap_status(config: Mapping[str, Any], pilot_path: Path) -> dict[str, Any]:
    pilot = read_json(pilot_path) if pilot_path.exists() else {}
    final_research = {int(item) for item in config.get("research_seeds", ())}
    pilot_research = {int(item) for item in pilot.get("research_seeds", ())}
    final_worlds = {
        phase: {int(item) for item in config.get("world_seeds", {}).get(phase, ())}
        for phase in ("train", "seen", "unseen")
    }
    pilot_all_worlds = {
        int(item)
        for values in pilot.get("world_seeds", {}).values()
        for item in values
    }
    return {
        "research_seed_overlap_pilot_final": sorted(final_research & pilot_research),
        "world_seed_overlap_pilot_final": sorted(
            set().union(*final_worlds.values()) & pilot_all_worlds
        ),
        "train_unseen_overlap_final": sorted(final_worlds["train"] & final_worlds["unseen"]),
        "train_seen_overlap_final": sorted(final_worlds["train"] & final_worlds["seen"]),
        "seen_unseen_overlap_final": sorted(final_worlds["seen"] & final_worlds["unseen"]),
    }


def copy_manifests(output: Path, experiment: str, config_path: Path, result_root: Path) -> None:
    target = output / "manifests" / experiment
    target.mkdir(parents=True, exist_ok=True)
    for source in (
        config_path,
        result_root / "manifests" / "protocol_manifest.json",
        result_root / "manifests" / "run_state.json",
        result_root / "manifests" / "resolved_config.json",
        result_root / "manifests" / "acceptance_gate_manifest.json",
        result_root / "manifests" / "frozen_creativity_rules.json",
        result_root / "manifests" / "human_dataset.json",
        result_root / "manifests" / "safe_application_world.json",
    ):
        if source.exists():
            shutil.copy2(source, target / source.name)


def fmt_number(value: Any, digits: int = 4) -> str:
    number = numeric(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def find_cross(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    condition: str,
    environment: str,
    phase: str,
    metric: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row.get("experiment") == experiment
            and row.get("condition") == condition
            and row.get("environment") == environment
            and row.get("phase") == phase
            and row.get("metric") == metric
        ),
        None,
    )


def make_svg_bar(
    path: Path,
    title: str,
    subtitle: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    x_label: str,
    max_value: float | None = None,
) -> None:
    width = 1000
    left = 310
    right = 80
    top = 100
    row_height = 34
    height = top + len(labels) * row_height + 90
    chart_width = width - left - right
    maximum = max_value if max_value is not None else max(max(values, default=1.0), 1e-9)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="38" font-family="Arial" font-size="23" fill="#20242b">{title}</text>',
        f'<text x="36" y="66" font-family="Arial" font-size="13" fill="#5b6470">{subtitle}</text>',
        f'<line x1="{left}" y1="{top-14}" x2="{left}" y2="{height-58}" stroke="#303640"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = top + index * row_height
        bar_width = max(0.0, value / maximum) * chart_width
        elements.extend(
            (
                f'<text x="{left-12}" y="{y+16}" text-anchor="end" font-family="Arial" font-size="12" fill="#303640">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="22" fill="#4c78a8" stroke="#2f4f6f"/>',
                f'<text x="{left+bar_width+8:.2f}" y="{y+16}" font-family="Consolas" font-size="12" fill="#20242b">{value:.4f}</text>',
            )
        )
    elements.extend(
        (
            f'<text x="{left+chart_width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13" fill="#303640">{x_label}</text>',
            "</svg>",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def make_svg_lines(
    path: Path,
    title: str,
    subtitle: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 240, 100, 80
    chart_width = width - left - right
    chart_height = height - top - bottom
    all_x = [x for values in series.values() for x, _ in values]
    all_y = [y for values in series.values() for _, y in values]
    min_x, max_x = min(all_x, default=0), max(all_x, default=1)
    max_y = max(max(all_y, default=1.0), 1e-9)
    colors = ["#4c78a8", "#f2a541", "#8c6bb1", "#6b8e23", "#d46a6a"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="38" font-family="Arial" font-size="23" fill="#20242b">{title}</text>',
        f'<text x="36" y="66" font-family="Arial" font-size="13" fill="#5b6470">{subtitle}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" stroke="#303640"/>',
        f'<line x1="{left}" y1="{top+chart_height}" x2="{left+chart_width}" y2="{top+chart_height}" stroke="#303640"/>',
    ]
    for tick in range(6):
        value = max_y * tick / 5
        y = top + chart_height - chart_height * tick / 5
        elements.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Consolas" font-size="11" fill="#5b6470">{value:.2f}</text>'
        )
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_width}" y2="{y:.1f}" stroke="#e4e7eb"/>'
        )
    for value in sorted(set(all_x)):
        x = left + ((value - min_x) / max(max_x - min_x, 1e-9)) * chart_width
        label = f"{value:g}"
        elements.extend(
            (
                f'<line x1="{x:.1f}" y1="{top+chart_height}" x2="{x:.1f}" y2="{top+chart_height+6}" stroke="#303640"/>',
                f'<text x="{x:.1f}" y="{top+chart_height+22}" text-anchor="middle" font-family="Consolas" font-size="11" fill="#5b6470">{label}</text>',
            )
        )
    for index, (label, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        points = []
        for x, y in values:
            px = left + ((x - min_x) / max(max_x - min_x, 1e-9)) * chart_width
            py = top + chart_height - (y / max_y) * chart_height
            points.append((px, py))
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>'
        )
        for x, y in points:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        legend_y = top + index * 28
        elements.extend(
            (
                f'<line x1="{left+chart_width+25}" y1="{legend_y+7}" x2="{left+chart_width+55}" y2="{legend_y+7}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{left+chart_width+65}" y="{legend_y+11}" font-family="Arial" font-size="11" fill="#303640">{label}</text>',
            )
        )
    elements.extend(
        (
            f'<text x="{left+chart_width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="13" fill="#303640">{x_label}</text>',
            f'<text transform="translate(22 {top+chart_height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13" fill="#303640">{y_label}</text>',
            "</svg>",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def generate_figures(
    output: Path,
    cross_rows: Sequence[Mapping[str, Any]],
    adaptation_rows: Sequence[Mapping[str, Any]],
    creativity: Mapping[str, Any],
    inventories: Mapping[str, Mapping[str, Any]],
) -> None:
    figure_dir = output / "figures"
    environments = ["opaque_dependency_l4", "opaque_dependency_l6", "opaque_dependency_l8"]
    conditions = ["full_aassr", "random", "contextual_policy", "q_learning", "dqn", "prophecy_no_imagination"]
    labels: list[str] = []
    values: list[float] = []
    for environment in environments:
        for condition in conditions:
            row = find_cross(
                cross_rows,
                experiment="autonomy",
                condition=condition,
                environment=environment,
                phase="evaluation_unseen_zero_shot",
                metric="success",
            )
            if row:
                labels.append(f"L{environment[-1]} {condition}")
                values.append(float(row["mean"]))
    make_svg_bar(
        figure_dir / "rq1_unseen_success.svg",
        "RQ1 unseen zero-shot success",
        "Seed-level mean success; 30 research seeds per condition and dependency length",
        labels,
        values,
        x_label="success rate",
        max_value=max(max(values, default=0.1), 0.1),
    )

    auc_labels: list[str] = []
    auc_values: list[float] = []
    for environment in environments:
        for condition in conditions:
            row = find_cross(
                cross_rows,
                experiment="autonomy",
                condition=condition,
                environment=environment,
                phase="training",
                metric="learning_auc",
            )
            if row:
                auc_labels.append(f"L{environment[-1]} {condition}")
                auc_values.append(float(row["mean"]))
    make_svg_bar(
        figure_dir / "rq1_learning_auc.svg",
        "RQ1 training success AUC",
        "Normalized success AUC inside each research seed, then averaged across 30 seeds",
        auc_labels,
        auc_values,
        x_label="training success AUC",
        max_value=1.0,
    )

    curve_groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in adaptation_rows:
        value = numeric(row.get("success_rate"))
        budget = numeric(row.get("adaptation_budget"))
        if value is not None and budget is not None:
            curve_groups[(str(row.get("condition", "")), budget)].append(value)
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (condition, budget), items in sorted(curve_groups.items()):
        series[condition].append((budget, statistics.fmean(items)))
    make_svg_lines(
        figure_dir / "transfer_adaptation_curves.svg",
        "Transfer adaptation curves",
        "Mean success after 0, 1, 4, 16, and 64 adaptation episodes; 30 paired seeds",
        series,
        x_label="adaptation episodes",
        y_label="success rate",
    )

    creativity_rows = list(creativity.get("by_condition", ()))
    full = next((row for row in creativity_rows if row.get("condition") == "full_aassr"), {})
    funnel_labels = ["successful", "novel", "utility-qualified", "reproduced", "creative candidate"]
    funnel_values = [
        float(full.get("successful", 0)),
        float(full.get("novel_above_threshold", 0)),
        float(full.get("utility_qualified", 0)),
        float(full.get("reproducible", 0)),
        float(full.get("creative_candidate", 0)),
    ]
    make_svg_bar(
        figure_dir / "creativity_funnel.svg",
        "Full AASSR creativity qualification funnel",
        "Training strategies; a final candidate must pass all four protocol gates",
        funnel_labels,
        funnel_values,
        x_label="strategy count",
    )

    storage_labels = list(sorted(inventories))
    storage_values = [
        float(inventories[name]["directory_size_bytes"]) / (1024**3)
        for name in storage_labels
    ]
    make_svg_bar(
        figure_dir / "storage_by_experiment.svg",
        "Final artifact storage by experiment",
        "Directory size includes suite run, resume cache, and merged raw artifacts",
        storage_labels,
        storage_values,
        x_label="GiB",
    )

    axis_rows = [
        row
        for row in cross_rows
        if row.get("experiment") == "ablation"
        and str(row.get("condition", "")).startswith("axis_")
        and row.get("metric") == "learning_auc"
    ]
    make_svg_bar(
        figure_dir / "ablation_axis_learning_auc.svg",
        "Ablation matrix marginal learning AUC",
        "Each axis level is averaged within seed across the other matrix axes; only dependency length L6 was run",
        [str(row["condition"]) for row in axis_rows],
        [float(row["mean"]) for row in axis_rows],
        x_label="training success AUC",
        max_value=1.0,
    )


def build_inventory_markdown(integrity: Mapping[str, Any]) -> str:
    rows = []
    for name, item in integrity["experiments"].items():
        rows.append(
            (
                name,
                f"`{item['config']['config_path']}`",
                item["run"]["started_at_utc"],
                item["run"]["pipeline_completed_at_utc"],
                f"{item['run']['planned_rows']:,}",
                f"{item['run']['actual_rows']:,}",
                item["run"]["completed_research_seed_count"],
                len(item["run"]["failed_runs"]),
                "PASS" if item["artifact_validator"]["passed"] else "FAIL",
            )
        )
    lines = [
        "# Final experiment inventory",
        "",
        "원본은 읽기 전용으로 조사했으며 이 디렉터리에는 경량 검토 산출물만 생성했다.",
        "",
        markdown_table(
            ["Experiment", "Config", "Started (UTC)", "Pipeline completed (UTC)", "Planned rows", "Actual rows", "Seeds", "Failed/retried runs", "Validator"],
            rows,
        ),
        "",
        "## Storage finding",
        "",
        "43GB를 넘는 것은 단일 CSV가 아니라 `paper-ablation-final-v1` 전체 디렉터리다. 최종 `raw/episodes.csv`는 약 0.74GiB이며, 대부분의 용량은 transition trace가 suite run, resume cache, merged raw에 반복 보관된 데서 발생한다.",
        "",
    ]
    for name, item in integrity["experiments"].items():
        profile = item["episode_csv"]
        lines.extend(
            (
                f"## {name}",
                "",
                f"- Directory: `{item['run']['result_path']}`",
                f"- Episode CSV: {profile['file_size_bytes'] / (1024**3):.3f} GiB, {profile['row_count']:,} rows, {profile['column_count']} columns, {profile['average_serialized_row_bytes']:.1f} bytes/row",
                f"- Exact simulated gzip size: {profile['gzip_size_bytes_exact'] / (1024**3):.3f} GiB ({profile['gzip_ratio']:.1%} of CSV)",
                f"- Full directory: {item['storage']['directory_size_bytes'] / (1024**3):.3f} GiB",
                "",
                "### Rows by condition, environment, and phase",
                "",
                markdown_table(
                    ["Condition", "Environment", "Phase", "Rows"],
                    [
                        (row["condition"], row["environment"], row["phase"], f"{row['rows']:,}")
                        for row in profile["condition_environment_phase_counts"]
                    ],
                ),
                "",
            )
        )
    return "\n".join(lines)


def result_report_markdown(
    integrity: Mapping[str, Any],
    cross_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    adaptation_rows: Sequence[Mapping[str, Any]],
    creativity: Mapping[str, Any],
    human_status: Mapping[str, Any],
) -> str:
    anomaly_analysis = integrity.get("anomaly_analysis", {})
    autonomy_auc_rows = []
    autonomy_eval_rows = []
    for length in (4, 6, 8):
        environment = f"opaque_dependency_l{length}"
        for condition in (
            "full_aassr",
            "random",
            "contextual_policy",
            "q_learning",
            "dqn",
            "prophecy_no_imagination",
        ):
            auc = find_cross(
                cross_rows,
                experiment="autonomy",
                condition=condition,
                environment=environment,
                phase="training",
                metric="learning_auc",
            )
            seen = find_cross(
                cross_rows,
                experiment="autonomy",
                condition=condition,
                environment=environment,
                phase="evaluation_seen",
                metric="success",
            )
            unseen = find_cross(
                cross_rows,
                experiment="autonomy",
                condition=condition,
                environment=environment,
                phase="evaluation_unseen_zero_shot",
                metric="success",
            )
            if auc:
                autonomy_auc_rows.append(
                    (length, condition, fmt_number(auc["mean"]), f"[{fmt_number(auc['ci95_low'])}, {fmt_number(auc['ci95_high'])}]", auc["seed_count"])
                )
            if seen and unseen:
                autonomy_eval_rows.append(
                    (length, condition, fmt_number(seen["mean"]), fmt_number(unseen["mean"]), seen["seed_count"])
                )

    full_eval_values = [
        float(row[2]) for row in autonomy_eval_rows if row[1] == "full_aassr"
    ] + [float(row[3]) for row in autonomy_eval_rows if row[1] == "full_aassr"]
    rq1_verdict = "Partial" if any(
        float(row[2]) > 0 for row in autonomy_auc_rows if row[1] == "full_aassr"
    ) and not any(full_eval_values) else ("Yes" if any(full_eval_values) else "No")
    rq2_verdict = "Yes" if int(creativity.get("creative_candidate_count", 0)) > 0 else "No"

    full_diagnostic_rows = []
    diagnostic_metrics = (
        "final_tail_success",
        "first_success_real_transitions",
        "repeat_rate",
        "error_rate",
        "steps",
        "runtime_seconds",
        "imagined_nodes",
        "prediction_score",
        "holdout_gain",
    )
    for length in (4, 6, 8):
        environment = f"opaque_dependency_l{length}"
        values = []
        for metric in diagnostic_metrics:
            row = find_cross(
                cross_rows,
                experiment="autonomy",
                condition="full_aassr",
                environment=environment,
                phase="training",
                metric=metric,
            )
            values.append(fmt_number(row.get("mean")) if row else "N/A")
        full_diagnostic_rows.append((length, *values))

    component_baselines = {
        "Prophecy/Imagination removed": "contextual_policy",
        "Imagination removed": "prophecy_no_imagination",
        "Validated information value removed": "full_no_validated_information",
        "Repeat penalty removed": "full_no_repeat_penalty",
        "Error penalty removed": "full_no_error_penalty",
        "Imagination without validated value": "imagination_no_validated_value",
    }
    component_rows = []
    for label, baseline in component_baselines.items():
        row = next(
            (
                item
                for item in comparisons
                if item.get("experiment") == "ablation"
                and item.get("phase") == "training"
                and item.get("metric") == "learning_auc"
                and item.get("target") == "full_aassr"
                and item.get("baseline") == baseline
            ),
            None,
        )
        if row:
            component_rows.append(
                (
                    label,
                    fmt_number(row.get("target_mean")),
                    fmt_number(row.get("baseline_mean")),
                    fmt_number(row.get("paired_mean_difference")),
                    f"[{fmt_number(row.get('ci95_low'))}, {fmt_number(row.get('ci95_high'))}]",
                    fmt_number(row.get("p_value_holm")),
                )
            )

    axis_metric_index = {
        (str(row.get("condition", "")), str(row.get("metric", ""))): row
        for row in cross_rows
        if row.get("experiment") == "ablation"
        and str(row.get("condition", "")).startswith("axis_")
    }
    axis_rows = []
    for condition in sorted({key[0] for key in axis_metric_index}):
        auc = axis_metric_index.get((condition, "learning_auc"))
        tail = axis_metric_index.get((condition, "final_tail_success"))
        prediction = axis_metric_index.get((condition, "prediction_score"))
        holdout = axis_metric_index.get((condition, "holdout_gain"))
        axis_rows.append(
            (
                condition,
                fmt_number(auc.get("mean")) if auc else "N/A",
                fmt_number(tail.get("mean")) if tail else "N/A",
                fmt_number(prediction.get("mean")) if prediction else "N/A",
                fmt_number(holdout.get("mean")) if holdout else "N/A",
            )
        )

    transfer_cross: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in adaptation_rows:
        value = numeric(row.get("success_rate"))
        budget = numeric(row.get("adaptation_budget"))
        if value is not None and budget is not None:
            transfer_cross[(str(row.get("condition", "")), budget)].append(value)
    transfer_table = [
        (
            condition,
            *[
                fmt_number(statistics.fmean(transfer_cross.get((condition, float(budget)), [0.0])))
                for budget in (0, 1, 4, 16, 64)
            ],
        )
        for condition in sorted({key[0] for key in transfer_cross})
    ]
    transfer_metric_rows = []
    for condition in sorted({key[0] for key in transfer_cross}):
        metric_values = []
        for metric in (
            "adaptation_auc",
            "episodes_to_50",
            "episodes_to_80",
            "sample_saving_to_50",
            "sample_saving_to_80",
            "transfer_gain_vs_from_scratch",
            "unseen_prediction_calibration_error",
        ):
            row = find_cross(
                cross_rows,
                experiment="transfer",
                condition=condition,
                environment="opaque_dependency_l6",
                phase="adaptation_curve",
                metric=metric,
            )
            if row:
                formatted = fmt_number(row.get("mean"))
                if metric in {
                    "episodes_to_50",
                    "episodes_to_80",
                    "sample_saving_to_50",
                    "sample_saving_to_80",
                }:
                    formatted += f" (n={row.get('seed_count')})"
                metric_values.append(formatted)
            else:
                metric_values.append("N/A")
        transfer_metric_rows.append((condition, *metric_values))

    creativity_table = [
        (
            row["condition"],
            row["strategies"],
            row["successful"],
            row["unique_causal_graphs"],
            row["novel_above_threshold"],
            row["utility_qualified"],
            row["reproducible"],
            row["creative_candidate"],
        )
        for row in creativity.get("by_condition", ())
    ]
    family_total = sum(
        int(value) for value in creativity.get("by_solution_family", {}).values()
    )
    family_rows = [
        (name, f"{int(count):,}", f"{int(count) / family_total:.1%}")
        for name, count in sorted(
            creativity.get("by_solution_family", {}).items(),
            key=lambda item: int(item[1]),
            reverse=True,
        )
    ]

    high_pred_rows = []
    for name, experiment in integrity["experiments"].items():
        for item in experiment["episode_csv"].get("condition_environment_phase_counts", ()):
            pass
        count = experiment["episode_csv"].get("high_prediction_failed_rows", 0)
        if count:
            high_pred_rows.append((name, count))

    lines = [
        "# AASSR Final 결과 무결성 및 경량 분석 보고서",
        "",
        "## 기술 요약",
        "",
        f"- **질문 1 — 희소 보상 환경에서 정답 시범 없이 스스로 목표에 도달했는가? `{rq1_verdict}`.** Full AASSR는 training AUC가 0보다 높아 학습 중 목표 도달은 관찰됐지만, frozen `evaluation_seen`과 `evaluation_unseen_zero_shot` 성공률은 모든 길이에서 0이었다.",
        f"- **질문 2 — 기준선과 다른 유효·유용·재현 가능한 전략을 만들었는가? `{rq2_verdict}`.** Full AASSR 전략은 성공하고 재현됐지만 frozen novelty threshold를 넘은 전략이 0건이어서 최종 creative candidate도 0건이다.",
        "- 다섯 Final suite는 계획 행 수와 실제 행 수가 일치하고 30개 research seed가 모두 존재한다. Autonomy에서 진행상태 파일 잠금으로 suite-level 재시도 1건이 있었지만 최종 데이터에는 누락 seed가 없다.",
        "- 43.75GiB Ablation 디렉터리의 주원인은 episode CSV가 아니라 transition trace의 실행본·cache·병합본 중복이다. 원본은 수정하거나 삭제하지 않았다.",
        "",
        "## RQ1 — 학습 중 도달했지만 frozen 평가에는 일반화되지 않았다",
        "",
        "Training success AUC는 각 research seed 내부 episode 곡선을 먼저 적분한 뒤 30 seed 사이에서 집계했다. 평가 성공률도 episode가 아니라 seed별 평균을 통계 단위로 사용했다.",
        "",
        "### Training success AUC",
        "",
        markdown_table(["Length", "Condition", "Mean AUC", "Bootstrap 95% CI", "Seeds"], autonomy_auc_rows),
        "",
        "### Frozen evaluation success",
        "",
        markdown_table(["Length", "Condition", "Seen", "Unseen zero-shot", "Seeds"], autonomy_eval_rows),
        "",
        "### Full AASSR training diagnostics",
        "",
        markdown_table(
            ["Length", "Final-tail success", "First success transitions", "Repeat rate", "Error rate", "Mean steps", "Runtime (s)", "Imagined nodes", "Prediction", "Holdout gain"],
            full_diagnostic_rows,
        ),
        "",
        "`evaluation_unseen_adaptation`은 Autonomy suite에서 생성되지 않았으므로 해당 평가지표는 N/A다. Baseline별 동일 지표와 paired 차이는 경량 CSV에 모두 포함했다.",
        "",
        "Full AASSR는 Random, Q-learning, DQN보다 training AUC가 높지만 Contextual Policy와 Prophecy without Imagination보다 낮다. 평가에서는 Full AASSR가 전 길이에서 0이고 L4의 Random/DQN은 0보다 높아, 자율 목표 달성 우수성은 지지되지 않는다.",
        "",
        "## Ablation — 제거 효과는 명확하지 않고 길이 변화는 측정되지 않았다",
        "",
        "Ablation Final config는 dependency length 6만 실행했다. 따라서 depth·branching·aggregation 효과가 환경 길이에 따라 어떻게 변하는지는 이 데이터로 답할 수 없다. 제거 조건 및 36개 imagination matrix 설정의 seed-first 비교는 `condition_comparisons.csv`와 `cross_seed_summary.csv`에 포함했다.",
        "",
        "### Component removal comparisons",
        "",
        markdown_table(["Removed/changed component", "Full AUC", "Comparator AUC", "Full - comparator", "95% CI", "Holm p"], component_rows),
        "",
        "### Imagination matrix marginal results (L6 only)",
        "",
        markdown_table(["Axis level", "Learning AUC", "Final-tail success", "Prediction", "Holdout gain"], axis_rows),
        "",
        "Prophecy/Imagination 제거군인 Contextual Policy 및 Prophecy without Imagination은 Full AASSR보다 training AUC가 높았다. validated information value, repeat penalty, error penalty 제거군과 Full 간 차이는 작고 bootstrap CI가 0을 포함한다. 이는 해당 구성요소의 긍정적 기여를 입증하지 못한다.",
        "",
        "## 구조 전이 — 작은 평균 차이는 있으나 유의한 sample-efficiency 이득은 없다",
        "",
        markdown_table(["Condition", "Budget 0", "Budget 1", "Budget 4", "Budget 16", "Budget 64"], transfer_table),
        "",
        markdown_table(["Condition", "Adaptation AUC", "Episodes to 50%", "Episodes to 80%", "Saving to 50%", "Saving to 80%", "Transfer gain", "Calibration error"], transfer_metric_rows),
        "",
        "Full transfer의 adaptation AUC는 from-scratch보다 소폭 높지만 paired bootstrap CI가 0을 포함하고 Holm 보정 후 유의하지 않다. Effect representation retained 조건도 명시적인 ID-retained 대조군이 없어 representation 자체의 인과적 이점을 판정할 수 없다. Effect-retained가 Prophecy-retained보다 우월하다는 증거도 없다.",
        "",
        "모든 adaptation branch는 동일한 시작 checkpoint fingerprint에서 분기했고 `[0,1,4,16,64]` budget을 모두 포함했다. Transfer runner는 `evaluation_seen`과 `evaluation_unseen_zero_shot` 행을 생성하지 않았으므로, transfer suite 단독으로 seen-vs-zero-shot 질문은 답할 수 없다.",
        "",
        "## RQ2 — 성공 전략은 많지만 기준선과 구조적으로 구별되지 않았다",
        "",
        markdown_table(["Condition", "Strategies", "Successful", "Unique graphs", "Novel", "Utility-qualified", "Reproduced", "Candidates"], creativity_table),
        "",
        markdown_table(["Solution family", "Strategies", "Share"], family_rows),
        "",
        f"Frozen novelty threshold는 {creativity.get('frozen_novelty_threshold')}이다. 총 {creativity.get('strategy_rows', 0):,}개 전략은 성공했지만 최소 baseline distance가 모두 0이어서 threshold를 넘은 전략과 최종 candidate는 0건이다. 행동 문자열은 graph key에 포함하지 않았고 causal effect graph 기준으로 중복 제거했다.",
        "",
        "## 인간 비교 데이터 — 포함되지 않았다",
        "",
        f"실제 인간 path: {human_status.get('human_paths_present')}, 인간 rating: {human_status.get('human_ratings_present')}, 참가자 수: {human_status.get('participant_count')}. Approval ID와 승인 dataset manifest가 없고 merge가 비활성화됐다. 따라서 모든 reference는 **baseline-generated reference**이며 인간 전략으로 해석하면 안 된다.",
        "",
        "## 무결성 및 이상 징후",
        "",
        "- 모든 suite에서 계획 행 수와 실제 행 수가 일치하고 Final research seed 30개가 모두 존재한다.",
        "- episode grain 및 exact row 중복, NaN/Inf, transition budget 초과, evaluation checkpoint mutation, agent-visible private-label leak은 `integrity_report.json`의 count로 기록했다.",
        "- 여러 평가 조건이 0% 또는 100%이고 일부 ablation 결과가 완전히 동일하다. 이는 ceiling/floor effect로 비교 검정의 식별력을 크게 낮춘다.",
        f"- Seed-level 성공 벡터가 완전히 동일한 조건 묶음은 {anomaly_analysis.get('identical_condition_vector_count', 'N/A')}개, 성공/AUC 절대합의 50%를 단일 seed가 차지한 묶음은 {anomaly_analysis.get('seed_dominance_group_count', 'N/A')}개다. 구체적인 조건과 seed는 `integrity_report.json`에 있다.",
        f"- 평균 prediction score가 0.8 이상인데 seed-level success가 0.1 이하인 조건/환경/phase 묶음은 {anomaly_analysis.get('high_prediction_low_success_group_count', 'N/A')}개다.",
        f"- Full AASSR가 Contextual/Prophecy no-imagination comparator보다 유의하게 낮은 비교는 {anomaly_analysis.get('imagination_degradation_comparison_count', 'N/A')}개다. 단, 조건 간 차이가 imagination 하나뿐은 아니므로 imagination의 단독 인과효과로 해석하지 않는다.",
        "- Creativity는 5개 solution family를 생성했지만 baseline pool에 동일한 causal graphs가 존재하여 novelty가 모두 0이다. Threshold가 낮아 대부분 통과한 것이 아니라 아무 것도 통과하지 못한 상태다.",
        f"- 가장 큰 solution family는 `{anomaly_analysis.get('largest_solution_family', 'N/A')}`이며 전체의 {float(anomaly_analysis.get('largest_solution_family_share', 0.0)):.1%}다. 단일 family가 과반을 차지하지는 않는다.",
        "- 원본 transition trace는 분석에 필요한 episode summary와 다른 grain이지만, 동일 merged/cache 파일이 중복 저장되어 전달 용량을 키운다.",
    ]
    if high_pred_rows:
        lines.extend(
            (
                "- prediction score >= 0.8인데 실패한 episode가 발견됐다: "
                + ", ".join(f"{name} {count:,}건" for name, count in high_pred_rows)
                + ". 식별자와 앞뒤 문맥 표본은 `anomaly_samples.csv`에 있다.",
            )
        )
    lines.extend(
        (
            "",
            "## 방법",
            "",
            "원본 CSV는 `csv.DictReader`로 한 행씩 읽었고, 중복 검사는 BLAKE2b-128 row/grain fingerprint를 임시 SQLite에 누적했다. gzip 크기는 원본 바이트를 메모리에 올리지 않고 zlib level 6으로 끝까지 통과시켜 실제 압축 바이트 수를 계산했다. Parquet은 pyarrow/duckdb가 설치되지 않아 생성하지 않았고 typed dictionary encoding을 가정한 범위만 제시했다.",
            "",
            "통계 단위는 research seed다. episode를 먼저 seed 내부에서 집계한 후 seed 평균·표준편차·5,000회 bootstrap 95% CI를 계산했다. 조건 비교는 paired seed difference, 20,000회 paired permutation test, Cohen's dz, 동일 experiment/environment/phase/metric family 내 Holm correction을 사용했다. Oracle은 추론 비교에서 제외했다.",
            "",
            "## 한계와 다음 단계",
            "",
            "1. Ablation을 L4/L8에서도 실행해야 imagination 설정과 환경 길이의 상호작용을 검정할 수 있다.",
            "2. Transfer에 명시적인 ID-retained 대조군과 zero-shot phase를 추가해야 representation advantage를 판정할 수 있다.",
            "3. Creativity reference pool을 독립 인간 데이터 또는 사전 동결된 외부 baseline으로 구성해야 자기 데이터와 동일 graph가 reference에 들어가는 문제를 피할 수 있다.",
            "4. Full AASSR의 frozen 평가 실패 원인을 training/evaluation world reset, learned policy use, reward propagation 관점에서 진단한 뒤 새로운 protocol version으로 재실행해야 한다. 현재 Final 결과를 사후 수정하면 안 된다.",
            "",
            "## 추가 질문",
            "",
            "- Training 성공이 frozen 평가에서 사라지는 원인이 checkpoint 복원, policy action selection, world-seed shift 중 어디에 있는가?",
            "- Baseline과 동일한 creative graph가 대량 생성되는 것이 환경의 해 공간 제한인지 agent canonicalization의 과도한 압축인지?",
        )
    )
    return "\n".join(lines)


def per_experiment_report(
    name: str,
    item: Mapping[str, Any],
    cross_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> str:
    profile = item["episode_csv"]
    run = item["run"]
    top_columns = sorted(
        profile["column_payload_bytes"].items(), key=lambda pair: pair[1], reverse=True
    )[:10]
    primary_rows = [
        (
            row.get("condition", ""),
            row.get("environment", ""),
            row.get("phase", ""),
            row.get("metric", ""),
            row.get("seed_count", ""),
            fmt_number(row.get("mean")),
            f"[{fmt_number(row.get('ci95_low'))}, {fmt_number(row.get('ci95_high'))}]",
        )
        for row in cross_rows
        if row.get("metric")
        in {"success", "learning_auc", "adaptation_auc", "prediction_score", "holdout_gain"}
        and not str(row.get("condition", "")).startswith("axis_")
    ]
    significant = [
        (
            row.get("phase", ""),
            row.get("metric", ""),
            f"{row.get('target')} vs {row.get('baseline')}",
            row.get("paired_seed_count", ""),
            fmt_number(row.get("paired_mean_difference")),
            f"[{fmt_number(row.get('ci95_low'))}, {fmt_number(row.get('ci95_high'))}]",
            fmt_number(row.get("p_value_holm")),
        )
        for row in comparisons
        if numeric(row.get("p_value_holm")) is not None
        and float(row["p_value_holm"]) < 0.05
    ]
    lines = [
            f"# {name} Final review",
            "",
            f"- Config: `{item['config']['config_path']}`",
            f"- Started: {run['started_at_utc']}",
            f"- Pipeline completed: {run['pipeline_completed_at_utc']}",
            f"- Planned/actual rows: {run['planned_rows']:,} / {run['actual_rows']:,}",
            f"- Completed research seeds: {run['completed_research_seed_count']} / {run['expected_research_seed_count']}",
            f"- Missing seeds: {run['missing_research_seeds']}",
            f"- Failed/retried runs: {run['failed_runs']}",
            f"- Artifact validator: {'PASS' if item['artifact_validator']['passed'] else 'FAIL'}",
            f"- Config/resolved hash matches manifest: {item['config']['config_hash_matches_manifest']} / {item['config']['resolved_config_hash_matches_manifest']}",
            f"- Final acceptance gate hash match: {item['config']['acceptance_gate_hash_matches']}",
            f"- Frozen creativity rule: {'applied and matched' if item['config']['frozen_creativity_rule_declared'] and item['config']['frozen_creativity_rule_hash_matches'] else 'not applicable'}",
            f"- Pilot/Final research seed overlap: {item['seed_overlap']['research_seed_overlap_pilot_final']}",
            f"- Pilot/Final world seed overlap: {item['seed_overlap']['world_seed_overlap_pilot_final']}",
            f"- Final train/unseen world overlap: {item['seed_overlap']['train_unseen_overlap_final']}",
            f"- Exact row duplicates: {profile['exact_duplicate_rows']:,}",
            f"- Grain duplicates: {profile['duplicate_episode_grain_rows']:,}",
            f"- NaN/Inf: {sum(profile['nan_or_infinite_counts'].values()):,}",
            f"- Invalid numeric: {sum(profile['invalid_numeric_counts'].values()):,}",
            f"- Abnormal domain values: {sum(profile['abnormal_counts'].values()):,}",
            f"- Agent-visible private/oracle label leaks: {item['transitions'].get('private_agent_visible_leak_rows', 0):,}",
            f"- Evaluation transitions with learning enabled: {item['transitions'].get('evaluation_learning_enabled_rows', 0):,}",
            "",
            "## Largest episode CSV payload columns",
            "",
            markdown_table(
                ["Column", "Payload MiB", "Share"],
                [
                    (
                        field,
                        f"{size / (1024**2):.2f}",
                        f"{profile['column_payload_share'][field]:.1%}",
                    )
                    for field, size in top_columns
                ],
            ),
            "",
            "Full per-condition/environment/phase counts and all integrity counters are in `integrity_report.json`.",
            "",
            "## Primary seed-level results",
            "",
            markdown_table(
                ["Condition", "Environment", "Phase", "Metric", "Seeds", "Mean", "Bootstrap 95% CI"],
                primary_rows,
            ),
            "",
            "## Holm-significant paired comparisons",
            "",
        ]
    if significant:
        lines.append(
            markdown_table(
                ["Phase", "Metric", "Comparison", "Paired seeds", "Difference", "95% CI", "Holm p"],
                significant,
            )
        )
    else:
        lines.append("No paired comparison remained significant after Holm correction.")
    lines.append("")
    return "\n".join(lines)


def headline_rows(
    cross_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    creativity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for environment in ("opaque_dependency_l4", "opaque_dependency_l6", "opaque_dependency_l8"):
        for phase in ("training", "evaluation_seen", "evaluation_unseen_zero_shot"):
            metric = "learning_auc" if phase == "training" else "success"
            row = find_cross(
                cross_rows,
                experiment="autonomy",
                condition="full_aassr",
                environment=environment,
                phase=phase,
                metric=metric,
            )
            if row:
                output.append(
                    {
                        "research_question": "RQ1 autonomy",
                        "experiment": "autonomy",
                        "environment": environment,
                        "phase": phase,
                        "condition": "full_aassr",
                        "metric": metric,
                        "seed_count": row["seed_count"],
                        "mean": row["mean"],
                        "ci95_low": row["ci95_low"],
                        "ci95_high": row["ci95_high"],
                        "verdict": "partial" if phase == "training" else "not_supported",
                    }
                )
    for row in comparisons:
        if (
            row.get("experiment") in {"autonomy", "transfer"}
            and row.get("target") in {"full_aassr", "full_transfer"}
            and row.get("metric") in {"learning_auc", "success", "adaptation_auc"}
        ):
            output.append(
                {
                    "research_question": "RQ1 comparison",
                    "experiment": row.get("experiment"),
                    "environment": row.get("environment"),
                    "phase": row.get("phase"),
                    "condition": f"{row.get('target')} vs {row.get('baseline')}",
                    "metric": row.get("metric"),
                    "seed_count": row.get("paired_seed_count"),
                    "mean": row.get("paired_mean_difference"),
                    "ci95_low": row.get("ci95_low"),
                    "ci95_high": row.get("ci95_high"),
                    "holm_p": row.get("p_value_holm"),
                    "verdict": (
                        "target_better"
                        if float(row.get("ci95_low", 0)) > 0
                        else "target_worse"
                        if float(row.get("ci95_high", 0)) < 0
                        else "no_clear_difference"
                    ),
                }
            )
    output.append(
        {
            "research_question": "RQ2 creativity",
            "experiment": "creativity",
            "environment": "multi_solution_dependency",
            "phase": "training",
            "condition": "full_aassr",
            "metric": "creative_candidate_count",
            "seed_count": 30,
            "mean": creativity.get("creative_candidate_count", 0),
            "ci95_low": "",
            "ci95_high": "",
            "verdict": "not_supported",
        }
    )
    return output


def derive_anomaly_analysis(
    seed_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    creativity: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    extreme_success_groups = [
        {
            "experiment": row.get("experiment"),
            "condition": row.get("condition"),
            "environment": row.get("environment"),
            "phase": row.get("phase"),
            "seed_count": row.get("seed_count"),
            "success_mean": row.get("mean"),
        }
        for row in cross_rows
        if row.get("metric") == "success"
        and numeric(row.get("mean")) in {0.0, 1.0}
        and not str(row.get("condition", "")).startswith("axis_")
    ]

    collapsed = collapse_seed_metric(seed_rows)
    vectors: dict[
        tuple[str, str, str, str, tuple[tuple[str, float], ...]], list[str]
    ] = defaultdict(list)
    group_values: dict[
        tuple[str, str, str, str, str], list[tuple[str, float]]
    ] = defaultdict(list)
    for (experiment, _suite, environment, phase, condition, seed), metrics in collapsed.items():
        if condition.startswith("axis_"):
            continue
        for metric in ("success", "learning_auc"):
            value = metrics.get(metric)
            if value is None:
                continue
            group_values[(experiment, environment, phase, condition, metric)].append(
                (seed, value)
            )
    for (experiment, environment, phase, condition, metric), values in group_values.items():
        vector = tuple(sorted((seed, round(value, 12)) for seed, value in values))
        vectors[(experiment, environment, phase, metric, vector)].append(condition)
    identical_condition_vectors = []
    for (experiment, environment, phase, metric, vector), conditions in vectors.items():
        unique_conditions = sorted(set(conditions))
        if len(unique_conditions) > 1:
            identical_condition_vectors.append(
                {
                    "experiment": experiment,
                    "environment": environment,
                    "phase": phase,
                    "metric": metric,
                    "conditions": unique_conditions,
                    "seed_count": len(vector),
                    "common_mean": statistics.fmean(value for _, value in vector),
                }
            )

    seed_dominance = []
    for (experiment, environment, phase, condition, metric), values in group_values.items():
        magnitudes = [abs(value) for _, value in values]
        total = sum(magnitudes)
        if len(values) < 10 or total <= 0:
            continue
        shares = [(seed, abs(value) / total, value) for seed, value in values]
        seed, share, value = max(shares, key=lambda item: item[1])
        if share > 0.5:
            seed_dominance.append(
                {
                    "experiment": experiment,
                    "environment": environment,
                    "phase": phase,
                    "condition": condition,
                    "metric": metric,
                    "dominant_seed": seed,
                    "dominant_value": value,
                    "share_of_absolute_total": share,
                    "seed_count": len(values),
                }
            )

    imagination_degradation = [
        {
            "experiment": row.get("experiment"),
            "environment": row.get("environment"),
            "phase": row.get("phase"),
            "metric": row.get("metric"),
            "target": row.get("target"),
            "baseline": row.get("baseline"),
            "paired_mean_difference": row.get("paired_mean_difference"),
            "ci95_low": row.get("ci95_low"),
            "ci95_high": row.get("ci95_high"),
            "p_value_holm": row.get("p_value_holm"),
            "interpretation": "Full AASSR underperformed a no-imagination/contextual comparator; this is associative because more than imagination differs between conditions.",
        }
        for row in comparisons
        if row.get("target") == "full_aassr"
        and row.get("baseline")
        in {"contextual_policy", "prophecy_no_imagination"}
        and row.get("metric") in {"learning_auc", "success"}
        and (numeric(row.get("ci95_high")) or 0.0) < 0.0
    ]

    high_prediction_failures = {
        name: int(item["episode_csv"].get("high_prediction_failed_rows", 0))
        for name, item in integrity["experiments"].items()
    }
    cross_index = {
        (
            str(row.get("experiment", "")),
            str(row.get("condition", "")),
            str(row.get("environment", "")),
            str(row.get("phase", "")),
            str(row.get("metric", "")),
        ): row
        for row in cross_rows
    }
    high_prediction_low_success_groups = []
    for key, prediction_row in cross_index.items():
        if key[-1] != "prediction_score":
            continue
        success_row = cross_index.get((*key[:-1], "success"))
        prediction_mean = numeric(prediction_row.get("mean"))
        success_mean = numeric(success_row.get("mean")) if success_row else None
        if (
            prediction_mean is not None
            and success_mean is not None
            and prediction_mean >= 0.8
            and success_mean <= 0.1
        ):
            high_prediction_low_success_groups.append(
                {
                    "experiment": key[0],
                    "condition": key[1],
                    "environment": key[2],
                    "phase": key[3],
                    "seed_count": prediction_row.get("seed_count"),
                    "prediction_score_mean": prediction_mean,
                    "success_mean": success_mean,
                }
            )
    family_counts = {
        str(name): int(count)
        for name, count in creativity.get("by_solution_family", {}).items()
    }
    family_total = sum(family_counts.values())
    largest_family = (
        max(family_counts.items(), key=lambda item: item[1])
        if family_counts
        else ("", 0)
    )
    observed_phase_gaps = {}
    for name, item in integrity["experiments"].items():
        config = read_json(CONFIG_ROOT / EXPERIMENTS[name]["config"])
        declared = {str(value) for value in config.get("phases", ())}
        observed = set(item["episode_csv"].get("phase_counts", {}))
        observed_phase_gaps[name] = {
            "declared": sorted(declared),
            "observed": sorted(observed),
            "declared_but_not_observed": sorted(declared - observed),
            "note": "Global protocol phases are not necessarily applicable to every suite; gaps are reported for auditability, not automatically classified as incomplete.",
        }
    return {
        "extreme_success_group_count": len(extreme_success_groups),
        "extreme_success_groups": extreme_success_groups,
        "identical_condition_vector_count": len(identical_condition_vectors),
        "identical_condition_vectors": identical_condition_vectors,
        "seed_dominance_group_count": len(seed_dominance),
        "seed_dominance_groups": seed_dominance,
        "seed_dominance_definition": "A single research seed contributes more than 50% of the group's absolute seed-level metric total.",
        "imagination_degradation_comparison_count": len(imagination_degradation),
        "imagination_degradation_comparisons": imagination_degradation,
        "high_prediction_failed_episode_counts": high_prediction_failures,
        "high_prediction_low_success_group_count": len(
            high_prediction_low_success_groups
        ),
        "high_prediction_low_success_groups": high_prediction_low_success_groups,
        "creativity_novelty_pass_count": sum(
            int(row.get("novel_above_threshold", 0))
            for row in creativity.get("by_condition", ())
        ),
        "creativity_strategy_count": int(creativity.get("strategy_rows", 0)),
        "largest_solution_family": largest_family[0],
        "largest_solution_family_count": largest_family[1],
        "largest_solution_family_share": (
            largest_family[1] / family_total if family_total else 0.0
        ),
        "observed_phase_gaps": observed_phase_gaps,
    }


def refresh_review_manifest(output: Path) -> None:
    manifest_path = output / "manifests" / "review_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "output": str(output),
            "source_results": str(RESULTS_ROOT.resolve()),
            "source_modified": False,
            "files": [
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(item for item in output.rglob("*") if item.is_file())
                if path != manifest_path and ".streaming_audit_tmp" not in path.parts
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper_final_review"),
        help="Output directory; source results are never modified.",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Rebuild lightweight reports from an existing review package without rescanning source data.",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.refresh_only:
        integrity = read_json(output / "integrity_report.json")
        seed_rows = load_csv(output / "seed_level_summary.csv")
        cross_rows = load_csv(output / "cross_seed_summary.csv")
        comparisons = load_csv(output / "condition_comparisons.csv")
        adaptation_rows = load_csv(output / "adaptation_curves.csv")
        creativity = integrity.get("creativity", {})
        human_status = integrity.get("human_data", {})
        integrity["anomaly_analysis"] = derive_anomaly_analysis(
            seed_rows,
            cross_rows,
            comparisons,
            creativity,
            integrity,
        )
        write_json(output / "integrity_report.json", integrity)
        (output / "result_report.md").write_text(
            result_report_markdown(
                integrity,
                cross_rows,
                comparisons,
                adaptation_rows,
                creativity,
                human_status,
            ),
            encoding="utf-8",
        )
        for experiment in EXPERIMENTS:
            target = output / "experiments" / experiment
            experiment_cross = [
                row for row in cross_rows if row.get("experiment") == experiment
            ]
            experiment_comparisons = [
                row for row in comparisons if row.get("experiment") == experiment
            ]
            write_json(
                target / "integrity_report.json",
                integrity["experiments"][experiment],
            )
            (target / "result_report.md").write_text(
                per_experiment_report(
                    experiment,
                    integrity["experiments"][experiment],
                    experiment_cross,
                    experiment_comparisons,
                ),
                encoding="utf-8",
            )
        refresh_review_manifest(output)
        print(f"Refreshed lightweight review: {output}")
        return 0
    temp_dir = output / ".streaming_audit_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    integrity: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(RESULTS_ROOT.resolve()),
        "read_only_source_policy": True,
        "experiments": {},
        "method": {
            "csv": "streaming csv.DictReader",
            "duplicates": "BLAKE2b-128 fingerprints counted in temporary SQLite",
            "gzip": "full streaming zlib level 6 byte count",
            "statistics_unit": "research_seed",
            "bootstrap_samples": 5000,
            "permutation_samples": 20000,
            "multiple_comparisons": "Holm within experiment/environment/phase/metric",
        },
    }
    all_seed_rows: list[dict[str, Any]] = []
    all_anomaly_samples: list[dict[str, Any]] = []
    inventories: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}

    for experiment, spec in EXPERIMENTS.items():
        result_root = RESULTS_ROOT / spec["protocol"]
        config_path = CONFIG_ROOT / spec["config"]
        pilot_path = CONFIG_ROOT / spec["pilot_config"]
        config = read_json(config_path)
        configs[experiment] = config
        manifest = read_json(result_root / "manifests" / "protocol_manifest.json")
        state = read_json(result_root / "manifests" / "run_state.json")
        episode_profile, samples, group_stats = profile_episode_csv(
            experiment,
            result_root / "raw" / "episodes.csv",
            config,
            temp_dir,
        )
        transition_profile, transition_samples = scan_transitions(
            experiment, result_root / "raw" / "transitions.jsonl"
        )
        all_anomaly_samples.extend(samples)
        all_anomaly_samples.extend(transition_samples)
        expected_seeds = {int(item) for item in config["research_seeds"]}
        actual_seeds = set(episode_profile["research_seeds"])
        validation_issues = validate_paper_artifacts(result_root)
        inventory = file_inventory(result_root)
        directory_size = sum(int(item["size_bytes"]) for item in inventory)
        storage = {
            "directory_size_bytes": directory_size,
            "files": inventory,
            "size_by_suffix": dict(
                Counter(
                    {
                        suffix: sum(
                            int(item["size_bytes"])
                            for item in inventory
                            if item["suffix"] == suffix
                        )
                        for suffix in sorted({item["suffix"] for item in inventory})
                    }
                )
            ),
        }
        cache_transition = result_root / "raw" / "suite_cache" / f"{experiment}_transitions.jsonl"
        if cache_transition.exists() and (result_root / "raw" / "transitions.jsonl").exists():
            storage["merged_cache_transition_same_size"] = (
                cache_transition.stat().st_size
                == (result_root / "raw" / "transitions.jsonl").stat().st_size
            )
            storage["merged_cache_transition_sha256_match"] = (
                file_sha256(cache_transition)
                == file_sha256(result_root / "raw" / "transitions.jsonl")
            )
        else:
            storage["merged_cache_transition_same_size"] = None
            storage["merged_cache_transition_sha256_match"] = None
        inventories[experiment] = storage
        lock_status = source_lock_status(config_path, result_root, config, manifest)
        overlap = seed_overlap_status(config, pilot_path)
        experiment_integrity = {
            "run": {
                "result_path": str(result_root.resolve()),
                "started_at_utc": state.get("started_at_utc", manifest.get("started_at_utc")),
                "experiment_completed_at_utc": manifest.get("completed_at_utc"),
                "pipeline_completed_at_utc": state.get("completed_at_utc"),
                "planned_rows": planned_paper_run_count(config),
                "actual_rows": episode_profile["row_count"],
                "state_row_count": state.get("row_count"),
                "expected_research_seed_count": len(expected_seeds),
                "completed_research_seed_count": len(actual_seeds),
                "missing_research_seeds": sorted(expected_seeds - actual_seeds),
                "unexpected_research_seeds": sorted(actual_seeds - expected_seeds),
                "failed_runs": manifest.get("failed_runs", ()),
                "completed_suites": state.get("completed_suites", ()),
            },
            "config": lock_status,
            "seed_overlap": overlap,
            "episode_csv": episode_profile,
            "transitions": transition_profile,
            "storage": storage,
            "artifact_validator": {
                "passed": not validation_issues,
                "issues": validation_issues,
            },
        }
        integrity["experiments"][experiment] = experiment_integrity
        seed_rows = load_seed_rows(experiment, result_root)
        all_seed_rows.extend(seed_rows)
        copy_manifests(output, experiment, config_path, result_root)

    axis_rows = add_ablation_axis_seed_rows(all_seed_rows)
    all_seed_rows.extend(axis_rows)
    cross_rows = cross_seed_rows(all_seed_rows)
    comparisons = condition_comparison_rows(all_seed_rows)
    adaptation_rows = adaptation_curve_rows(
        RESULTS_ROOT / EXPERIMENTS["transfer"]["protocol"]
    )
    creativity_root = RESULTS_ROOT / EXPERIMENTS["creativity"]["protocol"]
    frozen_path = creativity_root / "manifests" / "frozen_creativity_rules.json"
    frozen_rules = read_json(frozen_path) if frozen_path.exists() else None
    creativity, candidates = analyze_creativity(
        creativity_root,
        configs["creativity"],
        frozen_rules,
    )
    human_status = human_data_status(creativity_root, configs["creativity"])
    integrity["creativity"] = creativity
    integrity["human_data"] = human_status
    integrity["anomaly_analysis"] = derive_anomaly_analysis(
        all_seed_rows,
        cross_rows,
        comparisons,
        creativity,
        integrity,
    )

    write_json(output / "integrity_report.json", integrity)
    seed_fields = [
        "experiment",
        "suite",
        "condition",
        "environment",
        "model",
        "phase",
        "action_family",
        "seed",
        "episode_rows",
        *SEED_METRICS,
    ]
    append_csv_rows(output / "seed_level_summary.csv", all_seed_rows, seed_fields)
    write_csv(output / "cross_seed_summary.csv", cross_rows)
    write_csv(output / "condition_comparisons.csv", comparisons)
    write_csv(output / "adaptation_curves.csv", adaptation_rows)
    write_csv(output / "headline_results.csv", headline_rows(cross_rows, comparisons, creativity))
    with (output / "creativity_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    anomaly_fields = (
        "severity",
        "experiment",
        "anomaly_type",
        "identifier",
        "previous_identifier",
        "next_identifier",
        "metrics_json",
        "context_note",
    )
    append_csv_rows(output / "anomaly_samples.csv", all_anomaly_samples, anomaly_fields)

    inventory_md = build_inventory_markdown(integrity)
    (output / "experiment_inventory.md").write_text(inventory_md, encoding="utf-8")
    report_md = result_report_markdown(
        integrity,
        cross_rows,
        comparisons,
        adaptation_rows,
        creativity,
        human_status,
    )
    (output / "result_report.md").write_text(report_md, encoding="utf-8")

    experiments_dir = output / "experiments"
    for experiment in EXPERIMENTS:
        target = experiments_dir / experiment
        target.mkdir(parents=True, exist_ok=True)
        experiment_seed_rows = [row for row in all_seed_rows if row.get("experiment") == experiment]
        experiment_cross = [row for row in cross_rows if row.get("experiment") == experiment]
        experiment_comparisons = [row for row in comparisons if row.get("experiment") == experiment]
        append_csv_rows(target / "seed_level_summary.csv", experiment_seed_rows, seed_fields)
        write_csv(target / "cross_seed_summary.csv", experiment_cross)
        write_csv(target / "condition_comparisons.csv", experiment_comparisons)
        write_json(target / "integrity_report.json", integrity["experiments"][experiment])
        (target / "result_report.md").write_text(
            per_experiment_report(
                experiment,
                integrity["experiments"][experiment],
                experiment_cross,
                experiment_comparisons,
            ),
            encoding="utf-8",
        )

    generate_figures(output, cross_rows, adaptation_rows, creativity, inventories)
    chart_map = """# Chart map

| Figure | Question | Family | Fields | Supported claim |
|---|---|---|---|---|
| rq1_unseen_success.svg | Does Full AASSR generalize zero-shot? | Comparison/ranking bar | dependency length, condition, success | Full AASSR frozen unseen success is zero |
| rq1_learning_auc.svg | Does learning occur during training? | Comparison/ranking bar | dependency length, condition, AUC | Full learns but trails contextual/Prophecy |
| transfer_adaptation_curves.svg | Is adaptation faster after transfer? | Ordered line | budget, condition, success | Curves differ only slightly |
| ablation_axis_learning_auc.svg | Which imagination settings differ? | Comparison/ranking bar | axis level, marginal AUC | L6-only matrix effects |
| creativity_funnel.svg | How many strategies pass each creativity gate? | Progression bar | gate, count | Novelty gate removes all candidates |
| storage_by_experiment.svg | Why are artifacts large? | Comparison/ranking bar | experiment, GiB | Ablation and autonomy dominate storage |

Palette policy: single blue root with neutral axes and direct labels. All absolute-magnitude bars start at zero. SVGs were selected because the user requested a lightweight local review package and no plotting runtime is installed.
"""
    (output / "manifests" / "chart_map.md").write_text(chart_map, encoding="utf-8")
    refresh_review_manifest(output)
    try:
        temp_dir.rmdir()
    except OSError:
        pass
    print(f"Review package: {output}")
    print(f"Seed rows: {len(all_seed_rows):,}")
    print(f"Cross-seed rows: {len(cross_rows):,}")
    print(f"Comparison rows: {len(comparisons):,}")
    print(f"Anomaly samples: {len(all_anomaly_samples):,}")
    print(f"Creativity candidates: {len(candidates):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
