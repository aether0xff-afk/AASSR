from __future__ import annotations

import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from .experiment_statistics import (
    GROUP_FIELDS,
    cross_seed_summary,
    seed_level_rows,
)
from .experiment_runner import SUMMARY_METRICS
from .human_study import (
    human_automatic_concordance,
    inter_rater_agreement,
)
from .paper_protocol import PaperPaths
from .paper_types import StrategyRecord


def _numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(path: str | Path) -> Iterable[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def write_csv_rows(
    path: str | Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        if fields:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
    return destination


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 < confidence < 1.0 or samples <= 0:
        raise ValueError("invalid bootstrap settings")
    if len(values) == 1:
        return values[0], values[0]
    randomizer = random.Random(seed)
    means = sorted(
        fmean(randomizer.choice(values) for _ in values)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(samples - 1, int(tail * samples)))
    upper_index = max(
        0, min(samples - 1, int((1.0 - tail) * samples) - 1)
    )
    return means[lower_index], means[upper_index]


def paired_permutation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 20000,
    seed: int = 0,
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must be non-empty and equally sized")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    observed = abs(fmean(differences))
    if all(value == 0.0 for value in differences):
        return 1.0
    extreme = 0
    total = 0
    exact = len(differences) <= 18
    if exact:
        signs: Iterable[tuple[int, ...]] = itertools.product(
            (-1, 1), repeat=len(differences)
        )
    else:
        randomizer = random.Random(seed)
        signs = (
            tuple(
                1 if randomizer.random() < 0.5 else -1
                for _ in differences
            )
            for _ in range(samples)
        )
    for combination in signs:
        result = abs(
            fmean(
                sign * value
                for sign, value in zip(
                    combination, differences, strict=True
                )
            )
        )
        extreme += int(result >= observed - 1e-15)
        total += 1
    return (
        extreme / total
        if exact
        else (extreme + 1) / (total + 1)
    )


def holm_correction(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        corrected = min(1.0, (count - rank) * float(value))
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted


def trapezoid_auc(points: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(points)
    if len(ordered) < 2:
        return ordered[0][1] if ordered else 0.0
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(
            ordered, ordered[1:]
        )
    )
    width = ordered[-1][0] - ordered[0][0]
    return area / width if width else ordered[-1][1]


def threshold_episode(
    points: Sequence[tuple[int, float]], threshold: float
) -> int | None:
    return next(
        (episode for episode, success in sorted(points) if success >= threshold),
        None,
    )


def _add_seed_bootstrap_bounds(
    summaries: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            tuple(str(row.get(field, "")) for field in GROUP_FIELDS)
        ].append(row)
    result: list[dict[str, Any]] = []
    for summary in summaries:
        record = dict(summary)
        key = tuple(str(summary.get(field, "")) for field in GROUP_FIELDS)
        items = grouped.get(key, ())
        for metric in SUMMARY_METRICS:
            values = [
                numeric
                for item in items
                if (numeric := _numeric(item.get(metric))) is not None
            ]
            if values:
                low, high = bootstrap_confidence_interval(
                    values,
                    confidence=confidence,
                    samples=bootstrap_samples,
                    seed=int.from_bytes(
                        f"{key}:{metric}".encode("utf-8"), "little"
                    )
                    % (2**32),
                )
                record[f"{metric}_ci95_low"] = low
                record[f"{metric}_ci95_high"] = high
            else:
                record[f"{metric}_ci95_low"] = ""
                record[f"{metric}_ci95_high"] = ""
        result.append(record)
    return result


def _comparison_rows(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
    bootstrap_samples: int,
    permutation_samples: int,
) -> list[dict[str, Any]]:
    by_group: dict[
        tuple[str, str, str], dict[str, dict[str, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in seed_rows:
        success = _numeric(row.get("success"))
        if success is None:
            continue
        group = (
            str(row.get("suite", "")),
            str(row.get("environment", "")),
            str(row.get("phase", "")),
        )
        by_group[group][str(row.get("condition", ""))][
            str(row.get("seed", ""))
        ] = success

    result: list[dict[str, Any]] = []
    for group, conditions in sorted(by_group.items()):
        target_name = next(
            (
                name
                for name in ("full_aassr", "full", "full_transfer")
                if name in conditions
            ),
            None,
        )
        if target_name is None:
            continue
        target = conditions[target_name]
        for baseline_name, baseline in sorted(conditions.items()):
            if baseline_name == target_name or baseline_name.startswith("oracle"):
                continue
            common = sorted(set(target) & set(baseline))
            if not common:
                continue
            left = [target[key] for key in common]
            right = [baseline[key] for key in common]
            differences = [a - b for a, b in zip(left, right, strict=True)]
            low, high = bootstrap_confidence_interval(
                differences,
                confidence=confidence,
                samples=bootstrap_samples,
                seed=int.from_bytes(
                    "|".join((*group, baseline_name)).encode("utf-8"),
                    "little",
                )
                % (2**32),
            )
            result.append(
                {
                    "suite": group[0],
                    "environment": group[1],
                    "phase": group[2],
                    "model": "mixed_baselines",
                    "target": target_name,
                    "baseline": baseline_name,
                    "paired_seed_count": len(common),
                    "target_mean": fmean(left),
                    "baseline_mean": fmean(right),
                    "paired_mean_difference": fmean(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "effect_size_dz": (
                        fmean(differences) / stdev(differences)
                        if len(differences) > 1
                        and stdev(differences) > 0.0
                        else 0.0
                    ),
                    "p_value": paired_permutation_test(
                        left, right, samples=permutation_samples
                    ),
                    "oracle_excluded": True,
                }
            )
    adjusted = holm_correction(
        [float(row["p_value"]) for row in result]
    )
    for row, value in zip(result, adjusted, strict=True):
        row["p_value_holm"] = value
    return result


def _adaptation_rows(
    episode_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str], dict[int, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    calibration: dict[
        tuple[str, str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    for row in episode_rows:
        if str(row.get("phase", "")) != "evaluation_unseen_adaptation":
            continue
        budget = _numeric(row.get("adaptation_budget"))
        success = _numeric(row.get("success"))
        if budget is None or success is None:
            continue
        key = (
            str(row.get("condition", "")),
            str(row.get("environment", "")),
            str(row.get("seed", row.get("research_seed", ""))),
        )
        grouped[key][int(budget)].append(success)
        prediction = _numeric(row.get("prediction_score"))
        if prediction is not None:
            calibration[key].append((prediction, success))
    result: list[dict[str, Any]] = []
    for key, budgets in sorted(grouped.items()):
        points = sorted(
            (budget, fmean(values)) for budget, values in budgets.items()
        )
        result.append(
            {
                "condition": key[0],
                "environment": key[1],
                "seed": key[2],
                "adaptation_auc": trapezoid_auc(
                    [(float(x), y) for x, y in points]
                ),
                "episodes_to_50": threshold_episode(points, 0.5),
                "episodes_to_80": threshold_episode(points, 0.8),
                "curve_json": json.dumps(points, separators=(",", ":")),
                "mean_unseen_prediction_score": (
                    fmean(value for value, _ in calibration[key])
                    if calibration[key]
                    else ""
                ),
                "unseen_prediction_calibration_error": (
                    fmean(
                        abs(prediction - success)
                        for prediction, success in calibration[key]
                    )
                    if calibration[key]
                    else ""
                ),
            }
        )
    scratch = {
        (row["environment"], row["seed"]): row
        for row in result
        if row["condition"] == "from_scratch_full_aassr"
    }
    for row in result:
        baseline = scratch.get((row["environment"], row["seed"]))
        row["transfer_gain_vs_from_scratch"] = (
            float(row["adaptation_auc"])
            - float(baseline["adaptation_auc"])
            if baseline is not None
            else ""
        )
        for suffix in ("50", "80"):
            baseline_episode = (
                baseline.get(f"episodes_to_{suffix}")
                if baseline is not None
                else None
            )
            current_episode = row.get(f"episodes_to_{suffix}")
            row[f"sample_saving_to_{suffix}"] = (
                int(baseline_episode) - int(current_episode)
                if baseline_episode not in (None, "")
                and current_episode not in (None, "")
                else ""
            )
    return result


def _curve_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    target_names: Sequence[str],
    suite_label: str,
    confidence: float,
    bootstrap_samples: int,
    permutation_samples: int,
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str], dict[str, dict[str, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        value = _numeric(row.get(metric))
        if value is None:
            continue
        source_suite = str(row.get("suite", "")).strip()
        label = (
            f"{suite_label}:{source_suite}"
            if source_suite
            else suite_label
        )
        groups[(label, str(row.get("environment", "")))][
            str(row.get("condition", ""))
        ][str(row.get("seed", ""))] = value
    result: list[dict[str, Any]] = []
    for group, conditions in sorted(groups.items()):
        target_name = next(
            (name for name in target_names if name in conditions), None
        )
        if target_name is None:
            continue
        target = conditions[target_name]
        for baseline_name, baseline in sorted(conditions.items()):
            if baseline_name == target_name or baseline_name.startswith("oracle"):
                continue
            common = sorted(set(target) & set(baseline))
            if not common:
                continue
            left = [target[key] for key in common]
            right = [baseline[key] for key in common]
            differences = [
                a - b for a, b in zip(left, right, strict=True)
            ]
            low, high = bootstrap_confidence_interval(
                differences,
                confidence=confidence,
                samples=bootstrap_samples,
                seed=int.from_bytes(
                    f"{group}:{metric}:{baseline_name}".encode("utf-8"),
                    "little",
                )
                % (2**32),
            )
            deviation = stdev(differences) if len(differences) > 1 else 0.0
            result.append(
                {
                    "suite": group[0],
                    "environment": group[1],
                    "phase": metric,
                    "metric": metric,
                    "target": target_name,
                    "baseline": baseline_name,
                    "paired_seed_count": len(common),
                    "target_mean": fmean(left),
                    "baseline_mean": fmean(right),
                    "paired_mean_difference": fmean(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "effect_size_dz": (
                        fmean(differences) / deviation
                        if deviation > 0.0
                        else 0.0
                    ),
                    "p_value": paired_permutation_test(
                        left, right, samples=permutation_samples
                    ),
                    "oracle_excluded": True,
                }
            )
    adjusted = holm_correction(
        [float(row["p_value"]) for row in result]
    )
    for row, value in zip(result, adjusted, strict=True):
        row["p_value_holm"] = value
    return result


def _learning_curve_rows(
    episode_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str], list[tuple[int, float, int]]
    ] = defaultdict(list)
    for row in episode_rows:
        if str(row.get("phase", "")) != "training":
            continue
        if str(row.get("suite", "")) not in {"autonomy", "ablation"}:
            continue
        key = (
            str(row.get("suite", "")),
            str(row.get("condition", "")),
            str(row.get("environment", "")),
            str(row.get("seed", row.get("research_seed", ""))),
        )
        groups[key].append(
            (
                int(float(row["episode"])),
                float(row.get("success") or 0.0),
                int(
                    float(
                        row.get("real_transitions")
                        or row.get("steps")
                        or 0
                    )
                ),
            )
        )
    result = []
    for key, items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: item[0])
        success = [item[1] for item in ordered]
        transitions = [item[2] for item in ordered]
        cumulative = 0
        first_success_transitions: int | None = None
        for value, count in zip(success, transitions, strict=True):
            cumulative += count
            if value > 0.0 and first_success_transitions is None:
                first_success_transitions = cumulative
        tail_count = max(1, math.ceil(len(success) * 0.1))
        result.append(
            {
                "suite": key[0],
                "condition": key[1],
                "environment": key[2],
                "seed": key[3],
                "learning_auc": trapezoid_auc(
                    [
                        (float(index), value)
                        for index, value in enumerate(success)
                    ]
                ),
                "first_success_real_transitions": (
                    first_success_transitions
                    if first_success_transitions is not None
                    else ""
                ),
                "final_10_percent_success": fmean(success[-tail_count:]),
                "mean_primitive_steps": fmean(transitions),
                "episode_count": len(success),
            }
        )
    return result


def _creativity_rows(
    records: Sequence[StrategyRecord],
    *,
    frozen_rules: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []
    family_seeds: dict[str, set[int]] = defaultdict(set)
    family_worlds: dict[str, set[int]] = defaultdict(set)
    for record in records:
        family = record.graph.solution_family
        family_seeds[family].add(record.research_seed)
        family_worlds[family].add(record.world_seed)
    utility_profiles = {
        (
            int(record.success),
            record.primitive_steps,
            record.errors,
            record.resources_used,
            record.risk_entries,
        )
        for record in records
    }
    dominated_profiles: dict[tuple[Any, ...], bool] = {}
    for profile in utility_profiles:
        dominated_profiles[profile] = any(
            other[0] >= profile[0]
            and other[1] <= profile[1]
            and other[2] <= profile[2]
            and other[3] <= profile[3]
            and other[4] <= profile[4]
            and other != profile
            for other in utility_profiles
        )
    result = []
    for record in records:
        profile = (
            int(record.success),
            record.primitive_steps,
            record.errors,
            record.resources_used,
            record.risk_entries,
        )
        dominated = dominated_profiles[profile]
        family = record.graph.solution_family
        reproducible = (
            len(family_seeds[family]) >= 3
            or len(family_worlds[family]) >= 3
            or record.reusable_success_rate >= 0.5
        )
        components = dict(record.novelty_components)
        novelty_threshold = (
            _numeric(frozen_rules.get("novelty_threshold"))
            if frozen_rules
            else None
        )
        utility = (
            frozen_rules.get("utility_criteria", {})
            if frozen_rules
            else {}
        )
        minimum_reuse = float(
            utility.get("minimum_reusable_success_rate", 0.0)
        )
        utility_qualified = bool(
            record.valid
            and record.success
            and record.reusable_success_rate >= minimum_reuse
            and (
                not utility.get(
                    "must_not_be_dominated_on_all_utility_metrics", False
                )
                or not dominated
            )
        )
        novel_above_threshold = bool(
            novelty_threshold is not None
            and record.novelty_score is not None
            and record.novelty_score >= novelty_threshold
        )
        result.append(
            {
                "strategy_id": record.strategy_id,
                "source_kind": record.source_kind,
                "solution_family": family,
                "valid": int(record.valid and record.success),
                "novelty_score": record.novelty_score,
                "graph_edit_distance": components.get("graph_edit", ""),
                "motif_jaccard_distance": components.get(
                    "motif_jaccard", ""
                ),
                "prerequisite_edge_distance": components.get(
                    "prerequisite_edges", ""
                ),
                "solution_family_distance": components.get(
                    "solution_family", ""
                ),
                "effect_sequence_distance": components.get(
                    "effect_sequence", ""
                ),
                "utility_non_dominated": int(not dominated),
                "novel_above_frozen_threshold": (
                    int(novel_above_threshold)
                    if novelty_threshold is not None
                    else ""
                ),
                "utility_qualified": (
                    int(utility_qualified) if frozen_rules else ""
                ),
                "creative_candidate": (
                    int(
                        novel_above_threshold
                        and utility_qualified
                        and reproducible
                    )
                    if frozen_rules
                    else ""
                ),
                "reproducible": int(reproducible),
                "reproducing_seed_count": len(family_seeds[family]),
                "reproducing_world_count": len(family_worlds[family]),
                "reusable_success_rate": record.reusable_success_rate,
            }
        )
    return result


def analyze_paper_results(
    root: str | Path,
) -> dict[str, Path | int]:
    paths = PaperPaths.create(root)
    resolved_config_path = paths.manifests / "resolved_config.json"
    resolved_config = (
        json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if resolved_config_path.exists()
        else {}
    )
    statistics = resolved_config.get("statistics", {})
    confidence = float(statistics.get("confidence", 0.95))
    bootstrap_samples = int(
        statistics.get("bootstrap_samples", 5000)
    )
    permutation_samples = int(
        statistics.get("permutation_samples", 20000)
    )
    episodes_path = paths.raw / "episodes.csv"
    per_seed = seed_level_rows(iter_csv_rows(episodes_path))
    episode_count = sum(int(row["episode_rows"]) for row in per_seed)
    cross_seed = _add_seed_bootstrap_bounds(
        cross_seed_summary(per_seed),
        per_seed,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
    )
    comparisons = _comparison_rows(
        per_seed,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    adaptation = _adaptation_rows(iter_csv_rows(episodes_path))
    learning = _learning_curve_rows(iter_csv_rows(episodes_path))
    learning_comparisons = _curve_comparison_rows(
        learning,
        metric="learning_auc",
        target_names=("full_aassr", "full"),
        suite_label="learning_auc",
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    transfer_comparisons = _curve_comparison_rows(
        adaptation,
        metric="adaptation_auc",
        target_names=("full_transfer",),
        suite_label="transfer",
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    strategies_path = paths.raw / "strategies.jsonl"
    strategy_records = (
        [
            StrategyRecord.from_dict(json.loads(line))
            for line in strategies_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if strategies_path.exists()
        else []
    )
    human_paths = paths.raw / "human_paths.jsonl"
    human_records = (
        [
            StrategyRecord.from_dict(json.loads(line))
            for line in human_paths.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if human_paths.exists()
        else []
    )
    frozen_path = paths.manifests / "frozen_creativity_rules.json"
    frozen_rules = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.exists()
        else None
    )
    creativity = _creativity_rows(
        [*strategy_records, *human_records],
        frozen_rules=frozen_rules,
    )
    seed_path = write_csv_rows(paths.seed_level / "seed_summary.csv", per_seed)
    summary_path = write_csv_rows(
        paths.statistics / "cross_seed_summary.csv", cross_seed
    )
    comparisons_path = write_csv_rows(
        paths.statistics / "condition_comparisons.csv", comparisons
    )
    adaptation_path = write_csv_rows(
        paths.statistics / "adaptation_summary.csv", adaptation
    )
    learning_path = write_csv_rows(
        paths.statistics / "learning_summary.csv", learning
    )
    learning_comparisons_path = write_csv_rows(
        paths.statistics / "learning_auc_comparisons.csv",
        learning_comparisons,
    )
    transfer_comparisons_path = write_csv_rows(
        paths.statistics / "transfer_comparisons.csv",
        transfer_comparisons,
    )
    creativity_path = write_csv_rows(
        paths.statistics / "creativity_summary.csv", creativity
    )
    human_ratings = paths.raw / "human_ratings.csv"
    agreement: dict[str, Any] = {}
    if human_ratings.exists():
        rating_rows = read_csv_rows(human_ratings)
        agreement = {
            **inter_rater_agreement(rating_rows),
            "automatic_concordance": human_automatic_concordance(
                [*strategy_records, *human_records], rating_rows
            ),
        }
    summary_json = paths.statistics / "analysis_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "episode_rows": episode_count,
                "seed_rows": len(per_seed),
                "comparison_rows": len(comparisons),
                "adaptation_rows": len(adaptation),
                "learning_curve_rows": len(learning),
                "learning_auc_comparison_rows": len(
                    learning_comparisons
                ),
                "transfer_comparison_rows": len(transfer_comparisons),
                "creativity_rows": len(creativity),
                "human_agreement": agreement,
                "method": {
                    "unit": "research_seed",
                    "confidence_interval": "seed bootstrap 95%",
                    "test": "paired permutation",
                    "multiple_comparisons": "Holm",
                    "oracle_in_inference": False,
                    "confidence": confidence,
                    "bootstrap_samples": bootstrap_samples,
                    "permutation_samples": permutation_samples,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "episodes": episodes_path,
        "seed_summary": seed_path,
        "cross_seed_summary": summary_path,
        "comparisons": comparisons_path,
        "adaptation": adaptation_path,
        "learning": learning_path,
        "learning_comparisons": learning_comparisons_path,
        "transfer_comparisons": transfer_comparisons_path,
        "creativity": creativity_path,
        "analysis_summary": summary_json,
        "episode_rows": episode_count,
    }
