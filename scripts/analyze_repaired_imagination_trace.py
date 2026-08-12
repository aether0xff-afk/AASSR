from __future__ import annotations

import argparse
import json
from collections import Counter
from math import exp, log
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from aassr_v2.current_generation import relational_action_key
from aassr_v2.current_relational_codec import (
    descriptor,
    legal_action_mask,
    semantic_prediction_score,
    terminal_class,
)
from aassr_v2.skills import SKILL_VERB
from aassr_v2.types import Action, Prediction, StateSnapshot


ANALYSIS_VERSION = "repaired-imagination-trace-analysis-v2-probability"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.append(json.loads(line))
    return output


def _parse_parameters(text: str) -> dict[str, Any]:
    if not text:
        return {}
    output: dict[str, Any] = {}
    for item in text.split(","):
        key, encoded = item.split("=", 1)
        output[key] = json.loads(encoded)
    return output


def _action_from_signature(signature: str) -> Action:
    parts = signature.split("|", 4)
    if len(parts) < 4:
        raise ValueError(f"invalid action signature: {signature!r}")
    verb, target, tool, destination = parts[:4]
    parameters = _parse_parameters(parts[4]) if len(parts) == 5 else {}
    return Action(
        verb,
        target=None if target == "_" else target,
        tool=None if tool == "_" else tool,
        destination=None if destination == "_" else destination,
        parameters=parameters,
    )


def _action_from_dict(row: dict[str, Any] | None) -> Action | None:
    if not row:
        return None
    return Action(
        str(row.get("verb", "")),
        target=row.get("target"),
        tool=row.get("tool"),
        destination=row.get("destination"),
        metadata=dict(row.get("metadata") or {}),
        parameters=dict(row.get("parameters") or {}),
    )


def _state(row: dict[str, Any]) -> StateSnapshot:
    actions = tuple(
        _action_from_signature(str(signature))
        for signature in row.get("available_action_signatures", ())
    )
    return StateSnapshot(
        tuple(float(value) for value in row.get("vector", ())),
        facts=frozenset(str(fact) for fact in row.get("facts", ())),
        available_actions=actions,
        goal_progress=float(row.get("goal_progress", 0.0)),
        metadata=dict(row.get("metadata") or {}),
    )


def _prediction(row: dict[str, Any]) -> Prediction:
    return Prediction(
        _state(dict(row["next_state"])),
        float(row.get("probability", 0.0)),
        source=str(row.get("source", "trace")),
    )


def _normalized_masses(rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    raw = [max(0.0, float(row.get("outcome_probability", 1.0))) for row in rows]
    total = sum(raw)
    if total <= 1e-12:
        return [1.0 / len(rows)] * len(rows)
    return [value / total for value in raw]


def _structural_key(state: StateSnapshot, action: Action) -> tuple[Any, ...]:
    if action.verb_name == SKILL_VERB:
        return ("skill", str(action.target))
    return ("primitive", *relational_action_key(state, action))


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = sorted(float(value) for value in values)
    if not rows:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(rows),
        "mean": fmean(rows),
        "min": rows[0],
        "max": rows[-1],
    }


def _mask_jaccard(left: StateSnapshot, right: StateSnapshot) -> float:
    a = legal_action_mask(left)
    b = legal_action_mask(right)
    left_set = {index for index, value in enumerate(a) if value >= 0.5}
    right_set = {index for index, value in enumerate(b) if value >= 0.5}
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _analyze_transition(trace: dict[str, Any]) -> dict[str, Any] | None:
    raw_rows = [
        row
        for row in trace.get("predictions", ())
        if isinstance(row, dict) and "next_state" in row
    ]
    predictions = tuple(_prediction(row) for row in raw_rows)
    if not predictions:
        return None
    actual = _state(dict(trace["after"]))
    masses = _normalized_masses(raw_rows)
    individual_scores = [
        semantic_prediction_score((prediction,), actual)
        for prediction in predictions
    ]
    best_index = max(
        range(len(individual_scores)),
        key=lambda index: individual_scores[index],
    )
    weighted_score = sum(
        mass * score for mass, score in zip(masses, individual_scores, strict=True)
    )
    entropy = -sum(
        mass * log(max(1e-12, mass)) for mass in masses if mass > 0.0
    )
    raw_mass_sum = sum(
        max(0.0, float(row.get("outcome_probability", 1.0)))
        for row in raw_rows
    )
    distinct = {
        tuple(round(value, 8) for value in descriptor(item.next_state))
        for item in predictions
    }
    reliability = sum(
        mass * max(0.0, min(1.0, float(prediction.probability)))
        for mass, prediction in zip(masses, predictions, strict=True)
    )
    best_mode_mass = masses[best_index]
    return {
        "semantic_score": max(individual_scores),
        "probability_weighted_semantic_score": weighted_score,
        "actual_best_mode_probability": best_mode_mass,
        "actual_best_mode_surprisal": -log(max(1e-12, best_mode_mass)),
        "effective_outcome_count": exp(entropy),
        "outcome_mass_sum_error": abs(raw_mass_sum - 1.0),
        "reliability": reliability,
        "distinct_relational_futures": len(distinct),
        "topk_legal_mask_jaccard": max(
            _mask_jaccard(item.next_state, actual) for item in predictions
        ),
        "topk_terminal_match": int(
            any(terminal_class(item.next_state) == terminal_class(actual) for item in predictions)
        ),
        "error": int(bool(trace.get("error", False))),
        "reward": float(trace.get("real_reward", 0.0)),
    }


def _analyze_condition(events: list[dict[str, Any]]) -> dict[str, Any]:
    gate_reasons: Counter[str] = Counter()
    raw_gaps: list[float] = []
    structural_gaps: list[float] = []
    root_coverage: list[float] = []
    structural_root_counts: list[int] = []
    concrete_root_counts: list[int] = []
    alias_exact_ties = 0
    distinct_structure_exact_ties = 0
    near_structure_ties = 0
    transition_rows: list[dict[str, Any]] = []
    intervention_events = []

    for event in events:
        gate_reasons[str(event.get("gate_reason", "unknown"))] += 1
        for trace in event.get("real_traces", ()):
            row = _analyze_transition(trace)
            if row is not None:
                transition_rows.append(row)

        if bool(event.get("intervention_allowed", False)):
            intervention_events.append(event)

        roots = list(event.get("planner_root_evaluations", ()))
        if not roots:
            continue
        state = _state(dict(event["state"]))
        expected = len(state.available_actions)
        evaluated = len(roots)
        root_coverage.append(evaluated / max(1, expected))
        concrete_root_counts.append(evaluated)

        concrete_values = sorted(
            (float(root.get("planner_aggregate_value", 0.0)) for root in roots),
            reverse=True,
        )
        if len(concrete_values) > 1:
            raw_gaps.append(concrete_values[0] - concrete_values[1])

        grouped: dict[tuple[Any, ...], list[float]] = {}
        top_value = concrete_values[0]
        top_keys = set()
        top_concrete = 0
        for root in roots:
            action = _action_from_dict(root.get("action"))
            if action is None:
                continue
            key = _structural_key(state, action)
            value = float(root.get("planner_aggregate_value", 0.0))
            grouped.setdefault(key, []).append(value)
            if abs(value - top_value) <= 1e-12:
                top_concrete += 1
                top_keys.add(key)

        structural_root_counts.append(len(grouped))
        if top_concrete > 1:
            if len(top_keys) == 1:
                alias_exact_ties += 1
            else:
                distinct_structure_exact_ties += 1

        group_values = sorted(
            (fmean(values) for values in grouped.values()),
            reverse=True,
        )
        if len(group_values) > 1:
            gap = group_values[0] - group_values[1]
            structural_gaps.append(gap)
            if 1e-12 < gap < 0.01:
                near_structure_ties += 1

    semantic_scores = [row["semantic_score"] for row in transition_rows]
    weighted_semantic_scores = [
        row["probability_weighted_semantic_score"] for row in transition_rows
    ]
    best_mode_masses = [
        row["actual_best_mode_probability"] for row in transition_rows
    ]
    best_mode_surprisal = [
        row["actual_best_mode_surprisal"] for row in transition_rows
    ]
    effective_outcomes = [
        row["effective_outcome_count"] for row in transition_rows
    ]
    mass_errors = [row["outcome_mass_sum_error"] for row in transition_rows]
    reliabilities = [row["reliability"] for row in transition_rows]
    distinct_futures = [row["distinct_relational_futures"] for row in transition_rows]
    mask_scores = [row["topk_legal_mask_jaccard"] for row in transition_rows]
    terminal_matches = [row["topk_terminal_match"] for row in transition_rows]

    intervention_errors = sum(
        any(bool(trace.get("error", False)) for trace in event.get("real_traces", ()))
        for event in intervention_events
    )
    intervention_successes = sum(
        bool((event.get("episode_result") or {}).get("success", 0))
        for event in intervention_events
    )

    return {
        "events": len(events),
        "imagination_runs": sum(bool(event.get("used_imagination", False)) for event in events),
        "gate_reasons": dict(gate_reasons),
        "root_contract": {
            "plans": len(root_coverage),
            "coverage": _distribution(root_coverage),
            "all_roots_preserved": bool(root_coverage) and min(root_coverage) >= 1.0,
            "concrete_root_count": _distribution(concrete_root_counts),
            "structural_root_count": _distribution(structural_root_counts),
        },
        "critic_discrimination": {
            "raw_top1_top2_gap": _distribution(raw_gaps),
            "structural_top1_top2_gap": _distribution(structural_gaps),
            "exact_ties_equivalent_aliases": alias_exact_ties,
            "exact_ties_distinct_structures": distinct_structure_exact_ties,
            "near_ties_distinct_structures": near_structure_ties,
        },
        "prophecy": {
            "real_transitions_scored": len(transition_rows),
            "semantic_topk_score": _distribution(semantic_scores),
            "semantic_probability_weighted_score": _distribution(
                weighted_semantic_scores
            ),
            "actual_best_mode_probability": _distribution(best_mode_masses),
            "actual_best_mode_surprisal": _distribution(best_mode_surprisal),
            "effective_outcome_count": _distribution(effective_outcomes),
            "outcome_mass_sum_error": _distribution(mass_errors),
            "reliability": _distribution(reliabilities),
            "legal_action_mask_topk_jaccard": _distribution(mask_scores),
            "terminal_topk_match_rate": (
                fmean(terminal_matches) if terminal_matches else None
            ),
            "distinct_relational_futures": _distribution(distinct_futures),
            "multi_future_transition_count": sum(value > 1 for value in distinct_futures),
        },
        "interventions": {
            "count": len(intervention_events),
            "real_error_count": intervention_errors,
            "episode_success_count": intervention_successes,
        },
    }


def analyze(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    files = sorted(output.glob("decision_trace_*.jsonl"))
    conditions = {
        path.stem.removeprefix("decision_trace_"): _analyze_condition(_read_jsonl(path))
        for path in files
    }
    result = {
        "version": ANALYSIS_VERSION,
        "conditions": conditions,
    }
    (output / "repair_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
