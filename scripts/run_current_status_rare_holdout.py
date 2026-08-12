from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Sequence

from aassr_v2.current_relational_mixture_model import RelationalMixtureProphecyConfig
from aassr_v2.current_relational_state_v3 import (
    STATUS_CODES_V3,
    install_status_aware_relational_contract,
    latest_status_code,
)
from aassr_v2.current_status_models import (
    STATUS_OBJECTIVE,
    StatusAwareConditionalMixtureRelationalProphecy,
)
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_DIAGNOSTIC_SEEDS,
    TRANSFER_STAGES,
    TRANSFER_TRAIN_SEEDS,
    TransferDiagnosticWorld,
)
from aassr_v2.types import Action, StateSnapshot


DIAGNOSTIC_VERSION = "generic-rare-public-status-holdout-v2-broad-coverage"


@dataclass(frozen=True, slots=True)
class Transition:
    before: StateSnapshot
    action: Action
    after: StateSnapshot


def _collect(
    seeds: Sequence[int],
    *,
    stage_indices: Sequence[int],
    steps_per_episode: int,
    behavior_seed: int,
) -> list[Transition]:
    """Collect real public transitions without using task correctness or hidden state.

    The behavior policy is intentionally weak and generic: uniformly sample from
    the currently public legal action surface with a deterministic RNG. It never
    reads the target route/profile/object, audit score, session countdown, stage
    internals, or response status when choosing the next action.
    """

    rows: list[Transition] = []
    rng = random.Random(int(behavior_seed))
    for scenario_seed in seeds:
        for stage_index in stage_indices:
            world = TransferDiagnosticWorld(
                int(scenario_seed),
                stage=TRANSFER_STAGES[int(stage_index)],
            )
            for _ in range(int(steps_per_episode)):
                if world.success or (world.failed and world.locked) or world.rate_limited:
                    break
                before = world.snapshot()
                actions = tuple(before.available_actions)
                if not actions:
                    break
                action = actions[rng.randrange(len(actions))]
                world.step(action)
                after = world.snapshot()
                if latest_status_code(after) is not None:
                    rows.append(Transition(before, action, after))
    return rows


def _normalized_masses(predictions: Iterable[object]) -> list[float]:
    materialized = list(predictions)
    if not materialized:
        return []
    raw = [
        max(0.0, float(getattr(item, "outcome_probability", 1.0)))
        for item in materialized
    ]
    total = sum(raw)
    if total <= 1e-12:
        return [1.0 / len(materialized)] * len(materialized)
    return [value / total for value in raw]


def _frequency_groups(
    train_counts: dict[int, int],
    holdout_counts: dict[int, int],
    *,
    min_train: int = 4,
    min_holdout: int = 2,
) -> dict[str, list[int]]:
    """Split supported classes by empirical frequency only, never by code meaning."""

    eligible = sorted(
        code
        for code in STATUS_CODES_V3
        if int(train_counts.get(code, 0)) >= int(min_train)
        and int(holdout_counts.get(code, 0)) >= int(min_holdout)
    )
    if not eligible:
        return {"eligible": [], "rare": [], "common": []}
    pivot = float(median(train_counts[code] for code in eligible))
    rare = [code for code in eligible if float(train_counts[code]) <= pivot]
    common = [code for code in eligible if float(train_counts[code]) > pivot]
    return {"eligible": eligible, "rare": rare, "common": common}


def _evaluate(
    model: StatusAwareConditionalMixtureRelationalProphecy,
    rows: Sequence[Transition],
    *,
    samples: int,
) -> dict[str, Any]:
    by_status: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        actual = latest_status_code(row.after)
        if actual is None:
            continue
        predictions = tuple(model.predict(row.before, row.action, samples=int(samples)))
        masses = _normalized_masses(predictions)
        mass_by_status: Counter[int] = Counter()
        for prediction, mass in zip(predictions, masses, strict=True):
            predicted = latest_status_code(prediction.next_state)
            if predicted is not None:
                mass_by_status[int(predicted)] += float(mass)
        top1 = (
            max(mass_by_status, key=lambda code: (mass_by_status[code], -code))
            if mass_by_status
            else None
        )
        actual_mass = float(mass_by_status.get(int(actual), 0.0))
        by_status[int(actual)].append(
            {
                "top1": int(top1 == actual),
                "topk": int(int(actual) in mass_by_status),
                "actual_mass": actual_mass,
                "nll": -math.log(max(1e-12, actual_mass)),
            }
        )

    result: dict[str, Any] = {}
    all_rows: list[dict[str, float | int]] = []
    for code in STATUS_CODES_V3:
        items = by_status.get(int(code), [])
        all_rows.extend(items)
        result[str(code)] = {
            "count": len(items),
            "top1_accuracy": fmean(float(item["top1"]) for item in items) if items else None,
            "topk_recall": fmean(float(item["topk"]) for item in items) if items else None,
            "mean_actual_probability_mass": (
                fmean(float(item["actual_mass"]) for item in items) if items else None
            ),
            "mean_nll": fmean(float(item["nll"]) for item in items) if items else None,
        }
    result["overall"] = {
        "count": len(all_rows),
        "top1_accuracy": fmean(float(item["top1"]) for item in all_rows) if all_rows else None,
        "topk_recall": fmean(float(item["topk"]) for item in all_rows) if all_rows else None,
        "mean_actual_probability_mass": (
            fmean(float(item["actual_mass"]) for item in all_rows) if all_rows else None
        ),
        "mean_nll": fmean(float(item["nll"]) for item in all_rows) if all_rows else None,
    }
    return result


def _aggregate_group(
    status_metrics: dict[str, Any],
    codes: Sequence[int],
) -> dict[str, Any]:
    total = sum(int(status_metrics[str(code)]["count"]) for code in codes)
    if total <= 0:
        return {
            "codes": list(codes),
            "count": 0,
            "top1_accuracy": None,
            "topk_recall": None,
            "mean_actual_probability_mass": None,
        }
    output: dict[str, Any] = {"codes": list(codes), "count": total}
    for metric in (
        "top1_accuracy",
        "topk_recall",
        "mean_actual_probability_mass",
    ):
        weighted = 0.0
        for code in codes:
            row = status_metrics[str(code)]
            count = int(row["count"])
            value = row[metric]
            if count and value is not None:
                weighted += count * float(value)
        output[metric] = weighted / total
    return output


def run_diagnostic(
    *,
    research_seed: int = 7,
    train_seed_count: int = len(TRANSFER_TRAIN_SEEDS),
    holdout_seed_count: int = len(TRANSFER_DIAGNOSTIC_SEEDS),
    stage_indices: Sequence[int] = tuple(range(8)),
    steps_per_episode: int = 48,
    samples: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    install_status_aware_relational_contract()
    train_seeds = TRANSFER_TRAIN_SEEDS[: int(train_seed_count)]
    holdout_seeds = TRANSFER_DIAGNOSTIC_SEEDS[: int(holdout_seed_count)]
    train_rows = _collect(
        train_seeds,
        stage_indices=stage_indices,
        steps_per_episode=int(steps_per_episode),
        behavior_seed=int(research_seed) ^ 0x54524149,
    )
    holdout_rows = _collect(
        holdout_seeds,
        stage_indices=stage_indices,
        steps_per_episode=int(steps_per_episode),
        behavior_seed=int(research_seed) ^ 0x484F4C44,
    )
    if not train_rows or not holdout_rows:
        raise RuntimeError("rare-status diagnostic collected no usable real transitions")

    train_counts = Counter(
        int(code)
        for row in train_rows
        if (code := latest_status_code(row.after)) is not None
    )
    holdout_counts = Counter(
        int(code)
        for row in holdout_rows
        if (code := latest_status_code(row.after)) is not None
    )

    model = StatusAwareConditionalMixtureRelationalProphecy(
        seed=int(research_seed),
        device=device,
        config=RelationalMixtureProphecyConfig(
            hidden_units=48,
            ensemble_size=2,
            mixture_components=3,
            replay_capacity=max(8192, len(train_rows) + 1),
            batch_size=32,
            warmup_steps=64,
            gradient_steps_per_observation=1,
        ),
    )
    shuffled = list(train_rows)
    random.Random(int(research_seed) ^ 0x53544154).shuffle(shuffled)
    for row in shuffled:
        model.learn(row.before, row.action, row.after)

    metrics = _evaluate(model, holdout_rows, samples=int(samples))
    groups = _frequency_groups(dict(train_counts), dict(holdout_counts))
    rare_metrics = _aggregate_group(metrics, groups["rare"])
    common_metrics = _aggregate_group(metrics, groups["common"])

    eligible_zero_topk = [
        code
        for code in groups["eligible"]
        if metrics[str(code)]["topk_recall"] is not None
        and float(metrics[str(code)]["topk_recall"]) <= 0.0
    ]
    result = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status_objective": STATUS_OBJECTIVE,
        "behavior_policy": "seeded-uniform-over-public-legal-actions",
        "uses_hidden_correctness": False,
        "uses_final_blind_seeds": False,
        "train_seeds": list(train_seeds),
        "holdout_seeds": list(holdout_seeds),
        "stage_indices": list(stage_indices),
        "steps_per_episode": int(steps_per_episode),
        "train_transitions": len(train_rows),
        "holdout_transitions": len(holdout_rows),
        "train_status_counts": {str(code): int(train_counts.get(code, 0)) for code in STATUS_CODES_V3},
        "holdout_status_counts": {str(code): int(holdout_counts.get(code, 0)) for code in STATUS_CODES_V3},
        "frequency_groups": groups,
        "status_metrics": metrics,
        "rare_group": rare_metrics,
        "common_group": common_metrics,
        "eligible_zero_topk_statuses": eligible_zero_topk,
        "minimal_survival_gate": {
            "eligible_status_count": len(groups["eligible"]),
            "rare_status_count": len(groups["rare"]),
            "all_supported_statuses_have_nonzero_topk": not eligible_zero_topk,
            "rare_group_has_nonzero_topk": (
                rare_metrics["topk_recall"] is not None
                and float(rare_metrics["topk_recall"]) > 0.0
            ),
        },
        "model_diagnostics": model.diagnostics(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic held-out rare public-status diagnostic for current Prophecy."
    )
    parser.add_argument("--output", default="runs/current_status_rare_holdout.json")
    parser.add_argument("--research-seed", type=int, default=7)
    parser.add_argument("--train-seed-count", type=int, default=len(TRANSFER_TRAIN_SEEDS))
    parser.add_argument("--holdout-seed-count", type=int, default=len(TRANSFER_DIAGNOSTIC_SEEDS))
    parser.add_argument("--steps-per-episode", type=int, default=48)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_diagnostic(
        research_seed=args.research_seed,
        train_seed_count=args.train_seed_count,
        holdout_seed_count=args.holdout_seed_count,
        steps_per_episode=args.steps_per_episode,
        samples=args.samples,
        device=args.device,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"artifact: {path}")


if __name__ == "__main__":
    main()
