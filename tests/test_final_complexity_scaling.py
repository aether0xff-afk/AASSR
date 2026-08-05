from __future__ import annotations

import csv
import json

from aassr_v2.final_complexity_scaling import (
    COMPLEXITY_LEVELS,
    build_stratified_map_pool,
    profile_map_complexity,
    run_final_complexity_scaling,
)


def test_map_complexity_profile_is_deterministic() -> None:
    first = profile_map_complexity(7_000_000)
    second = profile_map_complexity(7_000_000)
    assert first == second
    if first is not None:
        assert first.oracle_shortest_steps > 0
        assert first.reachable_nonterminal_states > 0
        assert 0.0 <= first.irreversible_failure_ratio <= 1.0


def test_stratified_pool_is_balanced_and_structurally_ordered() -> None:
    pool = build_stratified_map_pool(
        start_seed=9_000_000,
        maps_per_level=2,
        selection_seed=123,
        oversample=2,
    )
    assert tuple(pool) == COMPLEXITY_LEVELS
    assert all(len(pool[level]) == 2 for level in COMPLEXITY_LEVELS)

    ordered_bins = [
        sorted(pool[level], key=lambda item: item.order_key)
        for level in COMPLEXITY_LEVELS
    ]
    for left, right in zip(ordered_bins, ordered_bins[1:]):
        assert max(item.order_key for item in left) <= min(
            item.order_key for item in right
        )


def test_tiny_final_run_has_no_abandonment_or_tick_limit(tmp_path) -> None:
    output = tmp_path / "tiny"
    payload = run_final_complexity_scaling(
        output,
        condition="legacy_aassr",
        seed=3,
        transition_budget=8,
        train_maps_per_level=1,
        evaluation_maps_per_level=1,
        checkpoints=(0, 8),
        pool_oversample=1,
    )
    config = payload["config"]
    assert config["abandonment_enabled"] is False
    assert config["artificial_step_limit"] is False
    assert config["oracle_used_for_agent_training_or_action"] is False
    assert config["actual_training_transitions"] >= 8

    with (output / "checkpoint_level_metrics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(COMPLEXITY_LEVELS) * 2

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["episode_termination"] == (
        "success_or_environment_failure_only"
    )
