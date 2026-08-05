from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from . import baseline_efficiency_benchmark as benchmark
from .bottleneck_sota_diagnostic import (
    CONDITIONS,
    BenchmarkOracleProgressScorer,
    BenchmarkOracleProphecy,
    HybridDQNImaginationAgent,
    make_bottleneck_agent as _base_factory,
)


def make_bottleneck_agent(
    condition: str,
    seed: int,
    *,
    train_episodes: int,
) -> Any:
    if condition != "hybrid_oracle_budgeted":
        return _base_factory(
            condition,
            seed,
            train_episodes=train_episodes,
        )
    return HybridDQNImaginationAgent(
        seed,
        train_episodes=train_episodes,
        prophecy=BenchmarkOracleProphecy(),
        scorer=BenchmarkOracleProgressScorer(),
        depth=4,
        branching_factor=2,
        beam_width=16,
        outcome_samples=1,
        imagination_interval=4,
        use_effect_composition=False,
        name=condition,
    )


def run_bottleneck_condition(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    train_episodes: int = 1000,
    train_map_count: int = 64,
    evaluation_episodes: int = 100,
    checkpoints: Sequence[int] = (0, 100, 250, 500, 1000),
) -> dict[str, Any]:
    original_factory = benchmark.make_benchmark_agent
    benchmark.make_benchmark_agent = make_bottleneck_agent
    try:
        payload = benchmark.run_gridpush_baseline_benchmark(
            output_dir,
            condition=condition,
            seed=seed,
            train_episodes=train_episodes,
            train_map_count=train_map_count,
            evaluation_episodes=evaluation_episodes,
            checkpoints=checkpoints,
        )
    finally:
        benchmark.make_benchmark_agent = original_factory

    spec = next(item for item in CONDITIONS if item.name == condition)
    payload["condition"] = asdict(spec)
    output = Path(output_dir)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
