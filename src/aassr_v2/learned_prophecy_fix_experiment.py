from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import baseline_efficiency_benchmark as benchmark
from .baseline_efficiency_benchmark import make_benchmark_agent
from .benchmark_neural_prophecy import NeuralProphecyHybridAgent


LEARNED_PROPHECY_CONDITIONS = (
    "dqn",
    "aassr_current",
    "hybrid_neural_prophecy",
    "hybrid_neural_prophecy_adaptive",
)


def make_learned_prophecy_agent(
    condition: str,
    seed: int,
    *,
    train_episodes: int,
) -> Any:
    if condition == "dqn":
        return make_benchmark_agent(
            "dqn",
            seed,
            train_episodes=train_episodes,
        )
    if condition == "aassr_current":
        return make_benchmark_agent(
            "aassr_full",
            seed,
            train_episodes=train_episodes,
        )
    if condition == "hybrid_neural_prophecy":
        return NeuralProphecyHybridAgent(
            seed,
            train_episodes=train_episodes,
            adaptive=False,
        )
    if condition == "hybrid_neural_prophecy_adaptive":
        return NeuralProphecyHybridAgent(
            seed,
            train_episodes=train_episodes,
            adaptive=True,
        )
    raise ValueError(f"unknown learned Prophecy condition: {condition}")


def run_learned_prophecy_fix(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    train_episodes: int = 2000,
    train_map_count: int = 128,
    evaluation_episodes: int = 100,
    checkpoints: Sequence[int] = (0, 100, 250, 500, 1000, 2000),
) -> dict[str, Any]:
    if condition not in LEARNED_PROPHECY_CONDITIONS:
        raise ValueError(f"unknown learned Prophecy condition: {condition}")
    original_factory = benchmark.make_benchmark_agent
    benchmark.make_benchmark_agent = make_learned_prophecy_agent
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
    payload["identity"] = {
        "policy_prophecy_imagination_separated": True,
        "prophecy_learned_from_real_transitions": True,
        "task_transition_rules_injected": False,
        "external_reward": "final_success_only",
        "neural_prophecy": condition.startswith("hybrid_neural_prophecy"),
        "adaptive_planning": condition.endswith("adaptive"),
    }
    output = Path(output_dir)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
