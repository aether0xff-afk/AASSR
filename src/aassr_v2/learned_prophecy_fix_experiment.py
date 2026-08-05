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
    "hybrid_neural_prophecy_calibrated",
    "hybrid_neural_prophecy_calibrated_conservative",
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
        )
    if condition == "hybrid_neural_prophecy_adaptive":
        return NeuralProphecyHybridAgent(
            seed,
            train_episodes=train_episodes,
            adaptive=True,
        )
    if condition == "hybrid_neural_prophecy_calibrated":
        return NeuralProphecyHybridAgent(
            seed,
            train_episodes=train_episodes,
            calibrated=True,
        )
    if condition == "hybrid_neural_prophecy_calibrated_conservative":
        return NeuralProphecyHybridAgent(
            seed,
            train_episodes=train_episodes,
            calibrated=True,
            conservative=True,
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
    created: dict[str, Any] = {}

    def factory(
        requested_condition: str,
        requested_seed: int,
        *,
        train_episodes: int,
    ) -> Any:
        agent = make_learned_prophecy_agent(
            requested_condition,
            requested_seed,
            train_episodes=train_episodes,
        )
        created["agent"] = agent
        return agent

    original_factory = benchmark.make_benchmark_agent
    benchmark.make_benchmark_agent = factory
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
        "oracle_used_for_behavior": False,
        "oracle_used_for_diagnostic_labels_only": condition.startswith(
            "hybrid_neural_prophecy"
        ),
        "external_reward": "final_success_only",
        "neural_prophecy": condition.startswith("hybrid_neural_prophecy"),
        "adaptive_planning": condition.endswith("adaptive"),
        "holdout_calibration": "calibrated" in condition,
    }
    agent = created.get("agent")
    diagnostics = getattr(agent, "diagnostics", None)
    payload["agent_diagnostics"] = (
        dict(diagnostics()) if callable(diagnostics) else {}
    )
    output = Path(output_dir)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "agent_diagnostics.json").write_text(
        json.dumps(payload["agent_diagnostics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload