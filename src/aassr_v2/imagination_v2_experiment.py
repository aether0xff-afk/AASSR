from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import baseline_efficiency_benchmark as benchmark
from .imagination_v2 import (
    IMAGINATION_V2_CONDITIONS,
    make_imagination_v2_agent,
)


def run_imagination_v2_experiment(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    train_episodes: int = 1000,
    train_map_count: int = 64,
    evaluation_episodes: int = 100,
    checkpoints: Sequence[int] = (0, 100, 250, 500, 1000),
) -> dict[str, Any]:
    if condition not in IMAGINATION_V2_CONDITIONS:
        raise ValueError(f"unknown Imagination v2 condition: {condition}")

    created: dict[str, Any] = {}

    def factory(
        requested_condition: str,
        requested_seed: int,
        *,
        train_episodes: int,
    ) -> Any:
        agent = make_imagination_v2_agent(
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

    agent = created.get("agent")
    diagnostics_fn = getattr(agent, "diagnostics", None)
    diagnostics = dict(diagnostics_fn()) if callable(diagnostics_fn) else {}
    payload["agent_diagnostics"] = diagnostics
    payload["identity"] = {
        "policy_prophecy_imagination_separated": condition not in {"dqn"},
        "neural_delta_prophecy": condition in {
            "neural_manual",
            "imagination_v2",
        },
        "gru_branch_critic": condition in {
            "legacy_gru_critic",
            "imagination_v2",
        },
        "hand_written_branch_scorer": condition in {
            "legacy_aassr",
            "controlled_legacy",
            "neural_manual",
        },
        "critic_target": (
            "real_episode_final_success_only"
            if condition in {"legacy_gru_critic", "imagination_v2"}
            else None
        ),
        "task_distance_or_oracle_used_for_training": False,
        "external_reward": "final_success_only",
    }
    output = Path(output_dir)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "agent_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
