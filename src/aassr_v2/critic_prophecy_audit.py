from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .baseline_efficiency_benchmark import solvable_map_seeds
from .critic_prophecy_common import (
    AuditConfig,
    collect_behavior_episodes,
    collect_random_episodes,
    train_behavior_dqn,
)
from .critic_pruning_audit import run_critic_audit
from .prophecy_one_step_audit import run_prophecy_audit


def run_audit(output_dir: str | Path, config: AuditConfig | None = None) -> dict[str, Any]:
    config = config or AuditConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_maps = solvable_map_seeds(
        10_000 + config.seed * 100, config.train_map_count
    )
    unseen_maps = solvable_map_seeds(
        90_000 + config.seed * 100, config.unseen_map_count
    )

    behavior = train_behavior_dqn(
        train_maps,
        episodes=config.behavior_train_episodes,
        seed=config.seed,
    )
    require_mixed_outcomes = config.behavior_train_episodes >= 200
    critic_train = collect_behavior_episodes(
        behavior, train_maps, episodes=config.critic_train_episodes,
        seed=config.seed ^ 0xC17, random_action_rate=0.15,
        minimum_successes=(
            max(8, config.critic_train_episodes // 10)
            if require_mixed_outcomes else 0
        ),
    )
    critic_eval = collect_behavior_episodes(
        behavior, unseen_maps, episodes=config.critic_eval_episodes,
        seed=config.seed ^ 0xE17, random_action_rate=0.15,
        minimum_successes=(
            max(4, config.critic_eval_episodes // 10)
            if require_mixed_outcomes else 0
        ),
    )
    critic_results = run_critic_audit(
        critic_train,
        critic_eval,
        unseen_maps,
        seed=config.seed,
        depth=config.pruning_depth,
        beam_width=config.pruning_beam_width,
    )

    prophecy_train = collect_random_episodes(
        train_maps, episodes=config.prophecy_train_episodes,
        seed=config.seed ^ 0xA11,
    )
    prophecy_seen = collect_random_episodes(
        train_maps, episodes=config.prophecy_eval_episodes,
        seed=config.seed ^ 0xA12,
    )
    prophecy_unseen = collect_random_episodes(
        unseen_maps, episodes=config.prophecy_eval_episodes,
        seed=config.seed ^ 0xA13,
    )
    prophecy_results = run_prophecy_audit(
        prophecy_train,
        prophecy_seen,
        prophecy_unseen,
        seed=config.seed,
        epochs=config.prophecy_epochs,
    )

    payload = {
        "config": asdict(config),
        "identity_constraints": {
            "critic_training_label": "real episode final success only",
            "critic_oracle_use": "evaluation only",
            "prophecy_training_tuple": (
                "real current state, real action, real next state"
            ),
            "reward_or_goal_distance_used_by_prophecy": False,
            "imagination_used_in_prophecy_audit": False,
            "other_critic_ideas_implemented": False,
        },
        "critic": critic_results,
        "prophecy_one_step": {
            "training_data": {
                "episodes": len(prophecy_train),
                "transitions": sum(
                    len(item.transitions) for item in prophecy_train
                ),
            },
            "models": prophecy_results,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return payload
