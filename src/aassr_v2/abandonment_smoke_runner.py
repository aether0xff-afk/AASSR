from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .abandonment_smoke import (
    AbandonmentEpisode,
    AbandonmentEvent,
    _pair_active_shadow,
    _run_frozen_episode,
    _summarize,
    _write_csv,
)
from .baseline_efficiency_benchmark import _run_episode, solvable_map_seeds
from .imagination_v2 import ImaginationV2Agent


def _agent_randomizers(agent: ImaginationV2Agent) -> tuple[random.Random, ...]:
    candidates = [
        getattr(agent.dqn, "randomizer", None),
        getattr(agent.agent, "randomizer", None),
        getattr(agent.critic, "randomizer", None),
        getattr(getattr(agent.agent, "base_prophecy", None), "randomizer", None),
        getattr(getattr(agent.agent, "prophecy", None), "randomizer", None),
        getattr(
            getattr(getattr(agent.agent, "prophecy", None), "base", None),
            "randomizer",
            None,
        ),
    ]
    unique: list[random.Random] = []
    seen: set[int] = set()
    for item in candidates:
        if not isinstance(item, random.Random) or id(item) in seen:
            continue
        seen.add(id(item))
        unique.append(item)
    return tuple(unique)


def _capture_random_states(
    agent: ImaginationV2Agent,
) -> tuple[tuple[random.Random, object], ...]:
    return tuple((item, item.getstate()) for item in _agent_randomizers(agent))


def _restore_random_states(
    captured: Sequence[tuple[random.Random, object]],
) -> None:
    for item, state in captured:
        item.setstate(state)


def run_abandonment_smoke(
    output_dir: str | Path,
    *,
    seed: int = 7,
    train_episodes: int = 300,
    train_map_count: int = 32,
    evaluation_episodes: int = 30,
    thresholds: Sequence[float] = (0.05, 0.15, 0.30),
    minimum_steps: int = 2,
    patience: int = 2,
    safety_cap: int = 128,
) -> dict[str, Any]:
    """Run paired shadow/active abandonment without cloning PyTorch modules."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agent = ImaginationV2Agent(seed, train_episodes=train_episodes)
    training_maps = solvable_map_seeds(seed * 1_000_000, train_map_count)
    environment_steps = 0
    training_successes = 0
    for episode in range(train_episodes):
        metric, environment_steps = _run_episode(
            agent,
            condition="imagination_v2_abandonment_training",
            seed=seed,
            phase="training",
            checkpoint_episode=train_episodes,
            episode=episode,
            map_seed=training_maps[episode % len(training_maps)],
            training=True,
            environment_steps_total=environment_steps,
        )
        training_successes += metric.success

    seen_maps = tuple(
        training_maps[index % len(training_maps)]
        for index in range(evaluation_episodes)
    )
    unseen_maps = solvable_map_seeds(
        seed * 1_000_000 + 500_000,
        evaluation_episodes,
    )
    rows: list[AbandonmentEpisode] = []
    events: list[AbandonmentEvent] = []
    for threshold in thresholds:
        for split, map_seeds in (("seen", seen_maps), ("unseen", unseen_maps)):
            for episode, map_seed in enumerate(map_seeds):
                captured = _capture_random_states(agent)
                shadow_row, shadow_event = _run_frozen_episode(
                    agent,
                    map_seed=map_seed,
                    seed=seed,
                    episode=episode,
                    split=split,
                    mode="shadow",
                    threshold=float(threshold),
                    minimum_steps=minimum_steps,
                    patience=patience,
                    safety_cap=safety_cap,
                )
                _restore_random_states(captured)
                active_row, active_event = _run_frozen_episode(
                    agent,
                    map_seed=map_seed,
                    seed=seed,
                    episode=episode,
                    split=split,
                    mode="active",
                    threshold=float(threshold),
                    minimum_steps=minimum_steps,
                    patience=patience,
                    safety_cap=safety_cap,
                )
                rows.extend((shadow_row, active_row))
                if shadow_event is not None:
                    events.append(shadow_event)
                if active_event is not None:
                    events.append(active_event)

    summary_rows = _summarize(rows)
    paired_rows = _pair_active_shadow(rows)
    critic_stats = asdict(agent.critic.stats())
    payload = {
        "config": {
            "seed": seed,
            "train_episodes": train_episodes,
            "train_map_count": train_map_count,
            "evaluation_episodes_per_split": evaluation_episodes,
            "thresholds": [float(value) for value in thresholds],
            "minimum_steps": minimum_steps,
            "patience": patience,
            "safety_cap": safety_cap,
            "environment": "strict_gridpush_final",
            "abandonment_training": "disabled; frozen post-training critic only",
            "paired_randomness": "python Random states restored before active run",
        },
        "training": {
            "success_rate": training_successes / train_episodes,
            "environment_steps": environment_steps,
            "critic_ready": bool(agent.critic_ready),
            "critic_stats": critic_stats,
        },
        "summary": summary_rows,
        "paired": {
            "prevented_successes": sum(
                item["prevented_success"] for item in paired_rows
            ),
            "saved_steps_on_shadow_failures": sum(
                item["saved_steps_on_shadow_failure"] for item in paired_rows
            ),
            "rows": paired_rows,
        },
        "interpretation_guardrails": {
            "oracle_used_for_training": False,
            "oracle_used_only_for_posthoc_abandonment_audit": True,
            "abandoned_episodes_used_to_train_critic": False,
            "fixed_episode_step_limit": False,
            "safety_cap_is_nontermination_guard_only": True,
        },
    }
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "abandonment_events.csv", events)
    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "paired_active_shadow.csv", paired_rows)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
