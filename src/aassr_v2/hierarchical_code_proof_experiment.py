from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Sequence

from .hierarchical_code_experiment_setup import install_hierarchical_code_agents
from .hierarchical_code_world import (
    HierarchicalCodeWorld,
    install_hierarchical_code_world,
)
from . import long_horizon_goal_experiment as experiment


CONDITIONS: tuple[str, ...] = (
    "policy_only",
    "short_imagination",
    "deep_imagination",
    "goal_maker_executor",
)


def _prophecy(agent: object) -> object:
    direct = getattr(agent, "prophecy", None)
    if direct is not None:
        return direct
    base = getattr(agent, "base", None)
    model = getattr(base, "prophecy", None)
    if model is None:
        raise TypeError("agent exposes no Prophecy")
    return model


def pretrain_prophecy_from_random_actions(
    agent: object,
    *,
    seed: int,
    episodes: int,
    map_count: int,
    stage_count: int,
    room_length: int,
) -> dict[str, int]:
    """Train only Prophecy from a matched stream of random real transitions."""

    if episodes <= 0 or map_count <= 0:
        raise ValueError("pretraining sizes must be positive")
    model = _prophecy(agent)
    learn = getattr(model, "learn")
    randomizer = random.Random(seed * 10_000_019 + 17)
    map_seeds = tuple(
        seed * 1_000_000 + index for index in range(map_count)
    )
    transitions = 0
    completed_checkpoints = 0
    successful_episodes = 0

    for episode in range(episodes):
        world = HierarchicalCodeWorld(
            map_seeds[episode % len(map_seeds)],
            stage_count=stage_count,
            room_length=room_length,
        )
        while world.snapshot().available_actions:
            before = world.snapshot()
            action = randomizer.choice(before.available_actions)
            outcome = world.step(action)
            learn(before, action, outcome.snapshot)
            transitions += 1
            if "checkpoint_transition" in outcome.added_facts:
                completed_checkpoints += 1
            if world.success or world.failed:
                break
        successful_episodes += int(world.success)

    return {
        "episodes": episodes,
        "transitions": transitions,
        "completed_checkpoints": completed_checkpoints,
        "successful_episodes": successful_episodes,
    }


def run_hierarchical_code_proof(
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (7, 13, 21, 42, 100),
    conditions: Sequence[str] = CONDITIONS,
    pretrain_episodes: int = 5000,
    pretrain_map_count: int = 256,
    evaluation_episodes: int = 100,
    stage_count: int = 20,
    room_length: int = 4,
) -> dict[str, object]:
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    install_hierarchical_code_world()
    install_hierarchical_code_agents()

    rows: list[experiment.LongHorizonEpisodeResult] = []
    pretraining: list[dict[str, int | str]] = []

    for seed in seeds:
        seen_maps = tuple(
            seed * 1_000_000 + index
            for index in range(evaluation_episodes)
        )
        unseen_maps = tuple(
            seed * 1_000_000 + 500_000 + index
            for index in range(evaluation_episodes)
        )
        for condition in conditions:
            agent = experiment.make_agent(
                condition,
                seed,
                room_length=room_length,
            )
            diagnostics = pretrain_prophecy_from_random_actions(
                agent,
                seed=seed,
                episodes=pretrain_episodes,
                map_count=pretrain_map_count,
                stage_count=stage_count,
                room_length=room_length,
            )
            pretraining.append(
                {
                    "condition": condition,
                    "seed": seed,
                    **diagnostics,
                }
            )

            for index, map_seed in enumerate(seen_maps):
                rows.append(
                    experiment.run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="evaluation_seen",
                        episode=index,
                        map_seed=map_seed,
                        stage_count=stage_count,
                        room_length=room_length,
                        learn=False,
                    )
                )
            for index, map_seed in enumerate(unseen_maps):
                rows.append(
                    experiment.run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="evaluation_unseen",
                        episode=evaluation_episodes + index,
                        map_seed=map_seed,
                        stage_count=stage_count,
                        room_length=room_length,
                        learn=False,
                    )
                )

    with (output / "episodes.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                experiment.LongHorizonEpisodeResult.__dataclass_fields__
            ),
        )
        writer.writeheader()
        writer.writerows(row.as_row() for row in rows)

    payload: dict[str, object] = {
        "config": {
            "seeds": list(seeds),
            "conditions": list(conditions),
            "pretrain_episodes": pretrain_episodes,
            "pretrain_map_count": pretrain_map_count,
            "evaluation_episodes": evaluation_episodes,
            "stage_count": stage_count,
            "room_length": room_length,
            "external_reward_used_in_pretraining": False,
            "correct_action_used_in_pretraining": False,
        },
        "pretraining": pretraining,
        "summary": experiment.summarize(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
