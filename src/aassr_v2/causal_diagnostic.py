from __future__ import annotations

import ast
import copy
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

from .causal_dependency_world import CausalDependencyWorldV2
from .paper_v2_protocol import checkpoint_fingerprint, clone_agent_from_checkpoint
from .paper_v2_types import FullAgentCheckpoint, RawCausalObservation


def identity_observation_key(observation: RawCausalObservation) -> str:
    return repr(
        (
            tuple(sorted(observation.inventory.items())),
            tuple(sorted(observation.observable_facts)),
            observation.available_actions,
            tuple(sorted(observation.spatial_observations.items())),
            observation.last_action_succeeded,
        )
    )


class MonteCarloIdentityAgent:
    """Strict-sparse contextual policy used for Diagnostic 1."""

    def __init__(self, *, seed: int, learning_rate: float = 0.2) -> None:
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.rng = random.Random(seed)
        self.values: dict[tuple[str, str], float] = defaultdict(float)
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.episode: list[tuple[str, str]] = []
        self.decision_counter = 0

    def select_action(
        self, observation: RawCausalObservation, *, epsilon: float
    ) -> str:
        actions = observation.available_actions
        if not actions:
            raise ValueError("no available action")
        if epsilon > 0.0:
            self.decision_counter += 1
            if self.rng.random() < epsilon:
                return self.rng.choice(actions)
        key = identity_observation_key(observation)
        return min(actions, key=lambda action: (-self.values[(key, action)], action))

    def observe_action(self, observation: RawCausalObservation, action: str) -> None:
        self.episode.append((identity_observation_key(observation), action))

    def finish_episode(self, terminal_success: bool, *, gamma: float = 0.97) -> None:
        target = float(terminal_success)
        for key in reversed(self.episode):
            self.counts[key] += 1
            self.values[key] += self.learning_rate * (target - self.values[key])
            target *= gamma
        self.episode.clear()

    def export_full_checkpoint(self) -> FullAgentCheckpoint:
        values = {repr(key): value for key, value in self.values.items()}
        counts = {repr(key): value for key, value in self.counts.items()}
        return FullAgentCheckpoint(
            policy={"values": values, "counts": counts},
            rng=repr(self.rng.getstate()),
            counters={"decision": self.decision_counter},
        )

    def import_full_checkpoint(self, checkpoint: FullAgentCheckpoint) -> None:
        self.values = defaultdict(
            float,
            {
                tuple(ast.literal_eval(key)): float(value)
                for key, value in checkpoint.policy.get("values", {}).items()
            },
        )
        self.counts = defaultdict(
            int,
            {
                tuple(ast.literal_eval(key)): int(value)
                for key, value in checkpoint.policy.get("counts", {}).items()
            },
        )
        if checkpoint.rng is not None:
            self.rng.setstate(ast.literal_eval(str(checkpoint.rng)))
        self.decision_counter = int(checkpoint.counters.get("decision", 0))
        self.episode.clear()


@dataclass(frozen=True, slots=True)
class DiagnosticOneResult:
    condition: str
    research_seed: int
    training_final_tail: float
    frozen_success: float
    random_success: float
    checkpoint_unchanged: bool
    learning_calls_during_evaluation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "research_seed": self.research_seed,
            "training_final_tail": self.training_final_tail,
            "frozen_success": self.frozen_success,
            "random_success": self.random_success,
            "checkpoint_unchanged": self.checkpoint_unchanged,
            "learning_calls_during_evaluation": self.learning_calls_during_evaluation,
        }


def _episode(
    agent: MonteCarloIdentityAgent,
    *,
    world_seed: int,
    episode: int,
    learn: bool,
    maximum_steps: int = 8,
) -> bool:
    world = CausalDependencyWorldV2(world_seed=world_seed)
    epsilon = max(0.05, 0.8 * (1.0 - episode / 400.0)) if learn else 0.0
    for _ in range(maximum_steps):
        observation = world.observe()
        if observation.terminal or not observation.available_actions:
            break
        action = agent.select_action(observation, epsilon=epsilon)
        if learn:
            agent.observe_action(observation, action)
        world.step(action)
        if world.terminal:
            break
    success = world.analysis_private_state.success
    if learn:
        agent.finish_episode(success)
    return success


def _random_success(
    *, research_seed: int, world_seeds: Sequence[int], episodes: int
) -> float:
    rng = random.Random(research_seed ^ 0xA11CE)
    successes = 0
    for episode in range(episodes):
        world = CausalDependencyWorldV2(
            world_seed=int(world_seeds[episode % len(world_seeds)])
        )
        for _ in range(8):
            actions = world.observe().available_actions
            if not actions or world.terminal:
                break
            world.step(rng.choice(actions))
        successes += int(world.analysis_private_state.success)
    return successes / episodes


def run_diagnostic_one(
    *,
    research_seeds: Sequence[int],
    train_world_seeds: Sequence[int],
    training_episodes: int = 500,
    evaluation_episodes: int = 100,
) -> list[DiagnosticOneResult]:
    results: list[DiagnosticOneResult] = []
    for research_seed in research_seeds:
        random_success = _random_success(
            research_seed=int(research_seed),
            world_seeds=train_world_seeds,
            episodes=evaluation_episodes,
        )
        for condition_offset, condition in enumerate(
            ("contextual_policy", "full_aassr")
        ):
            seed = int(research_seed) * 17 + condition_offset
            agent = MonteCarloIdentityAgent(seed=seed)
            successes = []
            for episode in range(training_episodes):
                successes.append(
                    _episode(
                        agent,
                        world_seed=int(
                            train_world_seeds[episode % len(train_world_seeds)]
                        ),
                        episode=episode,
                        learn=True,
                    )
                )
            clone, fingerprint_before = clone_agent_from_checkpoint(
                agent, lambda seed=seed: MonteCarloIdentityAgent(seed=seed)
            )
            frozen = [
                _episode(
                    clone,
                    world_seed=int(
                        train_world_seeds[index % len(train_world_seeds)]
                    ),
                    episode=training_episodes + index,
                    learn=False,
                )
                for index in range(evaluation_episodes)
            ]
            fingerprint_after = checkpoint_fingerprint(
                clone.export_full_checkpoint()
            )
            results.append(
                DiagnosticOneResult(
                    condition=condition,
                    research_seed=int(research_seed),
                    training_final_tail=fmean(
                        successes[-min(100, len(successes)) :]
                    ),
                    frozen_success=fmean(frozen),
                    random_success=random_success,
                    checkpoint_unchanged=(
                        fingerprint_before == fingerprint_after
                    ),
                    learning_calls_during_evaluation=0,
                )
            )
    return results


def diagnostic_one_gates(
    results: Sequence[DiagnosticOneResult],
) -> dict[str, Any]:
    contextual = [item for item in results if item.condition == "contextual_policy"]
    full = [item for item in results if item.condition == "full_aassr"]
    contextual_tail = fmean(item.training_final_tail for item in contextual)
    contextual_frozen = fmean(item.frozen_success for item in contextual)
    random_success = fmean(item.random_success for item in contextual)
    full_gap = fmean(
        abs(item.training_final_tail - item.frozen_success) for item in full
    )
    return {
        "checkpoint_immutable": all(item.checkpoint_unchanged for item in results),
        "evaluation_learning_calls_zero": all(
            item.learning_calls_during_evaluation == 0 for item in results
        ),
        "contextual_training_above_random": contextual_tail >= random_success + 0.20,
        "contextual_frozen_above_random": contextual_frozen >= random_success + 0.20,
        "contextual_replay_gap_within_0_10": abs(contextual_tail - contextual_frozen) <= 0.10,
        "full_replay_gap_within_0_10": full_gap <= 0.10,
        "metrics": {
            "contextual_training_final_tail": contextual_tail,
            "contextual_frozen_success": contextual_frozen,
            "random_success": random_success,
            "full_mean_absolute_replay_gap": full_gap,
        },
    }
