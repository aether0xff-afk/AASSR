from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .baseline_efficiency_benchmark import (
    CHOICE_ACTIONS,
    DQNBenchmarkAgent,
    BenchmarkGridPushWorld,
)
from .branch_critic import CriticTransition
from .neural_delta_prophecy import NeuralDeltaProphecy
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    transitions: tuple[CriticTransition, ...]
    success: bool
    map_seed: int


@dataclass(frozen=True, slots=True)
class AuditConfig:
    seed: int = 7
    train_map_count: int = 32
    unseen_map_count: int = 32
    behavior_train_episodes: int = 800
    critic_train_episodes: int = 600
    critic_eval_episodes: int = 200
    prophecy_train_episodes: int = 800
    prophecy_eval_episodes: int = 300
    prophecy_epochs: int = 1
    pruning_depth: int = 5
    pruning_beam_width: int = 4


class NeuralDirectProphecy(NeuralDeltaProphecy):
    """Same MLP, replay, and ensemble as Neural Delta; direct next-state target."""

    name = "neural-direct"

    def learn(self, state: StateSnapshot, action: Action, actual_next_state: StateSnapshot) -> None:
        before = self.codec.encode(state)
        after = self.codec.encode(actual_next_state)
        self.replay.append(
            (
                self._input(state, action),
                before,
                after,
                self._terminal_class(actual_next_state),
            )
        )
        self.observations += 1
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return
        for _ in range(self.config.gradient_steps_per_observation):
            self._train_step()

    def _raw_predictions(self, state: StateSnapshot, action: Action):
        input_tensor = self._tensor(self._input(state, action)).unsqueeze(0)
        states, terminal_probabilities = [], []
        with self.torch.no_grad():
            for model in self.models:
                output = model(input_tensor)[0]
                states.append(
                    [float(value) for value in output[: self.codec.dimension].tolist()]
                )
                terminal_probabilities.append(
                    [
                        float(value)
                        for value in self.torch.softmax(
                            output[self.codec.dimension :], dim=0
                        ).tolist()
                    ]
                )
        return states, terminal_probabilities


class BenchmarkGridPushVectorCodec:
    """16-value representation ablation without the explicit used-cell grid."""

    @property
    def dimension(self) -> int:
        return 16

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        return tuple(float(value) for value in state.vector)

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != 16:
            raise ValueError("vector codec requires 16 values")
        values = [min(1.0, max(0.0, float(value))) for value in encoded]
        for index in range(12):
            values[index] = round(values[index] * 2.0) / 2.0
        phase = min(6, max(0, int(round(values[12] * 6.0))))
        values[12] = phase / 6.0
        for index in range(13, 16):
            values[index] = float(values[index] >= 0.5)
        facts = {fact for fact in scaffold.facts if fact.startswith("used:")}
        facts.add(f"used:{int(round(values[0] * 2))}:{int(round(values[1] * 2))}")
        facts.add(f"phase:{phase}")
        if values[13] >= 0.5:
            facts.add("bridge_built")
        if values[14] >= 0.5:
            facts.add("key_held")
        if values[15] >= 0.5:
            facts.add("door_open")
        if terminal_class == 1:
            facts.add("success")
        elif terminal_class == 2:
            facts.add("failed")
        return StateSnapshot(
            vector=tuple(values),
            facts=frozenset(facts),
            available_actions=() if terminal_class else CHOICE_ACTIONS,
            goal_progress=1.0 if terminal_class == 1 else 0.0,
            metadata={**dict(scaffold.metadata), "prediction_source": source},
        )


def _run_episode(world: BenchmarkGridPushWorld, choose_action: Any) -> EpisodeRecord:
    transitions = []
    state = world.snapshot()
    while state.available_actions:
        action = choose_action(state)
        outcome = world.step(action)
        transitions.append(CriticTransition(state, action, outcome.snapshot, 1.0))
        state = outcome.snapshot
    return EpisodeRecord(tuple(transitions), world.success, world.seed)


def train_behavior_dqn(map_seeds: Sequence[int], *, episodes: int, seed: int) -> DQNBenchmarkAgent:
    agent = DQNBenchmarkAgent(seed, train_episodes=episodes)
    randomizer = random.Random(seed ^ 0xD011)
    for episode in range(episodes):
        world = BenchmarkGridPushWorld(randomizer.choice(map_seeds))
        agent.begin_episode(training=True)
        state = world.snapshot()
        while state.available_actions:
            decision = agent.select_action(state, episode=episode, training=True)
            outcome = world.step(decision.action)
            agent.observe(state, decision.action, outcome)
            state = outcome.snapshot
        agent.end_episode(success=world.success, training=True)
    return agent


def collect_behavior_episodes(
    agent: DQNBenchmarkAgent,
    map_seeds: Sequence[int],
    *,
    episodes: int,
    seed: int,
    random_action_rate: float = 0.10,
) -> tuple[EpisodeRecord, ...]:
    randomizer = random.Random(seed)
    records = []
    for episode in range(episodes):
        world = BenchmarkGridPushWorld(randomizer.choice(map_seeds))

        def choose(state: StateSnapshot) -> Action:
            if randomizer.random() < random_action_rate:
                return randomizer.choice(state.available_actions)
            return agent.select_action(state, episode=episode, training=False).action

        records.append(_run_episode(world, choose))
    return tuple(records)


def collect_random_episodes(
    map_seeds: Sequence[int], *, episodes: int, seed: int
) -> tuple[EpisodeRecord, ...]:
    randomizer = random.Random(seed)
    return tuple(
        _run_episode(
            BenchmarkGridPushWorld(randomizer.choice(map_seeds)),
            lambda state: randomizer.choice(state.available_actions),
        )
        for _ in range(episodes)
    )
