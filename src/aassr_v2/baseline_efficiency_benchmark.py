from __future__ import annotations

import csv
import io
import json
import math
import pickle
import random
import resource
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol, Sequence

from .autonomous_agent_core import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from .escape_reporting import serialize_agent_checkpoint
from .goal_gridpush_experiment import GridPushStep
from .imagination_tree import StateDeltaScorer
from .tabular_prophecy import TabularProphecy
from .types import Action, StateSnapshot


CHOICE_ACTIONS: tuple[Action, ...] = tuple(
    Action(f"choice_{index}") for index in range(4)
)
CHOICE_DIRECTIONS: tuple[str, ...] = (
    "north",
    "south",
    "west",
    "east",
)
DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}


def choice_index(action: Action) -> int:
    try:
        index = int(action.verb_name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"not a benchmark choice action: {action.signature}"
        ) from exc
    if not 0 <= index < len(CHOICE_ACTIONS):
        raise ValueError(f"choice index out of range: {index}")
    return index


class BenchmarkGridPushWorld:
    """Fixed-action sparse-reward GridPush benchmark.

    Every non-terminal state exposes the same four opaque choices. In movement
    and pushing phases, the choices map to north/south/west/east. During pickup
    and use phases, all choices execute the only meaningful interaction. Invalid
    movement ends the episode, and visited cells collapse, so episodes terminate
    through task dynamics without an artificial tick or energy limit.
    """

    grid_size = 3

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        randomizer = random.Random(seed)
        points = randomizer.sample(
            [
                (x, y)
                for y in range(self.grid_size)
                for x in range(self.grid_size)
            ],
            6,
        )
        (
            self.agent,
            self.crate,
            self.pit,
            self.key,
            self.door,
            self.exit,
        ) = points
        self.phase = 0
        self.bridge_built = False
        self.key_held = False
        self.door_open = False
        self.success = False
        self.failed = False
        self.used_cells: set[tuple[int, int]] = {self.agent}
        self.steps = 0
        self.optimal_steps = (
            self._distance(self.agent, self.crate)
            + self._distance(self.crate, self.pit)
            + self._distance(self.pit, self.key)
            + 1
            + self._distance(self.key, self.door)
            + 1
            + self._distance(self.door, self.exit)
        )

    @staticmethod
    def _distance(
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _normalize(self, point: tuple[int, int]) -> tuple[float, float]:
        scale = float(self.grid_size - 1)
        return point[0] / scale, point[1] / scale

    def _facts(self) -> frozenset[str]:
        facts = {
            f"phase:{self.phase}",
            *(f"used:{x}:{y}" for x, y in sorted(self.used_cells)),
        }
        if self.bridge_built:
            facts.add("bridge_built")
        if self.key_held:
            facts.add("key_held")
        if self.door_open:
            facts.add("door_open")
        if self.success:
            facts.add("success")
        if self.failed:
            facts.add("failed")
        return frozenset(facts)

    def _available_actions(self) -> tuple[Action, ...]:
        return () if self.success or self.failed else CHOICE_ACTIONS

    def _next_point(
        self,
        point: tuple[int, int],
        direction: str,
    ) -> tuple[int, int] | None:
        delta = DIRECTION_DELTAS[direction]
        candidate = point[0] + delta[0], point[1] + delta[1]
        if not (
            0 <= candidate[0] < self.grid_size
            and 0 <= candidate[1] < self.grid_size
        ):
            return None
        if candidate in self.used_cells:
            return None
        return candidate

    def _has_valid_motion(self, point: tuple[int, int]) -> bool:
        return any(
            self._next_point(point, direction) is not None
            for direction in CHOICE_DIRECTIONS
        )

    def _enter_new_room(self) -> None:
        self.used_cells = {self.agent}

    def snapshot(self) -> StateSnapshot:
        vector = (
            *self._normalize(self.agent),
            *self._normalize(self.crate),
            *self._normalize(self.pit),
            *self._normalize(self.key),
            *self._normalize(self.door),
            *self._normalize(self.exit),
            self.phase / 6.0,
            float(self.bridge_built),
            float(self.key_held),
            float(self.door_open),
        )
        return StateSnapshot(
            vector,
            self._facts(),
            self._available_actions(),
            1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "optimal_steps": self.optimal_steps,
                "termination": "fixed_choice_irreversible_path",
                "benchmark_action_count": len(CHOICE_ACTIONS),
            },
        )

    def step(self, action: Action) -> GridPushStep:
        before = self.snapshot()
        reward = 0.0
        error = False

        if action not in CHOICE_ACTIONS or not before.available_actions:
            self.failed = True
            error = True
        else:
            index = choice_index(action)
            direction = CHOICE_DIRECTIONS[index]
            self.steps += 1

            if self.phase in {0, 2, 4, 6}:
                candidate = self._next_point(self.agent, direction)
                if candidate is None:
                    self.failed = True
                    error = True
                else:
                    self.agent = candidate
                    self.used_cells.add(candidate)
                    if self.phase == 0 and self.agent == self.crate:
                        self.phase = 1
                        self._enter_new_room()
                    elif self.phase == 2 and self.agent == self.key:
                        self.phase = 3
                        self._enter_new_room()
                    elif self.phase == 4 and self.agent == self.door:
                        self.phase = 5
                        self._enter_new_room()
                    elif self.phase == 6 and self.agent == self.exit:
                        self.success = True
                    elif not self._has_valid_motion(self.agent):
                        self.failed = True

            elif self.phase == 1:
                candidate = self._next_point(self.crate, direction)
                if candidate is None:
                    self.failed = True
                    error = True
                else:
                    self.crate = candidate
                    self.agent = candidate
                    self.used_cells.add(candidate)
                    if self.crate == self.pit:
                        self.bridge_built = True
                        self.phase = 2
                        self._enter_new_room()
                    elif not self._has_valid_motion(self.crate):
                        self.failed = True

            elif self.phase == 3:
                if self.agent == self.key:
                    self.key_held = True
                    self.phase = 4
                    self._enter_new_room()
                else:
                    self.failed = True
                    error = True

            elif self.phase == 5:
                if self.agent == self.door and self.key_held:
                    self.door_open = True
                    self.phase = 6
                    self._enter_new_room()
                else:
                    self.failed = True
                    error = True

            else:
                self.failed = True
                error = True

        if self.success:
            reward = 1.0

        after = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        unlocked = tuple(
            item
            for item in after.available_actions
            if item.signature not in before_actions
        )
        return GridPushStep(
            after,
            after.facts - before.facts,
            before.facts - after.facts,
            unlocked,
            error,
            reward,
        )


_ORACLE_STEPS: dict[int, int | None] = {}


def _world_key(world: BenchmarkGridPushWorld) -> tuple[Any, ...]:
    return (
        world.agent,
        world.crate,
        world.phase,
        world.bridge_built,
        world.key_held,
        world.door_open,
        tuple(sorted(world.used_cells)),
        world.success,
        world.failed,
    )


def oracle_shortest_steps(seed: int) -> int | None:
    """Return the exact shortest successful path for one procedural map."""

    if int(seed) in _ORACLE_STEPS:
        return _ORACLE_STEPS[int(seed)]

    root = BenchmarkGridPushWorld(int(seed))
    queue: deque[tuple[BenchmarkGridPushWorld, int]] = deque([(root, 0)])
    visited = {_world_key(root)}
    while queue:
        world, depth = queue.popleft()
        for action in CHOICE_ACTIONS:
            candidate = deepcopy(world)
            candidate.step(action)
            if candidate.success:
                _ORACLE_STEPS[int(seed)] = depth + 1
                return depth + 1
            if candidate.failed:
                continue
            key = _world_key(candidate)
            if key in visited:
                continue
            visited.add(key)
            queue.append((candidate, depth + 1))

    _ORACLE_STEPS[int(seed)] = None
    return None


def solvable_map_seeds(start: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    result: list[int] = []
    candidate = int(start)
    while len(result) < count:
        if oracle_shortest_steps(candidate) is not None:
            result.append(candidate)
        candidate += 1
    return tuple(result)


def encode_gridpush_state(state: StateSnapshot) -> tuple[float, ...]:
    used = [0.0] * (BenchmarkGridPushWorld.grid_size ** 2)
    for fact in state.facts:
        if not fact.startswith("used:"):
            continue
        _, raw_x, raw_y = fact.split(":")
        x, y = int(raw_x), int(raw_y)
        used[y * BenchmarkGridPushWorld.grid_size + x] = 1.0
    return tuple(float(value) for value in state.vector) + tuple(used)


GRIDPUSH_OBSERVATION_SIZE = 16 + 9


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: Action
    imagined_nodes: int = 0
    used_imagination: bool = False


class BenchmarkAgent(Protocol):
    name: str

    def begin_episode(self, *, training: bool) -> None: ...

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision: ...

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None: ...

    def end_episode(self, *, success: bool, training: bool) -> None: ...

    def model_stats(self) -> dict[str, int | float]: ...


class RandomBenchmarkAgent:
    name = "random"

    def __init__(self, seed: int, *, train_episodes: int) -> None:
        del train_episodes
        self.randomizer = random.Random(seed)

    def begin_episode(self, *, training: bool) -> None:
        del training

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        del episode, training
        return AgentDecision(self.randomizer.choice(state.available_actions))

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None:
        del before, action, outcome

    def end_episode(self, *, success: bool, training: bool) -> None:
        del success, training

    def model_stats(self) -> dict[str, int | float]:
        return {"model_units": 0, "model_bytes": 0}


class TabularQLearningAgent:
    name = "q_learning"

    def __init__(
        self,
        seed: int,
        *,
        train_episodes: int,
        learning_rate: float = 0.20,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
    ) -> None:
        self.randomizer = random.Random(seed)
        self.train_episodes = int(train_episodes)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.q: dict[tuple[tuple[float, ...], int], float] = {}
        self.updates = 0

    def begin_episode(self, *, training: bool) -> None:
        del training

    def _epsilon(self, episode: int) -> float:
        horizon = max(1, int(self.train_episodes * 0.80))
        fraction = min(1.0, max(0.0, episode / horizon))
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def _value(self, state: StateSnapshot, index: int) -> float:
        return self.q.get((encode_gridpush_state(state), index), 0.0)

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        epsilon = self._epsilon(episode) if training else 0.0
        if training and self.randomizer.random() < epsilon:
            index = self.randomizer.randrange(len(CHOICE_ACTIONS))
        else:
            values = [
                self._value(state, index)
                for index in range(len(CHOICE_ACTIONS))
            ]
            best = max(values)
            candidates = [
                index
                for index, value in enumerate(values)
                if math.isclose(value, best, abs_tol=1e-12)
            ]
            index = self.randomizer.choice(candidates)
        return AgentDecision(CHOICE_ACTIONS[index])

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None:
        index = choice_index(action)
        key = (encode_gridpush_state(before), index)
        old = self.q.get(key, 0.0)
        terminal = not outcome.snapshot.available_actions
        next_value = 0.0
        if not terminal:
            next_value = max(
                self._value(outcome.snapshot, candidate)
                for candidate in range(len(CHOICE_ACTIONS))
            )
        target = float(outcome.reward) + self.gamma * next_value
        self.q[key] = old + self.learning_rate * (target - old)
        self.updates += 1

    def end_episode(self, *, success: bool, training: bool) -> None:
        del success, training

    def model_stats(self) -> dict[str, int | float]:
        encoded = pickle.dumps(self.q, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "model_units": len(self.q),
            "model_bytes": len(encoded),
            "gradient_updates": self.updates,
        }


class DQNBenchmarkAgent:
    name = "dqn"

    def __init__(
        self,
        seed: int,
        *,
        train_episodes: int,
        hidden_units: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        replay_capacity: int = 50_000,
        batch_size: int = 64,
        warmup_steps: int = 256,
        target_update_interval: int = 250,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DQN benchmark requires torch") from exc

        self.torch = torch
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover
            torch.use_deterministic_algorithms(True)

        self.train_episodes = int(train_episodes)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        self.replay: deque[
            tuple[
                tuple[float, ...],
                int,
                float,
                tuple[float, ...],
                bool,
            ]
        ] = deque(maxlen=int(replay_capacity))
        self.online = nn.Sequential(
            nn.Linear(GRIDPUSH_OBSERVATION_SIZE, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(CHOICE_ACTIONS)),
        )
        self.target = nn.Sequential(
            nn.Linear(GRIDPUSH_OBSERVATION_SIZE, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(CHOICE_ACTIONS)),
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(learning_rate),
        )
        self.loss = nn.SmoothL1Loss()
        self.environment_steps = 0
        self.gradient_updates = 0

    def begin_episode(self, *, training: bool) -> None:
        del training

    def _epsilon(self, episode: int) -> float:
        horizon = max(1, int(self.train_episodes * 0.80))
        fraction = min(1.0, max(0.0, episode / horizon))
        return 1.0 + fraction * (0.05 - 1.0)

    def _tensor(
        self,
        values: Sequence[Sequence[float]] | Sequence[float],
    ) -> Any:
        return self.torch.as_tensor(values, dtype=self.torch.float32)

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        epsilon = self._epsilon(episode) if training else 0.0
        if training and self.randomizer.random() < epsilon:
            index = self.randomizer.randrange(len(CHOICE_ACTIONS))
        else:
            with self.torch.no_grad():
                observation = self._tensor(
                    encode_gridpush_state(state)
                ).unsqueeze(0)
                values = self.online(observation)[0]
                index = int(self.torch.argmax(values).item())
        return AgentDecision(CHOICE_ACTIONS[index])

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None:
        terminal = not outcome.snapshot.available_actions
        self.replay.append(
            (
                encode_gridpush_state(before),
                choice_index(action),
                float(outcome.reward),
                encode_gridpush_state(outcome.snapshot),
                terminal,
            )
        )
        self.environment_steps += 1
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return

        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        observations = self._tensor([item[0] for item in batch])
        actions = self.torch.as_tensor(
            [item[1] for item in batch],
            dtype=self.torch.int64,
        )
        rewards = self._tensor([item[2] for item in batch])
        next_observations = self._tensor([item[3] for item in batch])
        terminals = self._tensor([float(item[4]) for item in batch])

        predicted = self.online(observations).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        with self.torch.no_grad():
            next_values = self.target(next_observations).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - terminals) * next_values

        loss = self.loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.gradient_updates += 1

        if self.gradient_updates % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def end_episode(self, *, success: bool, training: bool) -> None:
        del success, training

    def model_stats(self) -> dict[str, int | float]:
        handle = io.BytesIO()
        self.torch.save(self.online.state_dict(), handle)
        return {
            "model_units": sum(
                parameter.numel() for parameter in self.online.parameters()
            ),
            "model_bytes": len(handle.getvalue()),
            "gradient_updates": self.gradient_updates,
            "replay_size": len(self.replay),
        }


class AASSRBenchmarkAgent:
    def __init__(
        self,
        seed: int,
        *,
        train_episodes: int,
        use_imagination: bool,
    ) -> None:
        del train_episodes
        self.name = "aassr_full" if use_imagination else "policy_only"
        depth = 5 if use_imagination else 1
        self.agent = AutonomousLearningAgent(
            TabularProphecy(),
            config=AutonomousAgentConfig(
                use_imagination=use_imagination,
                imagination_depth=depth,
                imagination_branching_factor=2,
                imagination_beam_width=16,
                imagination_outcome_samples=2,
                imagination_minimum_coverage=0.0,
                imagination_intervention_margin=0.02,
                imagination_uncertainty_margin=0.25,
                imagination_aggregation="risk-adjusted",
                epsilon_start=0.90,
                epsilon_end=0.05,
                epsilon_decay_episodes=250,
                effect_minimum_samples=2,
            ),
            seed=seed,
        )
        self.agent.planner.scorer = StateDeltaScorer(
            goal_progress_weight=50.0,
            new_fact_weight=4.0,
            unlocked_action_weight=2.0,
            step_cost=0.01,
        )

    def begin_episode(self, *, training: bool) -> None:
        if not training:
            self.agent.discard_episode()

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        decision: ActionDecision = self.agent.select_action(
            state,
            episode=episode,
            explore=training,
        )
        return AgentDecision(
            decision.action,
            decision.imagined_nodes,
            decision.used_imagination,
        )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None:
        self.agent.observe(before, action, outcome)

    def end_episode(self, *, success: bool, training: bool) -> None:
        if training:
            self.agent.finish_episode(final_return=1.0 if success else 0.0)
        else:
            self.agent.discard_episode()

    def model_stats(self) -> dict[str, int | float]:
        base = self.agent.base_prophecy
        prophecy = self.agent.prophecy
        payload = serialize_agent_checkpoint(self.agent, episode=0)
        payload["effect_prophecy"] = {
            "exact_bucket_count": len(
                getattr(prophecy, "_exact_effects", {})
            ),
            "signature_bucket_count": len(
                getattr(prophecy, "_signature_effects", {})
            ),
            "family_bucket_count": len(
                getattr(prophecy, "_family_effects", {})
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        ).encode("utf-8")
        policy = self.agent.policy
        return {
            "model_units": (
                len(getattr(policy, "_local", {}))
                + len(getattr(base, "_exact", {}))
                + int(getattr(prophecy, "effect_bucket_count", 0))
            ),
            "model_bytes": len(encoded),
            "policy_entries": len(getattr(policy, "_local", {})),
            "prophecy_entries": len(getattr(base, "_exact", {})),
            "effect_buckets": int(
                getattr(prophecy, "effect_bucket_count", 0)
            ),
        }


def make_benchmark_agent(
    condition: str,
    seed: int,
    *,
    train_episodes: int,
) -> BenchmarkAgent:
    if condition == "random":
        return RandomBenchmarkAgent(seed, train_episodes=train_episodes)
    if condition == "q_learning":
        return TabularQLearningAgent(seed, train_episodes=train_episodes)
    if condition == "dqn":
        return DQNBenchmarkAgent(seed, train_episodes=train_episodes)
    if condition == "policy_only":
        return AASSRBenchmarkAgent(
            seed,
            train_episodes=train_episodes,
            use_imagination=False,
        )
    if condition == "aassr_full":
        return AASSRBenchmarkAgent(
            seed,
            train_episodes=train_episodes,
            use_imagination=True,
        )
    raise ValueError(f"unknown benchmark condition: {condition}")


@dataclass(frozen=True, slots=True)
class EpisodeMetric:
    condition: str
    seed: int
    phase: str
    checkpoint_episode: int
    episode: int
    map_seed: int
    success: int
    reward: float
    steps: int
    optimal_steps: int
    path_efficiency: float
    environment_steps_total: int
    select_seconds: float
    update_seconds: float
    imagined_nodes: int
    imagination_runs: int

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointMetric:
    condition: str
    seed: int
    checkpoint_episode: int
    environment_steps_total: int
    training_success_rate: float
    training_return_mean: float
    training_wall_seconds: float
    selection_seconds_total: float
    update_seconds_total: float
    peak_rss_mb: float
    model_units: int
    model_bytes: int
    gradient_updates: int
    seen_success_rate: float
    unseen_success_rate: float
    seen_mean_steps_on_success: float
    unseen_mean_steps_on_success: float
    seen_path_efficiency: float
    unseen_path_efficiency: float
    seen_eval_seconds: float
    unseen_eval_seconds: float

    def row(self) -> dict[str, Any]:
        return asdict(self)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if value < 10_000_000 else 1024.0 ** 2)


def _run_episode(
    agent: BenchmarkAgent,
    *,
    condition: str,
    seed: int,
    phase: str,
    checkpoint_episode: int,
    episode: int,
    map_seed: int,
    training: bool,
    environment_steps_total: int,
) -> tuple[EpisodeMetric, int]:
    oracle_steps = oracle_shortest_steps(map_seed)
    if oracle_steps is None:
        raise ValueError(f"benchmark map is not solvable: {map_seed}")
    world = BenchmarkGridPushWorld(map_seed)
    world.optimal_steps = oracle_steps
    agent.begin_episode(training=training)
    steps = 0
    select_seconds = 0.0
    update_seconds = 0.0
    imagined_nodes = 0
    imagination_runs = 0

    while world.snapshot().available_actions:
        before = world.snapshot()
        started = time.perf_counter()
        decision = agent.select_action(
            before,
            episode=episode,
            training=training,
        )
        select_seconds += time.perf_counter() - started

        outcome = world.step(decision.action)
        steps += 1
        environment_steps_total += int(training)
        imagined_nodes += int(decision.imagined_nodes)
        imagination_runs += int(decision.used_imagination)

        if training:
            started = time.perf_counter()
            agent.observe(before, decision.action, outcome)
            update_seconds += time.perf_counter() - started
        if world.success or world.failed:
            break

    agent.end_episode(success=world.success, training=training)
    metric = EpisodeMetric(
        condition=condition,
        seed=seed,
        phase=phase,
        checkpoint_episode=checkpoint_episode,
        episode=episode,
        map_seed=map_seed,
        success=int(world.success),
        reward=1.0 if world.success else 0.0,
        steps=steps,
        optimal_steps=world.optimal_steps,
        path_efficiency=(
            world.optimal_steps / steps if world.success and steps else 0.0
        ),
        environment_steps_total=environment_steps_total,
        select_seconds=select_seconds,
        update_seconds=update_seconds,
        imagined_nodes=imagined_nodes,
        imagination_runs=imagination_runs,
    )
    return metric, environment_steps_total


def _mean_success(rows: Sequence[EpisodeMetric]) -> float:
    return fmean(row.success for row in rows) if rows else 0.0


def _mean_success_field(
    rows: Sequence[EpisodeMetric],
    field: str,
) -> float:
    successful = [row for row in rows if row.success]
    return (
        fmean(float(getattr(row, field)) for row in successful)
        if successful
        else 0.0
    )


def _evaluate(
    agent: BenchmarkAgent,
    *,
    condition: str,
    seed: int,
    phase: str,
    checkpoint_episode: int,
    map_seeds: Sequence[int],
    environment_steps_total: int,
) -> tuple[list[EpisodeMetric], float]:
    rows: list[EpisodeMetric] = []
    started = time.perf_counter()
    for index, map_seed in enumerate(map_seeds):
        row, _ = _run_episode(
            agent,
            condition=condition,
            seed=seed,
            phase=phase,
            checkpoint_episode=checkpoint_episode,
            episode=index,
            map_seed=int(map_seed),
            training=False,
            environment_steps_total=environment_steps_total,
        )
        rows.append(row)
    return rows, time.perf_counter() - started


def _write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].row()),
        )
        writer.writeheader()
        writer.writerows(row.row() for row in rows)


def run_gridpush_baseline_benchmark(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    train_episodes: int = 1000,
    train_map_count: int = 64,
    evaluation_episodes: int = 50,
    checkpoints: Sequence[int] = (0, 50, 100, 250, 500, 1000),
) -> dict[str, Any]:
    if train_episodes <= 0 or train_map_count <= 0:
        raise ValueError("training sizes must be positive")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    normalized_checkpoints = tuple(
        sorted(
            {
                int(value)
                for value in checkpoints
                if 0 <= int(value) <= train_episodes
            }
            | {0, int(train_episodes)}
        )
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agent = make_benchmark_agent(
        condition,
        int(seed),
        train_episodes=train_episodes,
    )
    training_maps = solvable_map_seeds(
        seed * 1_000_000,
        train_map_count,
    )
    seen_maps = tuple(
        training_maps[index % len(training_maps)]
        for index in range(evaluation_episodes)
    )
    unseen_maps = solvable_map_seeds(
        seed * 1_000_000 + 500_000,
        evaluation_episodes,
    )

    training_rows: list[EpisodeMetric] = []
    evaluation_rows: list[EpisodeMetric] = []
    checkpoint_rows: list[CheckpointMetric] = []
    environment_steps_total = 0
    selection_seconds_total = 0.0
    update_seconds_total = 0.0
    training_started = time.perf_counter()

    for checkpoint in normalized_checkpoints:
        while len(training_rows) < checkpoint:
            episode = len(training_rows)
            row, environment_steps_total = _run_episode(
                agent,
                condition=condition,
                seed=seed,
                phase="training",
                checkpoint_episode=checkpoint,
                episode=episode,
                map_seed=training_maps[episode % len(training_maps)],
                training=True,
                environment_steps_total=environment_steps_total,
            )
            training_rows.append(row)
            selection_seconds_total += row.select_seconds
            update_seconds_total += row.update_seconds

        seen_rows, seen_seconds = _evaluate(
            agent,
            condition=condition,
            seed=seed,
            phase="evaluation_seen",
            checkpoint_episode=checkpoint,
            map_seeds=seen_maps,
            environment_steps_total=environment_steps_total,
        )
        unseen_rows, unseen_seconds = _evaluate(
            agent,
            condition=condition,
            seed=seed,
            phase="evaluation_unseen",
            checkpoint_episode=checkpoint,
            map_seeds=unseen_maps,
            environment_steps_total=environment_steps_total,
        )
        evaluation_rows.extend(seen_rows)
        evaluation_rows.extend(unseen_rows)

        recent = training_rows[-min(100, len(training_rows)) :]
        stats = agent.model_stats()
        checkpoint_rows.append(
            CheckpointMetric(
                condition=condition,
                seed=seed,
                checkpoint_episode=checkpoint,
                environment_steps_total=environment_steps_total,
                training_success_rate=_mean_success(recent),
                training_return_mean=(
                    fmean(row.reward for row in recent) if recent else 0.0
                ),
                training_wall_seconds=(
                    time.perf_counter() - training_started
                ),
                selection_seconds_total=selection_seconds_total,
                update_seconds_total=update_seconds_total,
                peak_rss_mb=_peak_rss_mb(),
                model_units=int(stats.get("model_units", 0)),
                model_bytes=int(stats.get("model_bytes", 0)),
                gradient_updates=int(stats.get("gradient_updates", 0)),
                seen_success_rate=_mean_success(seen_rows),
                unseen_success_rate=_mean_success(unseen_rows),
                seen_mean_steps_on_success=_mean_success_field(
                    seen_rows, "steps"
                ),
                unseen_mean_steps_on_success=_mean_success_field(
                    unseen_rows, "steps"
                ),
                seen_path_efficiency=_mean_success_field(
                    seen_rows, "path_efficiency"
                ),
                unseen_path_efficiency=_mean_success_field(
                    unseen_rows, "path_efficiency"
                ),
                seen_eval_seconds=seen_seconds,
                unseen_eval_seconds=unseen_seconds,
            )
        )

    _write_csv(output / "training_episodes.csv", training_rows)
    _write_csv(output / "evaluation_episodes.csv", evaluation_rows)
    _write_csv(output / "checkpoints.csv", checkpoint_rows)

    final = checkpoint_rows[-1]
    payload = {
        "config": {
            "condition": condition,
            "seed": seed,
            "train_episodes": train_episodes,
            "train_map_count": train_map_count,
            "evaluation_episodes": evaluation_episodes,
            "checkpoints": list(normalized_checkpoints),
            "observation_size": GRIDPUSH_OBSERVATION_SIZE,
            "action_count": len(CHOICE_ACTIONS),
            "external_reward": "final_success_only",
            "fixed_action_space": True,
            "artificial_tick_limit": False,
            "only_solvable_maps": True,
        },
        "final": final.row(),
        "checkpoints": [row.row() for row in checkpoint_rows],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
