from __future__ import annotations

import csv
import io
import json
import random
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .autonomous_agent_core import ActionDecision, AutonomousAgentConfig, AutonomousLearningAgent
from .branch_critic import CriticTransition, GRUBranchCritic
from .goal_gridpush_experiment import GridPushStep
from .neural_delta_prophecy import NeuralDeltaConfig, NeuralDeltaProphecy, StateCodec
from .policy import PolicyMemory, ScoredAction
from .types import Action, Prediction, StateSnapshot


GRID_SIZES: tuple[int, ...] = (3, 5, 7)
ACTION_COUNTS: tuple[int, ...] = (8, 12)
TOOLGRID_CONDITIONS: tuple[str, ...] = (
    "dqn",
    "neural_policy_only",
    "imagination_v2",
)
MAX_GRID_SIZE = 7
MAX_TOOL_COUNT = 8
STAGE_COUNT = 4
TOOLGRID_STATE_SIZE = 2 + 1 + 1 + STAGE_COUNT * 3 + MAX_GRID_SIZE**2

MOVE_NAMES: tuple[str, ...] = ("north", "south", "west", "east")
MOVE_DELTAS: Mapping[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}


def build_actions(action_count: int) -> tuple[Action, ...]:
    if action_count not in ACTION_COUNTS:
        raise ValueError(f"unsupported action_count: {action_count}")
    moves = tuple(Action(f"move_{name}") for name in MOVE_NAMES)
    tools = tuple(Action(f"tool_{index}") for index in range(action_count - 4))
    return moves + tools


def action_index(action: Action, actions: Sequence[Action]) -> int:
    for index, candidate in enumerate(actions):
        if candidate.signature == action.signature:
            return index
    raise ValueError(f"action is not in this ToolGrid action space: {action.signature}")


@dataclass(frozen=True, slots=True)
class ToolGridMap:
    map_seed: int
    start: tuple[int, int]
    stations: tuple[tuple[int, int], ...]
    required_tools: tuple[int, ...]
    oracle_shortest_steps: int


class ToolGridWorld:
    """Sparse-reward navigation and tool-selection benchmark.

    The map-size factor changes only spatial horizon. The action-count factor
    changes the number of globally meaningful tool actions: every tool is useful
    on some generated station, while choosing the wrong tool at the current
    station irreversibly damages it. Movement and tool actions are simultaneously
    available, so larger action spaces create genuine semantic branching rather
    than duplicate aliases or permanently useless buttons.
    """

    def __init__(self, seed: int, *, grid_size: int, action_count: int) -> None:
        if grid_size not in GRID_SIZES:
            raise ValueError(f"unsupported grid_size: {grid_size}")
        self.seed = int(seed)
        self.grid_size = int(grid_size)
        self.actions = build_actions(action_count)
        self.action_count = len(self.actions)
        self.tool_count = self.action_count - 4
        randomizer = random.Random(self.seed)
        points = randomizer.sample(
            [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)],
            STAGE_COUNT + 1,
        )
        self.agent = points[0]
        self.start = points[0]
        self.stations = tuple(points[1:])
        self.required_tools = tuple(
            randomizer.randrange(self.tool_count) for _ in range(STAGE_COUNT)
        )
        self.phase = 0
        self.success = False
        self.failed = False
        self.used_cells: set[tuple[int, int]] = {self.agent}
        self.steps = 0
        route = (self.start,) + self.stations
        self.optimal_steps = sum(
            self._distance(route[index], route[index + 1]) + 1
            for index in range(STAGE_COUNT)
        )

    @staticmethod
    def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    @property
    def current_station(self) -> tuple[int, int] | None:
        if self.phase >= STAGE_COUNT:
            return None
        return self.stations[self.phase]

    @property
    def current_tool(self) -> int | None:
        if self.phase >= STAGE_COUNT:
            return None
        return self.required_tools[self.phase]

    def map_profile(self) -> ToolGridMap:
        return ToolGridMap(
            map_seed=self.seed,
            start=self.start,
            stations=self.stations,
            required_tools=self.required_tools,
            oracle_shortest_steps=self.optimal_steps,
        )

    def _normalize_point(self, point: tuple[int, int]) -> tuple[float, float]:
        scale = float(self.grid_size - 1)
        return point[0] / scale, point[1] / scale

    def _vector(self) -> tuple[float, ...]:
        values: list[float] = [
            *self._normalize_point(self.agent),
            self.phase / float(STAGE_COUNT),
            self.tool_count / float(MAX_TOOL_COUNT),
        ]
        for station, tool in zip(self.stations, self.required_tools, strict=True):
            values.extend(
                (
                    *self._normalize_point(station),
                    tool / float(max(1, MAX_TOOL_COUNT - 1)),
                )
            )
        used = [0.0] * (MAX_GRID_SIZE**2)
        for x, y in self.used_cells:
            used[y * MAX_GRID_SIZE + x] = 1.0
        values.extend(used)
        if len(values) != TOOLGRID_STATE_SIZE:
            raise AssertionError("ToolGrid state encoder size drift")
        return tuple(values)

    def _facts(self) -> frozenset[str]:
        facts = {
            f"phase:{self.phase}",
            f"grid_size:{self.grid_size}",
            f"action_count:{self.action_count}",
            *(f"used:{x}:{y}" for x, y in sorted(self.used_cells)),
        }
        if self.current_tool is not None:
            facts.add(f"required_tool:{self.current_tool}")
        if self.success:
            facts.add("success")
        if self.failed:
            facts.add("failed")
        return frozenset(facts)

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            vector=self._vector(),
            facts=self._facts(),
            available_actions=() if self.success or self.failed else self.actions,
            goal_progress=1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "grid_size": self.grid_size,
                "action_count": self.action_count,
                "tool_count": self.tool_count,
                "stage_count": STAGE_COUNT,
                "stations": self.stations,
                "required_tools": self.required_tools,
                "optimal_steps": self.optimal_steps,
                "termination": "toolgrid_irreversible",
            },
        )

    def _move(self, name: str) -> bool:
        dx, dy = MOVE_DELTAS[name]
        candidate = self.agent[0] + dx, self.agent[1] + dy
        if not (
            0 <= candidate[0] < self.grid_size
            and 0 <= candidate[1] < self.grid_size
        ):
            return False
        if candidate in self.used_cells:
            return False
        self.agent = candidate
        self.used_cells.add(candidate)
        return True

    def _has_valid_move(self) -> bool:
        for name in MOVE_NAMES:
            dx, dy = MOVE_DELTAS[name]
            candidate = self.agent[0] + dx, self.agent[1] + dy
            if (
                0 <= candidate[0] < self.grid_size
                and 0 <= candidate[1] < self.grid_size
                and candidate not in self.used_cells
            ):
                return True
        return False

    def step(self, action: Action) -> GridPushStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        self.steps += 1

        try:
            index = action_index(action, self.actions)
        except ValueError:
            self.failed = True
            error = True
        else:
            if index < 4:
                moved = self._move(MOVE_NAMES[index])
                if not moved:
                    self.failed = True
                    error = True
                elif self.agent != self.current_station and not self._has_valid_move():
                    self.failed = True
            else:
                tool_index = index - 4
                if self.agent != self.current_station or tool_index != self.current_tool:
                    self.failed = True
                    error = True
                else:
                    self.phase += 1
                    if self.phase >= STAGE_COUNT:
                        self.success = True
                        reward = 1.0
                    else:
                        self.used_cells = {self.agent}

        after = self.snapshot()
        before_actions = {item.signature for item in before.available_actions}
        unlocked = tuple(
            item for item in after.available_actions if item.signature not in before_actions
        )
        return GridPushStep(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=unlocked,
            error=error,
            reward=reward,
        )

    def oracle_actions(self) -> tuple[Action, ...]:
        """Return one exact shortest solution, used only by tests/profiling."""

        actions: list[Action] = []
        position = self.start
        for station, tool in zip(self.stations, self.required_tools, strict=True):
            while position[0] < station[0]:
                actions.append(self.actions[3])
                position = position[0] + 1, position[1]
            while position[0] > station[0]:
                actions.append(self.actions[2])
                position = position[0] - 1, position[1]
            while position[1] < station[1]:
                actions.append(self.actions[1])
                position = position[0], position[1] + 1
            while position[1] > station[1]:
                actions.append(self.actions[0])
                position = position[0], position[1] - 1
            actions.append(self.actions[4 + tool])
        return tuple(actions)


def encode_toolgrid_state(state: StateSnapshot) -> tuple[float, ...]:
    values = tuple(float(value) for value in state.vector)
    if len(values) != TOOLGRID_STATE_SIZE:
        raise ValueError(
            f"ToolGrid state must contain {TOOLGRID_STATE_SIZE} values, got {len(values)}"
        )
    return values


@dataclass(frozen=True, slots=True)
class ToolGridCodec(StateCodec):
    action_count: int

    def __post_init__(self) -> None:
        build_actions(self.action_count)

    @property
    def dimension(self) -> int:
        return TOOLGRID_STATE_SIZE

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        return encode_toolgrid_state(state)

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != self.dimension:
            raise ValueError("ToolGrid neural state has an unexpected size")
        values = [self._bounded(value) for value in encoded]
        grid_size = int(scaffold.metadata.get("grid_size", MAX_GRID_SIZE))
        scale = float(grid_size - 1)
        values[0] = round(values[0] * scale) / scale
        values[1] = round(values[1] * scale) / scale
        phase = min(STAGE_COUNT, max(0, int(round(values[2] * STAGE_COUNT))))
        values[2] = phase / float(STAGE_COUNT)
        tool_count = self.action_count - 4
        values[3] = tool_count / float(MAX_TOOL_COUNT)
        offset = 4
        for _ in range(STAGE_COUNT):
            values[offset] = round(values[offset] * scale) / scale
            values[offset + 1] = round(values[offset + 1] * scale) / scale
            tool = min(
                tool_count - 1,
                max(0, int(round(values[offset + 2] * (MAX_TOOL_COUNT - 1)))),
            )
            values[offset + 2] = tool / float(MAX_TOOL_COUNT - 1)
            offset += 3
        for index in range(offset, len(values)):
            values[index] = float(values[index] >= 0.5)

        facts = {
            f"phase:{phase}",
            f"grid_size:{grid_size}",
            f"action_count:{self.action_count}",
        }
        for index, occupied in enumerate(values[offset:]):
            if occupied < 0.5:
                continue
            x = index % MAX_GRID_SIZE
            y = index // MAX_GRID_SIZE
            if x < grid_size and y < grid_size:
                facts.add(f"used:{x}:{y}")
        if phase < STAGE_COUNT:
            descriptor = 4 + phase * 3 + 2
            tool = min(
                tool_count - 1,
                max(0, int(round(values[descriptor] * (MAX_TOOL_COUNT - 1)))),
            )
            facts.add(f"required_tool:{tool}")
        if terminal_class == 1:
            facts.add("success")
        elif terminal_class == 2:
            facts.add("failed")
        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "imagined_neural_delta": True,
                "imagined_neural_delta_source": source,
                "action_count": self.action_count,
            }
        )
        return StateSnapshot(
            vector=tuple(values),
            facts=frozenset(facts),
            available_actions=() if terminal_class else build_actions(self.action_count),
            goal_progress=1.0 if terminal_class == 1 else 0.0,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: Action
    imagined_nodes: int = 0
    used_imagination: bool = False


class ToolGridDQNAgent:
    name = "dqn"

    def __init__(
        self,
        seed: int,
        *,
        action_count: int,
        train_transitions: int,
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
            raise RuntimeError("ToolGrid DQN requires torch") from exc
        self.torch = torch
        self.randomizer = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover
            torch.use_deterministic_algorithms(True)
        self.actions = build_actions(action_count)
        self.action_by_signature = {
            action.signature: index for index, action in enumerate(self.actions)
        }
        self.train_transitions = int(train_transitions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        self.replay: deque[
            tuple[tuple[float, ...], int, float, tuple[float, ...], bool]
        ] = deque(maxlen=int(replay_capacity))
        self.online = nn.Sequential(
            nn.Linear(TOOLGRID_STATE_SIZE, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(self.actions)),
        )
        self.target = nn.Sequential(
            nn.Linear(TOOLGRID_STATE_SIZE, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(self.actions)),
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.environment_steps = 0
        self.gradient_updates = 0

    def begin_episode(self, *, training: bool) -> None:
        del training

    def _epsilon(self, transition: int) -> float:
        horizon = max(1, int(self.train_transitions * 0.80))
        fraction = min(1.0, max(0.0, transition / horizon))
        return 1.0 + fraction * (0.05 - 1.0)

    def _tensor(self, values: Any) -> Any:
        return self.torch.as_tensor(values, dtype=self.torch.float32)

    def values(self, state: StateSnapshot) -> tuple[float, ...]:
        with self.torch.no_grad():
            tensor = self._tensor(encode_toolgrid_state(state)).unsqueeze(0)
            return tuple(float(item) for item in self.online(tensor)[0].tolist())

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        epsilon = self._epsilon(episode) if training else 0.0
        available = tuple(state.available_actions)
        if training and self.randomizer.random() < epsilon:
            action = self.randomizer.choice(available)
        else:
            values = self.values(state)
            action = max(
                available,
                key=lambda item: (
                    values[self.action_by_signature[item.signature]],
                    item.signature,
                ),
            )
        return AgentDecision(action)

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None:
        terminal = not outcome.snapshot.available_actions
        self.replay.append(
            (
                encode_toolgrid_state(before),
                self.action_by_signature[action.signature],
                float(outcome.reward),
                encode_toolgrid_state(outcome.snapshot),
                terminal,
            )
        )
        self.environment_steps += 1
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return
        batch = self.randomizer.sample(list(self.replay), self.batch_size)
        observations = self._tensor([item[0] for item in batch])
        actions = self.torch.as_tensor([item[1] for item in batch], dtype=self.torch.int64)
        rewards = self._tensor([item[2] for item in batch])
        next_observations = self._tensor([item[3] for item in batch])
        terminals = self._tensor([float(item[4]) for item in batch])
        predicted = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
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
            "model_units": sum(parameter.numel() for parameter in self.online.parameters()),
            "model_bytes": len(handle.getvalue()),
            "gradient_updates": self.gradient_updates,
            "replay_size": len(self.replay),
        }


class ToolGridDQNPolicyAdapter:
    def __init__(self, dqn: ToolGridDQNAgent) -> None:
        self.dqn = dqn

    def value(self, state: StateSnapshot, action: Action) -> float:
        return self.dqn.values(state)[self.dqn.action_by_signature[action.signature]]

    def rank(
        self,
        state: StateSnapshot,
        *,
        limit: int,
        memory: PolicyMemory | None = None,
    ) -> tuple[ScoredAction, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        deltas: Mapping[str, float] = {} if memory is None else memory.deltas
        values = self.dqn.values(state)
        ranked = sorted(
            (
                ScoredAction(
                    action,
                    values[self.dqn.action_by_signature[action.signature]]
                    + deltas.get(action.signature, 0.0),
                )
                for action in state.available_actions
            ),
            key=lambda item: (-item.score, item.action.signature),
        )
        return tuple(ranked[:limit])

    def select(
        self,
        state: StateSnapshot,
        *,
        randomizer: random.Random,
        epsilon: float,
        exploration_bonus: float,
    ) -> Action:
        del exploration_bonus
        if epsilon > 0.0 and randomizer.random() < epsilon:
            return randomizer.choice(state.available_actions)
        return self.rank(state, limit=1)[0].action

    def imagine_update(
        self,
        memory: PolicyMemory,
        action: Action,
        value: float,
    ) -> PolicyMemory:
        deltas = dict(memory.deltas)
        deltas[action.signature] = deltas.get(action.signature, 0.0) + 0.1 * value
        return PolicyMemory(deltas)

    def reinforce(self, action: Action, advantage: float) -> None:
        del action, advantage

    def observe_return(
        self,
        state: StateSnapshot,
        action: Action,
        target: float,
    ) -> None:
        del state, action, target


class ToolGridCalibratedProphecy:
    name = "toolgrid-calibrated-neural-delta"

    def __init__(
        self,
        base: NeuralDeltaProphecy,
        holdout: Any,
        actions: Sequence[Action],
        *,
        minimum_count: int = 8,
        evaluation_limit: int = 64,
        refresh_stride: int = 32,
        calibration_power: float = 1.35,
    ) -> None:
        self.base = base
        self.holdout = holdout
        self.actions = tuple(actions)
        self.minimum_count = int(minimum_count)
        self.evaluation_limit = int(evaluation_limit)
        self.refresh_stride = int(refresh_stride)
        self.calibration_power = float(calibration_power)
        self._cache: dict[tuple[int, str], float] = {}

    def learn(self, state: StateSnapshot, action: Action, actual_next_state: StateSnapshot) -> None:
        self.base.learn(state, action, actual_next_state)

    def _calibration(self, action: Action) -> float:
        items = [
            item
            for item in getattr(self.holdout, "_items", ())
            if item.action.signature == action.signature
        ]
        key = (len(items) // self.refresh_stride, action.signature)
        if key in self._cache:
            return self._cache[key]
        if len(items) < self.minimum_count:
            value = 0.0
        else:
            selected = items[-self.evaluation_limit :]
            scores: list[float] = []
            for item in selected:
                predictions = self.base.predict(item.before, item.action, samples=1)
                predicted = predictions[0].next_state
                vector_error = fmean(
                    abs(left - right)
                    for left, right in zip(predicted.vector, item.after.vector, strict=True)
                )
                terminal_match = bool(predicted.available_actions) == bool(item.after.available_actions)
                scores.append(max(0.0, 1.0 - vector_error) * (1.0 if terminal_match else 0.5))
            value = (fmean(scores) if scores else 0.0) ** self.calibration_power
        value = max(0.0, min(1.0, value))
        self._cache[key] = value
        return value

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        calibration = self._calibration(action)
        return tuple(
            Prediction(
                next_state=item.next_state,
                probability=item.probability * calibration,
                source=f"{item.source}:toolgrid-calibrated",
            )
            for item in self.base.predict(state, action, samples=samples)
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        return min(self.base.confidence(state, action), self._calibration(action))

    def coverage(self, state: StateSnapshot, actions: Iterable[Action]) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return fmean(self.confidence(state, action) for action in materialized)


class ToolGridHybridAgent:
    def __init__(
        self,
        seed: int,
        *,
        action_count: int,
        train_transitions: int,
        use_imagination: bool,
    ) -> None:
        self.name = "imagination_v2" if use_imagination else "neural_policy_only"
        self.actions = build_actions(action_count)
        self.dqn = ToolGridDQNAgent(
            seed,
            action_count=action_count,
            train_transitions=train_transitions,
        )
        prophecy = NeuralDeltaProphecy(
            ToolGridCodec(action_count),
            config=NeuralDeltaConfig(
                hidden_units=128,
                ensemble_size=3,
                replay_capacity=50_000,
                batch_size=64,
                warmup_steps=128,
                learning_rate=1e-3,
                gradient_steps_per_observation=1,
                confidence_prior=256.0,
            ),
            seed=seed ^ 0x544F4F4C,
        )
        policy = ToolGridDQNPolicyAdapter(self.dqn)
        self.agent = AutonomousLearningAgent(
            prophecy,
            config=AutonomousAgentConfig(
                learn_policy=False,
                learn_prophecy=True,
                use_imagination=use_imagination,
                use_effect_composition=False,
                imagination_depth=5,
                imagination_branching_factor=min(4, action_count),
                imagination_beam_width=24,
                imagination_outcome_samples=1,
                imagination_interval=1,
                imagination_minimum_coverage=0.70,
                imagination_intervention_margin=0.15,
                imagination_uncertainty_margin=1.25,
                imagination_aggregation="risk-adjusted",
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_episodes=max(1, int(train_transitions * 0.80)),
                effect_minimum_samples=2,
            ),
            seed=seed,
            policy=policy,
        )
        calibrated = ToolGridCalibratedProphecy(prophecy, self.agent.holdout, self.actions)
        self.agent.prophecy = calibrated
        self.agent.planner.prophecy = calibrated
        self.base_prophecy = prophecy
        self.use_imagination = bool(use_imagination)
        self.critic: GRUBranchCritic | None = None
        self._critic_trajectory: list[CriticTransition] = []
        self._critic_counts: Counter[str] = Counter()
        if self.use_imagination:
            self.critic = GRUBranchCritic(
                encode_toolgrid_state,
                TOOLGRID_STATE_SIZE,
                hidden_units=64,
                batch_size=16,
                replay_capacity=4_000,
                gradient_steps_per_episode=2,
                seed=seed ^ 0x43524954,
            )
            self.agent.planner.scorer = self.critic

    @property
    def critic_ready(self) -> bool:
        if self.critic is None:
            return False
        stats = self.critic.stats()
        return (
            self._critic_counts["episodes"] >= 64
            and self._critic_counts["successes"] >= 4
            and self._critic_counts["failures"] >= 4
            and stats.gradient_updates > 0
        )

    def begin_episode(self, *, training: bool) -> None:
        self.dqn.begin_episode(training=training)
        self._critic_trajectory.clear()
        if not training:
            self.agent.discard_episode()

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> AgentDecision:
        original = self.agent.config
        if self.use_imagination and not self.critic_ready:
            self.agent.config = replace(original, use_imagination=False)
        try:
            decision: ActionDecision = self.agent.select_action(
                state,
                episode=episode,
                explore=training,
            )
        finally:
            self.agent.config = original
        return AgentDecision(
            action=decision.action,
            imagined_nodes=decision.imagined_nodes,
            used_imagination=decision.used_imagination,
        )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> None:
        if self.critic is not None:
            self._critic_trajectory.append(
                CriticTransition(
                    before=before,
                    action=action,
                    after=outcome.snapshot,
                    prophecy_confidence=float(self.agent.prophecy.confidence(before, action)),
                )
            )
        self.dqn.observe(before, action, outcome)
        self.agent.observe(before, action, outcome)

    def end_episode(self, *, success: bool, training: bool) -> None:
        self.dqn.end_episode(success=success, training=training)
        if training:
            if self.critic is not None:
                self.critic.observe_episode(tuple(self._critic_trajectory), success=success)
                self._critic_counts["episodes"] += 1
                self._critic_counts["successes" if success else "failures"] += 1
            self.agent.finish_episode(final_return=1.0 if success else 0.0)
        else:
            self.agent.discard_episode()
        self._critic_trajectory.clear()

    def model_stats(self) -> dict[str, int | float]:
        stats = dict(self.dqn.model_stats())
        prophecy_stats = self.base_prophecy.stats()
        stats["model_units"] = int(stats["model_units"]) + prophecy_stats.parameter_count
        stats["gradient_updates"] = int(stats["gradient_updates"]) + prophecy_stats.gradient_updates
        stats["prophecy_gradient_updates"] = prophecy_stats.gradient_updates
        stats["prophecy_observations"] = prophecy_stats.observations
        stats["prophecy_mean_training_loss"] = prophecy_stats.mean_training_loss
        if self.critic is not None:
            critic = self.critic.stats()
            stats["model_units"] = int(stats["model_units"]) + critic.parameter_count
            stats["model_bytes"] = int(stats["model_bytes"]) + critic.model_bytes
            stats["gradient_updates"] = int(stats["gradient_updates"]) + critic.gradient_updates
            stats["critic_gradient_updates"] = critic.gradient_updates
            stats["critic_episodes"] = critic.episodes
            stats["critic_ready"] = int(self.critic_ready)
        return stats


def make_toolgrid_agent(
    condition: str,
    seed: int,
    *,
    action_count: int,
    train_transitions: int,
) -> ToolGridDQNAgent | ToolGridHybridAgent:
    if condition == "dqn":
        return ToolGridDQNAgent(
            seed,
            action_count=action_count,
            train_transitions=train_transitions,
        )
    if condition == "neural_policy_only":
        return ToolGridHybridAgent(
            seed,
            action_count=action_count,
            train_transitions=train_transitions,
            use_imagination=False,
        )
    if condition == "imagination_v2":
        return ToolGridHybridAgent(
            seed,
            action_count=action_count,
            train_transitions=train_transitions,
            use_imagination=True,
        )
    raise ValueError(f"unknown ToolGrid condition: {condition}")


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    condition: str
    seed: int
    phase: str
    grid_size: int
    action_count: int
    checkpoint_transition_target: int
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
    termination: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointRow:
    condition: str
    seed: int
    grid_size: int
    action_count: int
    checkpoint_transition_target: int
    actual_training_transitions: int
    training_episode_count: int
    seen_success_rate: float
    unseen_success_rate: float
    seen_mean_steps_on_success: float
    unseen_mean_steps_on_success: float
    imagined_nodes_mean: float
    imagination_use_rate: float
    training_wall_seconds: float
    model_units: int
    model_bytes: int
    gradient_updates: int

    def row(self) -> dict[str, Any]:
        return asdict(self)


def _write_rows(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].row()))
        writer.writeheader()
        writer.writerows(row.row() for row in rows)


def _run_episode(
    agent: Any,
    *,
    condition: str,
    research_seed: int,
    phase: str,
    grid_size: int,
    action_count: int,
    checkpoint_transition_target: int,
    episode_index: int,
    map_seed: int,
    training: bool,
    environment_steps_total: int,
    schedule_horizon: int,
) -> tuple[EpisodeRow, int]:
    world = ToolGridWorld(map_seed, grid_size=grid_size, action_count=action_count)
    agent.begin_episode(training=training)
    steps = 0
    select_seconds = 0.0
    update_seconds = 0.0
    imagined_nodes = 0
    imagination_runs = 0
    while world.snapshot().available_actions:
        before = world.snapshot()
        schedule_position = min(schedule_horizon, environment_steps_total)
        started = time.perf_counter()
        decision = agent.select_action(
            before,
            episode=schedule_position,
            training=training,
        )
        select_seconds += time.perf_counter() - started
        outcome = world.step(decision.action)
        steps += 1
        imagined_nodes += int(decision.imagined_nodes)
        imagination_runs += int(decision.used_imagination)
        if training:
            environment_steps_total += 1
            started = time.perf_counter()
            agent.observe(before, decision.action, outcome)
            update_seconds += time.perf_counter() - started
        if world.success or world.failed:
            break
    agent.end_episode(success=world.success, training=training)
    return (
        EpisodeRow(
            condition=condition,
            seed=research_seed,
            phase=phase,
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_transition_target,
            episode=episode_index,
            map_seed=map_seed,
            success=int(world.success),
            reward=1.0 if world.success else 0.0,
            steps=steps,
            optimal_steps=world.optimal_steps,
            path_efficiency=(world.optimal_steps / steps if world.success and steps else 0.0),
            environment_steps_total=environment_steps_total,
            select_seconds=select_seconds,
            update_seconds=update_seconds,
            imagined_nodes=imagined_nodes,
            imagination_runs=imagination_runs,
            termination="success" if world.success else "environment_failure",
        ),
        environment_steps_total,
    )


def _evaluate(
    agent: Any,
    *,
    condition: str,
    research_seed: int,
    phase: str,
    grid_size: int,
    action_count: int,
    checkpoint_transition_target: int,
    map_seeds: Sequence[int],
    environment_steps_total: int,
    schedule_horizon: int,
) -> list[EpisodeRow]:
    rows: list[EpisodeRow] = []
    for index, map_seed in enumerate(map_seeds):
        row, _ = _run_episode(
            agent,
            condition=condition,
            research_seed=research_seed,
            phase=phase,
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_transition_target,
            episode_index=index,
            map_seed=int(map_seed),
            training=False,
            environment_steps_total=environment_steps_total,
            schedule_horizon=schedule_horizon,
        )
        rows.append(row)
    return rows


def _mean_success(rows: Sequence[EpisodeRow]) -> float:
    return fmean(row.success for row in rows) if rows else 0.0


def _success_mean(rows: Sequence[EpisodeRow], field: str) -> float:
    successful = [row for row in rows if row.success]
    return fmean(float(getattr(row, field)) for row in successful) if successful else 0.0


def run_toolgrid_factorial(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    grid_size: int,
    action_count: int,
    transition_budget: int = 5_000,
    train_map_count: int = 48,
    evaluation_map_count: int = 100,
    checkpoints: Sequence[int] = (0, 2_500, 5_000),
) -> dict[str, Any]:
    if condition not in TOOLGRID_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if grid_size not in GRID_SIZES or action_count not in ACTION_COUNTS:
        raise ValueError("unsupported factorial cell")
    if min(transition_budget, train_map_count, evaluation_map_count) <= 0:
        raise ValueError("experiment sizes must be positive")
    normalized_checkpoints = tuple(
        sorted(
            {
                int(value)
                for value in checkpoints
                if 0 <= int(value) <= transition_budget
            }
            | {0, int(transition_budget)}
        )
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cell_base = (
        int(seed) * 10_000_000
        + int(grid_size) * 100_000
        + int(action_count) * 1_000
    )
    training_maps = tuple(cell_base + index for index in range(train_map_count))
    unseen_maps = tuple(
        cell_base + 500_000 + index for index in range(evaluation_map_count)
    )
    seen_maps = tuple(
        training_maps[index % len(training_maps)] for index in range(evaluation_map_count)
    )
    manifest = []
    for split, seeds in (("training", training_maps), ("unseen", unseen_maps)):
        for map_seed in seeds:
            profile = ToolGridWorld(
                map_seed,
                grid_size=grid_size,
                action_count=action_count,
            ).map_profile()
            manifest.append(
                {
                    "split": split,
                    "seed": seed,
                    "grid_size": grid_size,
                    "action_count": action_count,
                    "tool_count": action_count - 4,
                    "effective_branching_factor": action_count,
                    "map_seed": profile.map_seed,
                    "oracle_shortest_steps": profile.oracle_shortest_steps,
                    "start": json.dumps(profile.start),
                    "stations": json.dumps(profile.stations),
                    "required_tools": json.dumps(profile.required_tools),
                }
            )
    with (output / "map_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    agent = make_toolgrid_agent(
        condition,
        int(seed),
        action_count=action_count,
        train_transitions=transition_budget,
    )
    training_rows: list[EpisodeRow] = []
    evaluation_rows: list[EpisodeRow] = []
    checkpoint_rows: list[CheckpointRow] = []
    environment_steps_total = 0
    training_episode_count = 0
    training_started = time.perf_counter()

    for checkpoint_target in normalized_checkpoints:
        while environment_steps_total < checkpoint_target:
            map_seed = training_maps[training_episode_count % len(training_maps)]
            row, environment_steps_total = _run_episode(
                agent,
                condition=condition,
                research_seed=seed,
                phase="training",
                grid_size=grid_size,
                action_count=action_count,
                checkpoint_transition_target=checkpoint_target,
                episode_index=training_episode_count,
                map_seed=map_seed,
                training=True,
                environment_steps_total=environment_steps_total,
                schedule_horizon=transition_budget,
            )
            training_rows.append(row)
            training_episode_count += 1

        seen_rows = _evaluate(
            agent,
            condition=condition,
            research_seed=seed,
            phase="evaluation_seen",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_target,
            map_seeds=seen_maps,
            environment_steps_total=environment_steps_total,
            schedule_horizon=transition_budget,
        )
        unseen_rows = _evaluate(
            agent,
            condition=condition,
            research_seed=seed,
            phase="evaluation_unseen",
            grid_size=grid_size,
            action_count=action_count,
            checkpoint_transition_target=checkpoint_target,
            map_seeds=unseen_maps,
            environment_steps_total=environment_steps_total,
            schedule_horizon=transition_budget,
        )
        evaluation_rows.extend(seen_rows)
        evaluation_rows.extend(unseen_rows)
        stats = agent.model_stats()
        checkpoint_rows.append(
            CheckpointRow(
                condition=condition,
                seed=seed,
                grid_size=grid_size,
                action_count=action_count,
                checkpoint_transition_target=checkpoint_target,
                actual_training_transitions=environment_steps_total,
                training_episode_count=training_episode_count,
                seen_success_rate=_mean_success(seen_rows),
                unseen_success_rate=_mean_success(unseen_rows),
                seen_mean_steps_on_success=_success_mean(seen_rows, "steps"),
                unseen_mean_steps_on_success=_success_mean(unseen_rows, "steps"),
                imagined_nodes_mean=(
                    fmean(row.imagined_nodes for row in unseen_rows) if unseen_rows else 0.0
                ),
                imagination_use_rate=(
                    fmean(float(row.imagination_runs > 0) for row in unseen_rows)
                    if unseen_rows
                    else 0.0
                ),
                training_wall_seconds=time.perf_counter() - training_started,
                model_units=int(stats.get("model_units", 0)),
                model_bytes=int(stats.get("model_bytes", 0)),
                gradient_updates=int(stats.get("gradient_updates", 0)),
            )
        )

    _write_rows(output / "training_episodes.csv", training_rows)
    _write_rows(output / "evaluation_episodes.csv", evaluation_rows)
    _write_rows(output / "checkpoints.csv", checkpoint_rows)
    final = checkpoint_rows[-1]
    payload = {
        "config": {
            "condition": condition,
            "seed": seed,
            "grid_size": grid_size,
            "action_count": action_count,
            "tool_count": action_count - 4,
            "transition_budget": transition_budget,
            "train_map_count": train_map_count,
            "evaluation_map_count": evaluation_map_count,
            "checkpoints": list(normalized_checkpoints),
            "external_reward": "final_success_only",
            "artificial_tick_limit": False,
        },
        "final": final.row(),
        "model_stats": agent.model_stats(),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=repr),
        encoding="utf-8",
    )
    return payload
