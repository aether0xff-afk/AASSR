from __future__ import annotations

import json
import pickle
import random
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .autonomous_agent_core import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from .baseline_efficiency_benchmark import (
    CHOICE_ACTIONS,
    BenchmarkAgent,
    BenchmarkGridPushWorld,
    GRIDPUSH_OBSERVATION_SIZE,
    _world_key,
    encode_gridpush_state,
    make_benchmark_agent,
)
from .imagination_tree import StateDeltaScorer
from .policy import PolicyMemory, ScoredAction
from .tabular_prophecy import TabularProphecy
from .types import Action, Prediction, StateSnapshot


@dataclass(frozen=True, slots=True)
class BottleneckCondition:
    name: str
    identity_role: str
    diagnostic_question: str


CONDITIONS: tuple[BottleneckCondition, ...] = (
    BottleneckCondition(
        "dqn",
        "external_baseline",
        "Can a shared neural value function generalize across procedural maps?",
    ),
    BottleneckCondition(
        "aassr_current",
        "current_system",
        "What does the current Policy + Prophecy + Imagination stack achieve?",
    ),
    BottleneckCondition(
        "hybrid_current_model",
        "identity_preserving_policy_swap",
        "Does replacing only the tabular Policy with a neural Policy fix the bottleneck?",
    ),
    BottleneckCondition(
        "hybrid_oracle_model",
        "diagnostic_oracle",
        "With a perfect Prophecy, is the current scorer and search already sufficient?",
    ),
    BottleneckCondition(
        "hybrid_current_model_oracle_score",
        "diagnostic_oracle",
        "With perfect local progress scoring, does the learned Prophecy remain the bottleneck?",
    ),
    BottleneckCondition(
        "hybrid_oracle_model_oracle_score",
        "diagnostic_oracle",
        "What is the ceiling of the existing tree search with a perfect model and useful score?",
    ),
    BottleneckCondition(
        "hybrid_oracle_budgeted",
        "identity_preserving_compute_reduction",
        "Can sparse, shallow planning retain the oracle-model ceiling at lower cost?",
    ),
    BottleneckCondition(
        "oracle_bfs",
        "environment_upper_bound",
        "What are the solvability and shortest-path ceilings of the benchmark?",
    ),
)


def _decode_point(values: Sequence[float], offset: int) -> tuple[int, int]:
    scale = float(BenchmarkGridPushWorld.grid_size - 1)
    return (
        int(round(float(values[offset]) * scale)),
        int(round(float(values[offset + 1]) * scale)),
    )


def world_from_snapshot(state: StateSnapshot) -> BenchmarkGridPushWorld:
    """Reconstruct the benchmark simulator from a snapshot for oracle diagnosis."""

    map_seed = int(state.metadata.get("map_seed", 0))
    world = BenchmarkGridPushWorld(map_seed)
    values = state.vector
    if len(values) < 16:
        raise ValueError("benchmark snapshot vector must contain 16 values")
    world.agent = _decode_point(values, 0)
    world.crate = _decode_point(values, 2)
    world.pit = _decode_point(values, 4)
    world.key = _decode_point(values, 6)
    world.door = _decode_point(values, 8)
    world.exit = _decode_point(values, 10)
    world.phase = int(round(float(values[12]) * 6.0))
    world.bridge_built = bool(round(float(values[13])))
    world.key_held = bool(round(float(values[14])))
    world.door_open = bool(round(float(values[15])))
    world.success = "success" in state.facts or state.goal_progress >= 1.0
    world.failed = "failed" in state.facts
    used: set[tuple[int, int]] = set()
    for fact in state.facts:
        if not fact.startswith("used:"):
            continue
        _, raw_x, raw_y = fact.split(":")
        used.add((int(raw_x), int(raw_y)))
    world.used_cells = used or {world.agent}
    world.optimal_steps = int(state.metadata.get("optimal_steps", world.optimal_steps))
    return world


class BenchmarkOracleProphecy:
    """Perfect benchmark transition model used only to locate bottlenecks."""

    name = "benchmark-oracle"

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        del state, action, actual_next_state

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        if samples <= 0:
            raise ValueError("samples must be positive")
        world = world_from_snapshot(state)
        outcome = world.step(action)
        return (
            Prediction(
                next_state=outcome.snapshot,
                probability=1.0,
                source="benchmark-oracle:exact",
            ),
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state, action
        return 1.0

    def coverage(
        self,
        state: StateSnapshot,
        actions: Sequence[Action],
    ) -> float:
        del state
        return 1.0 if tuple(actions) else 1.0


_REMAINING_DISTANCE_CACHE: dict[tuple[Any, ...], int | None] = {}


def _snapshot_cache_key(state: StateSnapshot) -> tuple[Any, ...]:
    return (
        int(state.metadata.get("map_seed", 0)),
        tuple(round(float(value), 8) for value in state.vector),
        tuple(sorted(state.facts)),
        tuple(action.signature for action in state.available_actions),
    )


def remaining_oracle_steps(state: StateSnapshot) -> int | None:
    key = _snapshot_cache_key(state)
    if key in _REMAINING_DISTANCE_CACHE:
        return _REMAINING_DISTANCE_CACHE[key]
    root = world_from_snapshot(state)
    if root.success:
        _REMAINING_DISTANCE_CACHE[key] = 0
        return 0
    if root.failed:
        _REMAINING_DISTANCE_CACHE[key] = None
        return None
    queue: deque[tuple[BenchmarkGridPushWorld, int]] = deque([(root, 0)])
    visited = {_world_key(root)}
    while queue:
        world, depth = queue.popleft()
        for action in CHOICE_ACTIONS:
            candidate = deepcopy(world)
            candidate.step(action)
            if candidate.success:
                _REMAINING_DISTANCE_CACHE[key] = depth + 1
                return depth + 1
            if candidate.failed:
                continue
            candidate_key = _world_key(candidate)
            if candidate_key in visited:
                continue
            visited.add(candidate_key)
            queue.append((candidate, depth + 1))
    _REMAINING_DISTANCE_CACHE[key] = None
    return None


@dataclass(frozen=True, slots=True)
class BenchmarkOracleProgressScorer:
    """Task-aware diagnostic scorer; never a candidate for the final agent."""

    progress_weight: float = 5.0
    success_value: float = 100.0
    dead_end_value: float = -100.0
    step_cost: float = 0.01

    def score(
        self,
        before: StateSnapshot,
        action: Action,
        after: StateSnapshot,
    ) -> float:
        del action
        if after.goal_progress >= 1.0 or "success" in after.facts:
            return self.success_value
        if not after.available_actions or "failed" in after.facts:
            return self.dead_end_value
        before_distance = remaining_oracle_steps(before)
        after_distance = remaining_oracle_steps(after)
        if before_distance is None or after_distance is None:
            return self.dead_end_value
        return (
            self.progress_weight * (before_distance - after_distance)
            - self.step_cost
        )


class DQNPolicyAdapter:
    """Expose a DQN value network through the AASSR Policy interface."""

    def __init__(self, dqn_agent: Any) -> None:
        self.dqn_agent = dqn_agent

    def _values(self, state: StateSnapshot) -> tuple[float, ...]:
        torch = self.dqn_agent.torch
        with torch.no_grad():
            observation = self.dqn_agent._tensor(
                encode_gridpush_state(state)
            ).unsqueeze(0)
            values = self.dqn_agent.online(observation)[0]
        return tuple(float(value.item()) for value in values)

    def value(self, state: StateSnapshot, action: Action) -> float:
        index = int(action.verb_name.rsplit("_", 1)[1])
        return self._values(state)[index]

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
        values = self._values(state)
        ranked = sorted(
            (
                ScoredAction(
                    action,
                    values[index] + deltas.get(action.signature, 0.0),
                )
                for index, action in enumerate(state.available_actions)
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


class HybridDQNImaginationAgent:
    """DQN Policy plus AASSR Prophecy and Imagination.

    This keeps the AASSR separation between Policy, Prophecy, and Imagination,
    while replacing only the tabular Policy representation with the same neural
    value function used by the DQN baseline.
    """

    def __init__(
        self,
        seed: int,
        *,
        train_episodes: int,
        prophecy: object,
        scorer: object,
        depth: int,
        branching_factor: int,
        beam_width: int,
        outcome_samples: int,
        imagination_interval: int = 1,
        use_effect_composition: bool = True,
        name: str,
    ) -> None:
        self.name = name
        self.dqn = make_benchmark_agent(
            "dqn",
            seed,
            train_episodes=train_episodes,
        )
        policy = DQNPolicyAdapter(self.dqn)
        self.agent = AutonomousLearningAgent(
            prophecy,
            config=AutonomousAgentConfig(
                learn_policy=False,
                learn_prophecy=not isinstance(prophecy, BenchmarkOracleProphecy),
                use_imagination=True,
                use_effect_composition=use_effect_composition,
                imagination_depth=depth,
                imagination_branching_factor=branching_factor,
                imagination_beam_width=beam_width,
                imagination_outcome_samples=outcome_samples,
                imagination_interval=imagination_interval,
                imagination_minimum_coverage=0.0,
                imagination_intervention_margin=0.02,
                imagination_uncertainty_margin=0.25,
                imagination_aggregation="risk-adjusted",
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_episodes=max(1, int(train_episodes * 0.80)),
                effect_minimum_samples=2,
            ),
            seed=seed,
            policy=policy,
        )
        self.agent.planner.scorer = scorer

    def begin_episode(self, *, training: bool) -> None:
        self.dqn.begin_episode(training=training)
        if not training:
            self.agent.discard_episode()

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> Any:
        decision: ActionDecision = self.agent.select_action(
            state,
            episode=episode,
            explore=training,
        )
        from .baseline_efficiency_benchmark import AgentDecision

        return AgentDecision(
            action=decision.action,
            imagined_nodes=decision.imagined_nodes,
            used_imagination=decision.used_imagination,
        )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: Any,
    ) -> None:
        self.dqn.observe(before, action, outcome)
        self.agent.observe(before, action, outcome)

    def end_episode(self, *, success: bool, training: bool) -> None:
        self.dqn.end_episode(success=success, training=training)
        if training:
            self.agent.finish_episode(final_return=1.0 if success else 0.0)
        else:
            self.agent.discard_episode()

    def model_stats(self) -> dict[str, int | float]:
        stats = dict(self.dqn.model_stats())
        stats["model_units"] = int(stats.get("model_units", 0)) + int(
            getattr(self.agent.prophecy, "effect_bucket_count", 0)
        )
        payload = {
            "dqn": stats,
            "imagination": self.agent.imagination_diagnostics(),
            "effect_buckets": int(
                getattr(self.agent.prophecy, "effect_bucket_count", 0)
            ),
        }
        stats["model_bytes"] = int(stats.get("model_bytes", 0)) + len(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )
        return stats


class OracleBFSBenchmarkAgent:
    name = "oracle_bfs"

    def __init__(self, seed: int, *, train_episodes: int) -> None:
        del seed, train_episodes

    def begin_episode(self, *, training: bool) -> None:
        del training

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        training: bool,
    ) -> Any:
        del episode, training
        from .baseline_efficiency_benchmark import AgentDecision

        before_distance = remaining_oracle_steps(state)
        if before_distance is None:
            return AgentDecision(CHOICE_ACTIONS[0])
        candidates: list[tuple[int, str, Action]] = []
        for action in state.available_actions:
            prediction = BenchmarkOracleProphecy().predict(
                state,
                action,
                samples=1,
            )[0]
            distance = remaining_oracle_steps(prediction.next_state)
            if distance is None:
                continue
            candidates.append((distance, action.signature, action))
        if not candidates:
            return AgentDecision(CHOICE_ACTIONS[0])
        _, _, action = min(candidates)
        return AgentDecision(action)

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: Any,
    ) -> None:
        del before, action, outcome

    def end_episode(self, *, success: bool, training: bool) -> None:
        del success, training

    def model_stats(self) -> dict[str, int | float]:
        return {"model_units": 0, "model_bytes": 0, "gradient_updates": 0}


def _current_scorer() -> StateDeltaScorer:
    return StateDeltaScorer(
        goal_progress_weight=50.0,
        new_fact_weight=4.0,
        unlocked_action_weight=2.0,
        step_cost=0.01,
    )


def make_bottleneck_agent(
    condition: str,
    seed: int,
    *,
    train_episodes: int,
) -> BenchmarkAgent:
    if condition == "dqn":
        return make_benchmark_agent("dqn", seed, train_episodes=train_episodes)
    if condition == "aassr_current":
        return make_benchmark_agent(
            "aassr_full",
            seed,
            train_episodes=train_episodes,
        )
    if condition == "oracle_bfs":
        return OracleBFSBenchmarkAgent(seed, train_episodes=train_episodes)

    prophecy: object
    scorer: object
    use_effect_composition = True
    depth = 5
    branching = 2
    beam = 16
    samples = 2
    interval = 1

    if "oracle_model" in condition:
        prophecy = BenchmarkOracleProphecy()
        use_effect_composition = False
        samples = 1
    else:
        prophecy = TabularProphecy()

    if "oracle_score" in condition or condition == "hybrid_oracle_budgeted":
        scorer = BenchmarkOracleProgressScorer()
    else:
        scorer = _current_scorer()

    if condition == "hybrid_oracle_model_oracle_score":
        depth = 12
        branching = 4
        beam = 128
    elif condition == "hybrid_oracle_budgeted":
        depth = 4
        branching = 2
        beam = 16
        interval = 4

    if condition not in {item.name for item in CONDITIONS}:
        raise ValueError(f"unknown bottleneck condition: {condition}")

    return HybridDQNImaginationAgent(
        seed,
        train_episodes=train_episodes,
        prophecy=prophecy,
        scorer=scorer,
        depth=depth,
        branching_factor=branching,
        beam_width=beam,
        outcome_samples=samples,
        imagination_interval=interval,
        use_effect_composition=use_effect_composition,
        name=condition,
    )


def run_bottleneck_condition(
    output_dir: str | Path,
    *,
    condition: str,
    seed: int,
    train_episodes: int = 1000,
    train_map_count: int = 64,
    evaluation_episodes: int = 100,
    checkpoints: Sequence[int] = (0, 100, 250, 500, 1000),
) -> dict[str, Any]:
    """Run one condition through the shared baseline harness."""

    from . import baseline_efficiency_benchmark as benchmark

    original_factory = benchmark.make_benchmark_agent
    benchmark.make_benchmark_agent = make_bottleneck_agent
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

    spec = next(item for item in CONDITIONS if item.name == condition)
    payload["condition"] = asdict(spec)
    output = Path(output_dir)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
