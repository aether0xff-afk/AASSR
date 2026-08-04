from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from .autonomous_agent_core import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from .goal_gridpush_experiment import GridPushStep, GoalExecutor, GoalProposal
from .goals import Goal, GoalGenerator, GoalKind, GoalSet
from .imagination_tree import ImaginationConfig, ImaginationResult, ImaginationTree, StateDeltaScorer
from .tabular_prophecy import TabularProphecy
from .types import Action, ActionVerb, StateSnapshot


DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}
DIRECTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("east", "west"),
    ("north", "south"),
)


@dataclass(frozen=True, slots=True)
class CorridorRoom:
    choices: tuple[str, str]
    target_direction: str


class LongHorizonDependencyWorld:
    """Sparse-reward dependency chain whose local checkpoint is beyond depth four.

    Each room starts with two opposite directions. After the first choice the
    agent must continue in that direction for ``room_length`` moves. The target
    coordinate is observable, but no fact, action unlock or reward distinguishes
    the two branches before the final move. A correct branch opens the next room;
    a wrong branch ends naturally at its dead end. Several rooms must be solved
    before the only external reward is emitted.

    There is no tick, energy or episode-length cutoff. Every episode ends only
    through the world's dependency structure.
    """

    def __init__(
        self,
        seed: int,
        *,
        stage_count: int = 10,
        room_length: int = 6,
    ) -> None:
        if stage_count <= 1:
            raise ValueError("stage_count must exceed one")
        if room_length <= 4:
            raise ValueError("room_length must exceed the short imagination depth")
        self.seed = int(seed)
        self.stage_count = int(stage_count)
        self.room_length = int(room_length)
        randomizer = random.Random(seed)
        rooms: list[CorridorRoom] = []
        for _ in range(stage_count):
            choices = randomizer.choice(DIRECTION_PAIRS)
            rooms.append(CorridorRoom(choices, randomizer.choice(choices)))
        self.rooms = tuple(rooms)
        self.stage = 0
        self.path_step = 0
        self.chosen_direction: str | None = None
        self.agent = (0, 0)
        self.completed_checkpoints: set[int] = set()
        self.success = False
        self.failed = False
        self.optimal_steps = self.stage_count * self.room_length

    @property
    def room(self) -> CorridorRoom:
        return self.rooms[min(self.stage, self.stage_count - 1)]

    @property
    def target(self) -> tuple[int, int]:
        dx, dy = DIRECTION_DELTAS[self.room.target_direction]
        return dx * self.room_length, dy * self.room_length

    def _facts(self) -> frozenset[str]:
        facts = {
            f"stage:{self.stage}",
            *(f"checkpoint:{index}" for index in sorted(self.completed_checkpoints)),
        }
        if self.success:
            facts.add("success")
        if self.failed:
            facts.add("failed")
        return frozenset(facts)

    def _action(self, direction: str) -> Action:
        return Action(ActionVerb.MOVE, parameters={"direction": direction})

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        if self.chosen_direction is None:
            return tuple(self._action(direction) for direction in self.room.choices)
        return (self._action(self.chosen_direction),)

    def _normalize(self, point: tuple[int, int]) -> tuple[float, float]:
        scale = float(self.room_length)
        return point[0] / scale, point[1] / scale

    def snapshot(self) -> StateSnapshot:
        stage_scale = float(max(1, self.stage_count - 1))
        vector = (
            *self._normalize(self.agent),
            *self._normalize(self.target),
            self.stage / stage_scale,
            self.path_step / float(self.room_length),
            len(self.completed_checkpoints) / float(self.stage_count),
        )
        return StateSnapshot(
            vector,
            self._facts(),
            self._available_actions(),
            1.0 if self.success else 0.0,
            metadata={
                "map_seed": self.seed,
                "stage": self.stage,
                "stage_count": self.stage_count,
                "room_length": self.room_length,
                "optimal_steps": self.optimal_steps,
                "termination": "dependency_dead_end_or_final_success",
            },
        )

    def step(self, action: Action) -> GridPushStep:
        before = self.snapshot()
        error = False
        reward = 0.0
        available = {item.signature for item in before.available_actions}
        if action.signature not in available:
            self.failed = True
            error = True
        else:
            direction = str(action.parameters.get("direction", ""))
            if self.chosen_direction is None:
                self.chosen_direction = direction
            if direction != self.chosen_direction:
                self.failed = True
                error = True
            else:
                dx, dy = DIRECTION_DELTAS[direction]
                self.agent = self.agent[0] + dx, self.agent[1] + dy
                self.path_step += 1
                if self.path_step >= self.room_length:
                    if self.chosen_direction != self.room.target_direction:
                        self.failed = True
                    else:
                        completed = self.stage
                        self.completed_checkpoints.add(completed)
                        self.stage += 1
                        if self.stage >= self.stage_count:
                            self.success = True
                            reward = 1.0
                        else:
                            self.agent = (0, 0)
                            self.path_step = 0
                            self.chosen_direction = None

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


class WaypointGoalMaker:
    """Find a distant structural change, then expose only a nearby waypoint GOAL."""

    def __init__(
        self,
        policy: object,
        prophecy: object,
        *,
        search_depth: int = 6,
        waypoint_depth: int = 3,
    ) -> None:
        if waypoint_depth <= 0 or waypoint_depth > search_depth:
            raise ValueError("waypoint_depth must be in [1, search_depth]")
        self.waypoint_depth = waypoint_depth
        self.planner = ImaginationTree(
            policy,
            prophecy,
            config=ImaginationConfig(
                branching_factor=2,
                maximum_depth=search_depth,
                beam_width=32,
                outcome_samples=2,
                minimum_path_confidence=0.0,
                uncertainty_penalty=0.10,
                aggregation="risk_adjusted",
                update_policy=False,
                expand_all_root_actions=True,
            ),
            scorer=StateDeltaScorer(
                goal_progress_weight=100.0,
                new_fact_weight=8.0,
                unlocked_action_weight=2.0,
                step_cost=0.01,
            ),
        )

    @staticmethod
    def _structural_gain(
        before: StateSnapshot,
        after: StateSnapshot,
        depth: int,
    ) -> float:
        before_actions = {item.signature for item in before.available_actions}
        after_actions = {item.signature for item in after.available_actions}
        return (
            100.0 * (after.goal_progress - before.goal_progress)
            + 12.0 * len(after.facts - before.facts)
            + 3.0 * len(after_actions - before_actions)
            - 0.02 * depth
        )

    def _waypoint_for(
        self,
        selected_id: int,
        plan: ImaginationResult,
    ) -> StateSnapshot:
        by_id = {node.node_id: node for node in plan.nodes}
        path = []
        node = by_id[selected_id]
        while node.parent_id is not None:
            path.append(node)
            node = by_id[node.parent_id]
        path.reverse()
        index = min(self.waypoint_depth, len(path)) - 1
        return path[index].state

    def propose(self, state: StateSnapshot) -> GoalProposal | None:
        plan = self.planner.plan(state)
        candidates = [
            node
            for node in plan.nodes
            if node.depth > 0
            and "failed" not in node.state.facts
            and self._structural_gain(state, node.state, node.depth) > 0.0
        ]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda node: (
                self._structural_gain(state, node.state, node.depth),
                node.cumulative_confidence,
                node.cumulative_value,
                -node.depth,
            ),
        )
        waypoint = self._waypoint_for(selected.node_id, plan)
        goals = GoalSet()
        goals.add(
            Goal(
                "final:success",
                GoalKind.GOAL_PROGRESS,
                1.0,
                priority=5.0,
                source="external",
                final=True,
            )
        )
        goals.add(
            Goal(
                "maker:waypoint",
                GoalKind.VECTOR_TARGET,
                waypoint.vector,
                priority=4.0,
                threshold=0.999,
                source="imagined_path",
            )
        )
        for goal in GoalGenerator.from_desired_state(
            state,
            waypoint,
            parent_goal_id="final:success",
            prefix="maker",
        ):
            goals.add(goal)
        return GoalProposal(
            goals,
            waypoint,
            plan,
            self._structural_gain(state, selected.state, selected.depth),
        )


def _agent_config(*, use_imagination: bool, depth: int) -> AutonomousAgentConfig:
    return AutonomousAgentConfig(
        gamma=0.99,
        epsilon_start=0.90,
        epsilon_end=0.05,
        epsilon_decay_episodes=500,
        exploration_bonus=0.15,
        use_imagination=use_imagination,
        imagination_depth=depth,
        imagination_branching_factor=2,
        imagination_beam_width=32,
        imagination_outcome_samples=2,
        imagination_minimum_coverage=0.0,
        imagination_intervention_margin=0.0,
        imagination_uncertainty_margin=0.10,
        imagination_aggregation="risk-adjusted",
        validated_gain_weight=0.0,
        repeat_penalty=0.0,
        error_penalty=0.0,
        effect_novelty_weight=0.0,
        extrinsic_reward_weight=1.0,
        effect_minimum_samples=2,
    )


def _planning_scorer() -> StateDeltaScorer:
    return StateDeltaScorer(
        goal_progress_weight=100.0,
        new_fact_weight=8.0,
        unlocked_action_weight=2.0,
        step_cost=0.01,
    )


def make_direct_agent(condition: str, seed: int) -> AutonomousLearningAgent:
    if condition == "policy_only":
        use_imagination = False
        depth = 1
    elif condition == "short_imagination":
        use_imagination = True
        depth = 4
    elif condition == "deep_imagination":
        use_imagination = True
        depth = 6
    else:
        raise ValueError(f"unknown condition: {condition}")
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=_agent_config(use_imagination=use_imagination, depth=depth),
        seed=seed,
    )
    if use_imagination:
        agent.planner = ImaginationTree(
            agent.policy,
            agent.prophecy,
            config=ImaginationConfig(
                branching_factor=2,
                maximum_depth=depth,
                beam_width=32,
                outcome_samples=2,
                minimum_path_confidence=0.0,
                uncertainty_penalty=0.10,
                aggregation="risk_adjusted",
                update_policy=False,
                expand_all_root_actions=True,
            ),
            scorer=_planning_scorer(),
        )
    return agent


class HierarchicalGoalAgent:
    """Persistent GOAL Maker/Executor agent for the long dependency chain."""

    def __init__(self, seed: int, *, room_length: int = 6) -> None:
        self.base = AutonomousLearningAgent(
            TabularProphecy(),
            config=_agent_config(use_imagination=False, depth=1),
            seed=seed,
        )
        self.maker = WaypointGoalMaker(
            self.base.policy,
            self.base.prophecy,
            search_depth=room_length,
            waypoint_depth=3,
        )
        self.executor = GoalExecutor(
            self.base.policy,
            self.base.prophecy,
            depth=3,
        )
        self.active_goal: GoalProposal | None = None
        self.active_goal_age = 0
        self.maximum_goal_age = 4
        self.goal_proposals = 0
        self.goal_switches = 0
        self.goal_reuses = 0
        self.goal_completions = 0
        self.goal_abandons = 0

    @staticmethod
    def _target_reached(proposal: GoalProposal, state: StateSnapshot) -> bool:
        if len(proposal.desired_state.vector) != len(state.vector):
            return False
        return all(
            abs(left - right) <= 1e-9
            for left, right in zip(
                proposal.desired_state.vector,
                state.vector,
                strict=True,
            )
        )

    def _clear_goal(self, *, completed: bool) -> None:
        if self.active_goal is None:
            return
        if completed:
            self.goal_completions += 1
        else:
            self.goal_abandons += 1
        self.active_goal = None
        self.active_goal_age = 0

    def begin_episode(self) -> None:
        self.base.discard_episode()
        self.active_goal = None
        self.active_goal_age = 0

    def select_action(
        self,
        state: StateSnapshot,
        *,
        episode: int,
        explore: bool,
    ) -> ActionDecision:
        policy_decision = self.base.select_action(
            state,
            episode=episode,
            explore=explore,
        )
        if policy_decision.imagination_gate_reason in {
            "epsilon_random",
            "random_policy",
        }:
            return policy_decision
        if self.base.model_coverage(state) <= 0.0:
            return policy_decision

        if self.active_goal is not None and self._target_reached(
            self.active_goal,
            state,
        ):
            self._clear_goal(completed=True)
        if self.active_goal is not None and self.active_goal_age >= self.maximum_goal_age:
            self._clear_goal(completed=False)

        maker_nodes = 0
        if self.active_goal is None:
            self.active_goal = self.maker.propose(state)
            self.active_goal_age = 0
            if self.active_goal is not None:
                self.goal_proposals += 1
                maker_nodes = len(self.active_goal.maker_plan.nodes)
        else:
            self.goal_reuses += 1

        proposal = self.active_goal
        if proposal is None:
            return policy_decision

        plan = self.executor.plan(state, proposal)
        preferred = plan.root_evaluations[0]
        policy_evaluation = next(
            (
                item
                for item in plan.root_evaluations
                if item.action.signature == policy_decision.action.signature
            ),
            None,
        )
        if policy_evaluation is None:
            return policy_decision

        advantage = preferred.aggregate_value - policy_evaluation.aggregate_value
        changed = preferred.action.signature != policy_decision.action.signature
        if changed:
            self.goal_switches += 1
        self.active_goal_age += 1
        executed = preferred.action if changed else policy_decision.action
        return ActionDecision(
            executed,
            True,
            imagined_nodes=maker_nodes + len(plan.nodes),
            imagination_depth=max(
                proposal.maker_plan.maximum_depth_reached,
                plan.maximum_depth_reached,
            ),
            root_imagined_value=preferred.aggregate_value,
            policy_action_signature=policy_decision.action.signature,
            imagination_opportunity=True,
            imagination_eligible=True,
            imagination_gate_reason=(
                "goal_intervention" if changed else "goal_policy_agreement"
            ),
            imagination_changed_action=changed,
            model_coverage=self.base.model_coverage(state),
            imagination_preferred_action_signature=preferred.action.signature,
            imagination_policy_value=policy_evaluation.aggregate_value,
            imagination_preferred_value=preferred.aggregate_value,
            imagination_advantage=advantage,
            imagination_required_advantage=0.0,
            imagination_switch_candidate=changed,
            imagination_intervention_allowed=changed,
        )

    def observe(
        self,
        before: StateSnapshot,
        action: Action,
        outcome: GridPushStep,
    ) -> object:
        metrics = self.base.observe(before, action, outcome)
        if self.active_goal is not None:
            if self._target_reached(self.active_goal, outcome.snapshot):
                self._clear_goal(completed=True)
            elif not outcome.snapshot.available_actions:
                self._clear_goal(completed=False)
        return metrics

    def finish_episode(self, *, final_return: float) -> None:
        self.base.finish_episode(final_return=final_return)
        self.active_goal = None
        self.active_goal_age = 0

    def discard_episode(self) -> None:
        self.begin_episode()


Agent = AutonomousLearningAgent | HierarchicalGoalAgent


@dataclass(frozen=True, slots=True)
class LongHorizonEpisodeResult:
    condition: str
    seed: int
    phase: str
    episode: int
    map_seed: int
    success: int
    steps: int
    optimal_steps: int
    efficiency: float
    imagined_nodes: int
    imagination_runs: int
    interventions: int
    goal_proposals: int
    goal_completions: int
    goal_abandons: int

    def as_row(self) -> dict[str, int | float | str]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def make_agent(condition: str, seed: int, *, room_length: int) -> Agent:
    if condition == "goal_maker_executor":
        return HierarchicalGoalAgent(seed, room_length=room_length)
    return make_direct_agent(condition, seed)


def _counter(agent: Agent, name: str) -> int:
    return int(getattr(agent, name, 0))


def run_episode(
    agent: Agent,
    *,
    condition: str,
    seed: int,
    phase: str,
    episode: int,
    map_seed: int,
    stage_count: int,
    room_length: int,
    learn: bool,
) -> LongHorizonEpisodeResult:
    begin = getattr(agent, "begin_episode", None)
    if callable(begin):
        begin()
    elif not learn:
        agent.discard_episode()

    world = LongHorizonDependencyWorld(
        map_seed,
        stage_count=stage_count,
        room_length=room_length,
    )
    imagined_nodes = 0
    imagination_runs = 0
    interventions = 0
    goal_before = _counter(agent, "goal_proposals")
    complete_before = _counter(agent, "goal_completions")
    abandon_before = _counter(agent, "goal_abandons")
    steps = 0

    while world.snapshot().available_actions:
        before = world.snapshot()
        decision = agent.select_action(
            before,
            episode=episode,
            explore=learn,
        )
        outcome = world.step(decision.action)
        steps += 1
        imagined_nodes += decision.imagined_nodes
        imagination_runs += int(decision.used_imagination)
        interventions += int(decision.imagination_changed_action)
        if learn:
            agent.observe(before, decision.action, outcome)
        if world.success or world.failed:
            break

    if learn:
        agent.finish_episode(final_return=1.0 if world.success else 0.0)
    else:
        agent.discard_episode()

    return LongHorizonEpisodeResult(
        condition,
        seed,
        phase,
        episode,
        map_seed,
        int(world.success),
        steps,
        world.optimal_steps,
        world.optimal_steps / steps if world.success and steps else 0.0,
        imagined_nodes,
        imagination_runs,
        interventions,
        _counter(agent, "goal_proposals") - goal_before,
        _counter(agent, "goal_completions") - complete_before,
        _counter(agent, "goal_abandons") - abandon_before,
    )


def _mean(rows: Iterable[LongHorizonEpisodeResult], field: str) -> float:
    values = [float(getattr(row, field)) for row in rows]
    return fmean(values) if values else 0.0


def summarize(
    rows: Sequence[LongHorizonEpisodeResult],
) -> list[dict[str, int | float | str]]:
    groups: dict[tuple[str, str], list[LongHorizonEpisodeResult]] = {}
    for row in rows:
        groups.setdefault((row.condition, row.phase), []).append(row)
    summary: list[dict[str, int | float | str]] = []
    for (condition, phase), items in sorted(groups.items()):
        successful = [item for item in items if item.success]
        seed_rates = []
        for seed in sorted({item.seed for item in items}):
            seed_rows = [item for item in items if item.seed == seed]
            seed_rates.append(_mean(seed_rows, "success"))
        summary.append(
            {
                "condition": condition,
                "phase": phase,
                "episodes": len(items),
                "seed_count": len(seed_rates),
                "success_rate": _mean(items, "success"),
                "seed_mean_success_rate": fmean(seed_rates) if seed_rates else 0.0,
                "mean_steps_on_success": _mean(successful, "steps"),
                "mean_efficiency": _mean(items, "efficiency"),
                "mean_imagined_nodes": _mean(items, "imagined_nodes"),
                "mean_imagination_runs": _mean(items, "imagination_runs"),
                "mean_interventions": _mean(items, "interventions"),
                "mean_goal_proposals": _mean(items, "goal_proposals"),
                "mean_goal_completions": _mean(items, "goal_completions"),
                "mean_goal_abandons": _mean(items, "goal_abandons"),
            }
        )
    return summary


def run_long_horizon_goal_experiment(
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (7, 13, 21, 42, 100),
    conditions: Sequence[str] = (
        "policy_only",
        "short_imagination",
        "deep_imagination",
        "goal_maker_executor",
    ),
    train_episodes: int = 600,
    train_map_count: int = 48,
    evaluation_episodes: int = 80,
    training_tail: int = 100,
    stage_count: int = 10,
    room_length: int = 6,
) -> dict[str, object]:
    if train_episodes <= 0 or train_map_count <= 0 or evaluation_episodes <= 0:
        raise ValueError("experiment sizes must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[LongHorizonEpisodeResult] = []

    for seed in seeds:
        training_maps = tuple(
            seed * 1_000_000 + index for index in range(train_map_count)
        )
        seen_maps = tuple(
            training_maps[index % len(training_maps)]
            for index in range(evaluation_episodes)
        )
        unseen_maps = tuple(
            seed * 1_000_000 + 500_000 + index
            for index in range(evaluation_episodes)
        )
        for condition in conditions:
            agent = make_agent(condition, seed, room_length=room_length)
            training_rows = []
            for episode in range(train_episodes):
                training_rows.append(
                    run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="training",
                        episode=episode,
                        map_seed=training_maps[episode % len(training_maps)],
                        stage_count=stage_count,
                        room_length=room_length,
                        learn=True,
                    )
                )
            rows.extend(training_rows[-min(training_tail, len(training_rows)) :])
            for index, map_seed in enumerate(seen_maps):
                rows.append(
                    run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="evaluation_seen",
                        episode=train_episodes + index,
                        map_seed=map_seed,
                        stage_count=stage_count,
                        room_length=room_length,
                        learn=False,
                    )
                )
            for index, map_seed in enumerate(unseen_maps):
                rows.append(
                    run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="evaluation_unseen",
                        episode=train_episodes + evaluation_episodes + index,
                        map_seed=map_seed,
                        stage_count=stage_count,
                        room_length=room_length,
                        learn=False,
                    )
                )

    episodes_path = output / "episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(LongHorizonEpisodeResult.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(row.as_row() for row in rows)

    payload: dict[str, object] = {
        "config": {
            "seeds": list(seeds),
            "conditions": list(conditions),
            "train_episodes": train_episodes,
            "train_map_count": train_map_count,
            "evaluation_episodes": evaluation_episodes,
            "training_tail": training_tail,
            "stage_count": stage_count,
            "room_length": room_length,
        },
        "summary": summarize(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
