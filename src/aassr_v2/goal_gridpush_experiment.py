from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from .autonomous_agent_core import (
    ActionDecision,
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from .goals import Goal, GoalGenerator, GoalKind, GoalSet, GoalStateScorer
from .imagination_tree import ImaginationConfig, ImaginationResult, ImaginationTree, StateDeltaScorer
from .tabular_prophecy import TabularProphecy
from .types import Action, ActionVerb, StateSnapshot


DIRECTIONS: Mapping[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}


@dataclass(frozen=True, slots=True)
class GridPushStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str]
    removed_facts: frozenset[str]
    unlocked_actions: tuple[Action, ...]
    error: bool
    reward: float


class GoalGridPushWorld:
    """Small procedural GridPush dependency world with sparse final reward.

    The agent must navigate to a crate, push it into a pit, collect a key,
    open a door and reach an exit. Coordinates change with the map seed while
    movement, pushing, pickup and use rules stay fixed. Wrong movement is not
    rewarded; it only consumes the world's finite energy resource.
    """

    grid_size = 5

    def __init__(self, seed: int, *, slack: int = 8) -> None:
        if slack < 0:
            raise ValueError("slack must be non-negative")
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
        self.optimal_steps = (
            self._distance(self.agent, self.crate)
            + self._distance(self.crate, self.pit)
            + self._distance(self.pit, self.key)
            + 1
            + self._distance(self.key, self.door)
            + 1
            + self._distance(self.door, self.exit)
        )
        self.energy = self.optimal_steps + slack
        self.initial_energy = self.energy

    @staticmethod
    def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _normalize(self, point: tuple[int, int]) -> tuple[float, float]:
        scale = float(self.grid_size - 1)
        return point[0] / scale, point[1] / scale

    def _facts(self) -> frozenset[str]:
        facts = {f"phase:{self.phase}"}
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

    def _movement_actions(self, verb: ActionVerb | str) -> tuple[Action, ...]:
        return tuple(
            Action(verb, parameters={"direction": direction})
            for direction in DIRECTIONS
        )

    def _available_actions(self) -> tuple[Action, ...]:
        if self.success or self.failed:
            return ()
        if self.phase in {0, 2, 4, 6}:
            return self._movement_actions(ActionVerb.MOVE)
        if self.phase == 1:
            return self._movement_actions("push")
        if self.phase == 3:
            return (Action(ActionVerb.PICKUP),)
        if self.phase == 5:
            return (Action(ActionVerb.USE),)
        return ()

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
                "energy": self.energy,
                "optimal_steps": self.optimal_steps,
            },
        )

    def _move_point(
        self,
        point: tuple[int, int],
        direction: str,
    ) -> tuple[tuple[int, int], bool]:
        delta = DIRECTIONS.get(direction)
        if delta is None:
            return point, True
        candidate = point[0] + delta[0], point[1] + delta[1]
        if not (
            0 <= candidate[0] < self.grid_size
            and 0 <= candidate[1] < self.grid_size
        ):
            return point, True
        return candidate, False

    def _advance_phase_after_motion(self) -> None:
        if self.phase == 0 and self.agent == self.crate:
            self.phase = 1
        elif self.phase == 1 and self.crate == self.pit:
            self.bridge_built = True
            self.phase = 2
        elif self.phase == 2 and self.agent == self.key:
            self.phase = 3
        elif self.phase == 4 and self.agent == self.door:
            self.phase = 5
        elif self.phase == 6 and self.agent == self.exit:
            self.success = True

    def step(self, action: Action) -> GridPushStep:
        before = self.snapshot()
        error = False
        reward = 0.0

        if not before.available_actions:
            return GridPushStep(
                before,
                frozenset(),
                frozenset(),
                (),
                True,
                0.0,
            )

        self.energy -= 1
        direction = str(action.parameters.get("direction", ""))

        if self.phase in {0, 2, 4, 6} and action.verb_name == ActionVerb.MOVE.value:
            self.agent, error = self._move_point(self.agent, direction)
            self._advance_phase_after_motion()
        elif self.phase == 1 and action.verb_name == "push":
            next_crate, error = self._move_point(self.crate, direction)
            if not error:
                self.crate = next_crate
                self.agent = next_crate
                self._advance_phase_after_motion()
        elif self.phase == 3 and action.verb_name == ActionVerb.PICKUP.value:
            if self.agent == self.key:
                self.key_held = True
                self.phase = 4
            else:
                error = True
        elif self.phase == 5 and action.verb_name == ActionVerb.USE.value:
            if self.agent == self.door and self.key_held:
                self.door_open = True
                self.phase = 6
            else:
                error = True
        else:
            error = True

        if self.success:
            reward = 1.0
        elif self.energy <= 0:
            self.failed = True

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


@dataclass(frozen=True, slots=True)
class GoalProposal:
    goals: GoalSet
    desired_state: StateSnapshot
    maker_plan: ImaginationResult
    structural_gain: float


class ImaginedGoalMaker:
    """Generate a state GOAL by running Imagination without choosing reality actions."""

    def __init__(self, policy: object, prophecy: object, *, depth: int = 8) -> None:
        self.planner = ImaginationTree(
            policy,
            prophecy,
            config=ImaginationConfig(
                branching_factor=2,
                maximum_depth=depth,
                beam_width=64,
                outcome_samples=2,
                minimum_path_confidence=0.0,
                uncertainty_penalty=0.25,
                aggregation="risk_adjusted",
                update_policy=False,
                expand_all_root_actions=True,
            ),
            scorer=StateDeltaScorer(
                goal_progress_weight=50.0,
                new_fact_weight=4.0,
                unlocked_action_weight=2.0,
                step_cost=0.01,
            ),
        )

    @staticmethod
    def _gain(before: StateSnapshot, after: StateSnapshot, depth: int) -> float:
        before_actions = {item.signature for item in before.available_actions}
        after_actions = {item.signature for item in after.available_actions}
        return (
            100.0 * (after.goal_progress - before.goal_progress)
            + 6.0 * len(after.facts - before.facts)
            + 2.0 * len(after_actions - before_actions)
            - 0.02 * depth
        )

    def propose(self, state: StateSnapshot) -> GoalProposal | None:
        plan = self.planner.plan(state)
        candidates = [
            node
            for node in plan.nodes
            if node.depth > 0 and "failed" not in node.state.facts
        ]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda node: (
                self._gain(state, node.state, node.depth),
                node.cumulative_confidence,
                node.cumulative_value,
                -node.depth,
            ),
        )
        gain = self._gain(state, selected.state, selected.depth)
        if gain <= 0.0:
            return None

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
                "maker:vector_target",
                GoalKind.VECTOR_TARGET,
                selected.state.vector,
                priority=3.0,
                threshold=0.999,
                source="imagined_state",
            )
        )
        for goal in GoalGenerator.from_desired_state(
            state,
            selected.state,
            parent_goal_id="final:success",
            prefix="maker",
        ):
            goals.add(goal)
        return GoalProposal(goals, selected.state, plan, gain)


class GoalExecutor:
    """Choose a reality action for the GOAL supplied by the separate maker."""

    def __init__(self, policy: object, prophecy: object, *, depth: int = 4) -> None:
        self.policy = policy
        self.prophecy = prophecy
        self.depth = depth

    def plan(self, state: StateSnapshot, proposal: GoalProposal) -> ImaginationResult:
        planner = ImaginationTree(
            self.policy,
            self.prophecy,
            config=ImaginationConfig(
                branching_factor=2,
                maximum_depth=self.depth,
                beam_width=32,
                outcome_samples=2,
                minimum_path_confidence=0.0,
                uncertainty_penalty=0.25,
                aggregation="risk_adjusted",
                update_policy=False,
                expand_all_root_actions=True,
            ),
            scorer=GoalStateScorer(
                proposal.goals,
                final_goal_bonus=50.0,
                internal_goal_weight=4.0,
                step_cost=0.01,
            ),
        )
        return planner.plan(state)


class GoalSeparatedAgent:
    """Latest experimental path: GOAL Maker and GOAL Executor are separated."""

    def __init__(self, seed: int) -> None:
        self.base = AutonomousLearningAgent(
            TabularProphecy(),
            config=AutonomousAgentConfig(
                use_imagination=False,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay_episodes=250,
                imagination_minimum_coverage=0.0,
                effect_minimum_samples=2,
                effect_novelty_weight=0.0,
            ),
            seed=seed,
        )
        self.maker = ImaginedGoalMaker(self.base.policy, self.base.prophecy)
        self.executor = GoalExecutor(self.base.policy, self.base.prophecy)
        self.goal_proposals = 0
        self.goal_switches = 0

    @property
    def policy(self) -> object:
        return self.base.policy

    @property
    def prophecy(self) -> object:
        return self.base.prophecy

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

        proposal = self.maker.propose(state)
        if proposal is None:
            return policy_decision
        self.goal_proposals += 1
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
        changed = (
            preferred.action.signature != policy_decision.action.signature
            and advantage >= 0.02
        )
        if changed:
            self.goal_switches += 1
        action = preferred.action if changed else policy_decision.action
        return ActionDecision(
            action,
            True,
            imagined_nodes=len(proposal.maker_plan.nodes) + len(plan.nodes),
            imagination_depth=max(
                proposal.maker_plan.maximum_depth_reached,
                plan.maximum_depth_reached,
            ),
            root_imagined_value=(
                preferred.aggregate_value
                if changed
                else policy_evaluation.aggregate_value
            ),
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
            imagination_required_advantage=0.02,
            imagination_switch_candidate=(
                preferred.action.signature != policy_decision.action.signature
            ),
            imagination_intervention_allowed=changed,
        )

    def observe(self, before: StateSnapshot, action: Action, outcome: GridPushStep) -> object:
        return self.base.observe(before, action, outcome)

    def finish_episode(self, *, final_return: float) -> None:
        self.base.finish_episode(final_return=final_return)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
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
    imagination_interventions: int
    goal_proposals: int
    goal_switches: int

    def as_row(self) -> dict[str, int | float | str]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _standard_agent(condition: str, seed: int) -> AutonomousLearningAgent:
    if condition == "policy_only":
        depth = 1
        use_imagination = False
    elif condition == "prophecy_one_step":
        depth = 1
        use_imagination = True
    elif condition == "full_imagination":
        depth = 8
        use_imagination = True
    else:
        raise ValueError(f"unknown condition: {condition}")

    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            use_imagination=use_imagination,
            imagination_depth=depth,
            imagination_branching_factor=2,
            imagination_beam_width=64,
            imagination_outcome_samples=2,
            imagination_minimum_coverage=0.0,
            imagination_intervention_margin=0.02,
            imagination_uncertainty_margin=0.25,
            imagination_aggregation="risk-adjusted",
            epsilon_start=0.9,
            epsilon_end=0.05,
            epsilon_decay_episodes=250,
            effect_minimum_samples=2,
        ),
        seed=seed,
    )
    agent.planner.scorer = StateDeltaScorer(
        goal_progress_weight=50.0,
        new_fact_weight=4.0,
        unlocked_action_weight=2.0,
        step_cost=0.01,
    )
    return agent


def _run_episode(
    agent: AutonomousLearningAgent | GoalSeparatedAgent,
    *,
    condition: str,
    seed: int,
    phase: str,
    episode: int,
    map_seed: int,
    learn: bool,
) -> EpisodeResult:
    world = GoalGridPushWorld(map_seed)
    imagined_nodes = 0
    imagination_runs = 0
    interventions = 0
    goal_before = getattr(agent, "goal_proposals", 0)
    switches_before = getattr(agent, "goal_switches", 0)
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
        if outcome.snapshot.goal_progress >= 1.0 or "failed" in outcome.snapshot.facts:
            break

    if learn:
        agent.finish_episode(final_return=1.0 if world.success else 0.0)

    goal_after = getattr(agent, "goal_proposals", 0)
    switches_after = getattr(agent, "goal_switches", 0)
    return EpisodeResult(
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
        int(goal_after - goal_before),
        int(switches_after - switches_before),
    )


def _make_agent(condition: str, seed: int) -> AutonomousLearningAgent | GoalSeparatedAgent:
    if condition == "goal_maker_executor":
        return GoalSeparatedAgent(seed)
    return _standard_agent(condition, seed)


def _mean(rows: Iterable[EpisodeResult], field: str) -> float:
    values = [float(getattr(row, field)) for row in rows]
    return fmean(values) if values else 0.0


def summarize(rows: Sequence[EpisodeResult]) -> list[dict[str, int | float | str]]:
    groups: dict[tuple[str, str], list[EpisodeResult]] = {}
    for row in rows:
        groups.setdefault((row.condition, row.phase), []).append(row)
    summary = []
    for (condition, phase), items in sorted(groups.items()):
        successful = [item for item in items if item.success]
        seed_success = []
        for seed in sorted({item.seed for item in items}):
            seed_rows = [item for item in items if item.seed == seed]
            seed_success.append(_mean(seed_rows, "success"))
        summary.append(
            {
                "condition": condition,
                "phase": phase,
                "episodes": len(items),
                "seed_count": len(seed_success),
                "success_rate": _mean(items, "success"),
                "seed_mean_success_rate": fmean(seed_success) if seed_success else 0.0,
                "mean_steps_on_success": _mean(successful, "steps"),
                "mean_efficiency": _mean(items, "efficiency"),
                "mean_imagined_nodes": _mean(items, "imagined_nodes"),
                "mean_imagination_runs": _mean(items, "imagination_runs"),
                "mean_interventions": _mean(items, "imagination_interventions"),
                "mean_goal_proposals": _mean(items, "goal_proposals"),
                "mean_goal_switches": _mean(items, "goal_switches"),
            }
        )
    return summary


def run_goal_gridpush_experiment(
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (7, 13, 21, 42, 100),
    train_episodes: int = 300,
    train_map_count: int = 24,
    evaluation_episodes: int = 80,
    training_tail: int = 80,
) -> dict[str, object]:
    if train_episodes <= 0 or train_map_count <= 0 or evaluation_episodes <= 0:
        raise ValueError("episode and map counts must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions = (
        "policy_only",
        "prophecy_one_step",
        "full_imagination",
        "goal_maker_executor",
    )
    rows: list[EpisodeResult] = []

    for seed in seeds:
        train_maps = tuple(seed * 100_000 + index for index in range(train_map_count))
        unseen_maps = tuple(
            10_000_000 + seed * 100_000 + index
            for index in range(evaluation_episodes)
        )
        for condition in conditions:
            agent = _make_agent(condition, seed)
            training_rows: list[EpisodeResult] = []
            for episode in range(train_episodes):
                training_rows.append(
                    _run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="training",
                        episode=episode,
                        map_seed=train_maps[episode % len(train_maps)],
                        learn=True,
                    )
                )
            for item in training_rows[-min(training_tail, len(training_rows)):]:
                rows.append(
                    EpisodeResult(
                        item.condition,
                        item.seed,
                        "training_tail",
                        item.episode,
                        item.map_seed,
                        item.success,
                        item.steps,
                        item.optimal_steps,
                        item.efficiency,
                        item.imagined_nodes,
                        item.imagination_runs,
                        item.imagination_interventions,
                        item.goal_proposals,
                        item.goal_switches,
                    )
                )
            for episode in range(evaluation_episodes):
                rows.append(
                    _run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="evaluation_seen",
                        episode=episode,
                        map_seed=train_maps[episode % len(train_maps)],
                        learn=False,
                    )
                )
                rows.append(
                    _run_episode(
                        agent,
                        condition=condition,
                        seed=seed,
                        phase="evaluation_unseen",
                        episode=episode,
                        map_seed=unseen_maps[episode],
                        learn=False,
                    )
                )

    episodes_path = output / "episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EpisodeResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(row.as_row() for row in rows)

    summary_rows = summarize(rows)
    summary_path = output / "summary.json"
    payload = {
        "experiment": "goal_gridpush",
        "seeds": list(seeds),
        "train_episodes": train_episodes,
        "train_map_count": train_map_count,
        "evaluation_episodes": evaluation_episodes,
        "conditions": list(conditions),
        "sparse_reward_only_at_exit": True,
        "goal_maker_executor_separated": True,
        "root_expands_all_available_actions": True,
        "summary": summary_rows,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
