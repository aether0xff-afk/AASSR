from __future__ import annotations

import copy
import csv
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterator, Sequence

from . import toolgrid_factorial_masked as env
from .autonomous_agent_core import ActionDecision
from .types import Action, StateSnapshot


base = env.base


@dataclass(frozen=True, slots=True)
class OracleAssessment:
    viable: int
    remaining_steps: int


@dataclass(frozen=True, slots=True)
class PairedEpisodeRow:
    seed: int
    grid_size: int
    action_count: int
    map_seed: int
    mode: str
    success: int
    steps: int

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionDiagnosticRow:
    seed: int
    grid_size: int
    action_count: int
    map_seed: int
    decision_index: int
    policy_action: str
    preferred_action: str
    executed_action: str
    imagination_changed_action: int
    intervention_reason: str
    model_coverage: float
    policy_value: float
    preferred_value: float
    imagined_advantage: float
    required_advantage: float
    imagined_nodes: int
    policy_oracle_viable: int
    policy_oracle_remaining_steps: int
    preferred_oracle_viable: int
    preferred_oracle_remaining_steps: int
    executed_oracle_viable: int
    executed_oracle_remaining_steps: int
    intervention_effect: str
    predicted_critic_best_action: str
    actual_critic_best_action: str
    oracle_best_action: str
    predicted_critic_best_viable: int
    actual_critic_best_viable: int
    predicted_actual_rank_agree: int

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RootActionDiagnosticRow:
    seed: int
    grid_size: int
    action_count: int
    map_seed: int
    decision_index: int
    action: str
    is_policy_action: int
    is_preferred_action: int
    is_executed_action: int
    oracle_viable: int
    oracle_remaining_steps: int
    prophecy_vector_mae: float
    prophecy_terminal_match: int
    prophecy_available_actions_match: int
    prophecy_confidence: float
    critic_predicted_value: float
    critic_actual_value: float

    def row(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def _imagination_setting(agent: Any, enabled: bool) -> Iterator[None]:
    original = agent.agent.config
    agent.agent.config = replace(original, use_imagination=bool(enabled))
    try:
        yield
    finally:
        agent.agent.config = original


def _write_rows(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].row()))
        writer.writeheader()
        writer.writerows(item.row() for item in rows)


def _shortest_remaining_path(world: Any) -> int | None:
    station = world.current_station
    if station is None:
        return 0 if world.success else None
    if world.agent == station:
        return 1

    frontier: list[tuple[tuple[int, int], int]] = [(world.agent, 0)]
    seen = {world.agent}
    blocked = set(world.used_cells)
    blocked.discard(world.agent)
    for point, distance in frontier:
        if point == station:
            return distance + 1
        for dx, dy in base.MOVE_DELTAS.values():
            candidate = point[0] + dx, point[1] + dy
            if not (
                0 <= candidate[0] < world.grid_size
                and 0 <= candidate[1] < world.grid_size
            ):
                continue
            if candidate in blocked or candidate in seen:
                continue
            seen.add(candidate)
            frontier.append((candidate, distance + 1))
    return None


def assess_action(world: Any, action: Action) -> OracleAssessment:
    clone = copy.deepcopy(world)
    clone.step(action)
    if clone.success:
        return OracleAssessment(1, 0)
    if clone.failed:
        return OracleAssessment(0, -1)
    remaining = _shortest_remaining_path(clone)
    if remaining is None:
        return OracleAssessment(0, -1)
    return OracleAssessment(1, remaining)


def _actual_transition(world: Any, action: Action) -> StateSnapshot:
    clone = copy.deepcopy(world)
    return clone.step(action).snapshot


def _greedy_policy_action(agent: Any, state: StateSnapshot) -> Action:
    return agent.dqn.select_action(state, episode=0, training=False).action


def _imagination_decision(agent: Any, state: StateSnapshot) -> ActionDecision:
    with _imagination_setting(agent, True):
        return agent.agent.select_action(state, episode=0, explore=False)


def _train_without_imagination(
    agent: Any,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    transition_budget: int,
    train_map_count: int,
) -> dict[str, int | float]:
    cell_base = seed * 10_000_000 + grid_size * 100_000 + action_count * 1_000
    training_maps = tuple(cell_base + index for index in range(train_map_count))
    environment_steps = 0
    episode = 0
    successes = 0
    started = time.perf_counter()

    while environment_steps < transition_budget:
        world = env.ToolGridWorld(
            training_maps[episode % len(training_maps)],
            grid_size=grid_size,
            action_count=action_count,
        )
        agent.begin_episode(training=True)
        while world.snapshot().available_actions:
            before = world.snapshot()
            with _imagination_setting(agent, False):
                decision = agent.select_action(
                    before,
                    episode=min(transition_budget, environment_steps),
                    training=True,
                )
            outcome = world.step(decision.action)
            environment_steps += 1
            agent.observe(before, decision.action, outcome)
            if world.success or world.failed:
                break
        agent.end_episode(success=world.success, training=True)
        successes += int(world.success)
        episode += 1

    return {
        "training_transitions": environment_steps,
        "training_episodes": episode,
        "training_successes": successes,
        "training_success_rate": successes / episode if episode else 0.0,
        "training_wall_seconds": time.perf_counter() - started,
    }


def _evaluate_policy_episode(
    agent: Any,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    map_seed: int,
) -> PairedEpisodeRow:
    world = env.ToolGridWorld(map_seed, grid_size=grid_size, action_count=action_count)
    agent.begin_episode(training=False)
    steps = 0
    while world.snapshot().available_actions:
        action = _greedy_policy_action(agent, world.snapshot())
        world.step(action)
        steps += 1
        if world.success or world.failed:
            break
    agent.end_episode(success=world.success, training=False)
    return PairedEpisodeRow(
        seed=seed,
        grid_size=grid_size,
        action_count=action_count,
        map_seed=map_seed,
        mode="policy_only_same_checkpoint",
        success=int(world.success),
        steps=steps,
    )


def _effect_label(policy: OracleAssessment, executed: OracleAssessment, changed: bool) -> str:
    if not changed:
        return "no_change"
    if policy.viable and not executed.viable:
        return "harmful"
    if not policy.viable and executed.viable:
        return "beneficial"
    if policy.viable and executed.viable:
        if executed.remaining_steps < policy.remaining_steps:
            return "shorter_viable"
        if executed.remaining_steps > policy.remaining_steps:
            return "longer_viable"
    return "neutral"


def _best_action_by_oracle(
    assessments: dict[str, OracleAssessment],
) -> str:
    ranked = sorted(
        assessments.items(),
        key=lambda item: (
            -item[1].viable,
            item[1].remaining_steps if item[1].viable else 10**9,
            item[0],
        ),
    )
    return ranked[0][0]


def _evaluate_imagination_episode(
    agent: Any,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    map_seed: int,
    decision_offset: int,
) -> tuple[PairedEpisodeRow, list[DecisionDiagnosticRow], list[RootActionDiagnosticRow]]:
    world = env.ToolGridWorld(map_seed, grid_size=grid_size, action_count=action_count)
    agent.begin_episode(training=False)
    steps = 0
    decision_rows: list[DecisionDiagnosticRow] = []
    root_rows: list[RootActionDiagnosticRow] = []

    while world.snapshot().available_actions:
        state = world.snapshot()
        policy_action = _greedy_policy_action(agent, state)
        decision = _imagination_decision(agent, state)
        preferred_signature = (
            decision.imagination_preferred_action_signature
            or decision.action.signature
        )
        action_by_signature = {
            action.signature: action for action in state.available_actions
        }
        preferred_action = action_by_signature.get(
            preferred_signature,
            decision.action,
        )

        assessments = {
            action.signature: assess_action(world, action)
            for action in state.available_actions
        }
        predicted_values: dict[str, float] = {}
        actual_values: dict[str, float] = {}
        decision_index = decision_offset + len(decision_rows)

        for action in state.available_actions:
            prediction = agent.agent.prophecy.predict(state, action, samples=1)[0]
            actual_after = _actual_transition(world, action)
            vector_mae = fmean(
                abs(left - right)
                for left, right in zip(
                    prediction.next_state.vector,
                    actual_after.vector,
                    strict=True,
                )
            )
            predicted_score = agent.critic.score_step(
                state,
                action,
                prediction.next_state,
                prophecy_confidence=float(prediction.probability),
            ).value
            actual_score = agent.critic.score_step(
                state,
                action,
                actual_after,
                prophecy_confidence=float(prediction.probability),
            ).value
            predicted_values[action.signature] = float(predicted_score)
            actual_values[action.signature] = float(actual_score)
            assessment = assessments[action.signature]
            root_rows.append(
                RootActionDiagnosticRow(
                    seed=seed,
                    grid_size=grid_size,
                    action_count=action_count,
                    map_seed=map_seed,
                    decision_index=decision_index,
                    action=action.signature,
                    is_policy_action=int(action.signature == policy_action.signature),
                    is_preferred_action=int(action.signature == preferred_action.signature),
                    is_executed_action=int(action.signature == decision.action.signature),
                    oracle_viable=assessment.viable,
                    oracle_remaining_steps=assessment.remaining_steps,
                    prophecy_vector_mae=vector_mae,
                    prophecy_terminal_match=int(
                        bool(prediction.next_state.available_actions)
                        == bool(actual_after.available_actions)
                    ),
                    prophecy_available_actions_match=int(
                        {item.signature for item in prediction.next_state.available_actions}
                        == {item.signature for item in actual_after.available_actions}
                    ),
                    prophecy_confidence=float(prediction.probability),
                    critic_predicted_value=float(predicted_score),
                    critic_actual_value=float(actual_score),
                )
            )

        predicted_best = max(
            predicted_values,
            key=lambda signature: (predicted_values[signature], signature),
        )
        actual_best = max(
            actual_values,
            key=lambda signature: (actual_values[signature], signature),
        )
        oracle_best = _best_action_by_oracle(assessments)
        policy_assessment = assessments[policy_action.signature]
        preferred_assessment = assessments[preferred_action.signature]
        executed_assessment = assessments[decision.action.signature]
        changed = decision.action.signature != policy_action.signature
        decision_rows.append(
            DecisionDiagnosticRow(
                seed=seed,
                grid_size=grid_size,
                action_count=action_count,
                map_seed=map_seed,
                decision_index=decision_index,
                policy_action=policy_action.signature,
                preferred_action=preferred_action.signature,
                executed_action=decision.action.signature,
                imagination_changed_action=int(changed),
                intervention_reason=decision.imagination_gate_reason,
                model_coverage=decision.model_coverage,
                policy_value=decision.imagination_policy_value,
                preferred_value=decision.imagination_preferred_value,
                imagined_advantage=decision.imagination_advantage,
                required_advantage=decision.imagination_required_advantage,
                imagined_nodes=decision.imagined_nodes,
                policy_oracle_viable=policy_assessment.viable,
                policy_oracle_remaining_steps=policy_assessment.remaining_steps,
                preferred_oracle_viable=preferred_assessment.viable,
                preferred_oracle_remaining_steps=preferred_assessment.remaining_steps,
                executed_oracle_viable=executed_assessment.viable,
                executed_oracle_remaining_steps=executed_assessment.remaining_steps,
                intervention_effect=_effect_label(
                    policy_assessment,
                    executed_assessment,
                    changed,
                ),
                predicted_critic_best_action=predicted_best,
                actual_critic_best_action=actual_best,
                oracle_best_action=oracle_best,
                predicted_critic_best_viable=assessments[predicted_best].viable,
                actual_critic_best_viable=assessments[actual_best].viable,
                predicted_actual_rank_agree=int(predicted_best == actual_best),
            )
        )

        world.step(decision.action)
        steps += 1
        if world.success or world.failed:
            break

    agent.end_episode(success=world.success, training=False)
    return (
        PairedEpisodeRow(
            seed=seed,
            grid_size=grid_size,
            action_count=action_count,
            map_seed=map_seed,
            mode="imagination_same_checkpoint",
            success=int(world.success),
            steps=steps,
        ),
        decision_rows,
        root_rows,
    )


def _mean(rows: Sequence[Any], field: str) -> float:
    return fmean(float(getattr(item, field)) for item in rows) if rows else 0.0


def run_toolgrid_imagination_debug(
    output_dir: str | Path,
    *,
    seed: int,
    grid_size: int,
    action_count: int,
    transition_budget: int = 5_000,
    train_map_count: int = 48,
    evaluation_map_count: int = 100,
) -> dict[str, Any]:
    if grid_size not in env.GRID_SIZES or action_count not in env.ACTION_COUNTS:
        raise ValueError("unsupported ToolGrid diagnostic cell")
    if min(transition_budget, train_map_count, evaluation_map_count) <= 0:
        raise ValueError("diagnostic sizes must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agent = base.ToolGridHybridAgent(
        seed,
        action_count=action_count,
        train_transitions=transition_budget,
        use_imagination=True,
    )
    training = _train_without_imagination(
        agent,
        seed=seed,
        grid_size=grid_size,
        action_count=action_count,
        transition_budget=transition_budget,
        train_map_count=train_map_count,
    )
    if not agent.critic_ready:
        raise RuntimeError("critic did not become ready in the diagnostic training run")

    cell_base = seed * 10_000_000 + grid_size * 100_000 + action_count * 1_000
    evaluation_maps = tuple(
        cell_base + 500_000 + index for index in range(evaluation_map_count)
    )
    episode_rows: list[PairedEpisodeRow] = []
    decision_rows: list[DecisionDiagnosticRow] = []
    root_rows: list[RootActionDiagnosticRow] = []

    for map_seed in evaluation_maps:
        episode_rows.append(
            _evaluate_policy_episode(
                agent,
                seed=seed,
                grid_size=grid_size,
                action_count=action_count,
                map_seed=map_seed,
            )
        )
        imagination_row, decisions, roots = _evaluate_imagination_episode(
            agent,
            seed=seed,
            grid_size=grid_size,
            action_count=action_count,
            map_seed=map_seed,
            decision_offset=len(decision_rows),
        )
        episode_rows.append(imagination_row)
        decision_rows.extend(decisions)
        root_rows.extend(roots)

    policy_rows = [item for item in episode_rows if item.mode.startswith("policy")]
    imagination_rows = [
        item for item in episode_rows if item.mode.startswith("imagination")
    ]
    policy_by_map = {item.map_seed: item for item in policy_rows}
    imagination_by_map = {item.map_seed: item for item in imagination_rows}
    paired = [
        (policy_by_map[map_seed], imagination_by_map[map_seed])
        for map_seed in evaluation_maps
    ]
    changed = [item for item in decision_rows if item.imagination_changed_action]

    summary = {
        "config": {
            "seed": seed,
            "grid_size": grid_size,
            "action_count": action_count,
            "tool_count": action_count - 4,
            "transition_budget": transition_budget,
            "train_map_count": train_map_count,
            "evaluation_map_count": evaluation_map_count,
            "training_imagination": False,
            "paired_same_checkpoint": True,
        },
        "training": training,
        "model_stats": agent.model_stats(),
        "paired_evaluation": {
            "policy_success_rate": _mean(policy_rows, "success"),
            "imagination_success_rate": _mean(imagination_rows, "success"),
            "imagination_minus_policy": (
                _mean(imagination_rows, "success") - _mean(policy_rows, "success")
            ),
            "improved_maps": sum(
                1 for policy, imagined in paired if not policy.success and imagined.success
            ),
            "worsened_maps": sum(
                1 for policy, imagined in paired if policy.success and not imagined.success
            ),
            "unchanged_maps": sum(
                1 for policy, imagined in paired if policy.success == imagined.success
            ),
        },
        "interventions": {
            "decisions": len(decision_rows),
            "changed_actions": len(changed),
            "change_rate": len(changed) / len(decision_rows) if decision_rows else 0.0,
            "beneficial": sum(item.intervention_effect == "beneficial" for item in changed),
            "harmful": sum(item.intervention_effect == "harmful" for item in changed),
            "shorter_viable": sum(
                item.intervention_effect == "shorter_viable" for item in changed
            ),
            "longer_viable": sum(
                item.intervention_effect == "longer_viable" for item in changed
            ),
            "neutral": sum(item.intervention_effect == "neutral" for item in changed),
        },
        "component_diagnostics": {
            "prophecy_vector_mae": _mean(root_rows, "prophecy_vector_mae"),
            "prophecy_terminal_accuracy": _mean(
                root_rows,
                "prophecy_terminal_match",
            ),
            "prophecy_available_actions_accuracy": _mean(
                root_rows,
                "prophecy_available_actions_match",
            ),
            "predicted_critic_best_viable_rate": _mean(
                decision_rows,
                "predicted_critic_best_viable",
            ),
            "actual_state_critic_best_viable_rate": _mean(
                decision_rows,
                "actual_critic_best_viable",
            ),
            "predicted_actual_critic_rank_agreement": _mean(
                decision_rows,
                "predicted_actual_rank_agree",
            ),
        },
    }

    _write_rows(output / "paired_episodes.csv", episode_rows)
    _write_rows(output / "decision_diagnostics.csv", decision_rows)
    _write_rows(output / "root_action_diagnostics.csv", root_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=repr),
        encoding="utf-8",
    )
    return summary
