from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Any, Sequence

from .causal_agent_v2 import CausalAASSRAgent
from .causal_dependency_world import CausalDependencyWorldV2
from .causal_imagination import (
    CausalImaginationPlanner,
    ImaginationGateConfig,
    LearnedReturnModel,
    OracleReturnModel,
    RandomReturnModel,
    exact_root_action_values,
)
from .causal_representation import ObservableTransition, RelationalEffectEncoder
from .paper_v2_types import ImaginationDecisionRecord


def _transition(before, action: str, outcome) -> ObservableTransition:
    return ObservableTransition(
        before,
        action,
        outcome.observation,
        outcome.action_succeeded,
        outcome.inventory_delta,
        len(outcome.facts_added),
        len(outcome.facts_removed),
        len(outcome.unlocked_actions),
        outcome.resource_cost,
        outcome.damage,
        outcome.spatial_change is not None,
        outcome.reward,
    )


def train_causal_agent(
    *, research_seed: int, world_seeds: Sequence[int], episodes: int
) -> CausalAASSRAgent:
    agent = CausalAASSRAgent(RelationalEffectEncoder, seed=research_seed)
    for episode in range(episodes):
        world = CausalDependencyWorldV2(
            world_seed=int(world_seeds[episode % len(world_seeds)]),
            token_seed=92001,
        )
        epsilon = max(0.05, 0.8 * (1.0 - episode / 400.0))
        for _ in range(8):
            before = world.observe()
            if before.terminal:
                break
            action = agent.policy_action(before, epsilon=epsilon)
            outcome = world.step(action)
            agent.observe_transition(_transition(before, action, outcome))
            if world.terminal:
                break
        agent.finish_episode(world.analysis_private_state.success)
    return agent


def run_diagnostic_four(
    *,
    research_seeds: Sequence[int],
    world_seeds: Sequence[int],
    training_episodes: int = 500,
    evaluation_episodes: int = 100,
    gate: ImaginationGateConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = gate or ImaginationGateConfig()
    summaries = []
    decision_records = []
    for research_seed in research_seeds:
        agent = train_causal_agent(
            research_seed=int(research_seed),
            world_seeds=world_seeds,
            episodes=training_episodes,
        )
        models = {
            "random_model_gated": RandomReturnModel(int(research_seed)),
            "learned_model_gated": LearnedReturnModel(agent.prophecy),
            "oracle_model_upper_bound": OracleReturnModel(gamma=gate.gamma),
        }
        condition_metrics = {
            name: {"success": 0, "interventions": 0, "regret": [], "dead_ends": 0, "optimal": 0, "decisions": 0}
            for name in ("policy_only", *models)
        }
        transition_accuracy = []
        for episode in range(evaluation_episodes):
            seed = int(world_seeds[episode % len(world_seeds)])
            for name in condition_metrics:
                world = CausalDependencyWorldV2(world_seed=seed, token_seed=92001)
                planner = None if name == "policy_only" else CausalImaginationPlanner(
                    models[name], config=gate, gated=True
                )
                episode_records = []
                for step in range(8):
                    observation = world.observe()
                    if observation.terminal:
                        break
                    exact = exact_root_action_values(world, gamma=gate.gamma)
                    optimal_value = max(exact.values())
                    policy_action = agent.policy_action(observation, epsilon=0.0)
                    if planner is None:
                        action = policy_action
                        record = None
                    else:
                        record = planner.decide(
                            observation,
                            agent.policy,
                            world=world if name == "oracle_model_upper_bound" else None,
                        )
                        action = record.final_selected_action
                        condition_metrics[name]["interventions"] += int(record.intervened)
                        condition_metrics[name]["optimal"] += int(
                            abs(exact[action] - optimal_value) <= 1e-12
                        )
                        condition_metrics[name]["decisions"] += 1
                        if name == "oracle_model_upper_bound":
                            predicted = models[name].estimate(observation, action, world=world)
                            transition_accuracy.append(
                                predicted.next_observation == world.oracle_transition(action)
                            )
                    condition_metrics[name]["regret"].append(optimal_value - exact[action])
                    outcome = world.step(action)
                    if record is not None:
                        episode_records.append(
                            {
                                "research_seed": int(research_seed),
                                "episode": episode,
                                "step": step,
                                "condition": name,
                                **replace(
                                    record,
                                    actual_success=None,
                                ).to_dict(),
                            }
                        )
                    if world.analysis_private_state.dead_end:
                        condition_metrics[name]["dead_ends"] += 1
                    if world.terminal:
                        break
                success = world.analysis_private_state.success
                condition_metrics[name]["success"] += int(success)
                for item in episode_records:
                    item["actual_success"] = success
                    decision_records.append(item)
        for name, values in condition_metrics.items():
            summaries.append(
                {
                    "research_seed": int(research_seed),
                    "condition": name,
                    "success_rate": values["success"] / evaluation_episodes,
                    "intervention_rate": values["interventions"] / max(1, values["decisions"]),
                    "mean_decision_regret": fmean(values["regret"]) if values["regret"] else 0.0,
                    "dead_end_entry_rate": values["dead_ends"] / evaluation_episodes,
                    "root_action_optimality": values["optimal"] / max(1, values["decisions"]),
                }
            )
    def mean(condition: str, metric: str) -> float:
        return fmean(row[metric] for row in summaries if row["condition"] == condition)
    engineering = {
        "oracle_transition_accuracy": fmean(transition_accuracy) if transition_accuracy else 0.0,
        "oracle_root_action_optimality": mean("oracle_model_upper_bound", "root_action_optimality"),
        "oracle_regret_below_policy": mean("oracle_model_upper_bound", "mean_decision_regret")
        < mean("policy_only", "mean_decision_regret"),
        "oracle_dead_end_not_above_policy": mean("oracle_model_upper_bound", "dead_end_entry_rate")
        <= mean("policy_only", "dead_end_entry_rate"),
        "random_model_intervention_rate": mean("random_model_gated", "intervention_rate"),
    }
    return {"engineering": engineering, "summaries": summaries}, decision_records
