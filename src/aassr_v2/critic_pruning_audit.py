from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Sequence

from .baseline_efficiency_benchmark import BenchmarkGridPushWorld, encode_gridpush_state
from .bottleneck_sota_diagnostic import BenchmarkOracleProphecy, remaining_oracle_steps
from .branch_critic import GRUBranchCritic, ParentTransitionCritic
from .critic_prophecy_common import EpisodeRecord
from .imagination_tree import StateDeltaScorer
from .types import StateSnapshot


def _critic_predictions(critic: Any, episodes: Sequence[EpisodeRecord]) -> dict[str, float]:
    values, targets = [], []
    for episode in episodes:
        memory = critic.initial_memory()
        for transition in episode.transitions:
            step = critic.score_step(
                transition.before, transition.action, transition.after,
                memory=memory, prophecy_confidence=1.0,
            )
            memory = step.memory
            values.append(step.value)
            targets.append(float(episode.success))
    if not values:
        return {"count": 0, "accuracy": 0.0, "brier": 0.0}
    return {
        "count": len(values),
        "accuracy": fmean(
            float((value >= 0.5) == bool(target))
            for value, target in zip(values, targets, strict=True)
        ),
        "brier": fmean(
            (value - target) ** 2
            for value, target in zip(values, targets, strict=True)
        ),
        "mean_prediction": fmean(values),
        "positive_rate": fmean(targets),
    }


@dataclass(frozen=True, slots=True)
class _SearchNode:
    state: StateSnapshot
    score: float
    memory: Any


def evaluate_pruning(
    mode: str,
    map_seeds: Sequence[int],
    *,
    critic: Any | None,
    depth: int,
    beam_width: int,
) -> dict[str, float | int]:
    oracle = BenchmarkOracleProphecy()
    hand = StateDeltaScorer(
        goal_progress_weight=50.0,
        new_fact_weight=4.0,
        unlocked_action_weight=2.0,
        step_cost=0.01,
    )
    successes = retained = expanded = 0
    best_distances = []
    for map_seed in map_seeds:
        root = BenchmarkGridPushWorld(map_seed).snapshot()
        frontier = [
            _SearchNode(root, 0.0, critic.initial_memory() if critic else None)
        ]
        final_selected = frontier
        reached = False
        for _ in range(depth):
            children = []
            for node in frontier:
                expanded += 1
                for action in node.state.available_actions:
                    after = oracle.predict(node.state, action, samples=1)[0].next_state
                    if mode == "hand_scorer":
                        score = node.score + hand.score(node.state, action, after)
                        memory = None
                    else:
                        step = critic.score_step(
                            node.state, action, after, memory=node.memory,
                            prophecy_confidence=1.0,
                        )
                        score, memory = step.value, step.memory
                    children.append(_SearchNode(after, score, memory))
            if not children:
                break
            children.sort(
                key=lambda item: (
                    -item.score,
                    tuple(round(value, 8) for value in item.state.vector),
                    tuple(sorted(item.state.facts)),
                )
            )
            final_selected = children[:beam_width]
            if any("success" in node.state.facts for node in final_selected):
                reached = True
                break
            frontier = [node for node in final_selected if node.state.available_actions]
            if not frontier:
                break
        successes += int(reached)
        distances = []
        for node in final_selected:
            distance = remaining_oracle_steps(node.state)
            if "success" in node.state.facts:
                distance = 0
            if distance is not None:
                distances.append(distance)
        retained += int(bool(distances))
        if distances:
            best_distances.append(float(min(distances)))
    count = max(1, len(map_seeds))
    return {
        "map_count": len(map_seeds),
        "success_reached_rate": successes / count,
        "solvable_branch_retained_rate": retained / count,
        "mean_best_remaining_steps": (
            fmean(best_distances) if best_distances else float("inf")
        ),
        "expanded_parent_nodes": expanded,
    }


def run_critic_audit(
    train: Sequence[EpisodeRecord],
    heldout: Sequence[EpisodeRecord],
    unseen_maps: Sequence[int],
    *,
    seed: int,
    depth: int,
    beam_width: int,
) -> dict[str, Any]:
    parent = ParentTransitionCritic(encode_gridpush_state, 25, hidden_units=64, seed=seed)
    gru = GRUBranchCritic(encode_gridpush_state, 25, hidden_units=64, seed=seed)
    for episode in train:
        parent.observe_episode(episode.transitions, success=episode.success)
        gru.observe_episode(episode.transitions, success=episode.success)
    return {
        "training_data": {
            "episodes": len(train),
            "success_rate": fmean(float(item.success) for item in train),
            "transitions": sum(len(item.transitions) for item in train),
        },
        "heldout_episode_labels": {
            "parent": _critic_predictions(parent, heldout),
            "gru": _critic_predictions(gru, heldout),
        },
        "pruning": {
            "hand_scorer": evaluate_pruning(
                "hand_scorer", unseen_maps, critic=None,
                depth=depth, beam_width=beam_width,
            ),
            "parent": evaluate_pruning(
                "parent", unseen_maps, critic=parent,
                depth=depth, beam_width=beam_width,
            ),
            "gru": evaluate_pruning(
                "gru", unseen_maps, critic=gru,
                depth=depth, beam_width=beam_width,
            ),
        },
        "model_stats": {
            "parent": asdict(parent.stats()),
            "gru": asdict(gru.stats()),
        },
    }
