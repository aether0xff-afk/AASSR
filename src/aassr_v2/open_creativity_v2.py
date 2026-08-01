from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .causal_dependency_world import ACTION_LAWS, CausalDependencyWorldV2
from .causal_representation import RelationalEffectEncoder, RepresentedReturnAgent
from .creativity import novelty_against_references
from .paper_types import CausalEffectGraph
from .paper_v2_protocol import sha256_json
from .representation_diagnostic import _transition


EFFECT_BY_KEY = {law.key: law.effect for law in ACTION_LAWS}


def graph_from_analysis_path(
    world: CausalDependencyWorldV2,
    action_tokens: Sequence[str],
) -> CausalEffectGraph:
    replay = CausalDependencyWorldV2(
        world_seed=world.world_seed,
        token_seed=world.token_seed,
        observation_seed=world.observation_seed,
        composition_template=world.composition_template,
    )
    nodes = set()
    edges = set()
    sequence = []
    previous = None
    for token in action_tokens:
        effect, prerequisites = replay.analysis_effect_for_action(token)
        if effect == "no_effect":
            replay.step(token)
            continue
        nodes.add(effect)
        sequence.append(effect)
        for prerequisite in prerequisites:
            prerequisite_effect = EFFECT_BY_KEY[prerequisite]
            nodes.add(prerequisite_effect)
            edges.add((prerequisite_effect, effect, "inferred_after_analysis"))
        if previous is not None:
            edges.add((previous, effect, "observed_sequence"))
        previous = effect
        replay.step(token)
    return CausalEffectGraph(tuple(nodes), tuple(edges), tuple(sequence))


def _world_state_key(world: CausalDependencyWorldV2) -> tuple[Any, ...]:
    state = world.analysis_private_state
    return (
        state.completed,
        state.inventory,
        state.location,
        state.health,
        state.terminal,
        state.success,
    )


def enumerate_feasible_graphs(
    world: CausalDependencyWorldV2,
    *,
    maximum_depth: int = 12,
    maximum_paths: int = 10_000,
) -> tuple[CausalEffectGraph, ...]:
    queue = deque([(world.clone(), tuple())])
    seen: set[tuple[Any, ...]] = set()
    graphs: dict[str, CausalEffectGraph] = {}
    while queue and len(graphs) < maximum_paths:
        current, path = queue.popleft()
        state = current.analysis_private_state
        if state.success:
            graph = graph_from_analysis_path(world, path)
            graphs.setdefault(sha256_json(graph.to_dict()), graph)
            continue
        if state.terminal or len(path) >= maximum_depth:
            continue
        history_key = (_world_state_key(current), tuple(state.effect_history))
        if history_key in seen:
            continue
        seen.add(history_key)
        for action in current.observe().available_actions:
            effect, _ = current.analysis_effect_for_action(action)
            if effect == "no_effect":
                continue
            child = current.clone()
            child.step(action)
            queue.append((child, path + (action,)))
    return tuple(graphs.values())


def _random_baseline_graphs(
    *,
    world_seeds: Sequence[int],
    interaction_budget: int,
    maximum_steps: int = 12,
) -> list[CausalEffectGraph]:
    rng = random.Random(0xBA5E11)
    graphs = []
    for episode in range(interaction_budget):
        seed = int(world_seeds[episode % len(world_seeds)])
        world = CausalDependencyWorldV2(
            world_seed=seed,
            composition_template="open_creativity_v1",
        )
        path = []
        for _ in range(maximum_steps):
            actions = world.observe().available_actions
            if not actions or world.terminal:
                break
            action = rng.choice(actions)
            path.append(action)
            world.step(action)
        if world.analysis_private_state.success:
            graphs.append(graph_from_analysis_path(world, path))
    return graphs


def freeze_baseline_reference(
    path: str | Path,
    *,
    world_seeds: Sequence[int],
    interaction_budget: int,
) -> dict[str, Any]:
    graphs = _random_baseline_graphs(
        world_seeds=world_seeds, interaction_budget=interaction_budget
    )
    unique = {
        sha256_json(graph.to_dict()): graph.to_dict() for graph in graphs
    }
    payload = {
        "schema_version": 2,
        "status": "frozen",
        "source": "independent_random_baseline",
        "interaction_budget": int(interaction_budget),
        "world_seeds": [int(seed) for seed in world_seeds],
        "graphs": [unique[key] for key in sorted(unique)],
    }
    payload["reference_sha256"] = sha256_json(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return payload


def load_frozen_reference(path: str | Path) -> tuple[CausalEffectGraph, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(payload.pop("reference_sha256"))
    if payload.get("status") != "frozen" or sha256_json(payload) != expected:
        raise ValueError("creativity reference manifest changed")
    return tuple(CausalEffectGraph.from_dict(item) for item in payload["graphs"])


def creativity_environment_adequacy(
    *,
    world_seed: int,
    references: Sequence[CausalEffectGraph],
) -> dict[str, Any]:
    world = CausalDependencyWorldV2(
        world_seed=world_seed, composition_template="open_creativity_v1"
    )
    feasible = enumerate_feasible_graphs(world)
    reference_hashes = {sha256_json(graph.to_dict()) for graph in references}
    feasible_hashes = {sha256_json(graph.to_dict()) for graph in feasible}
    outside = feasible_hashes - reference_hashes
    family_signatures = {tuple(sorted(graph.nodes)) for graph in feasible}
    coverage = len(feasible_hashes & reference_hashes) / max(1, len(feasible_hashes))
    return {
        "feasible_graph_count": len(feasible),
        "causal_family_count": len(family_signatures),
        "outside_reference_count": len(outside),
        "random_reference_coverage": coverage,
        "structural_difference_exists": any(
            novelty_against_references(graph, references)["aggregate"] > 0.0
            for graph in feasible
        ),
        "adequate": bool(
            outside
            and len(family_signatures) >= 3
            and coverage <= 0.25
        ),
    }


@dataclass(frozen=True, slots=True)
class CreativeCandidate:
    graph_sha256: str
    graph: CausalEffectGraph
    novelty_score: float
    success_count: int
    research_seed_count: int
    mean_steps: float
    reusable_success_rate: float
    utility_qualified: bool
    candidate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_sha256": self.graph_sha256,
            "graph": self.graph.to_dict(),
            "novelty_score": self.novelty_score,
            "success_count": self.success_count,
            "research_seed_count": self.research_seed_count,
            "mean_steps": self.mean_steps,
            "reusable_success_rate": self.reusable_success_rate,
            "utility_qualified": self.utility_qualified,
            "candidate": self.candidate,
        }


def run_open_creativity_diagnostic(
    *,
    research_seeds: Sequence[int],
    world_seeds: Sequence[int],
    references: Sequence[CausalEffectGraph],
    interaction_budget: int,
) -> tuple[list[CreativeCandidate], dict[str, Any]]:
    observed: dict[str, list[tuple[int, int, CausalEffectGraph]]] = defaultdict(list)
    for research_seed in research_seeds:
        agent = RepresentedReturnAgent(
            RelationalEffectEncoder(), seed=int(research_seed)
        )
        for episode in range(interaction_budget):
            seed = int(world_seeds[episode % len(world_seeds)])
            world = CausalDependencyWorldV2(
                world_seed=seed,
                composition_template="open_creativity_v1",
            )
            path = []
            for _ in range(12):
                before = world.observe()
                if before.terminal:
                    break
                epsilon = max(0.05, 0.8 * (1.0 - episode / max(1, interaction_budget)))
                action = agent.select_action(before, epsilon=epsilon)
                outcome = world.step(action)
                agent.observe_transition(_transition(before, action, outcome))
                path.append(action)
                if world.terminal:
                    break
            success = world.analysis_private_state.success
            agent.finish_episode(success)
            if success:
                graph = graph_from_analysis_path(world, path)
                graph_hash = sha256_json(graph.to_dict())
                observed[graph_hash].append((int(research_seed), len(path), graph))
    baseline_steps = [len(graph.effect_sequence) for graph in references] or [12]
    candidates = []
    for graph_hash, records in observed.items():
        graph = records[0][2]
        novelty = novelty_against_references(graph, references)["aggregate"]
        seeds = {record[0] for record in records}
        mean_steps = sum(record[1] for record in records) / len(records)
        utility = mean_steps <= max(12.0, float(median(baseline_steps)) + 4.0)
        reusable = len(seeds) / max(1, len(research_seeds))
        candidates.append(
            CreativeCandidate(
                graph_hash,
                graph,
                novelty,
                len(records),
                len(seeds),
                mean_steps,
                reusable,
                utility,
                novelty > 0.0 and utility and len(seeds) >= 2 and reusable >= 0.5,
            )
        )
    return candidates, {
        "successful_unique_graphs": len(observed),
        "novel_graphs": sum(item.novelty_score > 0.0 for item in candidates),
        "creative_candidates": sum(item.candidate for item in candidates),
        "aassr_novelty_is_gate": False,
        "baseline_interaction_budget": interaction_budget,
        "aassr_interaction_budget_per_seed": interaction_budget,
    }
