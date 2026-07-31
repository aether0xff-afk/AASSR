from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .paper_types import CausalEffectGraph, StrategyRecord
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class CreativeStep:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    reward: float = 0.0
    effect_events: tuple[Mapping[str, Any], ...] = ()
    resource_cost: float = 0.0
    risk_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class _Operation:
    key: str
    action: Action
    effect: str
    prerequisites: tuple[str, ...] = ()
    relation: str = "prerequisite"
    resource_cost: float = 0.0
    risk_delta: float = 0.0
    terminal_family: str = ""


class MultiSolutionDependencyWorld:
    """Opaque sparse-reward world with several causally distinct solutions.

    Agent-visible snapshots contain only opaque observations and action tokens.
    The semantic operation table and terminal family remain private analysis
    state and are never placed in ``StateSnapshot`` or action metadata.
    """

    FAMILY_COUNT = 5

    def __init__(self, *, seed: int = 0, variant: int = 0) -> None:
        self.seed = int(seed)
        self.variant = int(variant)
        randomizer = random.Random(seed * 104729 + variant * 130363)
        keys = (
            "inspect",
            "decode",
            "gather",
            "unlock",
            "reroute",
            "traverse",
            "assemble",
            "remove",
            "synthesize",
        )
        tokens = {
            key: f"op_{randomizer.getrandbits(64):016x}" for key in keys
        }
        risk_bias = (variant % 3) * 0.1
        self._operations = {
            operation.key: operation
            for operation in (
                _Operation(
                    "inspect",
                    Action(tokens["inspect"]),
                    "information_acquisition",
                    resource_cost=0.5,
                ),
                _Operation(
                    "decode",
                    Action(tokens["decode"]),
                    "goal_achievement",
                    ("information_acquisition",),
                    "parameter_dependency",
                    resource_cost=0.5,
                    terminal_family="information_route",
                ),
                _Operation(
                    "gather",
                    Action(tokens["gather"]),
                    "resource_acquisition",
                    resource_cost=2.0,
                ),
                _Operation(
                    "unlock",
                    Action(tokens["unlock"]),
                    "obstacle_removal",
                    ("resource_acquisition",),
                    resource_cost=1.5,
                    terminal_family="resource_route",
                ),
                _Operation(
                    "reroute",
                    Action(tokens["reroute"]),
                    "risk_reduction",
                    risk_delta=-0.5,
                ),
                _Operation(
                    "traverse",
                    Action(tokens["traverse"]),
                    "state_transition",
                    ("risk_reduction",),
                    resource_cost=0.25,
                    terminal_family="bypass_route",
                ),
                _Operation(
                    "assemble",
                    Action(tokens["assemble"]),
                    "tool_formation",
                    resource_cost=1.0,
                    risk_delta=0.2 + risk_bias,
                ),
                _Operation(
                    "remove",
                    Action(tokens["remove"]),
                    "obstacle_removal",
                    ("tool_formation",),
                    resource_cost=1.0,
                    risk_delta=0.1 + risk_bias,
                    terminal_family="tool_route",
                ),
                _Operation(
                    "synthesize",
                    Action(tokens["synthesize"]),
                    "goal_achievement",
                    ("information_acquisition", "resource_acquisition"),
                    "enablement",
                    resource_cost=0.1,
                    terminal_family="emergent_combination",
                ),
            )
        }
        self._action_to_key = {
            operation.action.signature: key
            for key, operation in self._operations.items()
        }
        self._completed: list[str] = []
        self._effects: list[str] = []
        self._terminal_family = ""
        self._resource_total = 0.0
        self._risk = 0.0
        self.terminal = False
        self._observation_salt = randomizer.getrandbits(64)

    def _available_operations(self) -> tuple[_Operation, ...]:
        if self.terminal:
            return ()
        completed = set(self._completed)
        effects = set(self._effects)
        result: list[_Operation] = []
        for key in ("inspect", "gather", "reroute", "assemble"):
            if key not in completed:
                result.append(self._operations[key])
        for key in ("decode", "unlock", "traverse", "remove", "synthesize"):
            operation = self._operations[key]
            if key not in completed and set(operation.prerequisites) <= effects:
                result.append(operation)
        return tuple(result)

    def _opaque_fact(self, key: str) -> str:
        digest = hashlib.sha256(
            f"{self._observation_salt}:{key}".encode("utf-8")
        ).hexdigest()
        return f"obs_{digest[:16]}"

    def snapshot(self) -> StateSnapshot:
        operations = self._available_operations()
        vector = tuple(
            float(key in self._completed)
            for key in sorted(self._operations)
        ) + (
            round(self._resource_total / 10.0, 6),
            round(self._risk, 6),
            float(self.terminal),
        )
        facts = frozenset(self._opaque_fact(key) for key in self._completed)
        return StateSnapshot(
            vector=vector,
            facts=facts,
            available_actions=tuple(item.action for item in operations),
            goal_progress=1.0 if self.terminal else 0.0,
        )

    def primitive_action_descriptions(self) -> dict[str, str]:
        """Human-facing descriptions without solution or viability labels."""
        descriptions = (
            "inspect one observable feature",
            "apply collected information",
            "collect one available resource",
            "apply a collected resource",
            "prepare a lower-risk passage",
            "enter a prepared passage",
            "assemble available material",
            "apply an assembled object",
            "combine two observed effects",
        )
        return {
            operation.action.signature: description
            for operation, description in zip(
                self._operations.values(), descriptions, strict=True
            )
        }

    def step(self, action: Action) -> CreativeStep:
        if self.terminal:
            raise RuntimeError("cannot step a terminal world")
        before = self.snapshot()
        available = {
            item.action.signature: item for item in self._available_operations()
        }
        operation = available.get(action.signature)
        if operation is None:
            return CreativeStep(before, error=True)
        self._completed.append(operation.key)
        self._effects.append(operation.effect)
        self._resource_total += operation.resource_cost
        self._risk = max(0.0, self._risk + operation.risk_delta)
        reward = 0.0
        if operation.terminal_family:
            self.terminal = True
            self._terminal_family = operation.terminal_family
            reward = 1.0
            if operation.effect != "goal_achievement":
                self._effects.append("goal_achievement")
        after = self.snapshot()
        before_actions = {
            candidate.signature for candidate in before.available_actions
        }
        event = {
            "effect": operation.effect,
            "prerequisites": list(operation.prerequisites),
            "relation": operation.relation,
        }
        events: list[Mapping[str, Any]] = [event]
        if self.terminal and operation.effect != "goal_achievement":
            events.append(
                {
                    "effect": "goal_achievement",
                    "prerequisites": [operation.effect],
                    "relation": "enablement",
                }
            )
        return CreativeStep(
            after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=tuple(
                candidate
                for candidate in after.available_actions
                if candidate.signature not in before_actions
            ),
            reward=reward,
            effect_events=tuple(events),
            resource_cost=operation.resource_cost,
            risk_delta=operation.risk_delta,
        )

    @property
    def analysis_solution_family(self) -> str:
        """Analysis-only family; never serialize it into agent observations."""
        return self._terminal_family

    @property
    def analysis_resource_total(self) -> float:
        return self._resource_total

    @property
    def analysis_risk(self) -> float:
        return self._risk


def canonicalize_effect_trace(
    events: Iterable[Mapping[str, Any]],
    *,
    solution_family: str = "",
) -> CausalEffectGraph:
    nodes: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    sequence: list[str] = []
    for event in events:
        effect = str(event.get("effect", "")).strip()
        if not effect:
            continue
        nodes.add(effect)
        sequence.append(effect)
        relation = str(event.get("relation", "prerequisite"))
        for prerequisite in event.get("prerequisites", ()):
            source = str(prerequisite)
            nodes.add(source)
            edges.add((source, effect, relation))
    return CausalEffectGraph(
        tuple(nodes),
        tuple(edges),
        tuple(sequence),
        solution_family=solution_family,
    )


def _set_distance(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left ^ right) / len(union) if union else 0.0


def graph_edit_distance(
    left: CausalEffectGraph, right: CausalEffectGraph
) -> float:
    node_distance = _set_distance(set(left.nodes), set(right.nodes))
    edge_distance = _set_distance(set(left.edges), set(right.edges))
    return (node_distance + edge_distance) / 2.0


def motif_jaccard_distance(
    left: CausalEffectGraph, right: CausalEffectGraph
) -> float:
    return _set_distance(set(left.motifs), set(right.motifs))


def prerequisite_edge_distance(
    left: CausalEffectGraph, right: CausalEffectGraph
) -> float:
    relations = {"prerequisite", "parameter_dependency", "enablement"}
    left_edges = {edge for edge in left.edges if edge[2] in relations}
    right_edges = {edge for edge in right.edges if edge[2] in relations}
    return _set_distance(left_edges, right_edges)


def solution_family_distance(
    left: CausalEffectGraph, right: CausalEffectGraph
) -> float:
    if not left.solution_family or not right.solution_family:
        return 0.0
    return float(left.solution_family != right.solution_family)


def effect_sequence_distance(
    left: CausalEffectGraph, right: CausalEffectGraph
) -> float:
    first = left.effect_sequence
    second = right.effect_sequence
    if not first and not second:
        return 0.0
    previous = list(range(len(second) + 1))
    for index, left_item in enumerate(first, start=1):
        current = [index]
        for second_index, right_item in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1]
                    + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1] / max(len(first), len(second), 1)


def strategy_distance_components(
    left: CausalEffectGraph, right: CausalEffectGraph
) -> dict[str, float]:
    return {
        "graph_edit": graph_edit_distance(left, right),
        "motif_jaccard": motif_jaccard_distance(left, right),
        "prerequisite_edges": prerequisite_edge_distance(left, right),
        "solution_family": solution_family_distance(left, right),
        "effect_sequence": effect_sequence_distance(left, right),
    }


def novelty_against_references(
    strategy: CausalEffectGraph,
    references: Sequence[CausalEffectGraph],
) -> dict[str, float]:
    if not references:
        return {
            "graph_edit": 1.0,
            "motif_jaccard": 1.0,
            "prerequisite_edges": 1.0,
            "solution_family": 1.0,
            "effect_sequence": 1.0,
            "aggregate": 1.0,
        }
    components = [
        strategy_distance_components(strategy, reference)
        for reference in references
    ]
    closest = min(components, key=lambda item: fmean(item.values()))
    return {**closest, "aggregate": fmean(closest.values())}


def strategy_record_from_trace(
    *,
    strategy_id: str,
    source_kind: str,
    research_seed: int,
    world_seed: int,
    success: bool,
    primitive_steps: int,
    errors: int,
    resources_used: float,
    risk_entries: int,
    events: Sequence[Mapping[str, Any]],
    solution_family: str,
    references: Sequence[CausalEffectGraph] = (),
    reusable_success_rate: float = 0.0,
    trace: Sequence[Mapping[str, Any]] = (),
) -> StrategyRecord:
    graph = canonicalize_effect_trace(
        events, solution_family=solution_family
    )
    novelty = novelty_against_references(graph, references)
    return StrategyRecord(
        strategy_id=strategy_id,
        source_kind=source_kind,
        research_seed=research_seed,
        world_seed=world_seed,
        success=success,
        primitive_steps=primitive_steps,
        errors=errors,
        resources_used=resources_used,
        risk_entries=risk_entries,
        graph=graph,
        trace=tuple(dict(item) for item in trace),
        novelty_components={
            key: value for key, value in novelty.items() if key != "aggregate"
        },
        reusable_success_rate=reusable_success_rate,
        novelty_score=novelty["aggregate"],
        valid=success,
    )
