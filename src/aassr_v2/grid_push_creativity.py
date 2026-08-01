from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .creativity import novelty_against_references
from .grid_push_world import (
    GridPushSpec,
    GridPushWorld,
    GridSolution,
    SolverResult,
)
from .paper_types import CausalEffectGraph
from .paper_v2_protocol import sha256_json


@dataclass(frozen=True, slots=True)
class NormalizedGridStrategy:
    success: bool
    graph: CausalEffectGraph
    primitive_steps: int
    failed_actions: int
    meaningful_events: int
    block_moves: int
    irreversible_risk: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "graph": self.graph.to_dict(),
            "primitive_steps": self.primitive_steps,
            "failed_actions": self.failed_actions,
            "meaningful_events": self.meaningful_events,
            "block_moves": self.block_moves,
            "irreversible_risk": self.irreversible_risk,
        }


def normalize_grid_strategy(
    spec: GridPushSpec, actions: Sequence[str]
) -> NormalizedGridStrategy:
    world = GridPushWorld(spec)
    sequence: list[str] = []
    failed = 0
    block_moves = 0
    risk = 0
    for action in actions:
        if world.terminal:
            break
        before_open = world.open_doors()
        before_filled = world.analysis_private_state.filled_pits
        outcome = world.step(action)
        events = world.analysis_last_events
        if not outcome.action_succeeded:
            failed += 1
            continue
        moved = [event for event in events if event.kind == "block_moved"]
        block_moves += len(moved)
        for event in moved:
            if event.target in spec.pits:
                sequence.append("block_enters_pit")
                risk += 1
            elif event.target in spec.plates:
                sequence.append("block_occupies_plate")
        for event in events:
            if event.kind == "pit_filled":
                sequence.append("pit_becomes_passable")
            elif event.kind == "door_opened":
                sequence.append("door_becomes_passable")
            elif event.kind == "player_moved" and event.target in spec.doors and event.target in before_open:
                sequence.append("traverse_open_door")
            elif event.kind == "player_moved" and event.target in spec.pits and (
                event.target in before_filled
                or any(item.kind == "pit_filled" and item.target == event.target for item in events)
            ):
                sequence.append("traverse_filled_pit")
            elif event.kind == "goal_reached":
                sequence.append("goal_reached")
    # Repeated observations, failed moves, ordinary walking, and adjacent
    # duplicate milestones cannot manufacture novelty.
    compact: list[str] = []
    for item in sequence:
        if not compact or compact[-1] != item:
            compact.append(item)
    nodes = set(compact)
    edges = {
        (left, right, "observed_enablement")
        for left, right in zip(compact, compact[1:])
    }
    if "block_enters_pit" in nodes and "pit_becomes_passable" in nodes:
        edges.add(("block_enters_pit", "pit_becomes_passable", "physical_effect"))
    if "pit_becomes_passable" in nodes and "traverse_filled_pit" in nodes:
        edges.add(("pit_becomes_passable", "traverse_filled_pit", "enablement"))
    if "block_occupies_plate" in nodes and "door_becomes_passable" in nodes:
        edges.add(("block_occupies_plate", "door_becomes_passable", "physical_effect"))
    if "door_becomes_passable" in nodes and "traverse_open_door" in nodes:
        edges.add(("door_becomes_passable", "traverse_open_door", "enablement"))
    graph = CausalEffectGraph(tuple(nodes), tuple(edges), tuple(compact))
    return NormalizedGridStrategy(
        success=world.analysis_private_state.success,
        graph=graph,
        primitive_steps=min(len(actions), world.analysis_private_state.step_count),
        failed_actions=failed,
        meaningful_events=len(compact),
        block_moves=block_moves,
        irreversible_risk=risk,
    )


@dataclass(frozen=True, slots=True)
class FrozenSolverReference:
    world_sha256: str
    reference_sha256: str
    graphs: tuple[CausalEffectGraph, ...]
    entries: tuple[Mapping[str, Any], ...]


def _reference_entries(
    spec: GridPushSpec, result: SolverResult
) -> list[dict[str, Any]]:
    if not result.solutions:
        raise ValueError("cannot freeze an empty solver result")
    chosen: dict[str, tuple[GridSolution, set[str], CausalEffectGraph]] = {}

    def add(solution: GridSolution, reason: str) -> None:
        normalized = normalize_grid_strategy(spec, solution.actions)
        key = sha256_json(normalized.graph.to_dict())
        if key in chosen:
            chosen[key][1].add(reason)
        else:
            chosen[key] = (solution, {reason}, normalized.graph)

    minimum_actions = min(len(solution.actions) for solution in result.solutions)
    minimum_blocks = min(solution.block_moves for solution in result.solutions)
    minimum_risk = min(solution.irreversible_risk for solution in result.solutions)
    for solution in result.solutions:
        if len(solution.actions) == minimum_actions:
            add(solution, "minimum_actions")
        if solution.block_moves == minimum_blocks:
            add(solution, "minimum_block_moves")
        if solution.irreversible_risk == minimum_risk:
            add(solution, "minimum_risk")
        add(solution, "structural_representative")
    entries = []
    for graph_hash, (solution, reasons, graph) in sorted(chosen.items()):
        entries.append(
            {
                "graph_sha256": graph_hash,
                "graph": graph.to_dict(),
                "selection_reasons": sorted(reasons),
                "action_count": len(solution.actions),
                "block_moves": solution.block_moves,
                "irreversible_risk": solution.irreversible_risk,
                "actions_analysis_only": list(solution.actions),
            }
        )
    return entries


def freeze_solver_reference(
    path: str | Path,
    *,
    world: GridPushWorld,
    solver_result: SolverResult,
    maximum_actions: int,
) -> dict[str, Any]:
    entries = _reference_entries(world.analysis_spec, solver_result)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_agent_run",
        "source": "bounded_grid_solver",
        "world_sha256": world.world_sha256,
        "causal_law_sha256": world.causal_law_sha256,
        "maximum_actions": int(maximum_actions),
        "solver_explored_states": solver_result.explored_states,
        "solver_truncated": solver_result.truncated,
        "references": entries,
    }
    payload["reference_sha256"] = sha256_json(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return payload


def load_solver_reference(
    path: str | Path, *, expected_world_sha256: str | None = None
) -> FrozenSolverReference:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = str(payload.pop("reference_sha256"))
    if payload.get("status") != "frozen_before_agent_run":
        raise ValueError("solver reference is not frozen")
    if sha256_json(payload) != expected_hash:
        raise ValueError("solver reference hash mismatch")
    if expected_world_sha256 and payload.get("world_sha256") != expected_world_sha256:
        raise ValueError("solver reference belongs to another world")
    entries = tuple(dict(item) for item in payload["references"])
    return FrozenSolverReference(
        world_sha256=str(payload["world_sha256"]),
        reference_sha256=expected_hash,
        graphs=tuple(CausalEffectGraph.from_dict(item["graph"]) for item in entries),
        entries=entries,
    )


@dataclass(frozen=True, slots=True)
class GridCreativityScore:
    success: bool
    novelty: float
    utility: float
    reproducibility: float
    creativity: float
    novelty_components: Mapping[str, float]
    final_candidate: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_grid_creativity(
    strategy: NormalizedGridStrategy,
    *,
    reference: FrozenSolverReference,
    minimum_actions: int,
    reproducibility: float,
    novelty_threshold: float = 0.20,
    utility_threshold: float = 0.80,
    reproduction_threshold: float = 0.50,
) -> GridCreativityScore:
    if not strategy.success:
        return GridCreativityScore(
            False, 0.0, 0.0, float(reproducibility), 0.0, {}, False,
            ("strategy_did_not_reach_goal",),
        )
    novelty = novelty_against_references(strategy.graph, reference.graphs)
    movement_efficiency = min(1.0, minimum_actions / max(1, strategy.primitive_steps))
    error_efficiency = 1.0 / (1.0 + strategy.failed_actions)
    risk_efficiency = 1.0 / (1.0 + strategy.irreversible_risk)
    utility = fmean((movement_efficiency, error_efficiency, risk_efficiency))
    creativity = novelty["aggregate"] * utility * float(reproducibility)
    reasons = []
    if novelty["aggregate"] <= novelty_threshold:
        reasons.append("novelty_not_above_frozen_threshold")
    if utility < utility_threshold:
        reasons.append("utility_below_threshold")
    if reproducibility < reproduction_threshold:
        reasons.append("not_reproduced")
    return GridCreativityScore(
        True,
        novelty["aggregate"],
        utility,
        float(reproducibility),
        creativity,
        {key: value for key, value in novelty.items() if key != "aggregate"},
        not reasons,
        tuple(reasons),
    )
