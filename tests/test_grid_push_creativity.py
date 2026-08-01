from __future__ import annotations

from pathlib import Path

import pytest

from aassr_v2.grid_push_creativity import (
    FrozenSolverReference,
    NormalizedGridStrategy,
    freeze_solver_reference,
    load_solver_reference,
    normalize_grid_strategy,
    score_grid_creativity,
)
from aassr_v2.grid_push_world import GridPushSpec, GridPushWorld, solve_grid_world
from aassr_v2.paper_types import CausalEffectGraph


def _boundary(width: int, height: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (x, y)
        for x in range(width)
        for y in range(height)
        if x in {0, width - 1} or y in {0, height - 1}
    )


def _pit_spec(row: int = 2) -> GridPushSpec:
    return GridPushSpec(
        width=6,
        height=5,
        walls=_boundary(6, 5),
        start=(1, row),
        goal=(4, row),
        blocks=frozenset({(2, row)}),
        pits=frozenset({(3, row)}),
    )


def _plate_spec() -> GridPushSpec:
    walls = set(_boundary(7, 6))
    walls.update((4, y) for y in range(1, 5))
    walls.remove((4, 2))
    plate, door = (3, 3), (4, 2)
    return GridPushSpec(
        width=7,
        height=6,
        walls=frozenset(walls),
        start=(1, 3),
        goal=(5, 2),
        blocks=frozenset({(2, 3)}),
        plates=frozenset({plate}),
        doors=frozenset({door}),
        plate_links={plate: (door,)},
    )


def test_round_trips_and_failed_moves_do_not_create_novelty() -> None:
    spec = _pit_spec()
    direct = normalize_grid_strategy(spec, ("MOVE_EAST", "MOVE_EAST", "MOVE_EAST"))
    noisy = normalize_grid_strategy(
        spec,
        (
            "MOVE_WEST",  # wall failure
            "MOVE_NORTH", "MOVE_SOUTH",  # useless round trip
            "MOVE_EAST", "MOVE_EAST", "MOVE_EAST",
        ),
    )
    assert direct.success and noisy.success
    assert direct.graph == noisy.graph


def test_action_and_block_identity_do_not_enter_graph() -> None:
    first = normalize_grid_strategy(
        _pit_spec(2), ("MOVE_EAST", "MOVE_EAST", "MOVE_EAST")
    )
    second = normalize_grid_strategy(
        _pit_spec(3), ("MOVE_EAST", "MOVE_EAST", "MOVE_EAST")
    )
    assert first.graph == second.graph
    payload = repr(first.graph.to_dict()).lower()
    assert "move_east" not in payload
    assert "block_id" not in payload


def test_pit_and_plate_door_routes_are_structurally_distinct() -> None:
    pit = normalize_grid_strategy(
        _pit_spec(), ("MOVE_EAST", "MOVE_EAST", "MOVE_EAST")
    )
    plate = normalize_grid_strategy(
        _plate_spec(),
        ("MOVE_EAST", "MOVE_NORTH", "MOVE_EAST", "MOVE_EAST", "MOVE_EAST"),
    )
    assert plate.success
    assert pit.graph != plate.graph
    assert "pit_becomes_passable" in pit.graph.nodes
    assert "door_becomes_passable" in plate.graph.nodes


def test_failed_strategy_is_never_creative() -> None:
    strategy = normalize_grid_strategy(_pit_spec(), ("MOVE_WEST",))
    reference = FrozenSolverReference(
        "world", "hash", (strategy.graph,), ({},)
    )
    score = score_grid_creativity(
        strategy, reference=reference, minimum_actions=3, reproducibility=1.0
    )
    assert not score.final_candidate
    assert score.creativity == 0.0


def test_novel_but_inefficient_unreproduced_strategy_is_rejected() -> None:
    reference_graph = CausalEffectGraph(("reference",), (), ("reference",))
    novel_graph = CausalEffectGraph(("novel",), (), ("novel",))
    reference = FrozenSolverReference(
        "world", "hash", (reference_graph,), ({},)
    )
    strategy = NormalizedGridStrategy(True, novel_graph, 100, 5, 1, 10, 2)
    score = score_grid_creativity(
        strategy, reference=reference, minimum_actions=5, reproducibility=0.0
    )
    assert score.novelty > 0.0
    assert not score.final_candidate
    assert "utility_below_threshold" in score.rejection_reasons
    assert "not_reproduced" in score.rejection_reasons


def test_solver_reference_is_frozen_before_agent_use(tmp_path: Path) -> None:
    world = GridPushWorld(_pit_spec())
    result = solve_grid_world(world, maximum_actions=10)
    path = tmp_path / "solver_reference.json"
    payload = freeze_solver_reference(
        path, world=world, solver_result=result, maximum_actions=10
    )
    loaded = load_solver_reference(path, expected_world_sha256=world.world_sha256)
    assert loaded.reference_sha256 == payload["reference_sha256"]
    with pytest.raises(FileExistsError):
        freeze_solver_reference(
            path, world=world, solver_result=result, maximum_actions=10
        )
