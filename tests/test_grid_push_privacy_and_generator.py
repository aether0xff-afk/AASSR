from __future__ import annotations

import json

from aassr_v2.grid_push_world import (
    GRID_PUSH_LAW_SHA256,
    GridPushWorld,
    ProceduralGridPushGenerator,
)


def test_observation_has_tiles_but_no_solution_or_link_labels() -> None:
    spec, certification, _ = ProceduralGridPushGenerator().generate(
        6001, maximum_actions=30, random_rollouts=20
    )
    world = GridPushWorld(spec)
    payload = json.dumps(world.observe().to_dict(), sort_keys=True).lower()
    assert "block" in payload
    assert "pressure_plate" in payload
    assert "door_closed" in payload
    for forbidden in (
        "required", "useful", "correct", "bridge_block", "target_plate",
        "solution_door", "plate_links", "solver", "goal_distance",
        "goal_progress", "viability", "optimal",
    ):
        assert forbidden not in payload
    assert certification.private_observation_leaks == 0


def test_generator_changes_layout_but_preserves_physics_and_certification() -> None:
    generated = [
        ProceduralGridPushGenerator().generate(
            seed, maximum_actions=30, random_rollouts=20
        )
        for seed in (6001, 6002, 6003)
    ]
    assert len({cert.world_sha256 for _, cert, _ in generated}) == 3
    assert all(cert.causal_law_sha256 == GRID_PUSH_LAW_SHA256 for _, cert, _ in generated)
    assert all(cert.solvable and cert.adequate for _, cert, _ in generated)
    assert all(cert.minimum_actions and cert.minimum_actions >= 4 for _, cert, _ in generated)
    assert all(cert.bounded_dead_end_count > 0 for _, cert, _ in generated)
    assert all(cert.requires_block_push for _, cert, _ in generated)


def test_solver_paths_are_not_serialized_into_agent_observation() -> None:
    spec, _, result = ProceduralGridPushGenerator().generate(
        6004, maximum_actions=30, random_rollouts=20
    )
    assert result.solutions
    observation = repr(GridPushWorld(spec).observe().to_dict())
    assert repr(result.solutions[0].actions) not in observation
