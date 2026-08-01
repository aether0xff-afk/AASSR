from __future__ import annotations

from aassr_v2.grid_push_world import (
    GridPushSpec,
    GridPushWorld,
    solve_grid_world,
)
from aassr_v2.grid_push_development import _pressure_plate_rule_trace


def _boundary(width: int, height: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (x, y)
        for x in range(width)
        for y in range(height)
        if x in {0, width - 1} or y in {0, height - 1}
    )


def _world(
    *,
    width: int = 6,
    height: int = 5,
    player=(1, 2),
    goal=(4, 3),
    blocks=((2, 2),),
    extra_walls=(),
    pits=(),
    plates=(),
    doors=(),
    links=None,
) -> GridPushWorld:
    return GridPushWorld(
        GridPushSpec(
            width=width,
            height=height,
            walls=_boundary(width, height) | frozenset(extra_walls),
            start=player,
            goal=goal,
            blocks=frozenset(blocks),
            pits=frozenset(pits),
            plates=frozenset(plates),
            doors=frozenset(doors),
            plate_links=links or {},
        )
    )


def test_block_moves_one_cell_when_space_behind_is_passable() -> None:
    world = _world()
    outcome = world.step("MOVE_EAST")
    assert outcome.action_succeeded
    assert world.analysis_private_state.player == (2, 2)
    assert world.analysis_private_state.blocks == frozenset({(3, 2)})


def test_block_does_not_move_when_wall_is_behind_it() -> None:
    world = _world(extra_walls=((3, 2),))
    before = world.state_key()
    outcome = world.step("MOVE_EAST")
    assert not outcome.action_succeeded
    assert world.state_key() == before


def test_two_blocks_cannot_be_pushed_together() -> None:
    world = _world(blocks=((2, 2), (3, 2)))
    before = world.state_key()
    assert not world.step("MOVE_EAST").action_succeeded
    assert world.state_key() == before


def test_blocks_cannot_be_pulled() -> None:
    world = _world()
    world.step("MOVE_EAST")
    world.step("MOVE_WEST")
    assert world.analysis_private_state.player == (1, 2)
    assert world.analysis_private_state.blocks == frozenset({(3, 2)})


def test_block_fills_pit_and_pit_becomes_passable_irreversibly() -> None:
    world = _world(pits=((3, 2),))
    world.step("MOVE_EAST")
    assert not world.analysis_private_state.blocks
    assert world.analysis_private_state.filled_pits == frozenset({(3, 2)})
    assert any(event.kind == "pit_filled" for event in world.analysis_last_events)
    assert world.step("MOVE_EAST").action_succeeded
    assert world.analysis_private_state.player == (3, 2)


def test_block_on_plate_keeps_linked_door_open() -> None:
    plate, door = (3, 2), (4, 1)
    world = _world(plates=(plate,), doors=(door,), links={plate: (door,)})
    world.step("MOVE_EAST")
    assert plate in world.pressed_plates()
    assert door in world.open_doors()
    assert world.observe().spatial_observations["cell:4,1"] == "door_open"


def test_door_closes_after_block_and_player_leave_plate() -> None:
    plate, door = (3, 2), (4, 1)
    world = _world(plates=(plate,), doors=(door,), links={plate: (door,)})
    world.step("MOVE_EAST")
    world.step("MOVE_EAST")
    assert door in world.open_doors()  # player now occupies the plate
    world.step("MOVE_SOUTH")
    assert door not in world.open_doors()
    assert any(event.kind == "door_closed" for event in world.analysis_last_events)


def test_wrong_push_can_create_a_real_unsolvable_state() -> None:
    width, height = 7, 6
    walls = set(_boundary(width, height))
    walls.update((3, y) for y in range(1, height - 1))
    walls.remove((3, 2))
    world = GridPushWorld(
        GridPushSpec(
            width=width,
            height=height,
            walls=frozenset(walls),
            start=(1, 3),
            goal=(5, 2),
            blocks=frozenset({(2, 3)}),
            pits=frozenset({(3, 2)}),
        )
    )
    assert solve_grid_world(world, maximum_actions=20).solutions
    world.step("MOVE_NORTH")
    world.step("MOVE_EAST")
    world.step("MOVE_SOUTH")  # pushes the only block against the lower wall
    assert world.analysis_private_state.blocks == frozenset({(2, 4)})
    assert not solve_grid_world(world, maximum_actions=30).solutions


def test_pressure_rule_evidence_trace_opens_then_closes_door() -> None:
    trace = _pressure_plate_rule_trace()
    event_kinds = [
        event["kind"]
        for step in trace["steps"]
        for event in step["analysis_events"]
    ]
    assert "block_moved" in event_kinds
    assert "plate_pressed" in event_kinds
    assert "door_opened" in event_kinds
    assert "plate_released" in event_kinds
    assert "door_closed" in event_kinds
