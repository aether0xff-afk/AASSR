from __future__ import annotations

from typing import Any

from . import toolgrid_factorial_masked as env


def clone_toolgrid_world(world: Any) -> Any:
    """Clone only explicit ToolGrid environment state.

    Action objects contain immutable mapping proxies, so generic deepcopy cannot
    serialize them. Reconstructing from the map seed and copying the mutable
    episode fields gives an exact environment clone without copying actions.
    """

    clone = env.ToolGridWorld(
        world.seed,
        grid_size=world.grid_size,
        action_count=world.action_count,
    )
    clone.agent = tuple(world.agent)
    clone.phase = int(world.phase)
    clone.success = bool(world.success)
    clone.failed = bool(world.failed)
    clone.used_cells = set(world.used_cells)
    clone.steps = int(world.steps)
    return clone


def _deepcopy_toolgrid_world(world: Any, memo: dict[int, Any]) -> Any:
    clone = clone_toolgrid_world(world)
    memo[id(world)] = clone
    return clone


# The diagnostic module intentionally calls copy.deepcopy so every
# counterfactual is isolated. Register the explicit clone implementation on the
# benchmark class rather than attempting to pickle immutable Action internals.
env.ToolGridWorld.__deepcopy__ = _deepcopy_toolgrid_world
