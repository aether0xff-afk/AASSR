from __future__ import annotations

from collections.abc import Callable

from .action_plugins import ActionSchema
from .environment_plugin import EnvironmentPlugin, ObservableEnvironmentTransition
from .grid_push_world import GridPushSpec, GridPushWorld, MOVE_DELTAS
from .paper_v2_types import RawCausalObservation
from .types import Action


class GridPushEnvironmentPlugin(EnvironmentPlugin):
    """GridPush physics adapter with no solver or solution-facing API."""

    def __init__(self, world_factory: Callable[[int], GridPushSpec]) -> None:
        self._world_factory = world_factory
        self._world: GridPushWorld | None = None
        self._schemas = tuple(
            ActionSchema(
                plugin_id=self.plugin_id,
                action_id=action,
                capability_tags=frozenset(
                    {"movement", action.removeprefix("MOVE_").lower()}
                ),
                description=(
                    "Move one cell " + action.removeprefix("MOVE_").lower()
                ),
            )
            for action in MOVE_DELTAS
        )

    @property
    def plugin_id(self) -> str:
        return "grid_push"

    def _require_world(self) -> GridPushWorld:
        if self._world is None:
            raise RuntimeError("plugin must be reset before use")
        return self._world

    def reset(self, world_seed: int) -> RawCausalObservation:
        self._world = GridPushWorld(self._world_factory(int(world_seed)))
        return self._world.observe()

    def raw_observation(self) -> RawCausalObservation:
        return self._require_world().observe()

    def action_schemas(self) -> tuple[ActionSchema, ...]:
        return self._schemas

    def execute(self, action: Action) -> ObservableEnvironmentTransition:
        world = self._require_world()
        if action.metadata.get("plugin_id") != self.plugin_id:
            raise ValueError("action belongs to a different environment plugin")
        if action.verb_name not in MOVE_DELTAS or action.parameters:
            raise ValueError("GridPush accepts only parameterless cardinal movement")
        before = world.observe()
        outcome = world.step(action.verb_name)
        return ObservableEnvironmentTransition(
            before=before,
            action=action,
            after=outcome.observation,
            action_succeeded=outcome.action_succeeded,
            terminal=outcome.observation.terminal,
            final_sparse_reward=outcome.reward,
        )

    @property
    def terminal(self) -> bool:
        return self._require_world().terminal

    def final_sparse_reward(self) -> float:
        observation = self.raw_observation()
        return float(observation.terminal_reward if observation.terminal else 0.0)

    def render(self) -> str:
        return self._require_world().render_ascii()
