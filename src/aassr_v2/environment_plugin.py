from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .action_plugins import ActionSchema
from .feature_memory import HashEmbeddingProvider
from .paper_v2_types import RawCausalObservation
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class ObservableEnvironmentTransition:
    """Agent-visible result of one primitive plugin action."""

    before: RawCausalObservation
    action: Action
    after: RawCausalObservation
    action_succeeded: bool
    terminal: bool
    final_sparse_reward: float

    def __post_init__(self) -> None:
        if self.final_sparse_reward != 0.0 and not self.terminal:
            raise ValueError("non-terminal environment reward is forbidden")
        if self.final_sparse_reward not in {0.0, 1.0}:
            raise ValueError("environment reward must be sparse binary reward")


@runtime_checkable
class EnvironmentPlugin(Protocol):
    """Environment-only boundary consumed by :class:`AASSRCore`.

    Implementations expose observations and primitive execution.  Solver,
    optimal-action, progress-shaping and private causal data are intentionally
    absent from this contract.
    """

    @property
    def plugin_id(self) -> str: ...

    def reset(self, world_seed: int) -> RawCausalObservation: ...

    def raw_observation(self) -> RawCausalObservation: ...

    def action_schemas(self) -> tuple[ActionSchema, ...]: ...

    def execute(self, action: Action) -> ObservableEnvironmentTransition: ...

    @property
    def terminal(self) -> bool: ...

    def final_sparse_reward(self) -> float: ...

    def render(self) -> str: ...


class CoreObservationEncoder:
    """Environment-neutral encoder over only agent-visible raw observations."""

    def __init__(self, *, dimension: int = 64) -> None:
        self._embedding = HashEmbeddingProvider(dimension)

    @staticmethod
    def tokens(observation: RawCausalObservation) -> tuple[str, ...]:
        tokens = [
            *(f"inventory:{key}={value}" for key, value in observation.inventory.items()),
            *(f"fact:{fact}" for fact in observation.observable_facts),
            *(
                f"spatial:{key}={value}"
                for key, value in observation.spatial_observations.items()
            ),
            *(
                "affordance:"
                + action
                + "="
                + json.dumps(tuple(values), separators=(",", ":"))
                for action, values in observation.action_affordances.items()
            ),
        ]
        if observation.last_action_succeeded is not None:
            tokens.append(
                "last_action_succeeded:"
                + str(bool(observation.last_action_succeeded)).lower()
            )
        if observation.terminal:
            tokens.append("terminal")
        if observation.terminal_reward > 0.0:
            tokens.append("terminal_success")
        return tuple(sorted(tokens))

    def encode(
        self,
        observation: RawCausalObservation,
        actions: tuple[Action, ...],
    ) -> StateSnapshot:
        tokens = self.tokens(observation)
        embedded = self._embedding.embed(tokens)
        numeric = (
            float(observation.resource_cost),
            float(observation.health),
            float(observation.damage),
            0.0
            if observation.last_action_succeeded is None
            else (1.0 if observation.last_action_succeeded else -1.0),
            float(observation.terminal_reward),
            1.0 if observation.terminal else 0.0,
        )
        facts = frozenset(
            token
            for token in tokens
            if token.startswith(("fact:", "spatial:", "terminal"))
        )
        return StateSnapshot(
            vector=tuple(embedded) + numeric,
            facts=facts,
            available_actions=actions,
            # Explicit or normalized goal progress is never synthesized.
            goal_progress=0.0,
            metadata={"observation_contract": "raw-causal-observation-v2"},
        )


class CoreEnvironmentSession:
    """Adapts a plugin to the legacy evaluator without exposing the plugin."""

    def __init__(
        self,
        plugin: EnvironmentPlugin,
        encoder: CoreObservationEncoder,
    ) -> None:
        self._plugin = plugin
        self._encoder = encoder

    def _actions(self, observation: RawCausalObservation) -> tuple[Action, ...]:
        available = set(observation.available_actions)
        schemas = {
            schema.action_id: schema for schema in self._plugin.action_schemas()
        }
        missing = available - set(schemas)
        if missing:
            raise ValueError(f"observation has actions without schemas: {sorted(missing)}")
        return tuple(
            schemas[action_id].build({})
            for action_id in observation.available_actions
        )

    def snapshot(self) -> StateSnapshot:
        observation = self._plugin.raw_observation()
        return self._encoder.encode(observation, self._actions(observation))

    def raw_observation(self) -> RawCausalObservation:
        return self._plugin.raw_observation()

    def step(self, action: Action):
        from .action_plugins import PluginOutcome

        transition = self._plugin.execute(action)
        before_actions = self._actions(transition.before)
        after_actions = self._actions(transition.after)
        before = self._encoder.encode(transition.before, before_actions)
        after = self._encoder.encode(transition.after, after_actions)
        return PluginOutcome(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=tuple(
                candidate
                for candidate in after.available_actions
                if candidate.signature
                not in {item.signature for item in before.available_actions}
            ),
            error=not transition.action_succeeded,
            error_code=None if transition.action_succeeded else "action_failed",
            cost=1.0,
            reward=transition.final_sparse_reward,
            terminal=transition.terminal,
            raw={
                "action_succeeded": transition.action_succeeded,
                "terminal": transition.terminal,
                "final_sparse_reward": transition.final_sparse_reward,
            },
        )
