from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from ..types import Action


class ValueKind(str, Enum):
    """Pure data-shape categories exposed by an environment plugin.

    These values describe *how data is represented*, never what the data means
    for solving a task.
    """

    BOOLEAN = "boolean"
    SCALAR = "scalar"
    CATEGORICAL = "categorical"
    ENTITY = "entity"
    TEXT = "text"
    SET = "set"
    MAPPING = "mapping"
    BYTES = "bytes"


class TemporalKind(str, Enum):
    """Mechanical lifetime of an observation channel."""

    STATE = "state"
    EVENT = "event"
    COUNTER = "counter"
    MEASUREMENT = "measurement"


def _value_matches_kind(value: Any, kind: ValueKind) -> bool:
    if value is None:
        return True
    if kind is ValueKind.BOOLEAN:
        return isinstance(value, bool)
    if kind is ValueKind.SCALAR:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in {ValueKind.CATEGORICAL, ValueKind.ENTITY, ValueKind.TEXT}:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)
    if kind is ValueKind.SET:
        return isinstance(value, (tuple, list, set, frozenset))
    if kind is ValueKind.MAPPING:
        return isinstance(value, Mapping)
    if kind is ValueKind.BYTES:
        return isinstance(value, (bytes, bytearray))
    return True


def _validate_value_space(value_space: str | None, *, owner: str) -> None:
    if value_space is not None and not str(value_space).strip():
        raise ValueError(f"{owner} value_space must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class ObservationField:
    name: str
    kind: ValueKind
    temporal: TemporalKind = TemporalKind.STATE
    enum_values: tuple[str, ...] = ()
    item_kind: ValueKind | None = None
    value_space: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation field name must not be empty")
        if self.kind is ValueKind.CATEGORICAL and len(set(self.enum_values)) != len(
            self.enum_values
        ):
            raise ValueError(f"duplicate categorical values for {self.name!r}")
        if self.kind is not ValueKind.SET and self.item_kind is not None:
            raise ValueError("item_kind is valid only for set observations")
        _validate_value_space(self.value_space, owner=f"observation {self.name!r}")


@dataclass(frozen=True, slots=True)
class ActionParameter:
    name: str
    kind: ValueKind
    required: bool = True
    enum_values: tuple[str, ...] = ()
    value_space: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action parameter name must not be empty")
        if self.kind is ValueKind.CATEGORICAL and len(set(self.enum_values)) != len(
            self.enum_values
        ):
            raise ValueError(f"duplicate categorical values for {self.name!r}")
        _validate_value_space(self.value_space, owner=f"parameter {self.name!r}")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_id: str
    parameters: tuple[ActionParameter, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must not be empty")
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate parameter names in action {self.action_id!r}")

    def validate(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        by_name = {item.name: item for item in self.parameters}
        errors = [
            f"unknown parameter {name!r}"
            for name in arguments
            if name not in by_name
        ]
        for parameter in self.parameters:
            if parameter.required and parameter.name not in arguments:
                errors.append(f"missing parameter {parameter.name!r}")
            if parameter.name not in arguments:
                continue
            value = arguments[parameter.name]
            if not _value_matches_kind(value, parameter.kind):
                errors.append(
                    f"invalid type for {parameter.name!r}: expected {parameter.kind.value}"
                )
                continue
            if (
                parameter.kind is ValueKind.CATEGORICAL
                and parameter.enum_values
                and str(value) not in parameter.enum_values
            ):
                errors.append(
                    f"invalid categorical value for {parameter.name!r}: {value!r}"
                )
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class PluginSchema:
    plugin_id: str
    version: str
    observations: tuple[ObservationField, ...]
    actions: tuple[ActionSpec, ...]

    def __post_init__(self) -> None:
        if not self.plugin_id or not self.version:
            raise ValueError("plugin_id and version must not be empty")
        observation_names = [item.name for item in self.observations]
        action_names = [item.action_id for item in self.actions]
        if len(observation_names) != len(set(observation_names)):
            raise ValueError("duplicate observation field")
        if len(action_names) != len(set(action_names)):
            raise ValueError("duplicate action id")

    @property
    def observation_map(self) -> Mapping[str, ObservationField]:
        return MappingProxyType({item.name: item for item in self.observations})

    @property
    def action_map(self) -> Mapping[str, ActionSpec]:
        return MappingProxyType({item.action_id: item for item in self.actions})


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """One mechanically executable command selected/generated by the Core.

    The plugin receives this command only for syntax validation and real I/O.
    It has no authority to score the command or reinterpret its task value.
    """

    action_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def to_action(self, schema: PluginSchema) -> Action:
        try:
            spec = schema.action_map[self.action_id]
        except KeyError as exc:
            raise ValueError(f"unknown action_id {self.action_id!r}") from exc
        errors = spec.validate(self.arguments)
        if errors:
            raise ValueError("; ".join(errors))
        return Action(
            self.action_id,
            parameters=dict(self.arguments),
            metadata={
                "plugin_id": schema.plugin_id,
                "plugin_version": schema.version,
                "schema_id": f"{schema.plugin_id}:{self.action_id}",
            },
        )


@dataclass(frozen=True, slots=True)
class PluginObservation:
    """Only public environment data declared by the plugin schema.

    Plugins do not return candidate actions.  The Core derives candidate commands
    from the action syntax plus publicly observed typed values, preventing a
    plugin from strategically ranking or filtering the learner's choices.
    """

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class PluginStepResult:
    observation: PluginObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    error: bool = False
    error_code: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )
        if self.terminated and self.truncated:
            raise ValueError("a step cannot be both terminated and truncated")


def validate_observation(schema: PluginSchema, observation: PluginObservation) -> None:
    """Validate only declared public data shape, never task meaning."""

    declared = schema.observation_map
    unknown = tuple(sorted(set(observation.values) - set(declared)))
    if unknown:
        raise ValueError(f"plugin returned undeclared observation fields: {unknown!r}")
    for name, value in observation.values.items():
        field = declared[name]
        if not _value_matches_kind(value, field.kind):
            raise TypeError(
                f"observation field {name!r} has invalid type for {field.kind.value}"
            )
        if field.kind is ValueKind.CATEGORICAL and field.enum_values and value is not None:
            if str(value) not in field.enum_values:
                raise ValueError(
                    f"observation field {name!r} has undeclared categorical value {value!r}"
                )


# Plugin diagnostics are observability-only.  These names are consumed by the
# Core adapter as control channels and therefore must never be shadowable by a
# plugin-provided diagnostics mapping.
RESERVED_PLUGIN_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {"external_reward", "terminated", "truncated"}
)


def validate_step_result(schema: PluginSchema, result: PluginStepResult) -> None:
    """Validate the mechanical Plugin -> Core control boundary.

    This intentionally validates shape and control-channel integrity only.  It
    never decides whether an observation or action is useful for the task.
    """

    if not isinstance(result, PluginStepResult):
        raise TypeError("plugin reset/step must return PluginStepResult")
    validate_observation(schema, result.observation)

    if not isinstance(result.reward, Real) or isinstance(result.reward, bool):
        raise TypeError("plugin reward must be a real number")
    if not math.isfinite(float(result.reward)):
        raise ValueError("plugin reward must be finite")

    for name in ("terminated", "truncated", "error"):
        if not isinstance(getattr(result, name), bool):
            raise TypeError(f"plugin {name} must be bool")
    if result.terminated and result.truncated:
        raise ValueError("a step cannot be both terminated and truncated")
    if result.error_code is not None and not isinstance(result.error_code, str):
        raise TypeError("plugin error_code must be str or None")

    collisions = RESERVED_PLUGIN_DIAGNOSTIC_KEYS.intersection(result.diagnostics)
    if collisions:
        raise ValueError(
            "plugin diagnostics may not shadow Core control channels: "
            + ", ".join(sorted(collisions))
        )


class MinimalRuntimePlugin(Protocol):
    """The complete environment-facing authority allowed to a plugin.

    The plugin may describe syntax, public information types, and how to execute
    a command selected by the Core.  A ``value_space`` may additionally state
    mechanical compatibility (for example URL values may fill URL parameters).
    It must never encode whether a value is good, bad, target, decoy, or progress.
    The plugin does not return a strategic candidate list and must not define
    state representations, semantic identities, value functions, world models,
    planning scores, action priorities, or task heuristics.
    """

    @property
    def schema(self) -> PluginSchema: ...

    def reset(self, *, seed: int | None = None) -> PluginStepResult: ...

    def step(self, command: ActionCommand) -> PluginStepResult: ...


FORBIDDEN_PLUGIN_AUTHORITIES: tuple[str, ...] = (
    "state_vector",
    "state_key",
    "semantic_state_identity",
    "action_structure",
    "decode_state",
    "prediction_score",
    "install_world_model",
    "rank_actions",
    "score_action",
    "shape_reward",
)


def validate_minimal_plugin(plugin: MinimalRuntimePlugin) -> None:
    """Fail early when an object exposes powers reserved for the Core."""

    schema = plugin.schema
    if not isinstance(schema, PluginSchema):
        raise TypeError("plugin.schema must be a PluginSchema")
    for name in FORBIDDEN_PLUGIN_AUTHORITIES:
        if hasattr(plugin, name):
            raise TypeError(
                f"minimal plugin exposes Core-owned authority {name!r}"
            )
