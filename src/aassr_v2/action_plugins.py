from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from .types import Action, StateSnapshot

_MISSING = object()


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One plugin-owned command slot without solution semantics."""

    name: str
    role: str
    value_type: str = "any"
    required: bool = True
    default: Any = _MISSING
    repeatable: bool = False

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING


@dataclass(frozen=True, slots=True)
class ActionSchema:
    plugin_id: str
    action_id: str
    parameters: tuple[ParameterSpec, ...] = ()
    capability_tags: frozenset[str] = frozenset()
    description: str = ""

    @property
    def schema_id(self) -> str:
        return f"{self.plugin_id}:{self.action_id}"

    def validate(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        known = {parameter.name for parameter in self.parameters}
        errors = [f"unknown parameter: {name}" for name in arguments if name not in known]
        for parameter in self.parameters:
            if parameter.required and parameter.name not in arguments and not parameter.has_default:
                errors.append(f"missing parameter: {parameter.name}")
            if parameter.name in arguments and not parameter.repeatable:
                value = arguments[parameter.name]
                if isinstance(value, (list, tuple, set, frozenset)):
                    errors.append(f"parameter is not repeatable: {parameter.name}")
        return tuple(errors)

    def build(
        self,
        arguments: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Action:
        errors = self.validate(arguments)
        if errors:
            raise ValueError("; ".join(errors))
        values: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.name in arguments:
                values[parameter.name] = arguments[parameter.name]
            elif parameter.has_default:
                values[parameter.name] = parameter.default
        return Action(
            verb=self.action_id,
            parameters=values,
            metadata={
                "plugin_id": self.plugin_id,
                "schema_id": self.schema_id,
                **dict(metadata or {}),
            },
        )


@dataclass(frozen=True, slots=True)
class PluginOutcome:
    snapshot: StateSnapshot
    added_facts: frozenset[str] = frozenset()
    removed_facts: frozenset[str] = frozenset()
    unlocked_actions: tuple[Action, ...] = ()
    error: bool = False
    error_code: str | None = None
    cost: float = 1.0
    reward: float = 0.0
    terminal: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


class ActionPlugin(Protocol):
    @property
    def plugin_id(self) -> str: ...

    def schemas(self) -> tuple[ActionSchema, ...]: ...

    def enumerate_values(
        self,
        state: StateSnapshot,
        schema: ActionSchema,
        parameter: ParameterSpec,
    ) -> Iterable[Any]: ...

    def execute(self, action: Action) -> PluginOutcome: ...


class ActionRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ActionPlugin] = {}
        self._schemas: dict[str, ActionSchema] = {}

    def register(self, plugin: ActionPlugin) -> None:
        if plugin.plugin_id in self._plugins:
            raise ValueError(f"duplicate plugin: {plugin.plugin_id}")
        schemas = plugin.schemas()
        if any(schema.plugin_id != plugin.plugin_id for schema in schemas):
            raise ValueError("schema plugin_id does not match plugin")
        duplicate = [
            schema.schema_id
            for schema in schemas
            if schema.schema_id in self._schemas
        ]
        if duplicate:
            raise ValueError(f"duplicate schemas: {duplicate}")
        self._plugins[plugin.plugin_id] = plugin
        self._schemas.update({schema.schema_id: schema for schema in schemas})

    def schema(self, schema_id: str) -> ActionSchema:
        try:
            return self._schemas[schema_id]
        except KeyError as exc:
            raise KeyError(f"unknown action schema: {schema_id}") from exc

    def plugin_for(self, action: Action) -> ActionPlugin:
        plugin_id = action.metadata.get("plugin_id")
        if not isinstance(plugin_id, str) or plugin_id not in self._plugins:
            raise KeyError(f"action has no registered plugin: {action.signature}")
        return self._plugins[plugin_id]

    def execute(self, action: Action) -> PluginOutcome:
        return self.plugin_for(action).execute(action)

    def schemas(self) -> tuple[ActionSchema, ...]:
        return tuple(
            sorted(self._schemas.values(), key=lambda schema: schema.schema_id)
        )


@dataclass(frozen=True, slots=True)
class ParameterLesson:
    schema_id: str
    parameter: str
    required: bool
    value_type: str
    default_is_available: bool
    instruction: str


def parameter_lessons(schema: ActionSchema) -> tuple[ParameterLesson, ...]:
    """Generate syntax-only curriculum material from a plugin schema."""

    lessons = []
    for parameter in schema.parameters:
        necessity = "필수" if parameter.required and not parameter.has_default else "선택"
        lessons.append(
            ParameterLesson(
                schema_id=schema.schema_id,
                parameter=parameter.name,
                required=parameter.required,
                value_type=parameter.value_type,
                default_is_available=parameter.has_default,
                instruction=(
                    f"{schema.action_id}의 {parameter.name} 슬롯은 {necessity}이며 "
                    f"{parameter.value_type} 형식의 값을 받는다. "
                    "어떤 값이 유용한지는 알려주지 않는다."
                ),
            )
        )
    return tuple(lessons)


class CandidateSource(Protocol):
    def rank_for_slot(
        self,
        action_id: str,
        slot: str,
        candidate_ids: Iterable[str],
        *,
        limit: int,
    ) -> tuple[str, ...]: ...


class SlotCandidateResolver:
    """Two-stage slot filling: role/cluster retrieval, then concrete value."""

    def __init__(self, source: CandidateSource, *, per_slot_limit: int = 4) -> None:
        if per_slot_limit <= 0:
            raise ValueError("per_slot_limit must be positive")
        self.source = source
        self.per_slot_limit = per_slot_limit

    def resolve(
        self,
        schema: ActionSchema,
        candidates: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        assignments: list[dict[str, Any]] = [{}]
        for parameter in schema.parameters:
            ranked = self.source.rank_for_slot(
                schema.action_id,
                parameter.role,
                tuple(candidates),
                limit=self.per_slot_limit,
            )
            values = [candidates[candidate_id] for candidate_id in ranked]
            if not parameter.required:
                values = [None, *values]
            if not values and parameter.has_default:
                values = [parameter.default]
            if not values:
                return ()
            expanded: list[dict[str, Any]] = []
            for base in assignments:
                for value in values:
                    item = dict(base)
                    if value is not None:
                        item[parameter.name] = value
                    expanded.append(item)
            assignments = expanded
        return tuple(assignments)
