from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .action_plugins import (
    ActionSchema,
    ParameterSpec,
    PluginOutcome,
)
from .types import Action, StateSnapshot


class CommandTransport(Protocol):
    def invoke(
        self,
        command: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class DryRunTransport:
    calls: list[tuple[str, Mapping[str, Any]]]

    def __init__(self) -> None:
        self.calls = []

    def invoke(
        self,
        command: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append((command, dict(arguments)))
        return {
            "ok": True,
            "command": command,
            "arguments": dict(arguments),
        }


class SchemaOnlyPlugin:
    """Expose grammar while real execution remains transport-owned."""

    plugin_id = "schema-only"

    def __init__(
        self,
        transport: CommandTransport,
    ) -> None:
        self.transport = transport
        self._state = StateSnapshot(
            (0.0,),
            frozenset(),
            (),
            0.0,
        )

    def schemas(self) -> tuple[ActionSchema, ...]:
        raise NotImplementedError

    def enumerate_values(
        self,
        state: StateSnapshot,
        schema: ActionSchema,
        parameter: ParameterSpec,
    ) -> Iterable[Any]:
        del state, schema, parameter
        return ()

    def execute(self, action: Action) -> PluginOutcome:
        result = self.transport.invoke(
            action.verb_name,
            action.parameters,
        )
        error = not bool(result.get("ok", False))
        facts = frozenset(
            {
                f"result:{action.signature}:"
                f"{'error' if error else 'ok'}"
            }
        )
        self._state = StateSnapshot(
            self._state.vector,
            self._state.facts | facts,
            self._state.available_actions,
            self._state.goal_progress,
        )
        return PluginOutcome(
            self._state,
            facts,
            error=error,
            error_code=(
                "transport_error" if error else None
            ),
            raw=result,
        )


class MinecraftControlPlugin(SchemaOnlyPlugin):
    """Primitive game-control contract with no recipe knowledge."""

    plugin_id = "minecraft-control"

    def schemas(self) -> tuple[ActionSchema, ...]:
        return (
            ActionSchema(
                self.plugin_id,
                "move",
                (
                    ParameterSpec(
                        "forward",
                        "axis",
                        "number",
                    ),
                    ParameterSpec(
                        "strafe",
                        "axis",
                        "number",
                        required=False,
                        default=0.0,
                    ),
                    ParameterSpec(
                        "duration",
                        "duration",
                        "number",
                        required=False,
                        default=0.1,
                    ),
                ),
            ),
            ActionSchema(
                self.plugin_id,
                "look",
                (
                    ParameterSpec(
                        "yaw_delta",
                        "angle",
                        "number",
                    ),
                    ParameterSpec(
                        "pitch_delta",
                        "angle",
                        "number",
                    ),
                ),
            ),
            ActionSchema(
                self.plugin_id,
                "press",
                (
                    ParameterSpec(
                        "control",
                        "control",
                        "identifier",
                    ),
                    ParameterSpec(
                        "duration",
                        "duration",
                        "number",
                        required=False,
                        default=0.05,
                    ),
                ),
            ),
            ActionSchema(
                self.plugin_id,
                "interact",
                (
                    ParameterSpec(
                        "target",
                        "target",
                        "identifier",
                        required=False,
                    ),
                ),
            ),
        )


class AuthorizedAssessmentPlugin(SchemaOnlyPlugin):
    """Allowlisted abstract security-assessment actions.

    The plugin does not generate exploits or shell commands. A caller-supplied
    transport must enforce authorization and map abstract calls to tools.

    Targets and resources are separate authorization scopes. Every action is
    bound to one exact allowlisted target. ``read`` additionally requires the
    requested resource to be explicitly allowlisted for that target; resource
    names are opaque identifiers and never act as target aliases or prefixes.
    Runtime actions must use the canonical schema parameters; legacy action
    fields and unrecognized parameter aliases are rejected before transport.
    """

    plugin_id = "authorized-assessment"

    def __init__(
        self,
        transport: CommandTransport,
        *,
        allowlisted_targets: Iterable[str],
        allowlisted_resources: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        super().__init__(transport)
        self.allowlisted_targets = self._exact_string_set(
            allowlisted_targets,
            scope_name="target",
        )
        resource_scopes = (
            {} if allowlisted_resources is None else allowlisted_resources
        )
        if not isinstance(resource_scopes, Mapping):
            raise TypeError("resource allowlist must be a mapping")
        non_string_targets = [
            target
            for target in resource_scopes
            if type(target) is not str
        ]
        if non_string_targets:
            raise TypeError("resource allowlist target keys must be strings")
        unknown_targets = {
            target
            for target in resource_scopes
            if target not in self.allowlisted_targets
        }
        if unknown_targets:
            raise ValueError(
                "resource allowlist contains non-allowlisted targets: "
                f"{sorted(unknown_targets)!r}"
            )
        self.allowlisted_resources = {
            target: self._exact_string_set(
                resources,
                scope_name=f"resource for target {target!r}",
            )
            for target, resources in resource_scopes.items()
        }

    @staticmethod
    def _exact_string_set(
        values: Iterable[str],
        *,
        scope_name: str,
    ) -> frozenset[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{scope_name} allowlist must be an iterable of strings")
        normalized: set[str] = set()
        for value in values:
            if type(value) is not str:
                raise TypeError(f"{scope_name} allowlist entries must be strings")
            normalized.add(value)
        return frozenset(normalized)

    def schemas(self) -> tuple[ActionSchema, ...]:
        return (
            ActionSchema(
                self.plugin_id,
                "scan",
                (
                    ParameterSpec(
                        "target",
                        "target",
                        "identifier",
                    ),
                    ParameterSpec(
                        "profile",
                        "option",
                        "identifier",
                        required=False,
                        default="default",
                    ),
                ),
            ),
            ActionSchema(
                self.plugin_id,
                "connect",
                (
                    ParameterSpec(
                        "endpoint",
                        "target",
                        "identifier",
                    ),
                    ParameterSpec(
                        "credential",
                        "authentication",
                        "identifier",
                        required=False,
                    ),
                ),
            ),
            ActionSchema(
                self.plugin_id,
                "read",
                (
                    ParameterSpec(
                        "target",
                        "target",
                        "identifier",
                    ),
                    ParameterSpec(
                        "resource",
                        "resource",
                        "identifier",
                    ),
                ),
            ),
        )

    def execute(self, action: Action) -> PluginOutcome:
        # ``Action.verb_name`` intentionally stringifies generic core verbs. Do
        # not use it for an authorization decision: a non-string object with a
        # crafted ``__str__`` must not become an assessment command.
        if type(action.verb) is not str:
            return PluginOutcome(
                self._state,
                error=True,
                error_code="action_not_allowlisted",
            )
        verb = action.verb
        scope_parameter = {
            "scan": "target",
            "connect": "endpoint",
            "read": "target",
        }.get(verb)
        if scope_parameter is None:
            return PluginOutcome(
                self._state,
                error=True,
                error_code="action_not_allowlisted",
            )
        if any(type(name) is not str for name in action.parameters):
            return PluginOutcome(
                self._state,
                error=True,
                error_code="parameter_not_allowlisted",
            )
        expected_schema_id = f"{self.plugin_id}:{verb}"
        metadata_plugin_id = action.metadata.get("plugin_id")
        if (
            metadata_plugin_id is not None
            and (
                type(metadata_plugin_id) is not str
                or metadata_plugin_id != self.plugin_id
            )
        ):
            return PluginOutcome(
                self._state,
                error=True,
                error_code="action_not_allowlisted",
            )
        metadata_schema_id = action.metadata.get("schema_id")
        if (
            metadata_schema_id is not None
            and (
                type(metadata_schema_id) is not str
                or metadata_schema_id != expected_schema_id
            )
        ):
            return PluginOutcome(
                self._state,
                error=True,
                error_code="action_not_allowlisted",
            )
        if any(
            value is not None
            for value in (action.target, action.tool, action.destination)
        ):
            return PluginOutcome(
                self._state,
                error=True,
                error_code="parameter_not_allowlisted",
            )
        target = action.parameters.get(scope_parameter)
        if target is None:
            return PluginOutcome(
                self._state,
                error=True,
                error_code="target_scope_required",
            )
        if type(target) is not str or target not in self.allowlisted_targets:
            return PluginOutcome(
                self._state,
                error=True,
                error_code="target_not_allowlisted",
            )
        target_scope = target
        if verb == "read":
            resource = action.parameters.get("resource")
            if resource is None:
                return PluginOutcome(
                    self._state,
                    error=True,
                    error_code="resource_scope_required",
                )
            allowed = self.allowlisted_resources.get(
                target_scope,
                frozenset(),
            )
            if type(resource) is not str or resource not in allowed:
                return PluginOutcome(
                    self._state,
                    error=True,
                    error_code="resource_not_allowlisted",
                )
        schema = next(
            candidate
            for candidate in self.schemas()
            if candidate.action_id == verb
        )
        if schema.validate(action.parameters):
            return PluginOutcome(
                self._state,
                error=True,
                error_code="parameter_not_allowlisted",
            )
        for parameter in schema.parameters:
            if (
                parameter.value_type == "identifier"
                and parameter.name in action.parameters
                and type(action.parameters[parameter.name]) is not str
            ):
                return PluginOutcome(
                    self._state,
                    error=True,
                    error_code="parameter_not_allowlisted",
                )
        return super().execute(action)
