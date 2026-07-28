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
    """

    plugin_id = "authorized-assessment"

    def __init__(
        self,
        transport: CommandTransport,
        *,
        allowlisted_targets: Iterable[str],
    ) -> None:
        super().__init__(transport)
        self.allowlisted_targets = frozenset(
            allowlisted_targets
        )

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
                        "resource",
                        "resource",
                        "identifier",
                    ),
                ),
            ),
        )

    def execute(self, action: Action) -> PluginOutcome:
        target = action.parameters.get(
            "target",
            action.parameters.get("endpoint"),
        )
        if (
            target is not None
            and str(target) not in self.allowlisted_targets
        ):
            return PluginOutcome(
                self._state,
                error=True,
                error_code="target_not_allowlisted",
            )
        return super().execute(action)
