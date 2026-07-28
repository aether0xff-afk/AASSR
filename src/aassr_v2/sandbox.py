from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .action_plugins import (
    ActionSchema,
    ParameterSpec,
    PluginOutcome,
)
from .types import Action, StateSnapshot


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    resources: tuple[str, ...] = (
        "resource_a",
        "resource_b",
    )
    hidden_recipes: tuple[
        tuple[tuple[str, ...], str],
        ...,
    ] = ((("resource_a", "resource_b"), "composite_ab"),)
    goal_item: str = "composite_ab"


class SandboxEnv:
    """Plugin-driven world with break, place and combine primitives."""

    def __init__(
        self,
        spec: SandboxSpec | None = None,
    ) -> None:
        self.spec = spec or SandboxSpec()
        self.present = set(self.spec.resources)
        self.inventory: list[str] = []
        self.placed: set[str] = set()
        self.known: set[str] = set()
        self._plugin = SandboxActionPlugin(self)

    @property
    def plugin(self) -> SandboxActionPlugin:
        return self._plugin

    def snapshot(self) -> StateSnapshot:
        facts = {
            f"present:{item}"
            for item in self.present
        }
        facts |= {
            f"inventory:{item}"
            for item in self.inventory
        }
        facts |= {
            f"placed:{item}"
            for item in self.placed
        }
        facts |= self.known
        actions = self._plugin.available_actions()
        progress = (
            1.0
            if self.spec.goal_item in self.inventory
            or self.spec.goal_item in self.placed
            else 0.0
        )
        universe = tuple(
            sorted(
                set(self.spec.resources)
                | {
                    output
                    for _, output in self.spec.hidden_recipes
                }
            )
        )
        vector = tuple(
            1.0
            if f"inventory:{item}" in facts
            or f"placed:{item}" in facts
            else 0.0
            for item in universe
        ) + (progress,)
        return StateSnapshot(
            vector,
            frozenset(facts),
            actions,
            progress,
        )

    def step(self, action: Action) -> PluginOutcome:
        return self._plugin.execute(action)


class SandboxActionPlugin:
    plugin_id = "sandbox"

    def __init__(
        self,
        environment: SandboxEnv,
    ) -> None:
        self.environment = environment
        self._schemas = (
            ActionSchema(
                self.plugin_id,
                "observe",
                (
                    ParameterSpec(
                        "subject",
                        "subject",
                        "identifier",
                    ),
                ),
                frozenset({"inspect"}),
            ),
            ActionSchema(
                self.plugin_id,
                "break",
                (
                    ParameterSpec(
                        "subject",
                        "subject",
                        "identifier",
                    ),
                ),
                frozenset({"transform"}),
            ),
            ActionSchema(
                self.plugin_id,
                "place",
                (
                    ParameterSpec(
                        "item",
                        "item",
                        "identifier",
                    ),
                ),
                frozenset({"transform"}),
            ),
            ActionSchema(
                self.plugin_id,
                "combine",
                (
                    ParameterSpec(
                        "items",
                        "items",
                        "identifier",
                        repeatable=True,
                    ),
                ),
                frozenset({"combine"}),
            ),
        )

    def schemas(self) -> tuple[ActionSchema, ...]:
        return self._schemas

    def schema(self, action_id: str) -> ActionSchema:
        return next(
            schema
            for schema in self._schemas
            if schema.action_id == action_id
        )

    def available_actions(self) -> tuple[Action, ...]:
        actions = []
        for item in sorted(self.environment.present):
            actions.append(
                self.schema("observe").build(
                    {"subject": item}
                )
            )
            actions.append(
                self.schema("break").build(
                    {"subject": item}
                )
            )
        for item in sorted(
            set(self.environment.inventory)
        ):
            actions.append(
                self.schema("place").build(
                    {"item": item}
                )
            )
        if len(self.environment.inventory) >= 2:
            unique = tuple(
                dict.fromkeys(self.environment.inventory)
            )
            actions.append(
                self.schema("combine").build(
                    {"items": unique}
                )
            )
        return tuple(
            sorted(
                actions,
                key=lambda action: action.signature,
            )
        )

    def enumerate_values(
        self,
        state: StateSnapshot,
        schema: ActionSchema,
        parameter: ParameterSpec,
    ) -> Iterable[Any]:
        del schema
        if parameter.role == "subject":
            return tuple(
                fact.split(":", 1)[1]
                for fact in state.facts
                if fact.startswith("present:")
            )
        if parameter.role in {"item", "items"}:
            return tuple(
                fact.split(":", 1)[1]
                for fact in state.facts
                if fact.startswith("inventory:")
            )
        return ()

    def execute(self, action: Action) -> PluginOutcome:
        before = self.environment.snapshot()
        added: set[str] = set()
        removed: set[str] = set()
        error = False
        code = None
        if action.verb_name == "observe":
            subject = str(
                action.parameters.get("subject")
            )
            if subject in self.environment.present:
                fact = f"observed:{subject}"
                self.environment.known.add(fact)
                added.add(fact)
            else:
                error, code = True, "not_present"
        elif action.verb_name == "break":
            subject = str(
                action.parameters.get("subject")
            )
            if subject in self.environment.present:
                self.environment.present.remove(subject)
                self.environment.inventory.append(subject)
                removed.add(f"present:{subject}")
                added.add(f"inventory:{subject}")
            else:
                error, code = True, "not_present"
        elif action.verb_name == "place":
            item = str(action.parameters.get("item"))
            if item in self.environment.inventory:
                self.environment.inventory.remove(item)
                self.environment.placed.add(item)
                removed.add(f"inventory:{item}")
                added.add(f"placed:{item}")
            else:
                error, code = True, "not_owned"
        elif action.verb_name == "combine":
            items = tuple(
                action.parameters.get("items", ())
            )
            recipe = {
                tuple(sorted(inputs)): output
                for inputs, output
                in self.environment.spec.hidden_recipes
            }
            output = recipe.get(tuple(sorted(items)))
            if (
                output is not None
                and all(
                    item in self.environment.inventory
                    for item in items
                )
            ):
                for item in items:
                    self.environment.inventory.remove(item)
                    removed.add(f"inventory:{item}")
                self.environment.inventory.append(output)
                added.add(f"inventory:{output}")
            else:
                error, code = (
                    True,
                    "ineffective_combination",
                )
        else:
            error, code = True, "unknown_action"
        after = self.environment.snapshot()
        before_signatures = {
            item.signature
            for item in before.available_actions
        }
        unlocked = tuple(
            item
            for item in after.available_actions
            if item.signature not in before_signatures
        )
        return PluginOutcome(
            after,
            frozenset(added),
            frozenset(removed),
            unlocked,
            error,
            code,
        )
