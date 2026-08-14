from __future__ import annotations

from itertools import product
from typing import Any, Sequence

from .plugin_contract import ActionCommand, PluginObservation, ValueKind
from .representation import SchemaDrivenRepresentation as _BaseSchemaDrivenRepresentation
from .representation import _stable_json


class SchemaDrivenRepresentation(_BaseSchemaDrivenRepresentation):
    """Generic Core representation with mechanical value-space isolation.

    ``ValueKind`` answers "what shape is this value?". ``value_space`` answers
    "which protocol slot can this value mechanically fill?".  Neither carries a
    task-value judgement.  A Plugin may therefore declare that URL observations
    can fill URL parameters without saying which URL is useful or correct.

    ``value_space=None`` intentionally preserves the earlier kind-only behavior
    for schemas that have not opted into the stronger mechanical typing yet.
    """

    def _candidate_values(
        self,
        observation: PluginObservation,
        *,
        kind: ValueKind,
        enum_values: Sequence[str] = (),
        value_space: str | None = None,
        limit: int = 8,
    ) -> tuple[Any, ...]:
        values: dict[str, Any] = {}
        for item in enum_values:
            values[_stable_json(item)] = item
        if enum_values:
            return tuple(
                values[key]
                for key in sorted(values)[: max(1, int(limit))]
            )

        fields = self.schema.observation_map
        for name, raw in observation.values.items():
            field = fields.get(name)
            if field is None:
                continue
            if value_space is not None and field.value_space != value_space:
                continue
            if field.kind is kind:
                values[_stable_json(raw)] = raw
            elif field.kind is ValueKind.SET and field.item_kind is kind:
                try:
                    materialized = tuple(raw)
                except TypeError:
                    materialized = (raw,)
                for item in materialized:
                    values[_stable_json(item)] = item
        return tuple(
            values[key]
            for key in sorted(values)[: max(1, int(limit))]
        )

    def synthesize_commands(
        self,
        observation: PluginObservation,
        *,
        per_parameter_limit: int = 8,
        total_limit: int = 128,
    ) -> tuple[ActionCommand, ...]:
        """Generate type-compatible commands without strategic ranking."""

        commands: dict[str, ActionCommand] = {}
        for spec in self.schema.actions:
            names = [item.name for item in spec.parameters]
            candidate_rows: list[tuple[Any, ...]] = []
            impossible = False
            for parameter in spec.parameters:
                candidates = list(
                    self._candidate_values(
                        observation,
                        kind=parameter.kind,
                        enum_values=parameter.enum_values,
                        value_space=parameter.value_space,
                        limit=per_parameter_limit,
                    )
                )
                if not parameter.required:
                    candidates.insert(0, None)
                if not candidates:
                    impossible = True
                    break
                candidate_rows.append(tuple(candidates))
            if impossible:
                continue

            assignments = product(*candidate_rows) if candidate_rows else [()]
            for row in assignments:
                arguments = {
                    name: value
                    for name, value in zip(names, row, strict=True)
                    if value is not None
                }
                command = ActionCommand(spec.action_id, arguments)
                key = f"{command.action_id}:{_stable_json(dict(command.arguments))}"
                commands[key] = command
                if len(commands) >= total_limit:
                    break
            if len(commands) >= total_limit:
                break
        return tuple(commands[key] for key in sorted(commands))
