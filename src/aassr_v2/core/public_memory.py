from __future__ import annotations

from dataclasses import replace
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .plugin_contract import PluginObservation, PluginStepResult, TemporalKind, ValueKind
from .representation import (
    SchemaDrivenRepresentation,
    _add_hashed,
    _normalized,
    _stable_json,
)


class CorePublicKnowledge:
    """Generic memory of publicly observed typed values.

    Concrete values are retained only so the Core can act on things it actually
    observed earlier. The learned transfer vector sees type/count structure,
    not the concrete identifiers themselves.
    """

    def __init__(self, *, per_kind_capacity: int = 512) -> None:
        self.per_kind_capacity = int(per_kind_capacity)
        self._values: dict[ValueKind, dict[str, Any]] = {
            kind: {} for kind in ValueKind
        }
        self._semantic_evidence: set[str] = set()
        self.revision = 0

    def clear(self) -> None:
        for bucket in self._values.values():
            bucket.clear()
        self._semantic_evidence.clear()
        self.revision = 0

    def _remember_value(self, kind: ValueKind, value: Any) -> None:
        if value is None:
            return
        bucket = self._values[kind]
        key = _stable_json(value)
        if key in bucket:
            return
        if len(bucket) >= self.per_kind_capacity:
            oldest = next(iter(bucket))
            bucket.pop(oldest, None)
        bucket[key] = value

    def observe(self, schema, observation: PluginObservation) -> None:
        fields = schema.observation_map
        before = len(self._semantic_evidence)
        for name, raw in observation.values.items():
            field = fields.get(name)
            if field is None:
                continue

            if field.kind is ValueKind.SET:
                try:
                    materialized = tuple(raw)
                except TypeError:
                    materialized = (raw,)
                item_kind = field.item_kind
                if item_kind is not None:
                    for item in materialized:
                        self._remember_value(item_kind, item)
            elif field.kind in {
                ValueKind.BOOLEAN,
                ValueKind.SCALAR,
                ValueKind.CATEGORICAL,
                ValueKind.ENTITY,
                ValueKind.TEXT,
            }:
                self._remember_value(field.kind, raw)

            if field.temporal in {TemporalKind.COUNTER, TemporalKind.MEASUREMENT}:
                continue
            token = f"{name}:{_stable_json(raw)}"
            self._semantic_evidence.add(token)

        if len(self._semantic_evidence) > before:
            self.revision += 1

    def values(self, kind: ValueKind, *, limit: int) -> tuple[Any, ...]:
        bucket = self._values[kind]
        keys = sorted(bucket)
        return tuple(bucket[key] for key in keys[: max(1, int(limit))])

    def structural_vector(self, size: int) -> tuple[float, ...]:
        vector = [0.0] * int(size)
        for kind, bucket in self._values.items():
            count = len(bucket)
            _add_hashed(
                vector,
                f"public-knowledge:{kind.value}:present",
                float(count > 0),
            )
            _add_hashed(
                vector,
                f"public-knowledge:{kind.value}:count",
                min(1.0, count / 64.0),
            )
        _add_hashed(
            vector,
            "public-knowledge:revision",
            min(1.0, self.revision / 64.0),
        )
        return _normalized(vector)

    def diagnostics(self) -> Mapping[str, int]:
        result = {
            "revision": self.revision,
            "semantic_evidence": len(self._semantic_evidence),
        }
        for kind, bucket in self._values.items():
            result[f"values:{kind.value}"] = len(bucket)
        return MappingProxyType(result)


class MemoryBackedRepresentation(SchemaDrivenRepresentation):
    """Canonical Core representation with Core-owned public memory."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.public_knowledge = CorePublicKnowledge()

    def begin_episode(self, *, preserve: bool = False) -> None:
        if not preserve:
            self.public_knowledge.clear()

    def observation_vector(self, observation: PluginObservation) -> tuple[float, ...]:
        current = list(super().observation_vector(observation))
        memory = self.public_knowledge.structural_vector(self.state_size)
        return _normalized(
            tuple(left + right for left, right in zip(current, memory, strict=True))
        )

    def _candidate_values(
        self,
        observation: PluginObservation,
        *,
        kind: ValueKind,
        enum_values: Sequence[str] = (),
        limit: int = 8,
    ) -> tuple[Any, ...]:
        if enum_values:
            return super()._candidate_values(
                observation,
                kind=kind,
                enum_values=enum_values,
                limit=limit,
            )
        merged: dict[str, Any] = {
            _stable_json(value): value
            for value in super()._candidate_values(
                observation,
                kind=kind,
                limit=limit,
            )
        }
        for value in self.public_knowledge.values(kind, limit=limit):
            merged[_stable_json(value)] = value
        return tuple(
            merged[key]
            for key in sorted(merged)[: max(1, int(limit))]
        )

    def semantic_observation_fingerprint(self, observation: PluginObservation) -> str:
        identity = self.semantic_observation_identity(observation)
        return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()

    def to_snapshot(self, result: PluginStepResult):
        observation = result.observation
        self.public_knowledge.observe(self.schema, observation)
        snapshot = super().to_snapshot(result)
        metadata = dict(snapshot.metadata)
        semantic_identity = (
            self.semantic_observation_identity(observation),
            tuple(action.signature for action in snapshot.available_actions),
            self.public_knowledge.revision,
        )
        metadata.update(
            {
                "core_semantic_identity": semantic_identity,
                "core_public_knowledge_revision": self.public_knowledge.revision,
            }
        )
        return replace(snapshot, metadata=metadata)

    def observe_real_transition(
        self,
        *,
        before_result: PluginStepResult,
        action,
        after_result: PluginStepResult,
    ) -> None:
        self.experience.observe(
            action_signature=action.signature,
            before_semantic=self.semantic_observation_identity(
                before_result.observation
            ),
            after_semantic=self.semantic_observation_identity(
                after_result.observation
            ),
            outcome_fingerprint=self.semantic_observation_fingerprint(
                after_result.observation
            ),
            reward=float(after_result.reward),
            error=bool(after_result.error),
            terminated=bool(after_result.terminated),
        )
