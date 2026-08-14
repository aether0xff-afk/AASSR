from __future__ import annotations

from dataclasses import replace
import hashlib
import random
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .plugin_contract import (
    ActionCommand,
    PluginObservation,
    PluginStepResult,
    TemporalKind,
    ValueKind,
)
from .representation import (
    CoreExperienceMemory,
    SchemaDrivenRepresentation,
    _add_hashed,
    _bounded_scalar,
    _normalized,
    _stable_json,
)


_TRANSFER_TEXT_TOKEN_RE = re.compile(r"[\w./:@?&=+\-]+", flags=re.UNICODE)


def _evidence_tokens(schema, observation: PluginObservation) -> tuple[str, ...]:
    """Normalize exact episode evidence without volatile measurements."""

    rows: list[str] = []
    fields = schema.observation_map
    for name, raw in sorted(observation.values.items()):
        field = fields.get(name)
        if field is None or field.temporal in {
            TemporalKind.COUNTER,
            TemporalKind.MEASUREMENT,
        }:
            continue
        if field.temporal is TemporalKind.EVENT and field.kind is ValueKind.MAPPING:
            mapping = raw if isinstance(raw, Mapping) else {}
            rows.append(
                f"{name}:keys:{_stable_json(tuple(sorted(str(key) for key in mapping)))}"
            )
            continue
        if field.temporal is TemporalKind.EVENT and field.kind is ValueKind.SCALAR:
            rows.append(f"{name}:scalar:{round(_bounded_scalar(raw), 3)}")
            continue
        rows.append(f"{name}:{_stable_json(raw)}")
    return tuple(rows)


def _transfer_evidence_tokens(
    schema,
    observation: PluginObservation,
) -> tuple[str, ...]:
    """Build remembered evidence features without concrete entity identity.

    The exact public values can still be stored by ``CorePublicKnowledge`` for
    command construction inside an episode.  This second channel is what the
    transfer learners receive after the original observation disappears.  It
    preserves public content where rename-safe and deliberately masks concrete
    entity identifiers.
    """

    rows: list[str] = []
    fields = schema.observation_map
    for name, raw in sorted(observation.values.items()):
        field = fields.get(name)
        if field is None or field.temporal in {
            TemporalKind.COUNTER,
            TemporalKind.MEASUREMENT,
        }:
            continue

        space = field.value_space or "*"
        prefix = f"field:{name}:kind:{field.kind.value}:space:{space}"
        if raw is None:
            rows.append(f"{prefix}:none")
            continue

        if field.kind is ValueKind.BOOLEAN:
            rows.append(f"{prefix}:bool:{int(bool(raw))}")
            continue

        if field.kind is ValueKind.SCALAR:
            rows.append(f"{prefix}:scalar:{round(_bounded_scalar(raw), 3)}")
            continue

        if field.kind is ValueKind.CATEGORICAL:
            rows.append(f"{prefix}:category:{_stable_json(raw)}")
            continue

        if field.kind is ValueKind.ENTITY:
            rows.append(f"{prefix}:entity-present")
            continue

        if field.kind is ValueKind.TEXT:
            text = str(raw).lower()
            rows.append(f"{prefix}:text-present")
            rows.append(f"{prefix}:length:{min(32, len(text) // 32)}")
            for token in _TRANSFER_TEXT_TOKEN_RE.findall(text)[:256]:
                rows.append(f"{prefix}:token:{token}")
            continue

        if field.kind is ValueKind.BYTES:
            try:
                length = len(raw)
            except TypeError:
                length = len(bytes(str(raw), "utf-8"))
            rows.append(f"{prefix}:bytes-present")
            rows.append(f"{prefix}:length:{min(32, int(length) // 32)}")
            continue

        if field.kind is ValueKind.MAPPING:
            mapping = raw if isinstance(raw, Mapping) else {}
            rows.append(f"{prefix}:mapping-present")
            rows.append(f"{prefix}:count:{min(64, len(mapping))}")
            # Values may contain volatile IDs or transport-generated data. Until
            # mapping values have their own typed schema, remember key structure
            # only rather than inventing domain-specific interpretation rules.
            for key in sorted(str(key) for key in mapping)[:256]:
                rows.append(f"{prefix}:key:{key}")
            continue

        if field.kind is ValueKind.SET:
            try:
                materialized = tuple(raw)
            except TypeError:
                materialized = (raw,)
            rows.append(f"{prefix}:set-present")
            rows.append(f"{prefix}:count:{min(64, len(materialized))}")
            if field.item_kind is ValueKind.ENTITY:
                # Count/presence transfers; concrete member names do not.
                rows.append(f"{prefix}:entity-members")
                continue
            if field.item_kind is ValueKind.TEXT:
                for item in materialized[:128]:
                    for token in _TRANSFER_TEXT_TOKEN_RE.findall(str(item).lower())[:64]:
                        rows.append(f"{prefix}:member-token:{token}")
                continue
            for item in sorted((_stable_json(item) for item in materialized))[:256]:
                rows.append(f"{prefix}:member:{item}")
            continue

    return tuple(rows)


class EpisodicCoreExperienceMemory(CoreExperienceMemory):
    """Concrete-action evidence is local unless explicitly preserved."""

    def __init__(self) -> None:
        super().__init__()
        self.episodes = 0

    def begin_episode(self, *, preserve: bool = False) -> None:
        self.episodes += 1
        if preserve:
            return
        self._by_action.clear()
        self.revision = 0
        self.knowledge_revision = 0

    def diagnostics(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                **dict(super().diagnostics()),
                "episodes": self.episodes,
            }
        )


class CorePublicKnowledge:
    """Generic memory of public values plus rename-safe remembered evidence.

    Exact values are retained only when they are useful as mechanically typed
    command material.  A separate transfer-evidence channel preserves observed
    BOOLEAN/CATEGORICAL/TEXT/etc. content for Policy and Prophecy while masking
    concrete ENTITY identifiers.  This avoids both information loss and hidden
    raw-ID transfer.
    """

    def __init__(
        self,
        *,
        per_kind_capacity: int = 512,
        transfer_evidence_capacity: int = 4096,
    ) -> None:
        self.per_kind_capacity = int(per_kind_capacity)
        self.transfer_evidence_capacity = int(transfer_evidence_capacity)
        if self.per_kind_capacity <= 0 or self.transfer_evidence_capacity <= 0:
            raise ValueError("Core public knowledge capacities must be positive")
        self._values: dict[
            tuple[ValueKind, str | None],
            dict[str, Any],
        ] = {}
        self._semantic_evidence: set[str] = set()
        self._transfer_evidence: dict[str, None] = {}
        self.revision = 0

    def clear(self) -> None:
        self._values.clear()
        self._semantic_evidence.clear()
        self._transfer_evidence.clear()
        self.revision = 0

    def _bucket(
        self,
        kind: ValueKind,
        value_space: str | None,
    ) -> dict[str, Any]:
        return self._values.setdefault((kind, value_space), {})

    def _remember_value(
        self,
        kind: ValueKind,
        value: Any,
        *,
        value_space: str | None,
    ) -> None:
        if value is None:
            return
        bucket = self._bucket(kind, value_space)
        key = _stable_json(value)
        if key in bucket:
            return
        if len(bucket) >= self.per_kind_capacity:
            oldest = next(iter(bucket))
            bucket.pop(oldest, None)
        bucket[key] = value

    def _remember_transfer_evidence(self, tokens: Sequence[str]) -> None:
        for token in tokens:
            if token in self._transfer_evidence:
                continue
            if len(self._transfer_evidence) >= self.transfer_evidence_capacity:
                oldest = next(iter(self._transfer_evidence))
                self._transfer_evidence.pop(oldest, None)
            self._transfer_evidence[token] = None

    def observe(self, schema, observation: PluginObservation) -> None:
        fields = schema.observation_map
        before = len(self._semantic_evidence)
        for name, raw in observation.values.items():
            field = fields.get(name)
            if field is None:
                continue
            # Historical counters/measurements are not reusable problem-solving
            # knowledge. They remain fully visible in the *current* observation
            # representation, but are not accumulated here.
            if field.temporal in {
                TemporalKind.COUNTER,
                TemporalKind.MEASUREMENT,
            }:
                continue
            if field.kind is ValueKind.SET:
                try:
                    materialized = tuple(raw)
                except TypeError:
                    materialized = (raw,)
                if field.item_kind is not None:
                    for item in materialized:
                        self._remember_value(
                            field.item_kind,
                            item,
                            value_space=field.value_space,
                        )
            elif field.kind in {
                ValueKind.BOOLEAN,
                ValueKind.SCALAR,
                ValueKind.CATEGORICAL,
                ValueKind.ENTITY,
                ValueKind.TEXT,
            }:
                self._remember_value(
                    field.kind,
                    raw,
                    value_space=field.value_space,
                )

        self._semantic_evidence.update(_evidence_tokens(schema, observation))
        self._remember_transfer_evidence(
            _transfer_evidence_tokens(schema, observation)
        )
        if len(self._semantic_evidence) > before:
            self.revision += 1

    def _merged_values(
        self,
        kind: ValueKind,
        *,
        value_space: str | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for (stored_kind, stored_space), bucket in self._values.items():
            if stored_kind is not kind:
                continue
            if value_space is not None and stored_space != value_space:
                continue
            merged.update(bucket)
        return merged

    def values(
        self,
        kind: ValueKind,
        *,
        limit: int,
        value_space: str | None = None,
    ) -> tuple[Any, ...]:
        bucket = self._merged_values(kind, value_space=value_space)
        keys = sorted(bucket)
        return tuple(bucket[key] for key in keys[: max(1, int(limit))])

    def all_values(
        self,
        kind: ValueKind,
        *,
        value_space: str | None = None,
    ) -> tuple[Any, ...]:
        bucket = self._merged_values(kind, value_space=value_space)
        return tuple(bucket[key] for key in sorted(bucket))

    def structural_vector(self, size: int) -> tuple[float, ...]:
        vector = [0.0] * int(size)
        for kind in ValueKind:
            count = sum(
                len(bucket)
                for (stored_kind, _), bucket in self._values.items()
                if stored_kind is kind
            )
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
        for (kind, value_space), bucket in self._values.items():
            if value_space is None:
                continue
            _add_hashed(
                vector,
                f"public-knowledge-space:{kind.value}:{value_space}:present",
                float(bool(bucket)),
            )
            _add_hashed(
                vector,
                f"public-knowledge-space:{kind.value}:{value_space}:count",
                min(1.0, len(bucket) / 64.0),
            )
        for token in self._transfer_evidence:
            _add_hashed(vector, f"public-knowledge-evidence:{token}", 1.0)
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
            "transfer_evidence": len(self._transfer_evidence),
        }
        for kind in ValueKind:
            result[f"values:{kind.value}"] = sum(
                len(bucket)
                for (stored_kind, _), bucket in self._values.items()
                if stored_kind is kind
            )
        for (kind, value_space), bucket in self._values.items():
            if value_space is not None:
                result[f"space:{kind.value}:{value_space}"] = len(bucket)
        return MappingProxyType(result)


class MemoryBackedRepresentation(SchemaDrivenRepresentation):
    """Canonical Core representation with Core-owned public and local memory.

    Bounded candidate surfaces are sampled by a Core-owned episode seed rather
    than taking the lexicographically first concrete identifiers. This removes a
    hidden preference for names while keeping a repeated state stable inside an
    episode.
    """

    def __init__(
        self,
        *args: Any,
        candidate_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("experience") is None:
            kwargs["experience"] = EpisodicCoreExperienceMemory()
        super().__init__(*args, **kwargs)
        self.public_knowledge = CorePublicKnowledge()
        self.candidate_seed = int(candidate_seed)
        self.episode_index = 0
        self.candidate_sampling_events = 0
        self.last_candidate_population = 0
        self.last_candidate_surface = 0

    def begin_episode(self, *, preserve: bool = False) -> None:
        self.episode_index += 1
        if not preserve:
            self.public_knowledge.clear()
        begin = getattr(self.experience, "begin_episode", None)
        if callable(begin):
            begin(preserve=preserve)

    def observation_vector(self, observation: PluginObservation) -> tuple[float, ...]:
        current = list(super().observation_vector(observation))
        memory = self.public_knowledge.structural_vector(self.state_size)
        return _normalized(
            tuple(left + right for left, right in zip(current, memory, strict=True))
        )

    def _selection_seed(self, label: str, cardinalities: Sequence[int]) -> int:
        # The seed deliberately excludes concrete candidate values. The chosen
        # subset therefore has no lexical/name preference; under renaming its
        # distribution stays symmetric. Public knowledge revision changes the
        # surface only when genuinely new evidence appears.
        payload = (
            self.candidate_seed,
            self.episode_index,
            self.public_knowledge.revision,
            label,
            tuple(int(value) for value in cardinalities),
        )
        digest = hashlib.blake2b(
            repr(payload).encode("utf-8"),
            digest_size=16,
        ).digest()
        return int.from_bytes(digest, "big")

    def _all_candidate_values(
        self,
        observation: PluginObservation,
        *,
        kind: ValueKind,
        enum_values: Sequence[str] = (),
        value_space: str | None = None,
    ) -> tuple[Any, ...]:
        merged: dict[str, Any] = {}
        for value in enum_values:
            merged[_stable_json(value)] = value
        if not enum_values:
            fields = self.schema.observation_map
            for name, raw in observation.values.items():
                field = fields.get(name)
                if field is None:
                    continue
                if value_space is not None and field.value_space != value_space:
                    continue
                if field.kind is kind:
                    merged[_stable_json(raw)] = raw
                elif field.kind is ValueKind.SET and field.item_kind is kind:
                    try:
                        materialized = tuple(raw)
                    except TypeError:
                        materialized = (raw,)
                    for item in materialized:
                        merged[_stable_json(item)] = item
            for value in self.public_knowledge.all_values(
                kind,
                value_space=value_space,
            ):
                merged[_stable_json(value)] = value
        return tuple(merged[key] for key in sorted(merged))

    def _sample_candidate_values(
        self,
        observation: PluginObservation,
        *,
        kind: ValueKind,
        enum_values: Sequence[str],
        value_space: str | None,
        limit: int,
        label: str,
    ) -> tuple[Any, ...]:
        values = self._all_candidate_values(
            observation,
            kind=kind,
            enum_values=enum_values,
            value_space=value_space,
        )
        limit = max(1, int(limit))
        if len(values) <= limit:
            return values
        randomizer = random.Random(
            self._selection_seed(label, (len(values), limit))
        )
        indices = sorted(randomizer.sample(range(len(values)), limit))
        self.candidate_sampling_events += 1
        return tuple(values[index] for index in indices)

    def _candidate_values(
        self,
        observation: PluginObservation,
        *,
        kind: ValueKind,
        enum_values: Sequence[str] = (),
        value_space: str | None = None,
        limit: int = 8,
    ) -> tuple[Any, ...]:
        return self._sample_candidate_values(
            observation,
            kind=kind,
            enum_values=enum_values,
            value_space=value_space,
            limit=limit,
            label=f"kind:{kind.value}:space:{value_space or '*'}",
        )

    @staticmethod
    def _decode_product_index(
        index: int,
        rows: Sequence[Sequence[Any]],
    ) -> tuple[Any, ...]:
        values: list[Any] = [None] * len(rows)
        remainder = int(index)
        for slot in range(len(rows) - 1, -1, -1):
            size = len(rows[slot])
            remainder, position = divmod(remainder, size)
            values[slot] = rows[slot][position]
        return tuple(values)

    def synthesize_commands(
        self,
        observation: PluginObservation,
        *,
        per_parameter_limit: int = 8,
        total_limit: int = 128,
    ) -> tuple[ActionCommand, ...]:
        """Create a bounded, name-unbiased concrete command surface."""

        blocks: list[tuple[Any, tuple[tuple[Any, ...], ...], int]] = []
        total_population = 0
        for spec in self.schema.actions:
            rows: list[tuple[Any, ...]] = []
            impossible = False
            for parameter in spec.parameters:
                candidates = list(
                    self._sample_candidate_values(
                        observation,
                        kind=parameter.kind,
                        enum_values=parameter.enum_values,
                        value_space=parameter.value_space,
                        limit=per_parameter_limit,
                        label=(
                            f"{spec.action_id}:{parameter.name}:"
                            f"space:{parameter.value_space or '*'}"
                        ),
                    )
                )
                if not parameter.required:
                    candidates.insert(0, None)
                if not candidates:
                    impossible = True
                    break
                rows.append(tuple(candidates))
            if impossible:
                continue
            count = 1
            for row in rows:
                count *= len(row)
            blocks.append((spec, tuple(rows), count))
            total_population += count

        self.last_candidate_population = total_population
        if total_population <= 0:
            self.last_candidate_surface = 0
            return ()

        limit = min(max(1, int(total_limit)), total_population)
        if total_population <= limit:
            chosen = tuple(range(total_population))
        else:
            randomizer = random.Random(
                self._selection_seed(
                    "command-surface",
                    tuple(block[2] for block in blocks) + (limit,),
                )
            )
            chosen = tuple(sorted(randomizer.sample(range(total_population), limit)))
            self.candidate_sampling_events += 1

        commands: dict[str, ActionCommand] = {}
        offsets: list[int] = []
        cursor = 0
        for _, _, count in blocks:
            offsets.append(cursor)
            cursor += count

        for global_index in chosen:
            block_index = 0
            while (
                block_index + 1 < len(blocks)
                and global_index >= offsets[block_index + 1]
            ):
                block_index += 1
            spec, rows, _ = blocks[block_index]
            local_index = global_index - offsets[block_index]
            selected = self._decode_product_index(local_index, rows) if rows else ()
            arguments = {
                parameter.name: value
                for parameter, value in zip(spec.parameters, selected, strict=True)
                if value is not None
            }
            command = ActionCommand(spec.action_id, arguments)
            key = f"{command.action_id}:{_stable_json(dict(command.arguments))}"
            commands[key] = command

        self.last_candidate_surface = len(commands)
        return tuple(commands[key] for key in sorted(commands))

    def semantic_observation_identity(
        self,
        observation: PluginObservation,
    ) -> tuple[tuple[str, str], ...]:
        fields = self.schema.observation_map
        return tuple(
            (name, _stable_json(value))
            for name, value in sorted(observation.values.items())
            if name in fields and fields[name].temporal is TemporalKind.STATE
        )

    def semantic_observation_fingerprint(self, observation: PluginObservation) -> str:
        evidence = _evidence_tokens(self.schema, observation)
        return hashlib.sha256(repr(evidence).encode("utf-8")).hexdigest()

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
                "core_candidate_population": self.last_candidate_population,
                "core_candidate_surface": self.last_candidate_surface,
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
