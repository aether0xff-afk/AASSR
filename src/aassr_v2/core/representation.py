from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from itertools import product
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..action_plugins import PluginOutcome
from ..neural_delta_prophecy import StateCodec
from ..types import Action, StateSnapshot
from .plugin_contract import (
    ActionCommand,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    TemporalKind,
    ValueKind,
    validate_step_result,
)


_TOKEN_RE = re.compile(r"[\w./:@?&=+\-]+", flags=re.UNICODE)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _bounded_scalar(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number / (1.0 + abs(number))


def _add_hashed(
    vector: list[float],
    token: str,
    value: float = 1.0,
) -> None:
    digest = _digest(token)
    index = int.from_bytes(digest[:8], "big") % len(vector)
    sign = -1.0 if digest[8] & 1 else 1.0
    vector[index] += sign * float(value)


def _normalized(values: Sequence[float]) -> tuple[float, ...]:
    norm = sum(float(value) * float(value) for value in values) ** 0.5
    if norm <= 1.0:
        return tuple(float(value) for value in values)
    return tuple(float(value) / norm for value in values)


def _flatten_public_value(
    *,
    field: ObservationField,
    value: Any,
) -> tuple[tuple[str, float], ...]:
    """Convert public data into type-level tokens without task semantics."""

    prefix = f"field:{field.name}:kind:{field.kind.value}"
    rows: list[tuple[str, float]] = [(f"{prefix}:present", 1.0)]

    if value is None:
        rows.append((f"{prefix}:none", 1.0))
        return tuple(rows)

    if field.kind is ValueKind.BOOLEAN:
        rows.append((f"{prefix}:bool", 1.0 if bool(value) else -1.0))
        return tuple(rows)

    if field.kind is ValueKind.SCALAR:
        rows.append((f"{prefix}:scalar", _bounded_scalar(value)))
        return tuple(rows)

    if field.kind is ValueKind.CATEGORICAL:
        rows.append((f"{prefix}:category:{_stable_json(value)}", 1.0))
        return tuple(rows)

    if field.kind is ValueKind.ENTITY:
        # Concrete entity identity is execution-local. Transfer learners receive
        # only the typed presence signal; candidate-specific evidence enters via
        # CoreExperienceMemory after real interaction.
        rows.append((f"{prefix}:entity", 1.0))
        return tuple(rows)

    if field.kind is ValueKind.TEXT:
        text = str(value)
        rows.append((f"{prefix}:length", _bounded_scalar(len(text))))
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens[:256]:
            rows.append((f"{prefix}:token:{token}", 1.0))
        return tuple(rows)

    if field.kind is ValueKind.BYTES:
        raw = bytes(value) if isinstance(value, (bytes, bytearray)) else str(value).encode()
        rows.append((f"{prefix}:bytes-length", _bounded_scalar(len(raw))))
        rows.append((f"{prefix}:bytes:{hashlib.sha256(raw).hexdigest()[:24]}", 1.0))
        return tuple(rows)

    if field.kind is ValueKind.SET:
        try:
            materialized = tuple(value)
        except TypeError:
            materialized = (value,)
        rows.append((f"{prefix}:count", _bounded_scalar(len(materialized))))
        if field.item_kind is not ValueKind.ENTITY:
            for item in sorted((_stable_json(item) for item in materialized))[:256]:
                rows.append((f"{prefix}:member:{item}", 1.0))
        return tuple(rows)

    if field.kind is ValueKind.MAPPING:
        mapping = value if isinstance(value, Mapping) else {"value": value}
        rows.append((f"{prefix}:count", _bounded_scalar(len(mapping))))
        for key, item in sorted(
            ((str(key), item) for key, item in mapping.items()),
            key=lambda pair: pair[0],
        )[:256]:
            rows.append((f"{prefix}:key:{key}", 1.0))
            rows.append((f"{prefix}:pair:{key}={_stable_json(item)}", 1.0))
        return tuple(rows)

    rows.append((f"{prefix}:opaque:{_stable_json(value)}", 1.0))
    return tuple(rows)


def _canonical_observation(
    schema: PluginSchema,
    observation: PluginObservation,
    *,
    semantic: bool,
) -> tuple[tuple[str, str], ...]:
    fields = schema.observation_map
    output: list[tuple[str, str]] = []
    for name, value in sorted(observation.values.items()):
        field = fields.get(name)
        if field is None:
            continue
        if semantic and field.temporal in {
            TemporalKind.COUNTER,
            TemporalKind.MEASUREMENT,
        }:
            continue
        output.append((name, _stable_json(value)))
    return tuple(output)


@dataclass(slots=True)
class _ActionExperience:
    uses: int = 0
    changed: int = 0
    errors: int = 0
    terminated: int = 0
    positive_reward: int = 0
    negative_reward: int = 0
    zero_reward: int = 0
    novel_outcomes: int = 0
    outcomes: set[str] = field(default_factory=set)

    def vector(self) -> tuple[float, ...]:
        normalizer = float(max(1, self.uses))
        return (
            min(1.0, self.uses / 32.0),
            self.changed / normalizer,
            self.errors / normalizer,
            self.terminated / normalizer,
            self.positive_reward / normalizer,
            self.negative_reward / normalizer,
            self.zero_reward / normalizer,
            self.novel_outcomes / normalizer,
        )


class CoreExperienceMemory:
    """Generic action/outcome history owned entirely by the Core."""

    FEATURE_NAMES: tuple[str, ...] = (
        "uses",
        "state_change_rate",
        "error_rate",
        "terminal_rate",
        "positive_reward_rate",
        "negative_reward_rate",
        "zero_reward_rate",
        "novel_outcome_rate",
    )

    def __init__(self) -> None:
        self._by_action: dict[str, _ActionExperience] = {}
        self.transitions = 0
        self.revision = 0
        self.knowledge_revision = 0

    def features(self, action_signature: str) -> tuple[float, ...]:
        return self._by_action.get(action_signature, _ActionExperience()).vector()

    def observe(
        self,
        *,
        action_signature: str,
        before_semantic: object,
        after_semantic: object,
        outcome_fingerprint: str,
        reward: float,
        error: bool,
        terminated: bool,
    ) -> None:
        entry = self._by_action.setdefault(action_signature, _ActionExperience())
        entry.uses += 1
        if before_semantic != after_semantic:
            entry.changed += 1
        if error:
            entry.errors += 1
        if terminated:
            entry.terminated += 1
        if reward > 0.0:
            entry.positive_reward += 1
        elif reward < 0.0:
            entry.negative_reward += 1
        else:
            entry.zero_reward += 1
        novel = outcome_fingerprint not in entry.outcomes
        if novel:
            entry.novel_outcomes += 1
            entry.outcomes.add(outcome_fingerprint)
            self.knowledge_revision += 1
        self.transitions += 1
        self.revision += 1

    def diagnostics(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "transitions": self.transitions,
                "action_records": len(self._by_action),
                "revision": self.revision,
                "knowledge_revision": self.knowledge_revision,
            }
        )


@dataclass(frozen=True, slots=True)
class CoreRepresentationConfig:
    state_size: int = 256
    action_feature_size: int = 128

    def __post_init__(self) -> None:
        if self.state_size < 32 or self.action_feature_size < 16:
            raise ValueError("core representation dimensions are too small")


class SchemaDrivenRepresentation:
    """Core-owned representation generated only from public typed data."""

    def __init__(
        self,
        schema: PluginSchema,
        *,
        config: CoreRepresentationConfig | None = None,
        experience: CoreExperienceMemory | None = None,
    ) -> None:
        self.schema = schema
        self.config = config or CoreRepresentationConfig()
        self.experience = experience or CoreExperienceMemory()

    @property
    def state_size(self) -> int:
        return self.config.state_size

    @property
    def action_feature_size(self) -> int:
        return self.config.action_feature_size

    def observation_vector(self, observation: PluginObservation) -> tuple[float, ...]:
        vector = [0.0] * self.state_size
        fields = self.schema.observation_map
        for name, value in sorted(observation.values.items()):
            field = fields.get(name)
            if field is None:
                continue
            for token, weight in _flatten_public_value(field=field, value=value):
                _add_hashed(vector, token, weight)
        return _normalized(vector)

    def semantic_observation_identity(
        self,
        observation: PluginObservation,
    ) -> tuple[tuple[str, str], ...]:
        return _canonical_observation(self.schema, observation, semantic=True)

    def full_observation_fingerprint(self, observation: PluginObservation) -> str:
        canonical = _canonical_observation(self.schema, observation, semantic=False)
        return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()

    def _public_values(self, observation: PluginObservation) -> set[str]:
        output: set[str] = set()
        for value in observation.values.values():
            if isinstance(value, Mapping):
                for key, item in value.items():
                    output.add(_stable_json(key))
                    output.add(_stable_json(item))
            elif isinstance(value, (tuple, list, set, frozenset)):
                output.update(_stable_json(item) for item in value)
            else:
                output.add(_stable_json(value))
        return output

    def _candidate_values(
        self,
        observation: PluginObservation,
        *,
        kind: ValueKind,
        enum_values: Sequence[str] = (),
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
            if field.kind is kind:
                values[_stable_json(raw)] = raw
            elif (
                field.kind is ValueKind.SET
                and field.item_kind is kind
            ):
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
        """Generate mechanically typed commands without ranking their usefulness."""

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

    def command_to_action(
        self,
        command: ActionCommand,
    ) -> Action:
        return command.to_action(self.schema)

    def action_structure_from_observation(
        self,
        observation: PluginObservation,
        action: Action,
    ) -> tuple[float, ...]:
        """Stable task-meaning-free action structure.

        Concrete parameter identities and learned outcome statistics are excluded
        so this key can be reused by Skills and calibration across renamed values.
        """

        vector = [0.0] * self.action_feature_size
        _add_hashed(vector, f"action:{action.verb_name}", 1.0)
        public_values = self._public_values(observation)
        spec = self.schema.action_map.get(action.verb_name)
        parameter_kinds = {
            item.name: item.kind.value
            for item in (() if spec is None else spec.parameters)
        }
        for name, value in sorted(action.parameters.items()):
            kind = parameter_kinds.get(name, "unknown")
            serialized = _stable_json(value)
            _add_hashed(vector, f"parameter:{name}:kind:{kind}", 1.0)
            _add_hashed(
                vector,
                f"parameter:{name}:seen-in-public-observation",
                1.0 if serialized in public_values else 0.0,
            )
        return _normalized(vector)

    def action_features_from_observation(
        self,
        observation: PluginObservation,
        action: Action,
    ) -> tuple[float, ...]:
        """Full Core-owned model features for one concrete action."""

        vector = list(self.action_structure_from_observation(observation, action))
        # Do not feed concrete entity IDs to transfer learners. Initially
        # symmetric candidates stay symmetric; real interaction makes them
        # distinguishable through learned, candidate-local experience statistics.
        for name, value in zip(
            CoreExperienceMemory.FEATURE_NAMES,
            self.experience.features(action.signature),
            strict=True,
        ):
            _add_hashed(vector, f"experience:{name}", value)

        return _normalized(vector)

    def state_vector(self, state: StateSnapshot) -> tuple[float, ...]:
        values = tuple(float(value) for value in state.vector)
        if len(values) != self.state_size:
            raise ValueError(
                f"state vector size mismatch: {len(values)} != {self.state_size}"
            )
        return values

    def action_features(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        cached = state.metadata.get("core_action_features")
        if isinstance(cached, Mapping):
            row = cached.get(action.signature)
            if isinstance(row, (tuple, list)) and len(row) == self.action_feature_size:
                return tuple(float(value) for value in row)
        observation = PluginObservation(values={})
        return self.action_features_from_observation(observation, action)

    def action_structure(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[float, ...]:
        cached = state.metadata.get("core_action_structures")
        if isinstance(cached, Mapping):
            row = cached.get(action.signature)
            if isinstance(row, (tuple, list)) and len(row) == self.action_feature_size:
                return tuple(float(value) for value in row)
        observation = PluginObservation(values={})
        return self.action_structure_from_observation(observation, action)

    def state_key(self, state: StateSnapshot) -> tuple[float, ...]:
        return tuple(round(value, 8) for value in self.state_vector(state))

    def semantic_state_identity(self, state: StateSnapshot) -> object:
        identity = state.metadata.get("core_semantic_identity")
        if identity is not None:
            return identity
        return (
            tuple(sorted(state.facts)),
            tuple(action.signature for action in state.available_actions),
        )

    def to_snapshot(
        self,
        result: PluginStepResult,
    ) -> StateSnapshot:
        observation = result.observation
        commands = self.synthesize_commands(observation)
        actions = tuple(
            sorted(
                (self.command_to_action(command) for command in commands),
                key=lambda item: item.signature,
            )
        )
        if result.terminated or result.truncated:
            actions = ()

        action_features = {
            action.signature: self.action_features_from_observation(observation, action)
            for action in actions
        }
        action_structures = {
            action.signature: self.action_structure_from_observation(observation, action)
            for action in actions
        }
        semantic_identity = (
            self.semantic_observation_identity(observation),
            tuple(action.signature for action in actions),
            self.experience.knowledge_revision,
        )
        facts = {
            f"public:{name}:{hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()[:20]}"
            for name, value in observation.values.items()
            if name in self.schema.observation_map
        }
        if result.terminated:
            facts.add("core:terminated")
        if result.truncated:
            facts.add("core:truncated")
        if result.error:
            facts.add("core:error")

        metadata = {
            "core_plugin_id": self.schema.plugin_id,
            "core_plugin_version": self.schema.version,
            "core_semantic_identity": semantic_identity,
            "core_observation_fingerprint": self.full_observation_fingerprint(observation),
            "core_action_features": MappingProxyType(action_features),
            "core_action_structures": MappingProxyType(action_structures),
            "core_terminated": bool(result.terminated),
            "core_truncated": bool(result.truncated),
            "core_error": bool(result.error),
            "core_error_code": result.error_code,
            "core_external_reward": float(result.reward),
            "core_experience_revision": self.experience.revision,
        }
        return StateSnapshot(
            vector=self.observation_vector(observation),
            facts=frozenset(facts),
            available_actions=actions,
            goal_progress=1.0 if result.terminated and result.reward > 0.0 else 0.0,
            metadata=metadata,
        )

    def observe_real_transition(
        self,
        *,
        before_result: PluginStepResult,
        action: Action,
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
            outcome_fingerprint=self.full_observation_fingerprint(
                after_result.observation
            ),
            reward=float(after_result.reward),
            error=bool(after_result.error),
            terminated=bool(after_result.terminated),
        )


class SchemaDrivenStateCodec(StateCodec):
    """Core-owned codec; a plugin never supplies model decoding behavior."""

    def __init__(self, representation: SchemaDrivenRepresentation) -> None:
        self.representation = representation

    @property
    def dimension(self) -> int:
        return self.representation.state_size

    def encode(self, state: StateSnapshot) -> tuple[float, ...]:
        return self.representation.state_vector(state)

    def decode(
        self,
        encoded: Sequence[float],
        *,
        scaffold: StateSnapshot,
        terminal_class: int,
        source: str,
    ) -> StateSnapshot:
        if len(encoded) != self.dimension:
            raise ValueError(
                f"decoded vector size mismatch: {len(encoded)} != {self.dimension}"
            )
        vector = tuple(
            max(-1.0, min(1.0, float(value)))
            for value in encoded
        )
        facts = set(scaffold.facts)
        facts.discard("core:terminated")
        facts.discard("core:truncated")
        if terminal_class == 1:
            facts.add("core:terminated")
            available = ()
        elif terminal_class == 2:
            facts.add("core:truncated")
            available = ()
        else:
            available = scaffold.available_actions
        goal_progress = scaffold.goal_progress
        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "core_imagined": True,
                "core_imagined_source": source,
                "core_terminated": terminal_class == 1,
                "core_truncated": terminal_class == 2,
            }
        )
        return StateSnapshot(
            vector=vector,
            facts=frozenset(facts),
            available_actions=available,
            goal_progress=goal_progress,
            metadata=metadata,
        )


class PluginEnvironmentAdapter:
    """Translate a minimal plugin session into the legacy StateSnapshot protocol.

    This class belongs to the Core boundary.  Plugins never construct
    StateSnapshot objects and therefore cannot smuggle learned representations or
    semantic task labels into the learner.
    """

    def __init__(
        self,
        plugin: Any,
        representation: SchemaDrivenRepresentation,
    ) -> None:
        self.plugin = plugin
        self.representation = representation
        self._result: PluginStepResult | None = None
        self._snapshot: StateSnapshot | None = None

    def reset(self, *, seed: int | None = None) -> StateSnapshot:
        result = self.plugin.reset(seed=seed)
        validate_step_result(self.representation.schema, result)
        self._result = result
        self._snapshot = self.representation.to_snapshot(result)
        return self._snapshot

    def snapshot(self) -> StateSnapshot:
        if self._snapshot is None:
            raise RuntimeError("environment adapter has not been reset")
        return self._snapshot

    def step(self, action: Action) -> PluginOutcome:
        if self._result is None or self._snapshot is None:
            raise RuntimeError("environment adapter has not been reset")
        before_result = self._result
        command = ActionCommand(action.verb_name, dict(action.parameters))
        after_result = self.plugin.step(command)
        validate_step_result(self.representation.schema, after_result)
        self.representation.observe_real_transition(
            before_result=before_result,
            action=action,
            after_result=after_result,
        )
        after = self.representation.to_snapshot(after_result)
        before = self._snapshot
        before_signatures = {
            item.signature for item in before.available_actions
        }
        unlocked = tuple(
            item
            for item in after.available_actions
            if item.signature not in before_signatures
        )
        outcome = PluginOutcome(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=unlocked,
            error=bool(after_result.error),
            error_code=after_result.error_code,
            raw={
                "external_reward": float(after_result.reward),
                "terminated": bool(after_result.terminated),
                "truncated": bool(after_result.truncated),
                **dict(after_result.diagnostics),
            },
        )
        self._result = after_result
        self._snapshot = after
        return outcome

    @property
    def terminated(self) -> bool:
        return bool(self._result and self._result.terminated)

    @property
    def truncated(self) -> bool:
        return bool(self._result and self._result.truncated)

    @property
    def terminal(self) -> bool:
        return self.terminated or self.truncated

    @property
    def external_reward(self) -> float:
        return 0.0 if self._result is None else float(self._result.reward)
