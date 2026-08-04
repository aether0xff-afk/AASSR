from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from .effect_prophecy import (
    EffectComposedProphecy,
    EffectContextKey,
    EffectFingerprint,
    StateEffect,
    _EffectEntry,
    effect_context_key,
)
from .types import Action, Prediction, StateSnapshot


_EFFECT_METADATA_KEY = "aassr_effect_records_v1"


def _serialize_action(action: Action) -> dict[str, Any]:
    return {
        "verb": action.verb_name,
        "target": action.target,
        "tool": action.tool,
        "destination": action.destination,
        "metadata": dict(action.metadata),
        "parameters": dict(action.parameters),
    }


def _deserialize_action(payload: Mapping[str, Any]) -> Action:
    return Action(
        str(payload.get("verb", "")),
        target=payload.get("target"),
        tool=payload.get("tool"),
        destination=payload.get("destination"),
        metadata=dict(payload.get("metadata", {})),
        parameters=dict(payload.get("parameters", {})),
    )


def _serialize_context(context: EffectContextKey) -> list[Any]:
    vector, verbs, progress = context
    return [list(vector), list(verbs), progress]


def _deserialize_context(payload: Any) -> EffectContextKey:
    vector, verbs, progress = payload
    return (
        tuple(float(value) for value in vector),
        tuple(str(value) for value in verbs),
        float(progress),
    )


def _serialize_effect(effect: StateEffect) -> dict[str, Any]:
    return {
        "vector_delta": list(effect.vector_delta),
        "added_facts": sorted(effect.added_facts),
        "removed_facts": sorted(effect.removed_facts),
        "added_actions": [
            _serialize_action(action) for action in effect.added_actions
        ],
        "removed_action_signatures": sorted(
            effect.removed_action_signatures
        ),
        "goal_progress_delta": effect.goal_progress_delta,
    }


def _deserialize_effect(payload: Mapping[str, Any]) -> StateEffect:
    return StateEffect(
        vector_delta=tuple(
            float(value) for value in payload.get("vector_delta", ())
        ),
        added_facts=frozenset(
            str(value) for value in payload.get("added_facts", ())
        ),
        removed_facts=frozenset(
            str(value) for value in payload.get("removed_facts", ())
        ),
        added_actions=tuple(
            _deserialize_action(item)
            for item in payload.get("added_actions", ())
        ),
        removed_action_signatures=frozenset(
            str(value)
            for value in payload.get("removed_action_signatures", ())
        ),
        goal_progress_delta=float(
            payload.get("goal_progress_delta", 0.0)
        ),
    )


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    action = _deserialize_action(record.get("action", {}))
    effect = _deserialize_effect(record.get("effect", {}))
    return (
        repr(record.get("context", ())),
        action.signature,
        repr(effect.fingerprint),
    )


class PersistentEffectComposedProphecy(EffectComposedProphecy):
    """Effect-composed Prophecy whose learned deltas survive checkpoints.

    Existing portable checkpoints already serialize every Tabular Prophecy
    ``StateSnapshot``, including its metadata.  This class stores a compact,
    deduplicated effect ledger in that metadata.  After loading a model, the
    effect buckets are reconstructed lazily before the first prediction.

    Neural Prophecy checkpoints still require their own weight serializer; this
    class does not pretend that storing symbolic effect records also stores GRU
    parameters.
    """

    def __init__(
        self,
        base: object,
        *,
        minimum_samples: int = 2,
        capacity_per_bucket: int = 16,
    ) -> None:
        super().__init__(
            base,
            minimum_samples=minimum_samples,
            capacity_per_bucket=capacity_per_bucket,
        )
        self._effects_reconstructed = False

    @staticmethod
    def _insert_count(
        table: dict[Any, dict[EffectFingerprint, _EffectEntry]],
        key: Any,
        effect: StateEffect,
        count: int,
    ) -> None:
        bucket = table.setdefault(key, {})
        entry = bucket.get(effect.fingerprint)
        if entry is None:
            bucket[effect.fingerprint] = _EffectEntry(effect, count)
        else:
            entry.count += count

    def _ensure_reconstructed(self) -> None:
        if self._effects_reconstructed or self._effect_observations:
            self._effects_reconstructed = True
            return
        states = getattr(self.base, "_states", None)
        if not isinstance(states, Mapping):
            self._effects_reconstructed = True
            return

        observation_count = 0
        for snapshot in states.values():
            records = snapshot.metadata.get(_EFFECT_METADATA_KEY, ())
            if not isinstance(records, (list, tuple)):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                action = _deserialize_action(record.get("action", {}))
                effect = _deserialize_effect(record.get("effect", {}))
                context = _deserialize_context(record.get("context", ((), (), 0.0)))
                count = max(1, int(record.get("count", 1)))
                self._insert_count(
                    self._exact_effects,
                    (context, action.signature),
                    effect,
                    count,
                )
                self._insert_count(
                    self._signature_effects,
                    action.signature,
                    effect,
                    count,
                )
                self._insert_count(
                    self._family_effects,
                    action.verb_name,
                    effect,
                    count,
                )
                observation_count += count
        self._effect_observations = observation_count
        self._effects_reconstructed = True

    def _decorate_tabular_next_state(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
        effect: StateEffect,
    ) -> StateSnapshot:
        states = getattr(self.base, "_states", None)
        if not isinstance(states, Mapping):
            return actual_next_state
        try:
            from .tabular_prophecy import state_fingerprint

            fingerprint = state_fingerprint(actual_next_state)
        except (ImportError, AttributeError, TypeError, ValueError):
            return actual_next_state

        existing = states.get(fingerprint)
        metadata = dict(
            existing.metadata if existing is not None else actual_next_state.metadata
        )
        raw_records = metadata.get(_EFFECT_METADATA_KEY, ())
        records = [
            dict(record)
            for record in raw_records
            if isinstance(record, Mapping)
        ]
        new_record = {
            "context": _serialize_context(effect_context_key(state)),
            "action": _serialize_action(action),
            "effect": _serialize_effect(effect),
            "count": 1,
        }
        new_key = _record_key(new_record)
        for record in records:
            if _record_key(record) == new_key:
                record["count"] = int(record.get("count", 1)) + 1
                break
        else:
            records.append(new_record)
        metadata[_EFFECT_METADATA_KEY] = records
        return replace(actual_next_state, metadata=metadata)

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        self._ensure_reconstructed()
        effect = StateEffect.from_transition(state, actual_next_state)
        decorated = self._decorate_tabular_next_state(
            state,
            action,
            actual_next_state,
            effect,
        )
        learn = getattr(self.base, "learn", None)
        if callable(learn):
            learn(state, action, decorated)
        self._observe_effect(state, action, actual_next_state)

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ):
        self._ensure_reconstructed()
        return super().predict_step(
            state,
            action,
            memory=memory,
            samples=samples,
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        self._ensure_reconstructed()
        return super().predict(state, action, samples=samples)

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        self._ensure_reconstructed()
        return super().confidence(state, action)

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        self._ensure_reconstructed()
        return super().coverage(state, actions)

    def diagnostics(self) -> Mapping[str, int]:
        self._ensure_reconstructed()
        return super().diagnostics()
