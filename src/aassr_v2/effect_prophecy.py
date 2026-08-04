from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Mapping

from .empirical_confidence import empirical_confidence
from .prophecy import ProphecyStep
from .types import Action, Prediction, StateSnapshot


EffectContextKey = tuple[tuple[float, ...], tuple[str, ...], float]
EffectFingerprint = tuple[
    tuple[float, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    float,
]


def effect_context_key(state: StateSnapshot) -> EffectContextKey:
    """Return a compact, name-light context used for effect lookup.

    Numerical state and available action families are retained, while concrete
    fact names and action parameters are deliberately excluded. This allows an
    observed transition effect to be reused in a structurally similar state
    without pretending that two full snapshots are identical.
    """

    return (
        tuple(round(value, 4) for value in state.vector),
        tuple(sorted(action.verb_name for action in state.available_actions)),
        round(state.goal_progress, 4),
    )


@dataclass(frozen=True, slots=True)
class StateEffect:
    vector_delta: tuple[float, ...]
    added_facts: frozenset[str]
    removed_facts: frozenset[str]
    added_actions: tuple[Action, ...]
    removed_action_signatures: frozenset[str]
    goal_progress_delta: float

    @classmethod
    def from_transition(
        cls,
        before: StateSnapshot,
        after: StateSnapshot,
    ) -> StateEffect:
        before_actions = {
            action.signature: action for action in before.available_actions
        }
        after_actions = {
            action.signature: action for action in after.available_actions
        }
        return cls(
            vector_delta=tuple(
                round(right - left, 8)
                for left, right in zip(
                    before.vector,
                    after.vector,
                    strict=False,
                )
            ),
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            added_actions=tuple(
                after_actions[signature]
                for signature in sorted(after_actions.keys() - before_actions.keys())
            ),
            removed_action_signatures=frozenset(
                before_actions.keys() - after_actions.keys()
            ),
            goal_progress_delta=round(
                after.goal_progress - before.goal_progress,
                8,
            ),
        )

    @property
    def fingerprint(self) -> EffectFingerprint:
        return (
            self.vector_delta,
            tuple(sorted(self.added_facts)),
            tuple(sorted(self.removed_facts)),
            tuple(action.signature for action in self.added_actions),
            tuple(sorted(self.removed_action_signatures)),
            self.goal_progress_delta,
        )

    def apply(
        self,
        state: StateSnapshot,
        *,
        symbolic_scaffold: StateSnapshot | None = None,
        include_symbolic_delta: bool = True,
        source: str,
    ) -> StateSnapshot:
        scaffold = symbolic_scaffold or state
        vector = tuple(
            left + delta
            for left, delta in zip(
                state.vector,
                self.vector_delta,
                strict=False,
            )
        )
        if len(vector) < len(state.vector):
            vector += state.vector[len(vector) :]

        if include_symbolic_delta:
            facts = (state.facts - self.removed_facts) | self.added_facts
            action_map = {
                action.signature: action for action in state.available_actions
            }
            for signature in self.removed_action_signatures:
                action_map.pop(signature, None)
            for action in self.added_actions:
                action_map[action.signature] = action
            actions = tuple(
                action_map[signature] for signature in sorted(action_map)
            )
        else:
            facts = scaffold.facts
            actions = scaffold.available_actions

        metadata = dict(scaffold.metadata)
        metadata.update(
            {
                "imagined_effect_composed": True,
                "imagined_effect_source": source,
            }
        )
        return StateSnapshot(
            vector=vector,
            facts=frozenset(facts),
            available_actions=actions,
            goal_progress=min(
                1.0,
                max(0.0, state.goal_progress + self.goal_progress_delta),
            ),
            metadata=metadata,
        )


@dataclass(slots=True)
class _EffectEntry:
    effect: StateEffect
    count: int = 0


class EffectComposedProphecy:
    """Compose learned transition effects onto the current imagined state.

    Existing Prophecy implementations often retrieve a previously observed
    complete ``StateSnapshot`` after predicting its numerical vector. That is a
    useful fallback, but it cannot create a new combination of facts, actions,
    inventory and position. This adapter learns the observed *change* caused by
    each action and applies that change to the branch-local imagined state.

    The wrapped model still supplies uncertainty, recurrent memory and a
    fallback snapshot. The effect model supplies compositional state changes.
    """

    _DELEGATED_RESTORE_FIELDS = frozenset({"_exact", "_global", "_states"})

    def __init__(
        self,
        base: object,
        *,
        minimum_samples: int = 2,
        capacity_per_bucket: int = 16,
    ) -> None:
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if capacity_per_bucket <= 0:
            raise ValueError("capacity_per_bucket must be positive")
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "minimum_samples", int(minimum_samples))
        object.__setattr__(self, "capacity_per_bucket", int(capacity_per_bucket))
        object.__setattr__(self, "_exact_effects", {})
        object.__setattr__(self, "_signature_effects", {})
        object.__setattr__(self, "_family_effects", {})
        object.__setattr__(self, "_effect_observations", 0)
        object.__setattr__(self, "_composed_predictions", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in self._DELEGATED_RESTORE_FIELDS
            and "_base" in self.__dict__
        ):
            setattr(self._base, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def base(self) -> object:
        return self._base

    @property
    def name(self) -> str:
        return f"effect-composed:{getattr(self._base, 'name', 'prophecy')}"

    @property
    def effect_observations(self) -> int:
        return self._effect_observations

    @property
    def composed_predictions(self) -> int:
        return self._composed_predictions

    @property
    def effect_bucket_count(self) -> int:
        return (
            len(self._exact_effects)
            + len(self._signature_effects)
            + len(self._family_effects)
        )

    def initial_memory(self) -> Any:
        factory = getattr(self._base, "initial_memory", None)
        return factory() if callable(factory) else None

    def reset_sequence(self) -> None:
        reset = getattr(self._base, "reset_sequence", None)
        if callable(reset):
            reset()

    def reset_context(self) -> None:
        reset = getattr(self._base, "reset_context", None)
        if callable(reset):
            reset()

    def _bucket_observe(
        self,
        table: dict[Any, dict[EffectFingerprint, _EffectEntry]],
        key: Any,
        effect: StateEffect,
    ) -> None:
        bucket = table.setdefault(key, {})
        entry = bucket.get(effect.fingerprint)
        if entry is None:
            if len(bucket) >= self.capacity_per_bucket:
                least = min(
                    bucket,
                    key=lambda fingerprint: (
                        bucket[fingerprint].count,
                        repr(fingerprint),
                    ),
                )
                del bucket[least]
            entry = _EffectEntry(effect)
            bucket[effect.fingerprint] = entry
        entry.count += 1

    def _observe_effect(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        effect = StateEffect.from_transition(state, actual_next_state)
        self._bucket_observe(
            self._exact_effects,
            (effect_context_key(state), action.signature),
            effect,
        )
        self._bucket_observe(
            self._signature_effects,
            action.signature,
            effect,
        )
        self._bucket_observe(
            self._family_effects,
            action.verb_name,
            effect,
        )
        self._effect_observations += 1

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        learn = getattr(self._base, "learn", None)
        if callable(learn):
            learn(state, action, actual_next_state)
        self._observe_effect(state, action, actual_next_state)

    def _select_bucket(
        self,
        state: StateSnapshot,
        action: Action,
    ) -> tuple[dict[EffectFingerprint, _EffectEntry] | None, float, str, bool]:
        exact = self._exact_effects.get(
            (effect_context_key(state), action.signature)
        )
        if exact and sum(entry.count for entry in exact.values()) >= self.minimum_samples:
            return exact, 1.0, "effect-composed:exact", True

        signature = self._signature_effects.get(action.signature)
        if (
            signature
            and sum(entry.count for entry in signature.values())
            >= self.minimum_samples
        ):
            return signature, 0.85, "effect-composed:action-family", True

        family = self._family_effects.get(action.verb_name)
        if family and sum(entry.count for entry in family.values()) >= self.minimum_samples:
            return family, 0.45, "effect-composed:action-family", False
        return None, 0.0, "", False

    @staticmethod
    def _prediction_key(state: StateSnapshot) -> tuple[Any, ...]:
        return (
            tuple(round(value, 8) for value in state.vector),
            tuple(sorted(state.facts)),
            tuple(action.signature for action in state.available_actions),
            round(state.goal_progress, 8),
        )

    def _base_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        predict_step = getattr(self._base, "predict_step", None)
        if callable(predict_step):
            return predict_step(
                state,
                action,
                memory=memory,
                samples=samples,
            )
        predictions = self._base.predict(state, action, samples=samples)
        return ProphecyStep(predictions, memory)

    @staticmethod
    def _normalized_probabilities(
        predictions: tuple[Prediction, ...],
    ) -> tuple[float, ...]:
        total = sum(max(0.0, item.probability) for item in predictions)
        if total <= 0.0:
            return tuple(1.0 / len(predictions) for _ in predictions)
        return tuple(max(0.0, item.probability) / total for item in predictions)

    def predict_step(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        memory: Any,
        samples: int,
    ) -> ProphecyStep:
        if samples <= 0:
            raise ValueError("samples must be positive")
        base_step = self._base_step(
            state,
            action,
            memory=memory,
            samples=samples,
        )
        bucket, tier, source, include_symbolic = self._select_bucket(
            state,
            action,
        )
        if not bucket:
            return base_step

        ranked = sorted(
            bucket.values(),
            key=lambda entry: (-entry.count, repr(entry.effect.fingerprint)),
        )[:samples]
        effect_mass = min(
            0.95,
            empirical_confidence(
                (entry.count for entry in bucket.values()),
                prior_strength=1.0,
                tier=tier,
            ),
        )
        base_mass = 1.0 - effect_mass
        scaffold = max(
            base_step.predictions,
            key=lambda prediction: prediction.probability,
        ).next_state

        merged: dict[tuple[Any, ...], list[Any]] = {}

        def add_prediction(prediction: Prediction, probability: float) -> None:
            if probability <= 0.0:
                return
            key = self._prediction_key(prediction.next_state)
            current = merged.get(key)
            if current is None:
                merged[key] = [prediction.next_state, probability, prediction.source]
            else:
                current[1] += probability
                if probability > current[1] - probability:
                    current[2] = prediction.source

        ranked_total = sum(entry.count for entry in ranked)
        for entry in ranked:
            probability = effect_mass * entry.count / max(1, ranked_total)
            composed = entry.effect.apply(
                state,
                symbolic_scaffold=scaffold,
                include_symbolic_delta=include_symbolic,
                source=source,
            )
            add_prediction(Prediction(composed, probability, source), probability)

        base_probabilities = self._normalized_probabilities(base_step.predictions)
        for prediction, normalized in zip(
            base_step.predictions,
            base_probabilities,
            strict=True,
        ):
            add_prediction(prediction, base_mass * normalized)

        total = sum(float(item[1]) for item in merged.values())
        predictions = tuple(
            Prediction(
                next_state=item[0],
                probability=float(item[1]) / total,
                source=str(item[2]),
            )
            for _, item in sorted(
                merged.items(),
                key=lambda pair: (-float(pair[1][1]), repr(pair[0])),
            )
        )
        self._composed_predictions += 1
        return ProphecyStep(predictions, base_step.memory)

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        return self.predict_step(
            state,
            action,
            memory=self.initial_memory(),
            samples=samples,
        ).predictions

    def _effect_confidence(self, state: StateSnapshot, action: Action) -> float:
        bucket, tier, _, _ = self._select_bucket(state, action)
        if not bucket:
            return 0.0
        return empirical_confidence(
            (entry.count for entry in bucket.values()),
            prior_strength=1.0,
            tier=tier,
        )

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        base_confidence = 0.0
        confidence = getattr(self._base, "confidence", None)
        if callable(confidence):
            base_confidence = float(confidence(state, action))
        else:
            predictions = self._base.predict(state, action, samples=1)
            for prediction in predictions:
                source = prediction.source.lower()
                value = prediction.probability
                if source.endswith(":unseen"):
                    value = 0.0
                elif source.endswith(":action-family"):
                    value *= 0.5
                base_confidence = max(base_confidence, value)
        bucket, _, _, _ = self._select_bucket(state, action)
        if not bucket:
            return base_confidence
        effect_confidence = self._effect_confidence(state, action)
        if base_confidence <= 0.0:
            return effect_confidence
        if effect_confidence <= 0.0:
            return 0.0
        return sqrt(base_confidence * effect_confidence)

    def coverage(
        self,
        state: StateSnapshot,
        actions: Iterable[Action],
    ) -> float:
        materialized = tuple(actions)
        if not materialized:
            return 1.0
        return sum(
            self.confidence(state, action) for action in materialized
        ) / len(materialized)

    def diagnostics(self) -> Mapping[str, int]:
        return {
            "effect_observations": self.effect_observations,
            "effect_bucket_count": self.effect_bucket_count,
            "composed_predictions": self.composed_predictions,
        }
