from __future__ import annotations

from typing import Any, Sequence

from .empirical_confidence import empirical_confidence
from .integrated_agent import IntegratedProphecyView
from .types import Action, Prediction, StateSnapshot


class BatchedIntegratedProphecyView(IntegratedProphecyView):
    """Context-free holdout batch path through the full effect-composed stack.

    The expensive neural base is evaluated as one batch. Learned effect
    composition remains per-row Python logic because its buckets are symbolic;
    that preserves the canonical prediction semantics while moving the dense
    numerical work to CUDA.
    """

    def _effect_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        outer = self._prophecy
        ensure = getattr(outer, "_ensure_reconstructed", None)
        if callable(ensure):
            ensure()
        base = getattr(outer, "base", None)
        batch_predict = getattr(base, "predict_batch", None)
        if not callable(batch_predict):
            return tuple(
                tuple(outer.predict(state, action, samples=samples))
                for state, action in zip(states, actions, strict=True)
            )
        base_rows = batch_predict(states, actions, samples=samples)
        output: list[tuple[Prediction, ...]] = []

        for state, action, base_predictions in zip(
            states, actions, base_rows, strict=True
        ):
            bucket, tier, source, include_symbolic = outer._select_bucket(
                state, action
            )
            if not bucket:
                output.append(tuple(base_predictions))
                continue

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
                base_predictions,
                key=lambda prediction: prediction.probability,
            ).next_state
            merged: dict[tuple[Any, ...], list[Any]] = {}

            def add_prediction(prediction: Prediction, probability: float) -> None:
                if probability <= 0.0:
                    return
                key = outer._prediction_key(prediction.next_state)
                current = merged.get(key)
                if current is None:
                    merged[key] = [
                        prediction.next_state,
                        probability,
                        prediction.source,
                    ]
                else:
                    previous = float(current[1])
                    current[1] = previous + probability
                    if probability > previous:
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
                add_prediction(
                    Prediction(composed, probability, source), probability
                )

            normalized = outer._normalized_probabilities(tuple(base_predictions))
            for prediction, probability in zip(
                base_predictions, normalized, strict=True
            ):
                add_prediction(prediction, base_mass * probability)

            total = sum(float(item[1]) for item in merged.values())
            row = tuple(
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
            outer._composed_predictions += 1
            output.append(row)
        return tuple(output)

    def predict_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[Prediction, ...], ...]:
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        previous = self._contextual_skill_prophecy.knowledge
        self._contextual_skill_prophecy.bind_knowledge(None)
        try:
            return self._effect_batch(states, actions, samples=samples)
        finally:
            self._contextual_skill_prophecy.bind_knowledge(previous)


def enable_batched_integrated_prophecy(agent: object) -> object:
    """Install holdout batching and depth-wise Imagination batching.

    Learning objects and the frozen experiment runner remain unchanged. Only
    prediction execution is replaced by exact batch-capable views.
    """

    view = BatchedIntegratedProphecyView(
        agent.effect_prophecy,
        agent.skill_prophecy,
    )
    agent.prophecy = view
    agent.core.planner.prophecy = view
    agent.evaluator.prophecy = view

    from .native_batching import enable_depth_batched_imagination

    return enable_depth_batched_imagination(agent)
