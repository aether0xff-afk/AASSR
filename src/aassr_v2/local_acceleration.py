from __future__ import annotations

from typing import Any, Sequence

from .empirical_confidence import empirical_confidence
from .integrated_agent import IntegratedProphecyView
from .metrics import expected_prediction_vector
from .skills import SKILL_VERB
from .torch_gru_prophecy import TorchGRUProphecy
from .types import Action, Prediction, StateSnapshot


class BatchedIntegratedProphecyView(IntegratedProphecyView):
    """Batch-capable prediction view for local high-throughput evaluation.

    Full symbolic predictions remain available for Imagination. Replay holdout
    validation also gets an expected-vector-only path that skips constructing
    symbolic StateSnapshot/Prediction objects the validator never reads.
    """

    def __init__(self, prophecy: object, contextual_skill_prophecy: object) -> None:
        super().__init__(prophecy, contextual_skill_prophecy)
        self.expected_vector_batch_calls = 0
        self.expected_vector_batch_rows = 0

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

    @staticmethod
    def _torch_base_expected_tensor(
        model: TorchGRUProphecy,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ):
        torch = model.torch
        with torch.inference_mode():
            x = model._inputs(states, actions)
            hidden = torch.zeros(
                (len(states), model.hidden_size),
                device=model.device,
                dtype=model.dtype,
            )
            outputs, _ = model._forward_batch(x, hidden)
            current = torch.tensor(
                [tuple(state.vector) for state in states],
                device=model.device,
                dtype=model.dtype,
            )
            matrix, templates = model._templates()
            if matrix is None or not templates:
                return current

            out_norm = torch.linalg.vector_norm(outputs, dim=1, keepdim=True)
            template_norm = torch.linalg.vector_norm(matrix, dim=1, keepdim=True).T
            denom = out_norm * template_norm
            similarities = outputs @ matrix.T
            similarities = torch.where(
                denom > 0,
                similarities / torch.clamp_min(denom, 1e-12),
                torch.zeros_like(similarities),
            )
            k = min(int(samples), len(templates))
            order = torch.argsort(
                similarities,
                dim=1,
                descending=True,
                stable=True,
            )[:, :k]
            scores = torch.gather(similarities, 1, order)
            probabilities = torch.softmax(scores * 4.0, dim=1)
            selected = matrix[order]
            template_mean = (
                selected * probabilities.unsqueeze(-1)
            ).sum(dim=1)
            confidence = torch.tensor(
                [model.confidence(state, action) for state, action in zip(states, actions, strict=True)],
                device=model.device,
                dtype=model.dtype,
            ).unsqueeze(1)
            return confidence * template_mean + (1.0 - confidence) * current

    def expected_vector_batch(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
        *,
        samples: int,
    ) -> tuple[tuple[float, ...], ...]:
        """Return the exact probability-weighted vectors used by holdout scoring.

        This is mathematically equivalent to ``expected_prediction_vector`` over
        ``predict_batch`` for primitive TorchGRU transitions. Symbolic facts and
        action sets are intentionally not materialized because PredictionValidator
        only consumes the resulting numeric vectors.
        """

        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if not states:
            return ()
        if samples <= 0:
            raise ValueError("samples must be positive")

        outer = self._prophecy
        ensure = getattr(outer, "_ensure_reconstructed", None)
        if callable(ensure):
            ensure()
        skill_wrapper = getattr(outer, "base", None)
        neural = getattr(skill_wrapper, "base", None)
        if (
            not isinstance(neural, TorchGRUProphecy)
            or any(action.verb_name == SKILL_VERB for action in actions)
        ):
            rows = self.predict_batch(states, actions, samples=samples)
            return tuple(expected_prediction_vector(row) for row in rows)

        previous = self._contextual_skill_prophecy.knowledge
        self._contextual_skill_prophecy.bind_knowledge(None)
        try:
            base_expected = self._torch_base_expected_tensor(
                neural,
                states,
                actions,
                samples=samples,
            )
            effect_vectors: list[tuple[float, ...]] = []
            effect_masses: list[float] = []
            vector_size = neural.state_size

            for state, action in zip(states, actions, strict=True):
                bucket, tier, _, _ = outer._select_bucket(state, action)
                if not bucket:
                    effect_masses.append(0.0)
                    effect_vectors.append(tuple(float(value) for value in state.vector))
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
                ranked_total = max(1, sum(entry.count for entry in ranked))
                delta = [0.0] * vector_size
                for entry in ranked:
                    weight = entry.count / ranked_total
                    for index, value in enumerate(entry.effect.vector_delta[:vector_size]):
                        delta[index] += weight * float(value)
                effect_vectors.append(
                    tuple(
                        float(state.vector[index]) + delta[index]
                        for index in range(vector_size)
                    )
                )
                effect_masses.append(effect_mass)
                outer._composed_predictions += 1

            torch = neural.torch
            with torch.inference_mode():
                effects = torch.tensor(
                    effect_vectors,
                    device=neural.device,
                    dtype=neural.dtype,
                )
                masses = torch.tensor(
                    effect_masses,
                    device=neural.device,
                    dtype=neural.dtype,
                ).unsqueeze(1)
                expected = masses * effects + (1.0 - masses) * base_expected
                rows = tuple(
                    tuple(float(value) for value in row)
                    for row in expected.cpu().tolist()
                )
            self.expected_vector_batch_calls += 1
            self.expected_vector_batch_rows += len(states)
            return rows
        finally:
            self._contextual_skill_prophecy.bind_knowledge(previous)

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            "expected_vector_batch_calls": self.expected_vector_batch_calls,
            "expected_vector_batch_rows": self.expected_vector_batch_rows,
        }


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
