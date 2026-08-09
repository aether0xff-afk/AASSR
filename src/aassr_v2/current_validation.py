from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable, Sequence

from .metrics import prediction_similarity
from .pentest_agent_main_test import (
    AGENT_STATE_SIZE,
    CONTROL_SIZE,
    MAX_OBJECTS,
    MAX_ROUTES,
    PROFILE_ROLES,
    PROFILE_VECTOR_SIZE,
    ROUTE_ROLES,
)
from .replay import (
    PredictionValidator,
    ReplayTransition,
    ValidationScore,
    _prophecy_learning_revision,
)
from .skills import SKILL_VERB
from .types import Action, StateSnapshot


class CurrentVectorPredictionValidator(PredictionValidator):
    """Current-only holdout validator that skips symbolic prediction decoding.

    ``PredictionValidator`` scores only cosine similarity of the expected numeric
    next-state vector. The regular current Prophecy path decodes every predicted
    vector into a full ``StateSnapshot`` first, rebuilding facts and a potentially
    large Cartesian available-action surface that the validator immediately
    discards. At 2k real transitions this can create hundreds of thousands of
    unnecessary Python state/action objects.

    This validator preserves the exact holdout selection, before/after frequency,
    cache revision contract, samples setting, and the existing Python
    ``prediction_similarity`` implementation. It only computes the same decoded
    vector in one tensor batch and never materializes facts/actions for validation.
    """

    def __init__(
        self,
        agent: object,
        *,
        samples: int = 4,
        recent_limit: int = 64,
    ) -> None:
        super().__init__(samples=samples, recent_limit=recent_limit)
        self.agent = agent
        self.fast_calls = 0
        self.fast_rows = 0
        self.symbolic_fallback_calls = 0

    @staticmethod
    def _binary_slice(tensor: Any, start: int, stop: int) -> None:
        if stop <= start:
            return
        tensor[:, start:stop] = (tensor[:, start:stop] >= 0.5).to(tensor.dtype)

    def _decoded_expected_vectors(
        self,
        states: Sequence[StateSnapshot],
        actions: Sequence[Action],
    ) -> tuple[tuple[float, ...], ...]:
        neural = self.agent.base_neural_prophecy
        if len(states) != len(actions):
            raise ValueError("states/actions batch length mismatch")
        if not states:
            return ()

        # Match FullyRelationalNeuralDeltaProphecy.predict_batch warmup exactly:
        # before warmup it predicts the current state itself with zero confidence.
        if neural.observations < neural.config.warmup_steps:
            return tuple(tuple(float(value) for value in state.vector) for state in states)

        next_states, terminal, _ = neural._batch_outputs(states, actions)
        mean_states = next_states.mean(dim=0)
        terminal_classes = terminal.mean(dim=0).argmax(dim=1)

        # Tensor equivalent of HttpAgentCodec.decode(...).vector. Keep all
        # threshold/clamp semantics identical, but intentionally omit facts and
        # available_actions because vector cosine validation never reads them.
        bounded = neural.torch.clamp(mean_states, min=0.0, max=1.0).clone()
        if bounded.shape[1] != AGENT_STATE_SIZE:
            raise RuntimeError("current validation vector dimension drift")

        self._binary_slice(bounded, 0, 7)
        offset = CONTROL_SIZE
        self._binary_slice(bounded, offset, offset + MAX_ROUTES)
        offset += MAX_ROUTES
        self._binary_slice(bounded, offset, offset + PROFILE_VECTOR_SIZE)
        offset += PROFILE_VECTOR_SIZE
        self._binary_slice(bounded, offset, offset + MAX_OBJECTS)
        offset += MAX_OBJECTS

        role_end = (
            offset
            + len(ROUTE_ROLES) * MAX_ROUTES
            + len(PROFILE_ROLES) * PROFILE_VECTOR_SIZE
        )
        self._binary_slice(bounded, offset, role_end)
        offset = role_end
        object_role_end = offset + MAX_OBJECTS * 3
        self._binary_slice(bounded, offset, object_role_end)
        offset = object_role_end
        self._binary_slice(bounded, offset, AGENT_STATE_SIZE)

        success = (terminal_classes == 1) | (bounded[:, 3] >= 0.5)
        failed = (terminal_classes == 2) | (bounded[:, 4] >= 0.5)
        # Boolean advanced indexing is safe for an empty mask, so these stay on
        # device and require no batch-level ``.item()`` synchronization.
        bounded[success, 3] = 1.0
        bounded[success, 4] = 0.0
        bounded[success, 10] = 1.0
        failed_only = failed & ~success
        bounded[failed_only, 4] = 1.0

        host = bounded.detach().cpu().tolist()
        self.fast_calls += 1
        self.fast_rows += len(states)
        return tuple(tuple(float(value) for value in row) for row in host)

    def evaluate(
        self,
        prophecy: object,
        transitions: Iterable[ReplayTransition],
    ) -> ValidationScore:
        selected = tuple(transitions)[-self.recent_limit :]
        if not selected:
            return ValidationScore(0, 0.0)

        cache_key = (
            id(prophecy),
            self.samples,
            tuple(id(item) for item in selected),
            _prophecy_learning_revision(prophecy),
        )
        if cache_key == self._cache_key and self._cache_value is not None:
            self.cache_hits += 1
            return self._cache_value
        self.cache_misses += 1

        # Real evaluator holdout rows are primitive HTTP actions. Keep a generic
        # fallback so a future protocol that stores Skill macro rows cannot silently
        # use an invalid fast path.
        if any(item.action.verb_name == SKILL_VERB for item in selected):
            self.symbolic_fallback_calls += 1
            result = super().evaluate(prophecy, selected)
            self._cache_key = cache_key
            self._cache_value = result
            return result

        states = tuple(item.state for item in selected)
        actions = tuple(item.action for item in selected)
        expected_rows = self._decoded_expected_vectors(states, actions)
        scores = [
            prediction_similarity(expected, item.next_state.vector)
            for expected, item in zip(expected_rows, selected, strict=True)
        ]
        result = ValidationScore(len(scores), fmean(scores))
        self._cache_key = cache_key
        self._cache_value = result
        return result

    def runtime_diagnostics(self) -> dict[str, int]:
        return {
            **super().runtime_diagnostics(),
            "current_vector_fast_calls": self.fast_calls,
            "current_vector_fast_rows": self.fast_rows,
            "current_symbolic_fallback_calls": self.symbolic_fallback_calls,
        }


def install_current_fast_validation(agent: object) -> object:
    old = agent.evaluator.validator
    agent.evaluator.validator = CurrentVectorPredictionValidator(
        agent,
        samples=int(old.samples),
        recent_limit=int(old.recent_limit),
    )
    agent.current_fast_validation = True
    return agent
