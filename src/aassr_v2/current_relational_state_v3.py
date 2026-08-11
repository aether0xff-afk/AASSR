from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Sequence

from . import current_relational_state as state_v2
from .pentest_agent_main_test import AGENT_STATE_SIZE, STATUS_CODES
from .types import Prediction, StateSnapshot


BASE_REL_DESCRIPTOR_SIZE = int(state_v2.REL_DESCRIPTOR_SIZE)
STATUS_CODES_V3 = tuple(int(code) for code in STATUS_CODES)
STATUS_SIZE = len(STATUS_CODES_V3)
STATUS_START_INDEX = BASE_REL_DESCRIPTOR_SIZE
REL_DESCRIPTOR_SIZE_V3 = BASE_REL_DESCRIPTOR_SIZE + STATUS_SIZE
RAW_STATUS_START_INDEX = AGENT_STATE_SIZE - STATUS_SIZE


def _clamp01(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def latest_status_vector(state: StateSnapshot) -> tuple[float, ...]:
    """Return only the latest *publicly observed* HTTP status channel.

    The raw audited HTTP observation already contains these eight one-hot values.
    This function never reads audit score, lockout countdown, session TTL, hidden
    stage depth, or the hidden scenario.
    """
    explicit = state.metadata.get("relational_status_probabilities")
    if isinstance(explicit, (tuple, list)) and len(explicit) == STATUS_SIZE:
        return tuple(_clamp01(value) for value in explicit)

    explicit_code = state.metadata.get("relational_last_status")
    try:
        code = int(explicit_code) if explicit_code is not None else None
    except (TypeError, ValueError):
        code = None
    if code in STATUS_CODES_V3:
        return tuple(float(item == code) for item in STATUS_CODES_V3)

    for fact in state.facts:
        if not fact.startswith("last_status:"):
            continue
        try:
            code = int(fact.split(":", 1)[1])
        except (TypeError, ValueError):
            continue
        if code in STATUS_CODES_V3:
            return tuple(float(item == code) for item in STATUS_CODES_V3)

    if len(state.vector) >= AGENT_STATE_SIZE:
        values = state.vector[RAW_STATUS_START_INDEX : RAW_STATUS_START_INDEX + STATUS_SIZE]
        if len(values) == STATUS_SIZE:
            return tuple(_clamp01(value) for value in values)
    return (0.0,) * STATUS_SIZE


def latest_status_code(state: StateSnapshot) -> int | None:
    values = latest_status_vector(state)
    if not values or max(values) < 0.5:
        return None
    index = max(range(len(values)), key=lambda item: (values[item], -item))
    return STATUS_CODES_V3[index]


def relational_state_descriptor_v3(state: StateSnapshot) -> tuple[float, ...]:
    values = tuple(float(value) for value in state_v2.relational_state_descriptor_v2(state))
    if len(values) != BASE_REL_DESCRIPTOR_SIZE:
        raise AssertionError(
            f"relational v2 descriptor drift: {len(values)} != {BASE_REL_DESCRIPTOR_SIZE}"
        )
    descriptor = values + latest_status_vector(state)
    if len(descriptor) != REL_DESCRIPTOR_SIZE_V3:
        raise AssertionError("relational v3 descriptor size drift")
    return descriptor


def relational_state_vector_v3(state: StateSnapshot) -> tuple[float, ...]:
    descriptor = relational_state_descriptor_v3(state)
    if len(descriptor) > AGENT_STATE_SIZE:
        raise AssertionError("relational v3 descriptor exceeds DQN state size")
    return descriptor + (0.0,) * (AGENT_STATE_SIZE - len(descriptor))


def relational_state_key_v3(state: StateSnapshot) -> tuple[float, ...]:
    return tuple(round(float(value), 8) for value in relational_state_descriptor_v3(state))


def decode_relational_state_v3(
    predicted_descriptor: Sequence[float],
    mask_probabilities: Sequence[float],
    *,
    scaffold: StateSnapshot,
    predicted_terminal: int,
    source: str,
) -> StateSnapshot:
    """Decode v2 relational semantics, then restore the predicted public status."""
    from .current_relational_decode_v2 import decode_relational_state_v2

    values = tuple(_clamp01(value) for value in predicted_descriptor)
    if len(values) != REL_DESCRIPTOR_SIZE_V3:
        raise ValueError(
            f"unexpected relational v3 descriptor size {len(values)} != {REL_DESCRIPTOR_SIZE_V3}"
        )
    base = decode_relational_state_v2(
        values[:BASE_REL_DESCRIPTOR_SIZE],
        mask_probabilities,
        scaffold=scaffold,
        predicted_terminal=predicted_terminal,
        source=source,
    )
    status_probabilities = values[STATUS_START_INDEX:]
    status_code = None
    if status_probabilities and max(status_probabilities) >= 0.5:
        status_index = max(
            range(STATUS_SIZE),
            key=lambda item: (status_probabilities[item], -item),
        )
        status_code = STATUS_CODES_V3[status_index]

    vector = list(float(value) for value in base.vector)
    if len(vector) < AGENT_STATE_SIZE:
        vector.extend([0.0] * (AGENT_STATE_SIZE - len(vector)))
    for index in range(STATUS_SIZE):
        vector[RAW_STATUS_START_INDEX + index] = 0.0
    if status_code is not None:
        vector[RAW_STATUS_START_INDEX + STATUS_CODES_V3.index(status_code)] = 1.0

    facts = {fact for fact in base.facts if not fact.startswith("last_status:")}
    if status_code is not None:
        facts.add(f"last_status:{status_code}")

    metadata = dict(base.metadata)
    metadata.update(
        {
            "relational_state_version": "v3-public-status",
            "relational_last_status": status_code,
            "relational_status_probabilities": tuple(status_probabilities),
        }
    )
    return replace(
        base,
        vector=tuple(vector),
        facts=frozenset(facts),
        metadata=metadata,
    )


def status_match_score(predicted: StateSnapshot, actual: StateSnapshot) -> float:
    return float(latest_status_code(predicted) == latest_status_code(actual))


def semantic_prediction_score_v3(
    predictions: Sequence[Prediction],
    actual: StateSnapshot,
) -> float:
    """Decision-critical semantic score with HTTP status as an explicit term."""
    if not predictions:
        return 0.0
    from .current_relational_codec import legal_action_mask, terminal_class

    target = relational_state_descriptor_v3(actual)
    target_base = target[:BASE_REL_DESCRIPTOR_SIZE]
    target_mask = legal_action_mask(actual)
    target_terminal = terminal_class(actual)

    def mask_jaccard(left: StateSnapshot, right_mask: Sequence[float]) -> float:
        left_mask = legal_action_mask(left)
        a = {index for index, value in enumerate(left_mask) if float(value) >= 0.5}
        b = {index for index, value in enumerate(right_mask) if float(value) >= 0.5}
        return 1.0 if not (a or b) else len(a & b) / len(a | b)

    scores = []
    for prediction in predictions:
        predicted = prediction.next_state
        predicted_base = relational_state_descriptor_v3(predicted)[:BASE_REL_DESCRIPTOR_SIZE]
        semantic_error = fmean(
            abs(left - right)
            for left, right in zip(predicted_base, target_base, strict=True)
        )
        scores.append(
            0.35 * max(0.0, 1.0 - semantic_error)
            + 0.25 * mask_jaccard(predicted, target_mask)
            + 0.30 * status_match_score(predicted, actual)
            + 0.10 * float(terminal_class(predicted) == target_terminal)
        )
    return max(scores)


def install_status_aware_relational_contract() -> None:
    """Install relational v3 before Policy/Prophecy/Critic construction.

    Compatibility aliases are patched deliberately so historical modules can stay
    importable while the active current-generation builder uses the v3 contract.
    """
    from . import current_generation as generation
    from . import current_hardware as hardware
    from . import current_relational_codec as codec
    from . import current_relational_model as model
    from . import current_relational_mixture_model as mixture
    from . import current_repair as repair
    from . import current_semantic_calibration as calibration
    from . import current_semantic_evaluator as evaluator
    from . import dreamerv3_baseline as dreamer

    state_v2.REL_DESCRIPTOR_SIZE = REL_DESCRIPTOR_SIZE_V3
    state_v2.STATUS_START_INDEX = STATUS_START_INDEX
    state_v2.STATUS_CODES_V3 = STATUS_CODES_V3
    state_v2.relational_state_descriptor_v3 = relational_state_descriptor_v3
    state_v2.relational_state_vector_v3 = relational_state_vector_v3
    state_v2.relational_state_key_v3 = relational_state_key_v3

    generation.relational_state_descriptor = relational_state_descriptor_v3
    generation.relational_state_vector = relational_state_vector_v3
    generation.relational_state_key = relational_state_key_v3
    hardware.relational_state_key = relational_state_key_v3

    codec.REL_DESCRIPTOR_SIZE = REL_DESCRIPTOR_SIZE_V3
    codec.relational_state_descriptor_v2 = relational_state_descriptor_v3
    codec.relational_state_vector_v2 = relational_state_vector_v3
    codec.decode_relational_state = decode_relational_state_v3
    codec.semantic_prediction_score = semantic_prediction_score_v3

    model.REL_DESCRIPTOR_SIZE = REL_DESCRIPTOR_SIZE_V3
    model.relational_state_vector_v2 = relational_state_vector_v3
    model.decode_relational_state_v2 = decode_relational_state_v3

    mixture.REL_DESCRIPTOR_SIZE = REL_DESCRIPTOR_SIZE_V3
    mixture.relational_state_vector_v2 = relational_state_vector_v3
    mixture.decode_relational_state_v2 = decode_relational_state_v3

    calibration.semantic_prediction_score = semantic_prediction_score_v3
    evaluator.semantic_prediction_score = semantic_prediction_score_v3
    evaluator.relational_state_key_v2 = relational_state_key_v3
    repair.relational_state_key_v2 = relational_state_key_v3
    dreamer.relational_state_vector_v2 = relational_state_vector_v3

    try:
        from . import current_runtime as runtime
        if hasattr(runtime, "relational_state_vector"):
            runtime.relational_state_vector = relational_state_vector_v3
        if hasattr(runtime, "relational_state_key"):
            runtime.relational_state_key = relational_state_key_v3
    except ImportError:  # pragma: no cover
        pass
