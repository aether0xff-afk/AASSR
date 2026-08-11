from __future__ import annotations

from dataclasses import replace

from aassr_v2.current_relational_codec import legal_action_mask, terminal_class
from aassr_v2.current_relational_state import AUDIT_PRESSURE_INDEX, SESSION_REMAINING_INDEX
from aassr_v2.current_relational_state_v3 import (
    REL_DESCRIPTOR_SIZE_V3,
    STATUS_CODES_V3,
    STATUS_START_INDEX,
    decode_relational_state_v3,
    latest_status_code,
    relational_state_descriptor_v3,
    semantic_prediction_score_v3,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, Prediction, StateSnapshot


def _state(status: int, *, audit: float = 0.87, session: float = 0.63) -> StateSnapshot:
    action = Action(
        "request",
        parameters={"route_id": "route-01", "profile_id": "profile-browse"},
    )
    vector = [0.0] * AGENT_STATE_SIZE
    vector[AUDIT_PRESSURE_INDEX] = audit
    vector[SESSION_REMAINING_INDEX] = session
    vector[AGENT_STATE_SIZE - len(STATUS_CODES_V3) + STATUS_CODES_V3.index(status)] = 1.0
    return StateSnapshot(
        vector=tuple(vector),
        facts=frozenset({"known_route:route-01", "known_profile:profile-browse"}),
        available_actions=(action,),
        goal_progress=0.0,
        metadata={},
    )


def test_relational_v3_preserves_public_status_but_masks_hidden_pressure() -> None:
    state = _state(403)
    descriptor = relational_state_descriptor_v3(state)
    assert len(descriptor) == REL_DESCRIPTOR_SIZE_V3
    assert descriptor[AUDIT_PRESSURE_INDEX] == 0.0
    assert descriptor[SESSION_REMAINING_INDEX] == 0.0
    status = descriptor[STATUS_START_INDEX:]
    assert status == tuple(float(code == 403) for code in STATUS_CODES_V3)
    assert latest_status_code(state) == 403


def test_relational_v3_distinguishes_403_from_404_without_concrete_ids() -> None:
    forbidden = _state(403)
    missing = _state(404)
    left = relational_state_descriptor_v3(forbidden)
    right = relational_state_descriptor_v3(missing)
    assert left[:STATUS_START_INDEX] == right[:STATUS_START_INDEX]
    assert left[STATUS_START_INDEX:] != right[STATUS_START_INDEX:]


def test_status_mismatch_materially_reduces_semantic_score() -> None:
    forbidden = _state(403)
    missing = _state(404)
    correct = semantic_prediction_score_v3(
        (Prediction(forbidden, 1.0, source="correct"),),
        forbidden,
    )
    wrong = semantic_prediction_score_v3(
        (Prediction(missing, 1.0, source="wrong-status"),),
        forbidden,
    )
    assert correct == 1.0
    assert correct - wrong >= 0.29


def test_v3_decode_round_trips_latest_public_status() -> None:
    state = _state(429)
    descriptor = relational_state_descriptor_v3(state)
    decoded = decode_relational_state_v3(
        descriptor,
        legal_action_mask(state),
        scaffold=state,
        predicted_terminal=terminal_class(state),
        source="test",
    )
    assert latest_status_code(decoded) == 429
    assert "last_status:429" in decoded.facts
    decoded_descriptor = relational_state_descriptor_v3(decoded)
    assert decoded_descriptor[STATUS_START_INDEX:] == descriptor[STATUS_START_INDEX:]
