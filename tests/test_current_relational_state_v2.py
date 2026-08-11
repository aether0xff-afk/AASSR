from __future__ import annotations

import pytest

from aassr_v2.current_relational_state import (
    AUDIT_PRESSURE_INDEX,
    REL_DESCRIPTOR_SIZE,
    REQUEST_USAGE_INDEX,
    ROLE_START_INDEX,
    SESSION_REMAINING_INDEX,
    WORKFLOW_PROGRESS_INDEX,
    install_relational_state_contract,
    relational_state_descriptor_v2,
    relational_state_key_v2,
)

install_relational_state_contract()

from aassr_v2.current_relational_codec import (
    decode_relational_state,
    descriptor,
    transition_target,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.pentest_curriculum_env import PROFILE_RELATIONS, ROUTE_RELATIONS
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.types import Action, StateSnapshot


def _state(
    *,
    hidden_audit_noise: float = 0.0,
    request_usage: float = 0.0,
    hidden_session_noise: float = 0.0,
    workflow_fraction: float = 0.0,
) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request",
        parameters={"route_id": "route-05", "profile_id": "profile-browse"},
    )
    vector = [0.0] * AGENT_STATE_SIZE
    vector[7] = hidden_audit_noise
    vector[8] = request_usage
    vector[9] = hidden_session_noise
    vector[10] = workflow_fraction
    progress = int(round(workflow_fraction * 8.0))
    return (
        StateSnapshot(
            tuple(vector),
            facts=frozenset(
                {
                    "known_route:route-05",
                    "known_profile:profile-browse",
                    f"workflow_progress:{progress}",
                }
            ),
            available_actions=(action,),
            metadata={
                "observation_contract": "response_causal_observation_v3",
                "request_count_scale": 100.0,
                "workflow_progress_scale": 8.0,
            },
        ),
        action,
    )


def test_v2_descriptor_keeps_public_progress_and_masks_hidden_pressure() -> None:
    state, _ = _state(
        hidden_audit_noise=0.6,
        request_usage=0.4,
        hidden_session_noise=0.25,
        workflow_fraction=0.5,
    )
    values = relational_state_descriptor_v2(state)

    assert len(values) == REL_DESCRIPTOR_SIZE == 35
    assert values[AUDIT_PRESSURE_INDEX] == 0.0
    assert values[REQUEST_USAGE_INDEX] == pytest.approx(0.4)
    assert values[SESSION_REMAINING_INDEX] == 0.0
    assert values[WORKFLOW_PROGRESS_INDEX] == pytest.approx(0.5)


def test_unobserved_known_entities_contribute_explicit_unknown_role_mass() -> None:
    action = Action(
        "request",
        parameters={"route_id": "route-00", "profile_id": "profile-browse"},
    )
    facts = frozenset(
        {
            "known_route:route-00",
            "known_route:route-01",
            "known_route:route-02",
            "observed_route_role:route-00:catalog",
            "known_profile:profile-browse",
            "known_profile:profile-01",
        }
    )
    state = StateSnapshot(
        (0.0,) * AGENT_STATE_SIZE,
        facts=facts,
        available_actions=(action,),
    )
    values = relational_state_descriptor_v2(state)

    route_start = ROLE_START_INDEX
    profile_start = route_start + len(ROUTE_RELATIONS)
    assert values[route_start + ROUTE_RELATIONS.index("catalog")] == pytest.approx(1 / 3)
    assert values[route_start + ROUTE_RELATIONS.index("unknown")] == pytest.approx(2 / 3)
    assert values[profile_start + PROFILE_RELATIONS.index("browse")] == pytest.approx(1 / 2)
    assert values[profile_start + PROFILE_RELATIONS.index("unknown")] == pytest.approx(1 / 2)


def test_hidden_audit_and_session_noise_do_not_change_relational_identity() -> None:
    left, _ = _state(
        hidden_audit_noise=0.1,
        request_usage=0.2,
        hidden_session_noise=0.9,
        workflow_fraction=0.25,
    )
    right, _ = _state(
        hidden_audit_noise=0.9,
        request_usage=0.2,
        hidden_session_noise=0.1,
        workflow_fraction=0.25,
    )

    assert relational_state_key_v2(left) == relational_state_key_v2(right)


def test_public_request_and_workflow_progress_change_relational_identity() -> None:
    early, _ = _state(request_usage=0.1, workflow_fraction=0.125)
    later, _ = _state(request_usage=0.7, workflow_fraction=0.5)

    assert relational_state_key_v2(early) != relational_state_key_v2(later)


def test_v2_prophecy_target_and_decode_round_trip_public_progress() -> None:
    before, action = _state(request_usage=0.25, workflow_fraction=0.125)
    after, _ = _state(request_usage=0.5, workflow_fraction=0.25)
    _, next_descriptor, next_mask, next_terminal = transition_target(
        before,
        action,
        after,
    )
    decoded = decode_relational_state(
        next_descriptor,
        next_mask,
        scaffold=before,
        predicted_terminal=next_terminal,
        source="unit-test",
    )

    assert descriptor(decoded) == pytest.approx(next_descriptor)
    assert decoded.vector[7] == 0.0
    assert decoded.vector[8] == pytest.approx(0.5)
    assert decoded.vector[9] == 0.0
    assert decoded.vector[10] == pytest.approx(0.25)
    assert decoded.metadata["relational_workflow_progress"] == pytest.approx(0.25)
    assert decoded.metadata["workflow_progress"] == 2
    assert "workflow_depth" not in decoded.metadata


def test_real_transfer_snapshot_uses_fixed_public_scales_without_hidden_depth() -> None:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[5])
    world.request_count = 7
    world.workflow_progress = 2
    world.audit_score = 2
    world.session_requests_remaining = 3

    state = world.snapshot()
    values = relational_state_descriptor_v2(state)

    assert state.vector[7] == 0.0
    assert state.vector[8] == pytest.approx(0.07)
    assert state.vector[9] == 0.0
    assert state.vector[10] == pytest.approx(0.25)
    assert values[AUDIT_PRESSURE_INDEX] == 0.0
    assert values[REQUEST_USAGE_INDEX] == pytest.approx(0.07)
    assert values[SESSION_REMAINING_INDEX] == 0.0
    assert values[WORKFLOW_PROGRESS_INDEX] == pytest.approx(0.25)
    assert "workflow_depth" not in state.metadata
    assert "workflow_progress:2" in state.facts
