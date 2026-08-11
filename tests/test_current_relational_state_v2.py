from __future__ import annotations

import pytest

from aassr_v2.current_relational_state import (
    AUDIT_PRESSURE_INDEX,
    REL_DESCRIPTOR_SIZE,
    REQUEST_USAGE_INDEX,
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
from aassr_v2.types import Action, StateSnapshot


def _state(
    *,
    audit: float,
    requests: float,
    session: float,
    workflow_progress: int,
    workflow_depth: int = 4,
) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request",
        parameters={"route_id": "route-05", "profile_id": "profile-browse"},
    )
    vector = [0.0] * AGENT_STATE_SIZE
    vector[7] = audit
    vector[8] = requests
    vector[9] = session
    return (
        StateSnapshot(
            tuple(vector),
            facts=frozenset(
                {
                    "known_route:route-05",
                    "known_profile:profile-browse",
                    f"workflow_progress:{workflow_progress}:{workflow_depth}",
                }
            ),
            available_actions=(action,),
            metadata={
                "workflow_progress": workflow_progress,
                "workflow_depth": workflow_depth,
            },
        ),
        action,
    )


def test_v2_descriptor_keeps_all_public_resource_pressure_axes() -> None:
    state, _ = _state(
        audit=0.6,
        requests=0.4,
        session=0.25,
        workflow_progress=2,
    )
    values = relational_state_descriptor_v2(state)

    assert len(values) == REL_DESCRIPTOR_SIZE == 35
    assert values[AUDIT_PRESSURE_INDEX] == pytest.approx(0.6)
    assert values[REQUEST_USAGE_INDEX] == pytest.approx(0.4)
    assert values[SESSION_REMAINING_INDEX] == pytest.approx(0.25)
    assert values[WORKFLOW_PROGRESS_INDEX] == pytest.approx(0.5)


def test_resource_pressure_changes_relational_identity() -> None:
    safe, _ = _state(
        audit=0.1,
        requests=0.2,
        session=0.9,
        workflow_progress=1,
    )
    pressured, _ = _state(
        audit=0.9,
        requests=0.8,
        session=0.2,
        workflow_progress=3,
    )

    assert relational_state_key_v2(safe) != relational_state_key_v2(pressured)


def test_v2_prophecy_target_and_decode_round_trip_resource_pressure() -> None:
    before, action = _state(
        audit=0.2,
        requests=0.25,
        session=0.75,
        workflow_progress=1,
    )
    after, _ = _state(
        audit=0.7,
        requests=0.5,
        session=0.5,
        workflow_progress=2,
    )
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
    assert decoded.vector[7] == pytest.approx(0.7)
    assert decoded.vector[8] == pytest.approx(0.5)
    assert decoded.vector[9] == pytest.approx(0.5)
    assert decoded.metadata["relational_workflow_progress"] == pytest.approx(0.5)
    assert decoded.metadata["workflow_progress"] == 2
