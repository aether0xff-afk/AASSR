from __future__ import annotations

import pytest

from aassr_v2.current_relational_decode_v2 import decode_relational_state_v2 as stale_decode
from aassr_v2.current_relational_state_v3 import (
    REL_DESCRIPTOR_SIZE_V3,
    install_status_aware_relational_contract,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def _state(*, status: int | None = None) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request",
        parameters={"route_id": "route-05", "profile_id": "profile-browse"},
    )
    facts = {
        "known_route:route-05",
        "known_profile:profile-browse",
        "observed_route_role:route-05:catalog",
    }
    if status is not None:
        facts.add(f"last_status:{status}")
    return (
        StateSnapshot(
            vector=(0.0,) * AGENT_STATE_SIZE,
            facts=frozenset(facts),
            available_actions=(action,),
            metadata={},
        ),
        action,
    )


def test_stale_v2_decoder_dispatches_active_v3_target_after_install() -> None:
    # Reproduce the exact failure mode seen by the final runner: pytest collection
    # can bind a v2 decoder function before another module installs relational v3.
    # The already-imported function object must remain safe after that install.
    install_status_aware_relational_contract()

    from aassr_v2 import current_relational_codec as codec

    before, action = _state(status=200)
    after, _ = _state(status=404)
    _, next_descriptor, next_mask, next_terminal = codec.transition_target(
        before,
        action,
        after,
    )

    assert len(next_descriptor) == REL_DESCRIPTOR_SIZE_V3

    decoded = stale_decode(
        next_descriptor,
        next_mask,
        scaffold=before,
        predicted_terminal=next_terminal,
        source="import-order-regression",
    )

    assert "last_status:404" in decoded.facts
    assert codec.descriptor(decoded) == pytest.approx(next_descriptor)
    assert codec.legal_action_mask(decoded) == next_mask
