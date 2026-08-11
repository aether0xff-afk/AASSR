from __future__ import annotations

import pytest

from aassr_v2.current_relational_codec import (
    decode_relational_state,
    descriptor,
    legal_action_mask,
    transition_target,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def test_relational_decode_preserves_equivalent_action_multiplicity() -> None:
    actions = tuple(
        Action(
            "request",
            parameters={
                "route_id": f"route-{index:02d}",
                "profile_id": "profile-browse",
            },
        )
        for index in range(3)
    )
    facts = frozenset(
        {"known_profile:profile-browse"}
        | {f"known_route:route-{index:02d}" for index in range(3)}
    )
    actual = StateSnapshot(
        (0.0,) * AGENT_STATE_SIZE,
        facts=facts,
        available_actions=actions,
    )
    before = actual
    _, next_descriptor, next_mask, next_terminal = transition_target(
        before,
        actions[0],
        actual,
    )
    decoded = decode_relational_state(
        next_descriptor,
        next_mask,
        scaffold=before,
        predicted_terminal=next_terminal,
        source="unit-test",
    )

    assert sum(next_mask) == 1.0
    assert len(decoded.available_actions) == 3
    assert len({action.signature for action in decoded.available_actions}) == 3
    assert legal_action_mask(decoded) == next_mask
    assert descriptor(decoded) == pytest.approx(next_descriptor)
    assert not any(
        fact.endswith(":unknown") and fact.startswith("observed_route_role:")
        for fact in decoded.facts
    )
