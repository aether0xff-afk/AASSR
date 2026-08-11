from __future__ import annotations

import pytest

from aassr_v2.current_relational_state import install_relational_state_contract

install_relational_state_contract()

from aassr_v2.current_relational_codec import (
    decode_relational_state,
    descriptor,
    legal_action_mask,
    transition_target,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def test_relational_decode_keeps_multiplicity_as_summary_not_fake_actions() -> None:
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
    _, next_descriptor, next_mask, next_terminal = transition_target(
        actual,
        actions[0],
        actual,
    )
    decoded = decode_relational_state(
        next_descriptor,
        next_mask,
        scaffold=actual,
        predicted_terminal=next_terminal,
        source="unit-test",
    )

    # Three concrete aliases correspond to one observable structural action.
    assert sum(next_mask) == 1.0
    assert len(decoded.available_actions) == 1
    assert legal_action_mask(decoded) == next_mask
    # The public action-count summary is still preserved exactly for Policy/Critic.
    assert descriptor(decoded) == pytest.approx(next_descriptor)
    assert decoded.metadata["relational_action_count_fraction"] == pytest.approx(3 / 128)
    assert decoded.metadata["relational_unique_action_count_fraction"] == pytest.approx(1 / 32)
    assert all(
        "imagined_variant" not in action.parameters
        for action in decoded.available_actions
    )
    assert decoded.metadata["imagined_action_surface"] == (
        "one-action-per-relational-legal-slot"
    )
    assert not any(
        fact.endswith(":unknown") and fact.startswith("observed_route_role:")
        for fact in decoded.facts
    )
