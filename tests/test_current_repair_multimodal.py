from __future__ import annotations

import pytest

from aassr_v2.current_relational_state import install_relational_state_contract

install_relational_state_contract()

from aassr_v2.current_relational_codec import descriptor, transition_target
from aassr_v2.current_relational_model import (
    RelationalProphecyConfig,
    RelationalStochasticProphecy,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def _state(role: str | None = None) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request",
        parameters={"route_id": "route-05", "profile_id": "profile-browse"},
    )
    facts = {"known_route:route-05", "known_profile:profile-browse"}
    if role is not None:
        facts.add(f"observed_route_role:route-05:{role}")
    state = StateSnapshot(
        (0.0,) * AGENT_STATE_SIZE,
        facts=frozenset(facts),
        available_actions=(action,),
    )
    return state, action


def test_same_relational_input_with_two_real_outcomes_stays_multimodal() -> None:
    pytest.importorskip("torch")
    before, action = _state()
    catalog, _ = _state("catalog")
    auth, _ = _state("auth")

    prophecy = RelationalStochasticProphecy(
        seed=13,
        device="cpu",
        config=RelationalProphecyConfig(
            hidden_units=16,
            ensemble_size=3,
            replay_capacity=16,
            batch_size=1,
            warmup_steps=1,
            gradient_steps_per_observation=1,
        ),
    )

    catalog_target = transition_target(before, action, catalog)
    auth_target = transition_target(before, action, auth)
    assert catalog_target[0] == auth_target[0], (
        "same public relational state/action must have one common input key"
    )
    assert catalog_target[1:] != auth_target[1:], (
        "catalog/auth outcomes collapsed before reaching Prophecy replay",
        catalog_target[1],
        auth_target[1],
    )
    assert descriptor(catalog) != descriptor(auth), (
        "real catalog/auth semantic descriptors unexpectedly alias"
    )

    prophecy.learn(before, action, catalog)
    prophecy.learn(before, action, auth)

    input_key = prophecy._input(before, action)
    bucket = prophecy._outcomes.get(input_key, {})
    assert len(bucket) == 2, (
        "two distinct relational targets did not survive empirical outcome storage",
        tuple(bucket.items()),
    )

    predictions = prophecy.predict(before, action, samples=3)
    assert len(predictions) == 2, (
        "stored multimodal bucket was not emitted as two predictions",
        tuple((item.source, getattr(item, "outcome_probability", None)) for item in predictions),
    )
    assert all("empirical-outcome-" in item.source for item in predictions)
    predicted_descriptors = {
        descriptor(item.next_state) for item in predictions
    }
    assert len(predicted_descriptors) == 2, (
        "two empirical targets collapsed during relational decode",
        tuple(descriptor(item.next_state) for item in predictions),
    )
    diagnostics = prophecy.diagnostics()
    assert diagnostics["empirical_multimodal_input_keys"] == 1
    assert diagnostics["empirical_distinct_outcomes"] == 2
