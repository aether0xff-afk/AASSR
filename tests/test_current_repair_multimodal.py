from __future__ import annotations

import pytest

from aassr_v2.current_relational_codec import descriptor
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
    prophecy.learn(before, action, catalog)
    prophecy.learn(before, action, auth)

    predictions = prophecy.predict(before, action, samples=3)
    assert len(predictions) == 2
    assert all("empirical-outcome-" in item.source for item in predictions)
    assert len({descriptor(item.next_state) for item in predictions}) == 2
    diagnostics = prophecy.diagnostics()
    assert diagnostics["empirical_multimodal_input_keys"] == 1
    assert diagnostics["empirical_distinct_outcomes"] == 2
