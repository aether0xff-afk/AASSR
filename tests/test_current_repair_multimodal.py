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
    return (
        StateSnapshot(
            (0.0,) * AGENT_STATE_SIZE,
            facts=frozenset(facts),
            available_actions=(action,),
        ),
        action,
    )


def _prophecy() -> RelationalStochasticProphecy:
    pytest.importorskip("torch")
    return RelationalStochasticProphecy(
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


def _learn_two_outcomes() -> tuple[
    RelationalStochasticProphecy,
    StateSnapshot,
    Action,
]:
    before, action = _state()
    catalog, _ = _state("catalog")
    auth, _ = _state("auth")
    prophecy = _prophecy()
    prophecy.learn(before, action, catalog)
    prophecy.learn(before, action, auth)
    return prophecy, before, action


def test_multimodal_targets_share_input_but_keep_distinct_outputs() -> None:
    before, action = _state()
    catalog, _ = _state("catalog")
    auth, _ = _state("auth")
    catalog_target = transition_target(before, action, catalog)
    auth_target = transition_target(before, action, auth)

    assert catalog_target[0] == auth_target[0]
    assert catalog_target[1:] != auth_target[1:]
    assert descriptor(catalog) != descriptor(auth)


def test_multimodal_empirical_store_keeps_two_outcomes() -> None:
    prophecy, before, action = _learn_two_outcomes()
    bucket = prophecy._outcomes.get(prophecy._input(before, action), {})

    assert len(bucket) == 2, tuple(bucket.items())
    diagnostics = prophecy.diagnostics()
    assert diagnostics["empirical_multimodal_input_keys"] == 1
    assert diagnostics["empirical_distinct_outcomes"] == 2


def test_multimodal_empirical_store_emits_two_predictions() -> None:
    prophecy, before, action = _learn_two_outcomes()
    predictions = prophecy.predict(before, action, samples=3)

    assert len(predictions) == 2, tuple(
        (item.source, getattr(item, "outcome_probability", None))
        for item in predictions
    )
    assert all("empirical-outcome-" in item.source for item in predictions)
    assert sum(
        getattr(item, "outcome_probability", 0.0) for item in predictions
    ) == pytest.approx(1.0)


def test_multimodal_empirical_decode_keeps_two_semantic_modes() -> None:
    prophecy, before, action = _learn_two_outcomes()
    predictions = prophecy.predict(before, action, samples=3)
    predicted_descriptors = {
        descriptor(item.next_state) for item in predictions
    }

    assert len(predictions) == 2
    assert len(predicted_descriptors) == 2, tuple(
        descriptor(item.next_state) for item in predictions
    )
