from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_generation import relational_action_key
from aassr_v2.current_relational_model import (
    RelationalPrediction,
    RelationalProphecyConfig,
    RelationalStochasticProphecy,
)
from aassr_v2.current_semantic_calibration import SemanticCalibratedProphecy
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.replay import ReplayBuffer
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


def test_empirical_outcome_mass_is_separate_from_reliability() -> None:
    before, action = _state()
    catalog, _ = _state("catalog")
    auth, _ = _state("auth")
    prophecy = RelationalStochasticProphecy(
        seed=5,
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
    prophecy.learn(before, action, catalog)
    prophecy.learn(before, action, auth)
    rows = prophecy.predict(before, action, samples=3)

    assert all(isinstance(row, RelationalPrediction) for row in rows)
    assert sum(row.outcome_probability for row in rows) == pytest.approx(1.0)
    assert sorted(
        (row.outcome_probability for row in rows),
        reverse=True,
    ) == pytest.approx((2.0 / 3.0, 1.0 / 3.0))
    assert len({round(row.probability, 8) for row in rows}) == 1


def test_semantic_calibration_preserves_outcome_mass() -> None:
    before, action = _state()
    actual, _ = _state("catalog")
    base = RelationalStochasticProphecy(
        seed=6,
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
    base.learn(before, action, actual)
    calibrated = SemanticCalibratedProphecy(base, ReplayBuffer())
    raw = base.predict(before, action, samples=3)
    calibrated._cache[(
        relational_action_key(before, action),
        0,
        int(base.gradient_updates) // calibrated.refresh_stride,
    )] = 0.5
    rows = calibrated._calibrated(before, action, raw)

    assert [getattr(row, "outcome_probability", None) for row in rows] == [
        getattr(row, "outcome_probability", None) for row in raw
    ]
    assert [row.probability for row in rows] == pytest.approx(
        [row.probability * 0.5 for row in raw]
    )
