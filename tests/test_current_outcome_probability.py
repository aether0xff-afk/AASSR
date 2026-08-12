from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_relational_state import install_relational_state_contract

install_relational_state_contract()

from aassr_v2.current_generation import relational_action_key
from aassr_v2.current_relational_model import (
    RelationalPrediction,
    RelationalProphecyConfig,
    RelationalStochasticProphecy,
)
from aassr_v2.current_relational_skill_prophecy import (
    SKILL_COMPLETE_OUTCOME_LIMIT,
    RelationalStochasticSkillProphecy,
)
from aassr_v2.current_relational_state_v3 import relational_state_key_v3
from aassr_v2.current_semantic_calibration import (
    SemanticCalibratedProphecy,
    probability_weighted_semantic_score,
)
from aassr_v2.knowledge import KnowledgeStore
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.replay import ReplayBuffer
from aassr_v2.skills import SKILL_VERB
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


def test_probability_weighted_semantic_score_penalizes_wrong_outcome_mass() -> None:
    actual, _ = _state("catalog")
    wrong, _ = _state("auth")
    mostly_right = (
        RelationalPrediction(
            actual,
            0.9,
            source="right",
            outcome_probability=0.9,
        ),
        RelationalPrediction(
            wrong,
            0.9,
            source="wrong",
            outcome_probability=0.1,
        ),
    )
    mostly_wrong = (
        RelationalPrediction(
            actual,
            0.9,
            source="right",
            outcome_probability=0.1,
        ),
        RelationalPrediction(
            wrong,
            0.9,
            source="wrong",
            outcome_probability=0.9,
        ),
    )

    assert probability_weighted_semantic_score(
        mostly_right, actual
    ) > probability_weighted_semantic_score(mostly_wrong, actual)


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
        relational_state_key_v3(before),
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


def test_stochastic_skill_uses_defined_context_path_and_preserves_mass() -> None:
    before, primitive = _state()
    catalog, _ = _state("catalog")
    auth, _ = _state("auth")
    calls = {"context": 0}

    class Base:
        training_stats = object()

        def predict_with_context(self, state, action, *, knowledge, samples):
            del state, action, knowledge, samples
            calls["context"] += 1
            return (
                RelationalPrediction(
                    catalog,
                    0.8,
                    source="catalog",
                    outcome_probability=0.7,
                ),
                RelationalPrediction(
                    auth,
                    0.8,
                    source="auth",
                    outcome_probability=0.3,
                ),
            )

        def predict(self, state, action, *, samples):
            return self.predict_with_context(
                state,
                action,
                knowledge=KnowledgeStore(),
                samples=samples,
            )

        def confidence(self, state, action):
            del state, action
            return 0.8

        def diagnostics(self):
            return {}

    class Library:
        def template_length(self, skill_id):
            assert skill_id == "skill-test"
            return 1

        def resolve_primitive(self, skill_id, index, state):
            del state
            assert skill_id == "skill-test"
            assert index == 0
            return primitive

        @staticmethod
        def augment_state(state):
            return state

    knowledge = KnowledgeStore()
    skill_prophecy = RelationalStochasticSkillProphecy(
        Base(),
        Library(),
        knowledge,
    )
    skill_action = Action(SKILL_VERB, target="skill-test")
    rows = skill_prophecy.predict_with_context(
        before,
        skill_action,
        knowledge=knowledge,
        samples=3,
    )

    assert calls["context"] == 1
    assert len(rows) == 2
    assert sum(row.outcome_probability for row in rows) == pytest.approx(1.0)
    assert sorted(
        (row.outcome_probability for row in rows),
        reverse=True,
    ) == pytest.approx((0.7, 0.3))


def test_stochastic_skill_exposes_dropped_tail_and_fails_closed() -> None:
    before, primitive = _state()
    mode_count = SKILL_COMPLETE_OUTCOME_LIMIT + 6
    per_mode = 1.0 / mode_count

    class CompleteBase:
        complete_outcome_distribution = True
        training_stats = object()

        def predict_with_context(self, state, action, *, knowledge, samples):
            del action, knowledge, samples
            return tuple(
                RelationalPrediction(
                    state,
                    0.9,
                    source=f"mode-{index}",
                    outcome_probability=per_mode,
                )
                for index in range(mode_count)
            )

        def predict(self, state, action, *, samples):
            return self.predict_with_context(
                state,
                action,
                knowledge=KnowledgeStore(),
                samples=samples,
            )

        def confidence(self, state, action):
            del state, action
            return 0.9

        def diagnostics(self):
            return {}

    class Library:
        def template_length(self, skill_id):
            assert skill_id == "skill-tail"
            return 1

        def resolve_primitive(self, skill_id, index, state):
            del state
            assert skill_id == "skill-tail"
            assert index == 0
            return primitive

        @staticmethod
        def augment_state(state):
            return state

    skill_prophecy = RelationalStochasticSkillProphecy(
        CompleteBase(),
        Library(),
        KnowledgeStore(),
    )
    skill_action = Action(SKILL_VERB, target="skill-tail")
    rows = skill_prophecy.predict(before, skill_action, samples=1)

    assert len(rows) == SKILL_COMPLETE_OUTCOME_LIMIT + 1
    assert sum(row.outcome_probability for row in rows) == pytest.approx(1.0)
    tail = rows[-1]
    assert tail.source.endswith("unresolved-tail")
    assert tail.outcome_probability == pytest.approx(6.0 / mode_count)
    assert tail.probability == pytest.approx(0.0)
    assert skill_prophecy.confidence(before, skill_action) == pytest.approx(0.0)
