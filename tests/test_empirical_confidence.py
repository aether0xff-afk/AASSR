from __future__ import annotations

import pytest

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.effect_prophecy import EffectComposedProphecy
from aassr_v2.empirical_confidence import (
    empirical_confidence,
    outcome_consistency,
)
from aassr_v2.tabular_prophecy import TabularProphecy
from aassr_v2.types import Action, StateSnapshot


def _transition_states() -> tuple[StateSnapshot, Action, StateSnapshot, StateSnapshot]:
    action = Action("advance")
    before = StateSnapshot(
        vector=(0.0,),
        facts=frozenset({"same-observation"}),
        available_actions=(action,),
    )
    first = StateSnapshot(
        vector=(1.0,),
        facts=frozenset({"outcome-a"}),
        available_actions=(action,),
    )
    second = StateSnapshot(
        vector=(-1.0,),
        facts=frozenset({"outcome-b"}),
        available_actions=(action,),
    )
    return before, action, first, second


def test_outcome_consistency_detects_conflicting_empirical_results() -> None:
    assert outcome_consistency([10]) == 1.0
    assert outcome_consistency([5, 5]) == pytest.approx(0.0)
    assert 0.0 < outcome_consistency([9, 1]) < 1.0
    assert outcome_consistency([]) == 0.0


def test_empirical_confidence_needs_evidence_and_consistency() -> None:
    assert empirical_confidence([1], prior_strength=4.0) == pytest.approx(0.2)
    assert empirical_confidence([4], prior_strength=4.0) == pytest.approx(0.5)
    assert empirical_confidence([4, 4], prior_strength=4.0) == pytest.approx(0.0)


def test_tabular_confidence_falls_to_zero_for_observational_aliasing() -> None:
    before, action, first, second = _transition_states()
    prophecy = TabularProphecy()
    for _ in range(4):
        prophecy.learn(before, action, first)
        prophecy.learn(before, action, second)

    predictions = prophecy.predict(before, action, samples=2)

    assert len(predictions) == 2
    assert sum(item.probability for item in predictions) == pytest.approx(1.0)
    assert prophecy.confidence(before, action) == pytest.approx(0.0)
    assert prophecy.coverage(before, (action,)) == pytest.approx(0.0)


def test_tabular_confidence_rises_for_repeatable_transition() -> None:
    before, action, first, _ = _transition_states()
    prophecy = TabularProphecy()
    for _ in range(4):
        prophecy.learn(before, action, first)

    assert prophecy.confidence(before, action) == pytest.approx(0.5)


def test_effect_confidence_also_rejects_conflicting_deltas() -> None:
    before, action, first, second = _transition_states()
    prophecy = EffectComposedProphecy(
        TabularProphecy(),
        minimum_samples=2,
    )
    for _ in range(4):
        prophecy.learn(before, action, first)
        prophecy.learn(before, action, second)

    assert prophecy.confidence(before, action) == pytest.approx(0.0)
    assert prophecy.coverage(before, (action,)) == pytest.approx(0.0)


def test_ambiguous_transition_blocks_imagination_gate() -> None:
    before, action, first, second = _transition_states()
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_imagination=True,
            use_effect_composition=True,
            effect_minimum_samples=2,
            imagination_minimum_coverage=0.35,
        ),
        seed=5,
    )
    for _ in range(4):
        agent.prophecy.learn(before, action, first)
        agent.prophecy.learn(before, action, second)

    decision = agent.select_action(before, episode=0, explore=True)

    assert not decision.used_imagination
    assert decision.imagination_gate_reason == "coverage"
    assert decision.model_coverage == pytest.approx(0.0)
