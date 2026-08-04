from __future__ import annotations

from types import SimpleNamespace

import pytest

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.effect_prophecy import EffectComposedProphecy
from aassr_v2.types import Action, Prediction, StateSnapshot


class SelfStateProphecy:
    name = "self-state"

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        del action, samples
        return (Prediction(state, 1.0, source="self-state:unseen"),)

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        del state, action, actual_next_state

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state, action
        return 0.0


class TwoStepProphecy:
    name = "two-step"

    def __init__(self) -> None:
        self.bad = Action("bad")
        self.good = Action("good")
        self.finish = Action("finish")

    def _next(self, action: Action) -> StateSnapshot:
        if action.verb_name == "bad":
            return StateSnapshot(
                vector=(0.5,),
                facts=frozenset({"dead_end"}),
                available_actions=(),
                goal_progress=0.5,
            )
        if action.verb_name == "good":
            return StateSnapshot(
                vector=(0.0,),
                facts=frozenset({"setup"}),
                available_actions=(self.finish,),
                goal_progress=0.0,
            )
        return StateSnapshot(
            vector=(1.0,),
            facts=frozenset({"goal"}),
            available_actions=(),
            goal_progress=1.0,
        )

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        del state, samples
        return (Prediction(self._next(action), 1.0, source="two-step:exact"),)

    def learn(
        self,
        state: StateSnapshot,
        action: Action,
        actual_next_state: StateSnapshot,
    ) -> None:
        del state, action, actual_next_state

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state, action
        return 1.0

    def coverage(self, state: StateSnapshot, actions: tuple[Action, ...]) -> float:
        del state, actions
        return 1.0


def test_effect_prophecy_composes_delta_on_unseen_state() -> None:
    advance = Action("advance")
    wait = Action("wait")
    finish = Action("finish")
    before = StateSnapshot(
        vector=(0.0, 2.0),
        facts=frozenset({"closed", "keep"}),
        available_actions=(advance, wait),
        goal_progress=0.0,
    )
    after = StateSnapshot(
        vector=(1.0, 2.0),
        facts=frozenset({"open", "keep"}),
        available_actions=(finish, wait),
        goal_progress=0.5,
    )
    prophecy = EffectComposedProphecy(
        SelfStateProphecy(),
        minimum_samples=2,
    )
    prophecy.learn(before, advance, after)
    prophecy.learn(before, advance, after)

    novel = StateSnapshot(
        vector=(10.0, 2.0),
        facts=frozenset({"closed", "new"}),
        available_actions=(advance, wait),
        goal_progress=0.1,
    )
    predictions = prophecy.predict(novel, advance, samples=1)
    composed = predictions[0]

    assert composed.source == "effect-composed:action-family"
    assert composed.next_state.vector == pytest.approx((11.0, 2.0))
    assert composed.next_state.goal_progress == pytest.approx(0.6)
    assert "new" in composed.next_state.facts
    assert "open" in composed.next_state.facts
    assert "closed" not in composed.next_state.facts
    assert {action.verb_name for action in composed.next_state.available_actions} == {
        "finish",
        "wait",
    }
    assert prophecy.coverage(novel, (advance,)) > 0.5


def test_verb_family_fallback_does_not_copy_stale_symbolic_bindings() -> None:
    first = Action("advance", parameters={"stage": 1})
    second = Action("advance", parameters={"stage": 2})
    finish_stage_one = Action("finish", parameters={"stage": 1})
    before = StateSnapshot(
        vector=(0.0,),
        facts=frozenset({"stage:1"}),
        available_actions=(first,),
    )
    after = StateSnapshot(
        vector=(1.0,),
        facts=frozenset({"stage:2"}),
        available_actions=(finish_stage_one,),
    )
    prophecy = EffectComposedProphecy(
        SelfStateProphecy(),
        minimum_samples=1,
    )
    prophecy.learn(before, first, after)

    novel = StateSnapshot(
        vector=(5.0,),
        facts=frozenset({"stage:unknown"}),
        available_actions=(second,),
    )
    predictions = prophecy.predict(novel, second, samples=1)
    composed = next(
        prediction.next_state
        for prediction in predictions
        if prediction.source.startswith("effect-composed")
    )

    assert composed.vector == pytest.approx((6.0,))
    assert composed.facts == novel.facts
    assert composed.available_actions == novel.available_actions
    assert finish_stage_one not in composed.available_actions


def test_agent_reports_why_imagination_did_not_run() -> None:
    action = Action("opaque")
    state = StateSnapshot(
        vector=(0.0,),
        available_actions=(action,),
    )
    agent = AutonomousLearningAgent(
        SelfStateProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_imagination=True,
            imagination_minimum_coverage=0.5,
        ),
        seed=3,
    )

    decision = agent.select_action(state, episode=0, explore=True)
    diagnostics = agent.imagination_diagnostics()

    assert decision.action == action
    assert decision.imagination_opportunity
    assert not decision.imagination_eligible
    assert decision.imagination_gate_reason == "coverage"
    assert decision.model_coverage == 0.0
    assert diagnostics["opportunities"] == 1
    assert diagnostics["gate:coverage"] == 1


def test_imagination_changes_a_myopic_policy_action() -> None:
    prophecy = TwoStepProphecy()
    state = StateSnapshot(
        vector=(0.0,),
        available_actions=(prophecy.bad, prophecy.good),
        goal_progress=0.0,
    )
    agent = AutonomousLearningAgent(
        prophecy,
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_imagination=True,
            use_effect_composition=False,
            imagination_depth=2,
            imagination_branching_factor=2,
            imagination_minimum_coverage=0.0,
        ),
        seed=4,
    )

    decision = agent.select_action(state, episode=0, explore=True)
    diagnostics = agent.imagination_diagnostics()

    assert decision.policy_action_signature == prophecy.bad.signature
    assert decision.action == prophecy.good
    assert decision.used_imagination
    assert decision.imagination_changed_action
    assert decision.imagination_depth == 2
    assert diagnostics["runs"] == 1
    assert diagnostics["changed_actions"] == 1
    assert diagnostics["change_rate_per_run"] == 1.0


def test_agent_learns_effects_only_from_training_transitions() -> None:
    action = Action("advance")
    before = StateSnapshot(vector=(0.0,), available_actions=(action,))
    after = StateSnapshot(vector=(1.0,), available_actions=(action,))
    agent = AutonomousLearningAgent(
        SelfStateProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            holdout_stride=1000,
            effect_minimum_samples=2,
        ),
        seed=9,
    )
    outcome = SimpleNamespace(
        snapshot=after,
        error=False,
        unlocked_actions=(),
    )

    agent.observe(before, action, outcome)
    agent.observe(before, action, outcome)

    assert agent.prophecy_diagnostics()["effect_observations"] == 2
    prediction = agent.prophecy.predict(before, action, samples=1)[0]
    assert prediction.next_state.vector == pytest.approx((1.0,))
