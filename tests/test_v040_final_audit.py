from __future__ import annotations

from dataclasses import dataclass

import pytest

from aassr_v2.action_plugins import PluginOutcome
from aassr_v2.autonomous_agent import AutonomousAgentConfig
from aassr_v2.integrated_agent import build_full_aassr_core
from aassr_v2.knowledge import KnowledgeStore
from aassr_v2.learning import AdvancedTransitionEvaluator
from aassr_v2.replay import (
    PredictionValidator,
    ReplayBuffer,
    ReplayTransition,
    ValidationScore,
)
from aassr_v2.types import Action, Prediction, StateSnapshot, TransitionTrace


class FutureKnowledgeProphecy:
    """Only post-outcome knowledge would make the current prediction correct."""

    name = "future-knowledge-test"

    def __init__(self) -> None:
        self.learn_calls = 0
        self.advance_calls = 0
        self.reset_calls = 0

    def predict(self, state, action, *, samples):
        return (Prediction(state, 1.0, "no-context"),)

    def predict_with_context(self, state, action, *, knowledge, samples):
        if knowledge.get("future") is None:
            return self.predict(state, action, samples=samples)
        return (
            Prediction(
                StateSnapshot(
                    (1.0,),
                    facts=state.facts | frozenset({"future"}),
                    available_actions=(),
                    goal_progress=1.0,
                ),
                1.0,
                "future-context",
            ),
        )

    def learn(self, state, action, actual_next_state):
        self.learn_calls += 1

    def advance_sequence(self, state, action):
        self.advance_calls += 1

    def reset_sequence(self):
        self.reset_calls += 1


class FutureFactWorld:
    def __init__(self) -> None:
        self.done = False
        self.action = Action("discover")

    @property
    def terminal(self) -> bool:
        return self.done

    def snapshot(self) -> StateSnapshot:
        if self.done:
            return StateSnapshot(
                (1.0,),
                facts=frozenset({"future"}),
                available_actions=(),
                goal_progress=1.0,
            )
        return StateSnapshot(
            (0.0,),
            facts=frozenset(),
            available_actions=(self.action,),
            goal_progress=0.0,
        )

    def step(self, action: Action) -> PluginOutcome:
        assert action == self.action
        before = self.snapshot()
        self.done = True
        after = self.snapshot()
        return PluginOutcome(
            snapshot=after,
            added_facts=after.facts - before.facts,
            removed_facts=before.facts - after.facts,
            unlocked_actions=(),
            error=False,
        )


class LengthValidator(PredictionValidator):
    def evaluate(self, prophecy, transitions):
        materialized = tuple(transitions)
        return ValidationScore(len(materialized), float(len(materialized)))


def _transition(index: int) -> ReplayTransition:
    action = Action("a")
    before = StateSnapshot((float(index),), available_actions=(action,))
    after = StateSnapshot((float(index + 1),), available_actions=(action,))
    return ReplayTransition(before, action, after, f"pre-{index}")


def test_current_outcome_knowledge_cannot_retroactively_improve_same_transition() -> None:
    prophecy = FutureKnowledgeProphecy()
    evaluator = AdvancedTransitionEvaluator(prophecy, samples=1)
    knowledge = KnowledgeStore()
    world = FutureFactWorld()

    result = evaluator.execute(world, world.action, knowledge)

    # Before the action, both context-free and Knowledge-aware predictions had
    # exactly the same information. The fact learned from the outcome is not
    # allowed to travel backward and improve this transition's context score.
    assert result.effect.knowledge_only_gain == pytest.approx(0.0)
    assert result.effect.knowledge_context_score == pytest.approx(
        result.effect.latest_prediction_before
    )
    assert knowledge.get("future") is not None


def test_holdout_gain_uses_the_same_validation_set_before_and_after_update() -> None:
    replay = ReplayBuffer(capacity=16, holdout_stride=2)
    # Seen counts 1/2/3 => the second item is holdout. The next current sample
    # (seen count 4) will itself become holdout and must not appear only in the
    # post-update side of the same gain calculation.
    replay.add(_transition(0))
    replay.add(_transition(1))
    replay.add(_transition(2))
    assert len(replay.holdout()) == 1

    prophecy = FutureKnowledgeProphecy()
    evaluator = AdvancedTransitionEvaluator(
        prophecy,
        replay=replay,
        validator=LengthValidator(samples=1),
        samples=1,
    )
    world = FutureFactWorld()
    result = evaluator.execute(world, world.action, KnowledgeStore())

    assert len(replay.holdout()) == 2
    assert result.effect.holdout_before == pytest.approx(1.0)
    assert result.effect.holdout_after == pytest.approx(1.0)
    assert result.effect.holdout_gain == pytest.approx(0.0)
    # The held-out target was not learned, but recurrent context still advances.
    assert prophecy.learn_calls == 0
    assert prophecy.advance_calls == 1


def test_evaluation_episode_does_not_learn_features_or_promote_skills() -> None:
    prophecy = FutureKnowledgeProphecy()
    agent = build_full_aassr_core(
        prophecy,
        core_config=AutonomousAgentConfig(
            use_imagination=False,
            use_effect_composition=False,
        ),
        seed=17,
    )

    initial_feature_count = len(agent.feature_memory.snapshot())
    initial_skill_count = len(agent.skills.all())

    for episode in range(2):
        world = FutureFactWorld()
        agent.begin_episode(clear_knowledge=True)
        step = agent.step(world, episode=episode, training=False)
        assert step.newly_achieved_goal_ids == ("external:success",)
        assert step.promoted_skill is None
        assert agent.knowledge.get("future") is not None
        agent.finish_episode(final_return=1.0, training=False)

    assert len(agent.feature_memory.snapshot()) == initial_feature_count
    assert len(agent.skills.all()) == initial_skill_count
    assert prophecy.learn_calls == 0


def test_begin_episode_resets_recurrent_prophecy_context() -> None:
    prophecy = FutureKnowledgeProphecy()
    agent = build_full_aassr_core(
        prophecy,
        core_config=AutonomousAgentConfig(
            use_imagination=False,
            use_effect_composition=False,
        ),
        seed=23,
    )

    assert prophecy.reset_calls == 0
    agent.begin_episode()
    agent.begin_episode()
    assert prophecy.reset_calls == 2


@dataclass
class RateLimitMacroWorld:
    calls: int = 0
    rate_limited: bool = False
    success: bool = False
    failed: bool = False
    locked: bool = False

    @property
    def terminal(self) -> bool:
        return False

    def snapshot(self) -> StateSnapshot:
        primitive = Action("primitive")
        return StateSnapshot(
            (float(self.calls),),
            facts=frozenset(),
            available_actions=(primitive,),
        )

    def step(self, action: Action) -> PluginOutcome:
        before = self.snapshot()
        self.calls += 1
        self.rate_limited = self.calls >= 1
        after = self.snapshot()
        return PluginOutcome(
            snapshot=after,
            added_facts=frozenset(),
            removed_facts=frozenset(),
            unlocked_actions=(),
            error=False,
        )


def _trace(index: int, action: Action) -> TransitionTrace:
    before = StateSnapshot(
        (float(index),),
        facts=frozenset(),
        available_actions=(action,),
    )
    after = StateSnapshot(
        (float(index + 1),),
        facts=frozenset(),
        available_actions=(action,),
        goal_progress=1.0 if index == 2 else 0.0,
    )
    return TransitionTrace(
        f"skill-trace-{index}",
        before,
        action,
        (Prediction(after, 1.0, "test"),),
        after,
        frozenset(),
        frozenset(),
        (),
        False,
    )


def test_skill_macro_stops_at_semantic_terminal_and_respects_primitive_budget() -> None:
    prophecy = FutureKnowledgeProphecy()
    agent = build_full_aassr_core(
        prophecy,
        core_config=AutonomousAgentConfig(
            use_imagination=False,
            use_effect_composition=False,
        ),
        seed=31,
    )
    primitive = Action("primitive")
    traces = tuple(_trace(index, primitive) for index in range(3))
    # Two repeated successes promote a 3-primitive macro.
    assert agent.skills.observe_goal_completion(
        traces,
        achieved_goal_ids=("external:success",),
    ) is None
    skill = agent.skills.observe_goal_completion(
        traces,
        achieved_goal_ids=("external:success",),
    )
    assert skill is not None

    world = RateLimitMacroWorld()
    state = agent.skills.augment_state(world.snapshot())
    macro = skill.as_action()
    agent.policy.observe_return(state, macro, 10.0)

    agent.begin_episode()
    step = agent.step(
        world,
        episode=0,
        training=False,
        primitive_budget=2,
    )
    assert step.decision.action.verb_name == "skill"
    # Rate-limit became true after primitive one, so no second primitive may run
    # even though the explicit primitive budget still had capacity.
    assert world.calls == 1
    assert len(step.executed_actions) == 1
    assert step.terminal is True
