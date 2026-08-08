from __future__ import annotations

import pytest

from aassr_v2.action_plugins import PluginOutcome
from aassr_v2.autonomous_agent import AutonomousAgentConfig
from aassr_v2.integrated_agent import (
    IntegratedAASSRConfig,
    build_full_aassr_core,
    build_pentest_aassr_core,
)
from aassr_v2.knowledge import KnowledgeEntry
from aassr_v2.pentest_curriculum_causal import CausalRevisitAwareCurriculumWorld
from aassr_v2.pentest_curriculum_env import CURRICULUM_STAGES, CurriculumAgentWorld
from aassr_v2.pentest_curriculum_schedule import semantic_fingerprint
from aassr_v2.semantic_control import SemanticContextualPolicy, SemanticSelfLoopASEQ
from aassr_v2.types import Action, Prediction, StateSnapshot


class ExactLearningProphecy:
    name = "unit-exact"

    def __init__(self) -> None:
        self.learn_calls = 0

    @staticmethod
    def _next(state: StateSnapshot, action: Action) -> StateSnapshot:
        if action.verb_name != "discover":
            return state
        return StateSnapshot(
            (1.0,),
            facts=frozenset({"ready", "done"}),
            available_actions=(),
            goal_progress=1.0,
            metadata=dict(state.metadata),
        )

    def predict(self, state, action, *, samples):
        return (
            Prediction(
                self._next(state, action),
                1.0,
                "unit:exact",
            ),
        )

    def learn(self, state, action, actual_next_state):
        self.learn_calls += 1


class ContextAwareProphecy:
    name = "unit-context-aware"

    def predict(self, state, action, *, samples):
        return (Prediction(state, 1.0, "unit:no-context"),)

    def predict_with_context(self, state, action, *, knowledge, samples):
        if knowledge.get("permit") is None:
            return self.predict(state, action, samples=samples)
        return (
            Prediction(
                StateSnapshot(
                    state.vector,
                    facts=state.facts | frozenset({"context_used"}),
                    available_actions=(),
                    goal_progress=1.0,
                    metadata=dict(state.metadata),
                ),
                1.0,
                "unit:context",
            ),
        )

    def learn(self, state, action, actual_next_state):
        pass


class OneStepWorld:
    def __init__(self) -> None:
        self.done = False

    @property
    def terminal(self) -> bool:
        return self.done

    def snapshot(self) -> StateSnapshot:
        if self.done:
            return StateSnapshot(
                (1.0,),
                facts=frozenset({"ready", "done"}),
                available_actions=(),
                goal_progress=1.0,
            )
        return StateSnapshot(
            (0.0,),
            facts=frozenset({"ready"}),
            available_actions=(Action("discover"),),
            goal_progress=0.0,
        )

    def step(self, action: Action) -> PluginOutcome:
        assert action.verb_name == "discover"
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


def test_semantic_policy_reuses_value_across_admin_noise() -> None:
    action = Action("inspect")

    def key(state: StateSnapshot):
        return tuple(
            sorted(
                fact for fact in state.facts if not fact.startswith("requests:")
            )
        )

    first = StateSnapshot(
        (0.0, 0.1),
        frozenset({"known:item", "requests:1"}),
        (action,),
    )
    noisy = StateSnapshot(
        (99.0, -42.0),
        frozenset({"known:item", "requests:999"}),
        (action,),
    )
    policy = SemanticContextualPolicy(key)
    policy.observe_return(first, action, 2.5)
    learned = policy.value(first, action)
    assert learned > 0.0
    assert policy.value(noisy, action) == pytest.approx(learned)


def test_semantic_aseq_guards_only_confirmed_self_loop_and_restores_freedom() -> None:
    a = Action("a")
    b = Action("b")
    state = StateSnapshot((0.0,), available_actions=(a, b))
    aseq = SemanticSelfLoopASEQ(repeat_threshold=2)

    aseq.observe("S", a, "S")
    assert aseq.filter_state(state, "S")[0].available_actions == (a, b)
    aseq.observe("S", a, "S")
    filtered, guarded, fallback = aseq.filter_state(state, "S")
    assert guarded == 1
    assert not fallback
    assert filtered.available_actions == (b,)

    # Repeated state-changing ASEQ remains legal.
    aseq.observe("S", b, "S2")
    aseq.observe("S", b, "S2")
    assert aseq.guarded_signatures("S") == frozenset({a.signature})

    # A pair with more than one observed S' is never treated as an exact loop.
    aseq.observe("S", b, "S")
    aseq.observe("S", b, "S")
    assert aseq.guarded_signatures("S") == frozenset({a.signature})

    isolated = SemanticSelfLoopASEQ(repeat_threshold=2)
    for action in (a, b):
        isolated.observe("S", action, "S")
        isolated.observe("S", action, "S")
    restored, guarded, fallback = isolated.filter_state(state, "S")
    assert guarded == 2
    assert fallback
    assert restored.available_actions == state.available_actions


def test_integrated_closed_loop_learns_each_real_transition_once_and_promotes_skill() -> None:
    prophecy = ExactLearningProphecy()
    agent = build_full_aassr_core(
        prophecy,
        core_config=AutonomousAgentConfig(
            use_imagination=True,
            use_effect_composition=True,
            imagination_depth=2,
            imagination_branching_factor=2,
            imagination_beam_width=4,
            imagination_outcome_samples=1,
            imagination_minimum_coverage=0.0,
        ),
        integration_config=IntegratedAASSRConfig(
            use_aseq=True,
            information_value_weight=0.25,
        ),
        seed=7,
    )

    promoted = None
    for episode in range(2):
        world = OneStepWorld()
        agent.begin_episode()
        step = agent.step(world, episode=episode, training=True)
        assert step.executed_actions == (Action("discover"),)
        assert step.newly_achieved_goal_ids == ("external:success",)
        promoted = step.promoted_skill or promoted
        agent.finish_episode(final_return=1.0, training=True)

    # The integrated evaluator owns Prophecy learning; the historical autonomous
    # loop is disabled for learning, so each real transition is learned once.
    assert prophecy.learn_calls == 2
    assert agent.knowledge.get("done") is not None
    assert agent.feature_memory.record("done") is not None
    assert agent.policy.value(
        StateSnapshot(
            (0.0,),
            facts=frozenset({"ready"}),
            available_actions=(Action("discover"),),
        ),
        Action("discover"),
    ) > 0.0
    assert promoted is not None
    assert len(agent.skills.all()) == 1
    diagnostics = agent.diagnostics()
    assert diagnostics["closed_loop"] is True
    assert diagnostics["semantic_state_contract_shared"] is True
    assert diagnostics["knowledge_bound_to_planning"] is True
    assert diagnostics["skills"] == 1


def test_live_knowledge_is_visible_to_planner_facing_prophecy() -> None:
    agent = build_full_aassr_core(
        ContextAwareProphecy(),
        core_config=AutonomousAgentConfig(
            use_imagination=True,
            use_effect_composition=False,
            imagination_minimum_coverage=0.0,
        ),
        seed=11,
    )
    state = StateSnapshot(
        (0.0,),
        facts=frozenset({"ready"}),
        available_actions=(Action("use"),),
    )
    before = agent.core.planner.prophecy.predict(
        state,
        Action("use"),
        samples=1,
    )[0].next_state
    assert before.goal_progress == 0.0

    agent.knowledge.apply((KnowledgeEntry("permit", True, "test"),))
    after = agent.core.planner.prophecy.predict(
        state,
        Action("use"),
        samples=1,
    )[0].next_state
    assert after.goal_progress == 1.0
    assert "context_used" in after.facts


def test_pentest_factory_requires_v3_and_clears_seed_local_knowledge() -> None:
    prophecy = ExactLearningProphecy()
    agent = build_pentest_aassr_core(
        prophecy,
        core_config=AutonomousAgentConfig(
            use_imagination=False,
            use_effect_composition=False,
        ),
        seed=3,
    )
    audited = CausalRevisitAwareCurriculumWorld(
        90_001,
        stage=CURRICULUM_STAGES[0],
    )
    state = audited.snapshot()
    assert agent.semantic_state_key(state) == semantic_fingerprint(state)
    decision = agent.select_action(state, episode=0, explore=False)
    assert decision.action in state.available_actions

    agent.knowledge.apply((KnowledgeEntry("seed-local-id", True, "test"),))
    assert agent.knowledge.get("seed-local-id") is not None
    agent.begin_episode()
    assert agent.knowledge.get("seed-local-id") is None
    assert agent.diagnostics()["preserve_knowledge_across_episodes"] is False

    legacy = CurriculumAgentWorld(90_001, stage=CURRICULUM_STAGES[0])
    with pytest.raises(ValueError, match="observation contract mismatch"):
        agent.select_action(legacy.snapshot(), episode=0, explore=False)