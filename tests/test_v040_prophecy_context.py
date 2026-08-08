from __future__ import annotations

from aassr_v2.autonomous_agent import AutonomousAgentConfig
from aassr_v2.integrated_agent import build_full_aassr_core
from aassr_v2.knowledge import KnowledgeEntry
from aassr_v2.types import Action, Prediction, StateSnapshot


class ContextAwareProphecy:
    name = "context-effect-test"

    def predict(self, state, action, *, samples):
        return (Prediction(state, 1.0, "base:no-context"),)

    def predict_with_context(self, state, action, *, knowledge, samples):
        if knowledge.get("permit") is None:
            return self.predict(state, action, samples=samples)
        return (
            Prediction(
                StateSnapshot(
                    tuple(1.0 for _ in state.vector),
                    facts=state.facts | frozenset({"context_used"}),
                    available_actions=state.available_actions,
                    goal_progress=1.0,
                    metadata=dict(state.metadata),
                ),
                1.0,
                "base:context",
            ),
        )

    def learn(self, state, action, actual_next_state):
        pass


def test_context_prediction_keeps_effect_composition_active() -> None:
    action = Action("use")
    state = StateSnapshot(
        (0.0,),
        facts=frozenset({"ready"}),
        available_actions=(action,),
    )
    observed = StateSnapshot(
        (0.5,),
        facts=frozenset({"ready", "observed_effect"}),
        available_actions=(action,),
        goal_progress=0.5,
    )
    agent = build_full_aassr_core(
        ContextAwareProphecy(),
        core_config=AutonomousAgentConfig(
            use_imagination=True,
            use_effect_composition=True,
            effect_minimum_samples=2,
            imagination_minimum_coverage=0.0,
        ),
        seed=19,
    )

    # Populate the effect bucket through the exact model stack used by the
    # integrated evaluator/planner.
    agent.prophecy.learn(state, action, observed)
    agent.prophecy.learn(state, action, observed)
    agent.knowledge.apply((KnowledgeEntry("permit", True, "test"),))

    predictions = agent.prophecy.predict_with_context(
        state,
        action,
        knowledge=agent.knowledge,
        samples=2,
    )
    sources = {prediction.source for prediction in predictions}
    assert any(source.startswith("effect-composed:") for source in sources)
    assert any(
        "context_used" in prediction.next_state.facts
        for prediction in predictions
    )
    assert agent.effect_prophecy.composed_predictions >= 1
    assert agent.diagnostics()["knowledge_effect_composition_aligned"] is True


def test_holdout_prediction_is_context_free_but_planner_uses_live_knowledge() -> None:
    action = Action("use")
    state = StateSnapshot(
        (0.0,),
        facts=frozenset({"ready"}),
        available_actions=(action,),
    )
    agent = build_full_aassr_core(
        ContextAwareProphecy(),
        core_config=AutonomousAgentConfig(
            use_imagination=True,
            use_effect_composition=False,
            imagination_minimum_coverage=0.0,
        ),
        seed=23,
    )
    agent.knowledge.apply((KnowledgeEntry("permit", True, "test"),))

    # Plain predict is the interface used by replay holdout validation. It must
    # not see Knowledge obtained after a held-out transition was recorded.
    holdout_prediction = agent.prophecy.predict(
        state,
        action,
        samples=1,
    )[0].next_state
    assert holdout_prediction.vector == (0.0,)
    assert holdout_prediction.goal_progress == 0.0
    assert "context_used" not in holdout_prediction.facts

    # Imagination uses predict_step, which stays bound to live Knowledge.
    planning_step = agent.core.planner.prophecy.predict_step(
        state,
        action,
        memory=agent.core.planner._initial_prophecy_memory(),
        samples=1,
    )
    planned = planning_step.predictions[0].next_state
    assert planned.vector == (1.0,)
    assert planned.goal_progress == 1.0
    assert "context_used" in planned.facts
