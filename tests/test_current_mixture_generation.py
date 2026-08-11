from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_mixture_entrypoint import (
    MIXTURE_CURRENT_COMPONENTS,
    build_current_mixture_pentest_aassr_core,
)
from aassr_v2.current_relational_codec import descriptor
from aassr_v2.current_relational_mixture_model import (
    ConditionalMixtureRelationalProphecy,
    RelationalMixtureProphecyConfig,
)
from aassr_v2.current_relational_state import install_relational_state_contract
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.types import Action, StateSnapshot


install_relational_state_contract()


def _state(role: str | None = None, *, request_usage: float = 0.0) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request",
        parameters={"route_id": "route-05", "profile_id": "profile-browse"},
    )
    facts = {"known_route:route-05", "known_profile:profile-browse"}
    if role is not None:
        facts.add(f"observed_route_role:route-05:{role}")
    vector = [0.0] * AGENT_STATE_SIZE
    vector[8] = request_usage
    vector[10] = 0.0
    return (
        StateSnapshot(
            tuple(vector),
            facts=frozenset(facts),
            available_actions=(action,),
            metadata={
                "observation_contract": "response_causal_observation_v3",
                "request_count_scale": 100.0,
                "workflow_progress_scale": 8.0,
            },
        ),
        action,
    )


def test_mixture_builder_installs_conditional_world_model() -> None:
    agent = build_current_mixture_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    assert isinstance(
        agent.base_neural_prophecy,
        ConditionalMixtureRelationalProphecy,
    )
    assert agent.current_components == MIXTURE_CURRENT_COMPONENTS
    diagnostics = agent.base_neural_prophecy.diagnostics()
    assert diagnostics["conditional_mixture_components"] == 3
    assert diagnostics["terminal_classes"] == 4
    assert diagnostics["mixture_training_objective"] == "soft-mixture-likelihood"
    assert diagnostics["epistemic_confidence"] == "ensemble-mode-set-disagreement"
    assert agent.planner.config.aggregation == "mean"
    assert agent.planner.config.discount == pytest.approx(agent.config.gamma)
    assert agent.diagnostics()["current_repairs"]["planner_discount_matches_gamma"] is True


def test_mixture_builder_executes_and_learns_one_real_transition() -> None:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0])
    agent = build_current_mixture_pentest_aassr_core(
        seed=11,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    agent.begin_episode()
    step = agent.step(
        world,
        episode=0,
        training=True,
        primitive_budget=1,
    )
    assert len(step.traces) == 1
    agent.finish_episode(final_return=0.0, training=True)
    assert agent.base_neural_prophecy.observations == 1
    assert agent.critic.stats().episodes == 1


def test_learned_mixture_returns_normalized_distinct_modes_on_novel_related_input() -> None:
    model = ConditionalMixtureRelationalProphecy(
        seed=13,
        device="cpu",
        config=RelationalMixtureProphecyConfig(
            hidden_units=24,
            ensemble_size=2,
            mixture_components=3,
            replay_capacity=32,
            batch_size=2,
            warmup_steps=2,
            gradient_steps_per_observation=2,
        ),
    )
    before, action = _state(request_usage=0.0)
    catalog, _ = _state("catalog", request_usage=0.1)
    auth, _ = _state("auth", request_usage=0.1)
    for _ in range(4):
        model.learn(before, action, catalog)
        model.learn(before, action, auth)

    novel, novel_action = _state(request_usage=0.25)
    rows = model.predict(novel, novel_action, samples=3)
    assert rows
    assert len(rows) <= 3
    assert all(":mixture-" in row.source for row in rows)
    assert sum(
        getattr(row, "outcome_probability", 0.0) for row in rows
    ) == pytest.approx(1.0)
    semantic_modes = {
        tuple(round(value, 4) for value in descriptor(row.next_state))
        for row in rows
    }
    assert len(semantic_modes) >= 2
    assert model.diagnostics()["learned_mixture_prediction_rows"] >= 1
