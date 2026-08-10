from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.branch_critic import CriticTransition
from aassr_v2.current_confidence_gate import _confidence_gated_core_select_action
from aassr_v2.types import Action, StateSnapshot


def _state() -> tuple[StateSnapshot, Action, Action]:
    policy = Action("request", parameters={"route_id": "route-policy"})
    imagined = Action("request", parameters={"route_id": "route-imagined"})
    state = StateSnapshot(
        vector=(0.0,) * 32,
        facts=frozenset({"known_route:route-policy", "known_route:route-imagined"}),
        available_actions=(policy, imagined),
        goal_progress=0.0,
        metadata={},
    )
    return state, policy, imagined


class _FakePolicy:
    def __init__(self, selected: Action, values: dict[str, float] | None = None) -> None:
        self.selected = selected
        self.values = values or {}

    def select(self, state: StateSnapshot, **_: object) -> Action:
        del state
        return self.selected

    def value(self, state: StateSnapshot, action: Action) -> float:
        del state
        return float(self.values.get(action.signature, 0.0))


class _FakeProphecy:
    def __init__(self, confidence: dict[str, float], *, coverage: float = 1.0) -> None:
        self._confidence = confidence
        self._coverage = float(coverage)

    def coverage(self, state: StateSnapshot, actions: object) -> float:
        del state, actions
        return self._coverage

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state
        return float(self._confidence[action.signature])


def _evaluation(action: Action, value: float) -> SimpleNamespace:
    return SimpleNamespace(action=action, aggregate_value=float(value))


def _fake_agent(
    *,
    policy_action: Action,
    policy_value: float,
    imagined_action: Action,
    imagined_value: float,
    policy_confidence: float,
    imagined_confidence: float,
    intervention_margin: float = 0.05,
) -> SimpleNamespace:
    plan = SimpleNamespace(
        root_evaluations=(
            _evaluation(policy_action, policy_value),
            _evaluation(imagined_action, imagined_value),
        ),
        nodes=(object(), object()),
        maximum_depth_reached=1,
    )
    agent = SimpleNamespace(
        config=SimpleNamespace(
            imagination_minimum_coverage=0.55,
            imagination_intervention_margin=float(intervention_margin),
            # Deliberately absurd: the new rule must ignore this as a score/margin term.
            imagination_uncertainty_margin=999.0,
        ),
        policy=_FakePolicy(
            policy_action,
            values={
                policy_action.signature: 0.0,
                imagined_action.signature: 1.0,
            },
        ),
        skill_prophecy=_FakeProphecy(
            {
                policy_action.signature: policy_confidence,
                imagined_action.signature: imagined_confidence,
            },
            coverage=1.0,
        ),
        planner=SimpleNamespace(plan=lambda state: plan),
        randomizer=random.Random(7),
        requested_imagination=True,
        training_imagination=False,
        critic_ready=True,
        _decision_index=0,
        epsilon=lambda episode: 0.0,
        _record_decision=lambda decision: decision,
    )
    return agent


def test_current_builder_separates_confidence_from_planner_and_critic_value() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    state, action, _ = _state()

    assert agent.current_confidence_gate is True
    assert agent.current_imagination_value_contract == "critic_only_after_confidence_gate"
    assert agent.planner.config.uncertainty_penalty == 0.0
    assert agent.config.imagination_uncertainty_margin == 0.0
    assert agent.current_confidence_gate_threshold == agent.config.imagination_minimum_coverage

    low = agent.critic.encoder.encode(CriticTransition(state, action, state, 0.1))
    high = agent.critic.encoder.encode(CriticTransition(state, action, state, 0.9))
    assert low == high
    assert len(low) == agent.critic.encoder.feature_size
    assert agent.critic.gru.input_size == agent.critic.encoder.feature_size


def test_confidence_difference_alone_cannot_create_intervention() -> None:
    state, policy, imagined = _state()
    agent = _fake_agent(
        policy_action=policy,
        policy_value=0.60,
        imagined_action=imagined,
        imagined_value=0.60,
        policy_confidence=0.56,
        imagined_confidence=0.99,
    )

    decision = _confidence_gated_core_select_action(
        agent,
        state,
        episode=0,
        explore=False,
    )

    assert decision.imagination_switch_candidate is True
    assert decision.imagination_advantage == pytest.approx(0.0)
    assert decision.imagination_required_advantage == pytest.approx(0.05)
    assert decision.imagination_intervention_allowed is False
    assert decision.action.signature == policy.signature


def test_low_confidence_raw_winner_is_rejected_before_value_comparison() -> None:
    state, policy, imagined = _state()
    agent = _fake_agent(
        policy_action=policy,
        policy_value=0.60,
        imagined_action=imagined,
        imagined_value=0.95,
        policy_confidence=0.90,
        imagined_confidence=0.40,
    )

    decision = _confidence_gated_core_select_action(
        agent,
        state,
        episode=0,
        explore=False,
    )

    assert decision.imagination_gate_reason == "candidate_prediction_low_confidence"
    assert decision.imagination_intervention_allowed is False
    assert decision.action.signature == policy.signature


def test_reliable_candidates_are_ranked_by_critic_only() -> None:
    state, policy, imagined = _state()
    agent = _fake_agent(
        policy_action=policy,
        policy_value=0.60,
        imagined_action=imagined,
        imagined_value=0.66,
        policy_confidence=0.99,
        imagined_confidence=0.56,
    )

    decision = _confidence_gated_core_select_action(
        agent,
        state,
        episode=0,
        explore=False,
    )

    assert decision.imagination_advantage == pytest.approx(0.06)
    assert decision.imagination_required_advantage == pytest.approx(0.05)
    assert decision.imagination_intervention_allowed is True
    assert decision.imagination_gate_reason == "intervention"
    assert decision.action.signature == imagined.signature


def test_unreliable_policy_prediction_blocks_override_fail_closed() -> None:
    state, policy, imagined = _state()
    agent = _fake_agent(
        policy_action=policy,
        policy_value=0.50,
        imagined_action=imagined,
        imagined_value=0.95,
        policy_confidence=0.40,
        imagined_confidence=0.95,
    )

    decision = _confidence_gated_core_select_action(
        agent,
        state,
        episode=0,
        explore=False,
    )

    assert decision.imagination_gate_reason == "policy_prediction_low_confidence"
    assert decision.imagination_switch_candidate is True
    assert decision.imagination_intervention_allowed is False
    assert decision.action.signature == policy.signature
