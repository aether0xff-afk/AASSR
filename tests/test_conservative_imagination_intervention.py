from __future__ import annotations

from types import SimpleNamespace

import pytest

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.experiment_runner import RESULT_FIELDS, SUMMARY_METRICS
from aassr_v2.imagination_tree import RootActionEvaluation
from aassr_v2.types import Action, Prediction, StateSnapshot


class CoveredSelfProphecy:
    name = "covered-self"

    def __init__(self, coverage: float = 1.0) -> None:
        self._coverage = coverage

    def predict(
        self,
        state: StateSnapshot,
        action: Action,
        *,
        samples: int,
    ) -> tuple[Prediction, ...]:
        del action, samples
        return (Prediction(state, 1.0, source="covered-self:exact"),)

    def confidence(self, state: StateSnapshot, action: Action) -> float:
        del state, action
        return self._coverage

    def coverage(
        self,
        state: StateSnapshot,
        actions: tuple[Action, ...],
    ) -> float:
        del state, actions
        return self._coverage


class FixedPlan:
    def __init__(
        self,
        first: Action,
        second: Action,
        *,
        first_value: float,
        second_value: float,
    ) -> None:
        self.first = first
        self.second = second
        self.first_value = first_value
        self.second_value = second_value

    def plan(self, state: StateSnapshot) -> SimpleNamespace:
        del state
        evaluations = (
            RootActionEvaluation(
                action=self.first,
                leaf_values=(self.first_value,),
                aggregate_value=self.first_value,
                best_path=(self.first.signature,),
                best_leaf_id=1,
            ),
            RootActionEvaluation(
                action=self.second,
                leaf_values=(self.second_value,),
                aggregate_value=self.second_value,
                best_path=(self.second.signature,),
                best_leaf_id=2,
            ),
        )
        return SimpleNamespace(
            root_evaluations=evaluations,
            nodes=(object(), object()),
            maximum_depth_reached=2,
        )


def _agent(
    *,
    coverage: float = 1.0,
    margin: float = 0.05,
    uncertainty_margin: float = 0.20,
) -> tuple[AutonomousLearningAgent, StateSnapshot, Action, Action]:
    first = Action("a")
    second = Action("b")
    state = StateSnapshot(
        vector=(0.0,),
        available_actions=(first, second),
    )
    agent = AutonomousLearningAgent(
        CoveredSelfProphecy(coverage),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            exploration_bonus=0.0,
            use_imagination=True,
            use_effect_composition=False,
            imagination_minimum_coverage=0.0,
            imagination_intervention_margin=margin,
            imagination_uncertainty_margin=uncertainty_margin,
        ),
        seed=1,
    )
    return agent, state, first, second


def test_small_imagined_advantage_does_not_override_policy() -> None:
    agent, state, policy_action, preferred = _agent(
        margin=0.05,
        uncertainty_margin=0.0,
    )
    agent.planner = FixedPlan(
        policy_action,
        preferred,
        first_value=1.0,
        second_value=1.04,
    )

    decision = agent.select_action(state, episode=0, explore=True)
    diagnostics = agent.imagination_diagnostics()

    assert decision.used_imagination
    assert decision.action == policy_action
    assert decision.policy_action_signature == policy_action.signature
    assert decision.imagination_preferred_action_signature == preferred.signature
    assert decision.imagination_switch_candidate
    assert not decision.imagination_intervention_allowed
    assert not decision.imagination_changed_action
    assert decision.imagination_gate_reason == "insufficient_advantage"
    assert decision.imagination_advantage == pytest.approx(0.04)
    assert decision.imagination_required_advantage == pytest.approx(0.05)
    assert diagnostics["switch_candidates"] == 1
    assert diagnostics["suppressed_switches"] == 1
    assert diagnostics.get("interventions", 0) == 0


def test_large_imagined_advantage_overrides_policy() -> None:
    agent, state, policy_action, preferred = _agent(
        margin=0.05,
        uncertainty_margin=0.0,
    )
    agent.planner = FixedPlan(
        policy_action,
        preferred,
        first_value=1.0,
        second_value=1.20,
    )

    decision = agent.select_action(state, episode=0, explore=True)
    diagnostics = agent.imagination_diagnostics()

    assert decision.action == preferred
    assert decision.imagination_switch_candidate
    assert decision.imagination_intervention_allowed
    assert decision.imagination_changed_action
    assert decision.imagination_gate_reason == "intervention"
    assert decision.imagination_advantage == pytest.approx(0.20)
    assert diagnostics["interventions"] == 1
    assert diagnostics["changed_actions"] == 1
    assert diagnostics["intervention_rate_per_candidate"] == 1.0


def test_low_coverage_increases_required_advantage() -> None:
    agent, state, policy_action, preferred = _agent(
        coverage=0.5,
        margin=0.05,
        uncertainty_margin=0.20,
    )
    agent.planner = FixedPlan(
        policy_action,
        preferred,
        first_value=1.0,
        second_value=1.12,
    )

    decision = agent.select_action(state, episode=0, explore=True)

    assert decision.imagination_required_advantage == pytest.approx(0.15)
    assert decision.imagination_advantage == pytest.approx(0.12)
    assert decision.action == policy_action
    assert decision.imagination_gate_reason == "insufficient_advantage"


def test_policy_action_must_be_evaluated_before_override() -> None:
    agent, state, policy_action, preferred = _agent(
        margin=0.0,
        uncertainty_margin=0.0,
    )
    third = Action("c")
    agent.planner = FixedPlan(
        preferred,
        third,
        first_value=1.0,
        second_value=2.0,
    )

    decision = agent.select_action(state, episode=0, explore=True)

    assert decision.action == policy_action
    assert decision.imagination_switch_candidate
    assert not decision.imagination_intervention_allowed
    assert decision.imagination_gate_reason == "policy_not_evaluated"


def test_intervention_margins_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="intervention_margin"):
        AutonomousAgentConfig(imagination_intervention_margin=-0.01)
    with pytest.raises(ValueError, match="uncertainty_margin"):
        AutonomousAgentConfig(imagination_uncertainty_margin=-0.01)


def test_standard_schemas_include_intervention_gate_metrics() -> None:
    metrics = {
        "imagination_switch_candidates",
        "imagination_interventions",
        "imagination_suppressed_switches",
        "imagination_intervention_rate",
        "imagination_advantage_mean",
        "imagination_required_advantage_mean",
    }

    assert metrics <= set(RESULT_FIELDS)
    assert metrics <= set(SUMMARY_METRICS)
