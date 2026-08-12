from __future__ import annotations

from collections import Counter
from types import MethodType

from aassr_v2.autonomous_agent_core import ActionDecision
from aassr_v2.branch_critic import CriticTransition
from aassr_v2.current_critic_support import (
    DEFAULT_CRITIC_SUPPORT_THRESHOLD,
    install_critic_support_gate,
)
from aassr_v2.current_relational_state_v3 import STATUS_CODES_V3
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


POLICY = Action(
    "request",
    parameters={"route_id": "route-00", "profile_id": "profile-browse"},
)
CANDIDATE = Action(
    "request_object",
    parameters={
        "route_id": "route-01",
        "profile_id": "profile-01",
        "object_id": "object-00",
    },
)


def _state(object_count: int) -> StateSnapshot:
    vector = [0.0] * AGENT_STATE_SIZE
    vector[AGENT_STATE_SIZE - len(STATUS_CODES_V3) + STATUS_CODES_V3.index(200)] = 1.0
    facts = {
        "known_route:route-00",
        "known_route:route-01",
        "known_profile:profile-browse",
        "known_profile:profile-01",
        "observed_route_role:route-01:object",
        "observed_profile_role:profile-01:read",
    }
    facts.update(f"known_object:object-{index:02d}" for index in range(object_count))
    return StateSnapshot(
        vector=tuple(vector),
        facts=frozenset(facts),
        available_actions=(POLICY, CANDIDATE),
        goal_progress=0.0,
        metadata={},
    )


class _Critic:
    def __init__(self) -> None:
        self.calls = 0

    def observe_episode(self, trajectory, *, success: bool) -> None:
        del trajectory, success
        self.calls += 1


class _Agent:
    def __init__(self) -> None:
        self.critic = _Critic()
        self._core_select_action = MethodType(_raw_intervention, self)

    def diagnostics(self):
        return {}


class _RecordedAgent(_Agent):
    def __init__(self) -> None:
        super().__init__()
        self._imagination_diagnostics: Counter[str] = Counter()
        self._core_select_action = MethodType(_recorded_intervention, self)


def _raw_intervention(self, state, *, episode: int, explore: bool):
    del self, state, episode, explore
    return ActionDecision(
        CANDIDATE,
        True,
        policy_action_signature=POLICY.signature,
        imagination_opportunity=True,
        imagination_eligible=True,
        imagination_gate_reason="intervention",
        imagination_changed_action=True,
        imagination_policy_value=0.20,
        imagination_preferred_value=0.50,
        imagination_advantage=0.30,
        imagination_required_advantage=0.05,
        imagination_switch_candidate=True,
        imagination_intervention_allowed=True,
    )


def _recorded_intervention(self, state, *, episode: int, explore: bool):
    decision = _raw_intervention(self, state, episode=episode, explore=explore)
    diagnostics = self._imagination_diagnostics
    diagnostics["opportunities"] += 1
    diagnostics["eligible"] += 1
    diagnostics["runs"] += 1
    diagnostics["switch_candidates"] += 1
    diagnostics["interventions"] += 1
    diagnostics["changed_actions"] += 1
    diagnostics["gate:intervention"] += 1
    return decision


def _observe_support(agent: _Agent, state: StateSnapshot, action: Action, count: int = 20):
    trajectory = tuple(
        CriticTransition(state, action, state, 1.0)
        for _ in range(count)
    )
    agent.critic.observe_episode(trajectory, success=False)


def test_candidate_without_local_critic_support_cannot_override_policy() -> None:
    agent = _Agent()
    install_critic_support_gate(agent)
    state = _state(3)
    _observe_support(agent, state, POLICY)

    decision = agent._core_select_action(state, episode=0, explore=False)
    assert decision.action.signature == POLICY.signature
    assert decision.imagination_intervention_allowed is False
    assert decision.imagination_gate_reason == "critic_candidate_out_of_distribution"

    _observe_support(agent, state, CANDIDATE)
    decision = agent._core_select_action(state, episode=0, explore=False)
    assert decision.action.signature == CANDIDATE.signature
    assert decision.imagination_intervention_allowed is True


def test_support_block_reconciles_pre_recorded_intervention() -> None:
    agent = _RecordedAgent()
    install_critic_support_gate(agent)
    state = _state(3)
    _observe_support(agent, state, POLICY)

    decision = agent._core_select_action(state, episode=0, explore=False)
    assert decision.action.signature == POLICY.signature
    assert decision.imagination_intervention_allowed is False
    assert decision.imagination_changed_action is False
    assert decision.imagination_gate_reason == "critic_candidate_out_of_distribution"

    counters = agent._imagination_diagnostics
    assert counters["runs"] == 1
    assert counters["switch_candidates"] == 1
    assert counters["interventions"] == 0
    assert counters["changed_actions"] == 0
    assert counters["suppressed_switches"] == 1
    assert counters["gate:intervention"] == 0
    assert counters["gate:critic_candidate_out_of_distribution"] == 1
    assert (
        agent._critic_support_diagnostics["reconciled_pre_support_interventions"]
        == 1
    )


def test_critic_support_drops_when_public_problem_scale_is_unseen() -> None:
    agent = _Agent()
    install_critic_support_gate(agent)
    training_state = _state(3)
    larger_state = _state(6)
    _observe_support(agent, training_state, CANDIDATE)

    in_distribution = agent.critic.support_confidence(training_state, CANDIDATE)
    out_of_distribution = agent.critic.support_confidence(larger_state, CANDIDATE)
    assert in_distribution >= DEFAULT_CRITIC_SUPPORT_THRESHOLD
    assert out_of_distribution < DEFAULT_CRITIC_SUPPORT_THRESHOLD
