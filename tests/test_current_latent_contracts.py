from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from aassr_v2.branch_critic import CriticTransition
from aassr_v2.current_decision_optimization import _memoized_relational_coverage
from aassr_v2.current_entrypoint import (
    CURRENT_INTERVENTION_MARGIN,
    build_current_pentest_aassr_core,
)
from aassr_v2.current_semantic_calibration import _public_state_distance
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.types import Action, StateSnapshot


def test_canonical_current_margin_is_frozen_to_repaired_2k_contract() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        device="cpu",
        allow_tf32=False,
    )
    assert CURRENT_INTERVENTION_MARGIN == pytest.approx(0.05)
    assert agent.config.imagination_intervention_margin == pytest.approx(0.05)
    assert agent.config.imagination_uncertainty_margin == pytest.approx(0.0)


def test_zero_return_no_longer_counts_as_negative_critic_readiness_support() -> None:
    agent = build_current_pentest_aassr_core(
        seed=9,
        train_transitions=64,
        device="cpu",
        allow_tf32=False,
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    action = state.available_actions[0]
    transition = CriticTransition(state, action, state, 1.0)

    agent._critic_trajectory[:] = [transition]
    agent.finish_episode(final_return=0.0, training=True)
    assert agent._critic_counts["zero_returns"] == 1
    assert agent._critic_counts["non_successes"] == 0

    agent._critic_trajectory[:] = [transition]
    agent.finish_episode(final_return=-1.0, training=True)
    assert agent._critic_counts["negative_returns"] == 1
    assert agent._critic_counts["non_successes"] == 1


def test_critic_readiness_expires_when_recent_signed_evidence_disappears() -> None:
    agent = build_current_pentest_aassr_core(
        seed=10,
        train_transitions=64,
        device="cpu",
        allow_tf32=False,
    )
    agent._critic_counts.update(
        {
            "episodes": 64,
            "successes": 8,
            "non_successes": 8,
            "positive_returns": 8,
            "negative_returns": 8,
        }
    )
    agent.critic.gradient_updates = 1

    agent.critic.recent_episode_returns.clear()
    agent.critic.recent_episode_returns.extend((1.0,) * 4 + (-1.0,) * 4)
    assert agent.critic_ready is True
    assert agent.critic_reliably_ready() is True

    # Historical counts remain sufficient, but the active/recent regime has lost
    # all negative-return evidence. Imagination must fail closed again.
    agent.critic.recent_episode_returns.clear()
    agent.critic.recent_episode_returns.extend((1.0,) * 8 + (0.0,) * 8)
    assert agent.critic_ready is True
    assert agent.critic_reliably_ready() is False

    # Frozen checkpoints restore compact recent counts rather than training replay.
    agent.critic.recent_episode_returns.clear()
    agent._critic_counts.update(
        {
            "recent_return_window": 128,
            "recent_positive_returns": 4,
            "recent_zero_returns": 0,
            "recent_negative_returns": 4,
        }
    )
    assert agent.critic_reliably_ready() is True


def test_coverage_is_invariant_to_concrete_alias_multiplicity() -> None:
    alias_a = Action(
        "request",
        parameters={"route_id": "route-a", "profile_id": "profile-browse"},
    )
    alias_b = Action(
        "request",
        parameters={"route_id": "route-b", "profile_id": "profile-browse"},
    )
    distinct = Action(
        "request_object",
        parameters={
            "route_id": "route-c",
            "profile_id": "profile-browse",
            "object_id": "object-c",
        },
    )
    state = StateSnapshot(
        vector=(0.0,) * AGENT_STATE_SIZE,
        facts=frozenset(),
        available_actions=(alias_a, alias_b, distinct),
        metadata={},
    )
    calls: list[str] = []

    def confidence(_state, action):
        calls.append(action.signature)
        return 0.2 if action.verb_name == "request" else 0.8

    fake = SimpleNamespace(confidence=confidence)
    coverage = _memoized_relational_coverage(
        fake,
        state,
        (alias_a, alias_b, distinct),
    )
    assert coverage == pytest.approx(0.5)
    assert len(calls) == 2


def test_calibration_locality_rejects_status_action_surface_and_count_regime_shift() -> None:
    base_action = Action(
        "request",
        parameters={"route_id": "route-a", "profile_id": "profile-browse"},
    )
    new_action = Action(
        "request",
        parameters={"route_id": "route-b", "profile_id": "profile-browse"},
    )
    vector = [0.0] * AGENT_STATE_SIZE
    base_facts = {"last_status:200", "known_object:object-a"}
    base = StateSnapshot(
        vector=tuple(vector),
        facts=frozenset(base_facts),
        available_actions=(base_action,),
        metadata={},
    )
    status_shift = StateSnapshot(
        vector=tuple(vector),
        facts=frozenset({"last_status:403", "known_object:object-a"}),
        available_actions=(base_action,),
        metadata={},
    )
    surface_shift = StateSnapshot(
        vector=tuple(vector),
        facts=frozenset(base_facts),
        available_actions=(base_action, new_action),
        metadata={},
    )
    count_shift = StateSnapshot(
        vector=tuple(vector),
        facts=frozenset(
            {
                "last_status:200",
                "known_object:object-a",
                "known_object:object-b",
            }
        ),
        available_actions=(base_action,),
        metadata={},
    )
    assert _public_state_distance(base, base) == pytest.approx(0.0)
    assert _public_state_distance(base, status_shift) == pytest.approx(1.0)
    assert _public_state_distance(base, surface_shift) > 0.0
    # 1 -> 2 observed objects is a 50% public regime change. It must not collapse
    # to roughly 1/35 merely because only one descriptor channel changed.
    assert _public_state_distance(base, count_shift) >= 0.5 - 1e-12
