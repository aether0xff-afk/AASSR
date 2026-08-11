from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def _renamed(route: str, raw_slot: int) -> StateSnapshot:
    action = Action(
        "request",
        parameters={"route_id": route, "profile_id": "profile-browse"},
    )
    vector = [0.0] * AGENT_STATE_SIZE
    vector[raw_slot] = 1.0
    return StateSnapshot(
        tuple(vector),
        facts=frozenset(
            {
                f"known_route:{route}",
                "known_profile:profile-browse",
            }
        ),
        available_actions=(action,),
    )


def test_imagination_state_key_is_rename_invariant() -> None:
    agent = build_current_pentest_aassr_core(
        seed=17,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    left = _renamed("route-05", 100)
    right = _renamed("route-28", 300)

    assert left.vector != right.vector
    assert agent.planner._state_key(left) == agent.planner._state_key(right)
    assert agent.diagnostics()["current_repairs"]["relational_imagination_state_key"] is True
