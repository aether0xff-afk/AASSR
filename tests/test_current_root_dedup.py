from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.types import Action, StateSnapshot


def _alias_state(count: int = 24) -> StateSnapshot:
    actions = tuple(
        Action(
            "request",
            parameters={
                "route_id": f"route-{index:02d}",
                "profile_id": "profile-browse",
            },
        )
        for index in range(count)
    )
    facts = {"known_profile:profile-browse"}
    facts.update(f"known_route:route-{index:02d}" for index in range(count))
    return StateSnapshot(
        vector=(0.0,) * AGENT_STATE_SIZE,
        facts=frozenset(facts),
        available_actions=actions,
        goal_progress=0.0,
        metadata={},
    )


def test_root_aliases_share_prophecy_and_critic_compute_but_keep_concrete_evaluations() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    state = _alias_state(24)
    result = agent.planner.plan(state, maximum_depth=1)

    assert len(result.root_evaluations) == len(state.available_actions) == 24
    assert {item.action.signature for item in result.root_evaluations} == {
        action.signature for action in state.available_actions
    }

    diagnostics = agent.diagnostics()["structural_root_dedup"]
    assert diagnostics["prophecy_requested_rows"] >= 24
    assert diagnostics["prophecy_unique_rows"] == 1
    assert diagnostics["prophecy_alias_rows_removed"] >= 23
    assert diagnostics["critic_root_requested_rows"] >= 24
    assert diagnostics["critic_root_unique_rows"] == 1
    assert diagnostics["critic_root_alias_rows_removed"] >= 23
