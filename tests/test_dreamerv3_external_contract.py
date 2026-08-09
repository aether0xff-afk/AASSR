from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.dreamerv3_baseline import DREAMERV3_ACTION_SLOT_COUNT
from aassr_v2.dreamerv3_external import _DreamerPentestEnv
from aassr_v2.pentest_agent_main_test import ACTION_FEATURE_SIZE, AGENT_STATE_SIZE


class _FakeNP:
    float32 = float

    @staticmethod
    def asarray(values, dtype=None):
        del dtype
        return tuple(float(value) for value in values)


def test_dreamer_env_adapter_executes_exact_real_transition_budget() -> None:
    env = _DreamerPentestEnv(
        elements=object(),
        np=_FakeNP,
        research_seed=7,
        stage_index=0,
        scenario_seed=90_001,
        transition_cap=1,
        phase="contract",
    )

    first = env.step({"reset": True})
    assert first["is_first"] is True
    assert first["is_last"] is False
    assert len(first["state"]) == AGENT_STATE_SIZE
    assert len(first["action_mask"]) == DREAMERV3_ACTION_SLOT_COUNT
    assert sum(first["action_mask"]) > 0
    assert env.result is None

    terminal = env.step(
        {
            "reset": False,
            "action": [0.0] * ACTION_FEATURE_SIZE,
        }
    )
    assert terminal["is_first"] is False
    assert terminal["is_last"] is True
    assert terminal["is_terminal"] is True
    assert env.result is not None
    assert env.result.primitive_transitions == 1
    assert env.result.status in {"success", "failure", "stalled", "truncation"}
    assert env.result.reward in {-1.0, 0.0, 1.0}
    assert env.projection_distances


def test_dreamer_env_reset_restarts_episode_without_carrying_budget_state() -> None:
    env = _DreamerPentestEnv(
        elements=object(),
        np=_FakeNP,
        research_seed=42,
        stage_index=0,
        scenario_seed=90_002,
        transition_cap=1,
        phase="contract",
    )
    env.step({"reset": True})
    env.step({"reset": False, "action": [0.0] * ACTION_FEATURE_SIZE})
    assert env.result is not None
    assert env.transitions == 1

    restarted = env.step({"reset": True})
    assert restarted["is_first"] is True
    assert env.result is None
    assert env.transitions == 0
    assert env.projection_distances == []
