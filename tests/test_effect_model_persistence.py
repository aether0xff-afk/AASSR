from __future__ import annotations

from pathlib import Path

import pytest

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.model_io import load_agent_model, save_agent_model
from aassr_v2.tabular_prophecy import TabularProphecy
from aassr_v2.types import Action, StateSnapshot


def _agent() -> AutonomousLearningAgent:
    return AutonomousLearningAgent(
        TabularProphecy(),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            effect_minimum_samples=2,
            imagination_minimum_coverage=0.0,
        ),
        seed=17,
    )


def test_effect_memory_survives_portable_model_reload(tmp_path: Path) -> None:
    advance = Action("advance")
    wait = Action("wait")
    before = StateSnapshot(
        vector=(0.0, 4.0),
        facts=frozenset({"closed", "persistent"}),
        available_actions=(advance, wait),
        goal_progress=0.0,
    )
    after = StateSnapshot(
        vector=(1.0, 4.0),
        facts=frozenset({"open", "persistent"}),
        available_actions=(wait,),
        goal_progress=0.5,
    )
    original = _agent()
    original.prophecy.learn(before, advance, after)
    original.prophecy.learn(before, advance, after)

    path = save_agent_model(
        original,
        tmp_path / "effect-roundtrip",
        completed_episodes=2,
    )
    restored = _agent()
    load_agent_model(restored, path)

    novel = StateSnapshot(
        vector=(10.0, 4.0),
        facts=frozenset({"closed", "novel"}),
        available_actions=(advance, wait),
        goal_progress=0.1,
    )
    prediction = restored.prophecy.predict(novel, advance, samples=1)[0]

    assert prediction.source == "effect-composed:action-family"
    assert prediction.next_state.vector == pytest.approx((11.0, 4.0))
    assert prediction.next_state.goal_progress == pytest.approx(0.6)
    assert "novel" in prediction.next_state.facts
    assert "open" in prediction.next_state.facts
    assert "closed" not in prediction.next_state.facts
    assert restored.prophecy_diagnostics()["effect_observations"] == 2
