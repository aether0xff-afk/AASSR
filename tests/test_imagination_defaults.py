from __future__ import annotations

import pytest

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.tabular_prophecy import TabularProphecy


def test_agent_uses_multi_outcome_risk_adjusted_planning_by_default() -> None:
    config = AutonomousAgentConfig()
    agent = AutonomousLearningAgent(
        TabularProphecy(),
        config=config,
        seed=3,
    )

    assert config.imagination_outcome_samples == 2
    assert config.imagination_aggregation == "risk-adjusted"
    assert agent.planner.config.outcome_samples == 2
    assert agent.planner.config.aggregation == "risk_adjusted"


def test_imagination_outcome_samples_must_be_positive() -> None:
    with pytest.raises(ValueError, match="imagination_outcome_samples"):
        AutonomousAgentConfig(imagination_outcome_samples=0)
