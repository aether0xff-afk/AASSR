from __future__ import annotations

import json

from aassr_v2.autonomous_agent import (
    AutonomousAgentConfig,
    AutonomousLearningAgent,
)
from aassr_v2.autonomous_experiment import _agent_config, _run_episode
from aassr_v2.experiment_runner import RESULT_FIELDS, SUMMARY_METRICS
from aassr_v2.tabular_prophecy import TabularProphecy


def test_agent_config_exposes_effect_composition_controls() -> None:
    config = _agent_config(
        {
            "name": "diagnostic",
            "use_effect_composition": False,
            "effect_minimum_samples": 5,
        },
        length=4,
    )

    assert not config.use_effect_composition
    assert config.effect_minimum_samples == 5
    assert config.imagination_minimum_coverage == 0.35


def test_standard_episode_row_contains_imagination_diagnostics() -> None:
    agent = AutonomousLearningAgent(
        TabularProphecy(name="observability"),
        config=AutonomousAgentConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_imagination=False,
            holdout_stride=1000,
        ),
        seed=7,
    )

    result = _run_episode(
        agent,
        length=2,
        world_seed=101,
        episode=0,
        phase="training",
        learn=True,
    )

    assert result["steps"] == 2
    assert result["imagination_opportunities"] == 0
    assert result["imagination_eligible"] == 0
    assert result["imagination_runs"] == 0
    assert result["imagination_changed_actions"] == 0
    assert result["imagination_change_rate"] == 0.0
    assert result["imagination_eligibility_rate"] == 0.0
    assert result["imagination_coverage_mean"] == 0.0
    assert json.loads(result["imagination_gate_reasons"]) == {
        "disabled": 2
    }

    transitions = result["_transitions"]
    assert len(transitions) == 2
    required = {
        "policy_action_signature",
        "imagination_opportunity",
        "imagination_eligible",
        "imagination_gate_reason",
        "imagination_changed_action",
        "model_coverage",
    }
    assert all(required <= set(row) for row in transitions)


def test_standard_csv_and_summary_schemas_include_numeric_diagnostics() -> None:
    numeric = {
        "imagination_opportunities",
        "imagination_eligible",
        "imagination_runs",
        "imagination_changed_actions",
        "imagination_change_rate",
        "imagination_eligibility_rate",
        "imagination_coverage_mean",
    }

    assert numeric <= set(RESULT_FIELDS)
    assert numeric <= set(SUMMARY_METRICS)
    assert "imagination_gate_reasons" in RESULT_FIELDS
    assert "imagination_gate_reasons" not in SUMMARY_METRICS
