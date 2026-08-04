from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aassr_v2.autonomous_experiment import (
    _run_episode,
    run_autonomous_experiment,
)
from aassr_v2.experiment_runner import RESULT_FIELDS, SUMMARY_METRICS


class OracleInterventionAgent:
    def __init__(self, *, correct_policy: bool) -> None:
        self.correct_policy = correct_policy

    def select_action_for_environment(
        self,
        environment: object,
        state: object,
        *,
        episode: int,
        explore: bool,
    ) -> SimpleNamespace:
        del episode, explore
        oracle = environment.oracle_action()
        alternative = next(
            action
            for action in state.available_actions
            if action.signature != oracle.signature
        )
        policy_action = oracle if self.correct_policy else alternative
        executed_action = alternative if self.correct_policy else oracle
        return SimpleNamespace(
            action=executed_action,
            used_imagination=True,
            imagined_nodes=2,
            imagination_depth=2,
            root_imagined_value=1.0,
            policy_action_signature=policy_action.signature,
            imagination_opportunity=True,
            imagination_eligible=True,
            imagination_gate_reason="eligible",
            imagination_changed_action=True,
            model_coverage=1.0,
        )

    def observe(self, before: object, action: object, outcome: object) -> SimpleNamespace:
        del before, action, outcome
        return SimpleNamespace(
            prediction_score=1.0,
            holdout_after=1.0,
            holdout_gain=0.0,
            intrinsic_value=0.0,
            error=False,
            repeated=False,
        )

    def finish_episode(self, *, final_return: float) -> None:
        del final_return

    def discard_episode(self) -> None:
        return None


def test_oracle_audit_counts_imagination_corrections() -> None:
    result = _run_episode(
        OracleInterventionAgent(correct_policy=False),
        length=3,
        world_seed=101,
        episode=0,
        phase="training",
        learn=True,
    )

    assert result["steps"] == 3
    assert result["policy_oracle_agreements"] == 0
    assert result["executed_oracle_agreements"] == 3
    assert result["imagination_corrections"] == 3
    assert result["imagination_harms"] == 0
    assert result["imagination_neutral_changes"] == 0
    assert result["imagination_net_corrections"] == 3
    assert result["imagination_oracle_gain"] == 3
    assert result["imagination_correction_rate"] == 1.0
    assert result["imagination_harm_rate"] == 0.0

    for transition in result["_transitions"]:
        privileged = transition["privileged_analysis"]
        assert privileged["policy_oracle_agreement"] is False
        assert privileged["executed_oracle_agreement"] is True
        assert privileged["imagination_oracle_delta"] == 1
        assert "oracle_action_signature" not in transition["before"]
        assert "oracle_action_signature" not in transition["after"]


def test_oracle_audit_counts_imagination_harms() -> None:
    result = _run_episode(
        OracleInterventionAgent(correct_policy=True),
        length=4,
        world_seed=202,
        episode=0,
        phase="training",
        learn=True,
    )

    assert result["steps"] == 4
    assert result["policy_oracle_agreements"] == 4
    assert result["executed_oracle_agreements"] == 0
    assert result["imagination_corrections"] == 0
    assert result["imagination_harms"] == 4
    assert result["imagination_net_corrections"] == -4
    assert result["imagination_oracle_gain"] == -4
    assert result["imagination_correction_rate"] == 0.0
    assert result["imagination_harm_rate"] == 1.0


def test_standard_schemas_include_counterfactual_audit_metrics() -> None:
    metrics = {
        "policy_oracle_agreements",
        "executed_oracle_agreements",
        "policy_oracle_agreement_rate",
        "executed_oracle_agreement_rate",
        "imagination_corrections",
        "imagination_harms",
        "imagination_neutral_changes",
        "imagination_net_corrections",
        "imagination_oracle_gain",
        "imagination_correction_rate",
        "imagination_harm_rate",
    }

    assert metrics <= set(RESULT_FIELDS)
    assert metrics <= set(SUMMARY_METRICS)


def test_protocol_manifest_declares_oracle_analysis_boundaries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "oracle-manifest"
    config = {
        "name": "oracle_manifest_regression",
        "runner": "autonomous_main",
        "seeds": [7],
        "train_episodes": 1,
        "eval_episodes": 1,
        "evaluation_modes": ["seen"],
        "execution": {
            "workers": 1,
            "cuda_workers": 1,
            "device": "cpu",
            "allow_cpu_fallback": True,
        },
        "progress": {"every_episodes": 1, "every_seconds": 1},
        "environments": [
            {
                "name": "opaque_dependency_l2",
                "length": 2,
                "seed_offset": 2000,
                "train_worlds_per_seed": 1,
                "eval_worlds_per_seed": 1,
            }
        ],
        "conditions": [
            {
                "name": "contextual_policy",
                "algorithm": "contextual_policy",
                "model": "tabular",
                "learn_policy": True,
                "learn_prophecy": False,
                "use_imagination": False,
                "validated_gain_weight": 0.0,
            }
        ],
    }

    run_autonomous_experiment(
        config,
        output_dir=output,
        overwrite=False,
        progress_console=False,
    )
    manifest = json.loads(
        (output / "protocol_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["privileged_oracle_analysis_only"] is True
    assert manifest["oracle_labels_agent_visible"] is False
    assert manifest["oracle_labels_used_for_learning"] is False
