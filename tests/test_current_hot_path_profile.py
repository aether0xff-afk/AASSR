from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_hot_path_profile import (
    HOT_PATH_CATEGORIES,
    current_hot_path_phase,
    current_hot_path_snapshot,
)
from aassr_v2.pentest_current_generation_smoke import _learning_counters
from aassr_v2.pentest_current_generation_main import (
    run_current_generation_condition,
)
from aassr_v2.pentest_transfer_stages import (
    TRANSFER_STAGES,
    TransferDiagnosticWorld,
)


def _exercise(agent: object) -> tuple[tuple[object, ...], tuple[object, ...], int]:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0])
    observations: list[object] = []
    transitions = 0
    with current_hot_path_phase(agent, "training"):
        agent.begin_episode()
        while transitions < 4 and not world.success and not world.rate_limited:
            step = agent.step(
                world,
                episode=transitions,
                training=True,
                primitive_budget=1,
            )
            assert len(step.traces) == 1
            evaluation = step.evaluations[0]
            observations.append(
                (
                    step.decision.action.signature,
                    step.decision.core_decision,
                    evaluation.trace,
                    evaluation.effect,
                    evaluation.uncertainty_before,
                    evaluation.uncertainty_after,
                    dict(evaluation.features),
                    evaluation.predicted_information_value,
                    evaluation.immediate_information_value,
                )
            )
            transitions += 1
        final_return = 1.0 if world.success else 0.0
        agent.finish_episode(final_return=final_return, training=True)
    return tuple(observations), _learning_counters(agent), transitions


def test_hot_path_profiler_is_observationally_equivalent_to_reference() -> None:
    reference = build_current_pentest_aassr_core(
        seed=2026,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
        allow_tf32=False,
        profile_hot_path=False,
    )
    profiled = build_current_pentest_aassr_core(
        seed=2026,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
        allow_tf32=False,
        profile_hot_path=True,
    )

    reference_outputs, reference_counters, reference_transitions = _exercise(reference)
    profiled_outputs, profiled_counters, profiled_transitions = _exercise(profiled)

    assert profiled_outputs == reference_outputs
    assert profiled_counters == reference_counters
    assert profiled_transitions == reference_transitions
    assert current_hot_path_snapshot(
        reference,
        training_transitions=reference_transitions,
    ) == {"enabled": False}

    profile = current_hot_path_snapshot(
        profiled,
        training_transitions=profiled_transitions,
    )
    assert profile["enabled"] is True
    assert profile["host_observed"] is True
    assert profile["cuda_synchronized"] is False
    assert tuple(profile["categories"]) == HOT_PATH_CATEGORIES
    assert set(profile["phases"]) == {"training"}
    assert profile["training_transitions"] == profiled_transitions

    categories = profile["categories"]
    assert categories["policy_selection"]["training_calls"] == profiled_transitions
    assert categories["evaluator_pre_prediction"]["training_calls"] == (
        2 * profiled_transitions
    )
    assert categories["evaluator_post_prediction"]["training_calls"] == (
        profiled_transitions
    )
    assert categories["holdout_before"]["training_calls"] == profiled_transitions
    assert categories["holdout_after"]["training_calls"] == profiled_transitions
    assert categories["environment_step"]["training_calls"] == profiled_transitions
    assert categories["dqn_observe_train"]["training_calls"] == profiled_transitions
    assert categories["critic_episode_learning"]["training_calls"] == 1
    for row in categories.values():
        assert row["total_seconds"] >= 0.0
        assert row["seconds_per_call"] >= 0.0
        assert row["training_seconds_per_transition"] >= 0.0

    validator = profile["validator_runtime_diagnostics"]
    assert "cache_hits" in validator
    assert "cache_misses" in validator
    assert "batch_calls" in validator
    assert validator["expected_vector_calls"] == 0


def test_main_summary_exposes_phase_scoped_hot_path_profile(tmp_path) -> None:
    result = run_current_generation_condition(
        tmp_path,
        research_seed=7,
        transition_budget=2,
        block_target=2,
        train_seeds=(90_001,),
        validation_seeds=(93_001,),
        diagnostic_seeds=(92_001,),
        diagnostic_stage_indices=(0,),
        device="cpu",
        allow_tf32=False,
        profile_hot_path=True,
    )

    profile = result["hot_path_profile"]
    assert profile["enabled"] is True
    assert set(profile["phases"]) == {
        "training",
        "curriculum_validation",
        "no_imagination_diagnostic",
        "full_diagnostic",
    }
    assert profile["training_transitions"] == 2
    assert profile["categories"]["environment_step"][
        "training_calls"
    ] == 2
    persisted = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert persisted["hot_path_profile"] == profile
