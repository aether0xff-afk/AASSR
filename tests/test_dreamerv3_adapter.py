from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_experiment_suite import (
    CURRENT_FINAL_EXPERIMENT_CONDITIONS,
    assemble_current_generation_suite,
)
from aassr_v2.dreamerv3_baseline import (
    DREAMERV3_ACTION_SLOT_COUNT,
    DREAMERV3_ACTION_SLOTS,
    DREAMERV3_UPSTREAM_COMMIT,
    dreamer_action_slot_key,
    dreamer_action_surface_mask,
    dreamer_adapter_manifest,
    dreamer_observation_vector,
    dreamer_relational_action_features,
    project_dreamer_action,
)
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.pentest_curriculum_causal import OBSERVATION_CONTRACT
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.types import Action, StateSnapshot


def _renamed_state(
    *,
    route: str,
    profile: str,
    object_id: str,
    raw_slot: int,
) -> tuple[StateSnapshot, Action]:
    action = Action(
        "request_object",
        parameters={
            "route_id": route,
            "profile_id": profile,
            "object_id": object_id,
        },
    )
    facts = frozenset(
        {
            "authenticated",
            f"known_route:{route}",
            f"known_profile:{profile}",
            f"known_object:{object_id}",
            f"observed_route_role:{route}:object",
            f"observed_profile_role:{profile}:read",
        }
    )
    vector = [0.0] * AGENT_STATE_SIZE
    vector[0] = 1.0
    vector[8] = 0.04
    vector[11 + raw_slot % max(1, AGENT_STATE_SIZE - 11)] = 1.0
    return (
        StateSnapshot(
            vector=tuple(vector),
            facts=facts,
            available_actions=(action,),
            goal_progress=0.0,
            metadata={"observation_contract": OBSERVATION_CONTRACT},
        ),
        action,
    )


def test_dreamer_action_vocabulary_is_fixed_unique_and_complete_size() -> None:
    assert DREAMERV3_ACTION_SLOT_COUNT == 240
    assert len(DREAMERV3_ACTION_SLOTS) == len(set(DREAMERV3_ACTION_SLOTS))
    assert all(slot[0] in {"request", "request_object"} for slot in DREAMERV3_ACTION_SLOTS)
    assert all(slot[3] == "none" for slot in DREAMERV3_ACTION_SLOTS if slot[0] == "request")
    assert all(slot[3] != "none" for slot in DREAMERV3_ACTION_SLOTS if slot[0] == "request_object")


def test_dreamer_adapter_is_seed_rename_invariant() -> None:
    left, left_action = _renamed_state(
        route="route-28",
        profile="profile-14",
        object_id="object-07",
        raw_slot=3,
    )
    right, right_action = _renamed_state(
        route="route-05",
        profile="profile-02",
        object_id="object-31",
        raw_slot=17,
    )

    assert dreamer_observation_vector(left) == dreamer_observation_vector(right)
    assert dreamer_action_slot_key(left, left_action) == dreamer_action_slot_key(
        right, right_action
    )
    assert dreamer_relational_action_features(
        left, left_action
    ) == dreamer_relational_action_features(right, right_action)
    assert dreamer_action_surface_mask(left) == dreamer_action_surface_mask(right)


def test_exact_relational_proposal_projects_to_a_currently_legal_action() -> None:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0])
    state = world.snapshot()
    assert state.available_actions
    chosen = state.available_actions[-1]
    proposal = dreamer_relational_action_features(state, chosen)

    projection = project_dreamer_action(state, proposal)
    assert projection.action in state.available_actions
    assert projection.squared_distance == pytest.approx(0.0)
    assert projection.relational_features == proposal
    assert dreamer_action_surface_mask(state)[projection.slot_index] == 1.0


def test_every_predeclared_stage_surface_is_covered_without_hidden_scenario_access() -> None:
    for stage in TRANSFER_STAGES:
        for seed in (90_001, 90_007, 90_016):
            state = TransferDiagnosticWorld(seed, stage=stage).snapshot()
            mask = dreamer_action_surface_mask(state)
            assert len(mask) == DREAMERV3_ACTION_SLOT_COUNT
            for action in state.available_actions:
                slot = dreamer_action_slot_key(state, action)
                assert slot in DREAMERV3_ACTION_SLOTS
                projection = project_dreamer_action(
                    state,
                    dreamer_relational_action_features(state, action),
                )
                assert projection.action in state.available_actions
                assert mask[projection.slot_index] == 1.0


def test_adapter_manifest_declares_official_unmodified_upstream() -> None:
    manifest = dreamer_adapter_manifest()
    assert manifest["condition"] == "dreamerv3_relational"
    assert manifest["upstream_repository"] == "danijar/dreamerv3"
    assert manifest["upstream_agent_modified"] is False
    assert manifest["oracle_information"] is False
    assert manifest["slot_count"] == 240


def _fake_current_summary() -> dict[str, object]:
    front = {"highest_contiguous_nonzero_level": 0, "first_zero_success_level": 1}
    return {
        "architecture_version": "aassr-current-generation-v1",
        "research_seed": 7,
        "transition_budget_per_training_condition": 10_000,
        "diagnostic_successes": {
            "dqn_raw": 1,
            "dqn_relational": 2,
            "aassr_current_no_imagination": 3,
            "aassr_current_full": 4,
        },
        "frontier": {
            "dqn_raw": front,
            "dqn_relational": front,
            "aassr_current_no_imagination": front,
            "aassr_current_full": front,
        },
        "train_seeds": [90_001],
        "validation_seeds": [93_001],
        "diagnostic_seeds": [7],
        "stage_manifest": [{"level": 0}],
        "diagnostic_full_stage_sweep": True,
        "final_blind_consumed": False,
    }


def _fake_dreamer_summary() -> dict[str, object]:
    return {
        "condition": "dreamerv3_relational",
        "research_seed": 7,
        "transitions_used": 10_000,
        "exact_budget": True,
        "diagnostic_successes": 5,
        "frontier": {
            "highest_contiguous_nonzero_level": 1,
            "first_zero_success_level": 2,
        },
        "official_upstream": {
            "actual_commit": DREAMERV3_UPSTREAM_COMMIT,
            "commit_matches_pin": True,
        },
        "train_seeds": [90_001],
        "validation_seeds": [93_001],
        "diagnostic_seeds": [7],
        "stage_manifest": [{"level": 0}],
        "final_blind_consumed": False,
    }


def test_five_condition_suite_requires_matching_scientific_contracts() -> None:
    result = assemble_current_generation_suite(
        _fake_current_summary(),
        _fake_dreamer_summary(),
    )
    assert tuple(result["experiment_conditions"]) == CURRENT_FINAL_EXPERIMENT_CONDITIONS
    assert result["training_checkpoint_count"] == 4
    assert result["nominal_total_training_transitions"] == 40_000
    assert result["diagnostic_successes"]["dreamerv3_relational"] == 5
    assert result["comparison_contract"]["dreamerv3_upstream_algorithm_modified"] is False


def test_five_condition_suite_rejects_noncanonical_dreamer_checkout() -> None:
    dreamer = _fake_dreamer_summary()
    dreamer["official_upstream"] = {
        "actual_commit": "deadbeef",
        "commit_matches_pin": False,
    }
    with pytest.raises(ValueError, match="pinned official DreamerV3"):
        assemble_current_generation_suite(_fake_current_summary(), dreamer)
