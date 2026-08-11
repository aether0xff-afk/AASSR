from __future__ import annotations

import pytest

pytest.importorskip("torch")

import aassr_v2
from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.autonomous_agent_core import AutonomousLearningAgent
from aassr_v2.current_agent import CurrentStandalonePentestAASSRAgent
from aassr_v2.current_generation import (
    CurrentRelationalPolicy,
    RelationalInvariantDQN,
    RelationalSkillLibrary,
    relational_action_key,
    relational_state_key,
)
from aassr_v2.current_manifest import (
    CURRENT_COMPONENTS,
    CURRENT_GENERATION_VERSION,
    LEGACY_COMPONENTS_ACTIVE,
)
from aassr_v2.current_planner import CurrentFullyBatchedImaginationTree
from aassr_v2.current_relational_model import RelationalStochasticProphecy
from aassr_v2.current_relational_skill_prophecy import RelationalStochasticSkillProphecy
from aassr_v2.current_return_critic import (
    ReturnAwareHardwareRelationalGRUBranchCritic,
)
from aassr_v2.current_semantic_calibration import (
    RelationalDepthBatchedProphecyView,
    SemanticCalibratedProphecy,
)
from aassr_v2.goals import GoalStateScorer
from aassr_v2.gru_prophecy import OnlineGRUProphecy
from aassr_v2.integrated_agent import IntegratedAASSRAgent
from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
from aassr_v2.pentest_curriculum_causal import OBSERVATION_CONTRACT
from aassr_v2.pentest_curriculum_schedule import semantic_fingerprint
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.semantic_control import SemanticContextualPolicy
from aassr_v2.types import Action, StateSnapshot


def _renamed_state(
    *,
    route: str,
    profile: str,
    object_id: str,
    raw_slot: int = 0,
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
    identifier_start = 11
    identifier_width = max(1, AGENT_STATE_SIZE - identifier_start)
    vector[identifier_start + raw_slot % identifier_width] = 1.0
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


def test_current_manifest_has_no_active_legacy_components() -> None:
    assert CURRENT_GENERATION_VERSION == "aassr-current-generation-v2"
    assert LEGACY_COMPONENTS_ACTIVE == ()
    assert CURRENT_COMPONENTS["policy"].startswith("relational-invariant-dqn")
    assert CURRENT_COMPONENTS["prophecy"].startswith("relational-stochastic-world-model")
    assert "relational-descriptor" in CURRENT_COMPONENTS["prophecy_output"]
    assert CURRENT_COMPONENTS["critic"].startswith(
        "relational-gru-discounted-sparse-return"
    )
    assert CURRENT_COMPONENTS["effect_composition"].endswith("disabled")
    assert CURRENT_COMPONENTS["training_imagination"] == "disabled-same-checkpoint"


def test_transfer_identity_is_rename_invariant_but_aseq_identity_is_concrete() -> None:
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

    assert relational_state_key(left) == relational_state_key(right)
    assert relational_action_key(left, left_action) == relational_action_key(
        right,
        right_action,
    )
    assert semantic_fingerprint(left) != semantic_fingerprint(right)


def test_public_current_builder_constructs_repaired_current_runtime() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        use_imagination=True,
        device="cpu",
    )

    assert isinstance(agent, CurrentStandalonePentestAASSRAgent)
    assert aassr_v2.CurrentPentestAASSRAgent is CurrentStandalonePentestAASSRAgent
    assert IntegratedAASSRAgent not in type(agent).mro()
    assert AutonomousLearningAgent not in type(agent).mro()

    assert isinstance(agent.policy, CurrentRelationalPolicy)
    assert isinstance(agent.dqn, RelationalInvariantDQN)
    assert isinstance(agent.base_neural_prophecy, RelationalStochasticProphecy)
    assert isinstance(agent.calibrated_prophecy, SemanticCalibratedProphecy)
    assert isinstance(agent.skill_prophecy, RelationalStochasticSkillProphecy)
    assert isinstance(agent.critic, ReturnAwareHardwareRelationalGRUBranchCritic)
    assert isinstance(agent.skills, RelationalSkillLibrary)
    assert isinstance(agent.planner, CurrentFullyBatchedImaginationTree)
    assert agent.core.planner is agent.planner
    assert isinstance(agent.current_batched_prophecy, RelationalDepthBatchedProphecyView)
    assert agent.current_depth_batching is True
    assert agent.current_critic_batching is True
    assert agent.current_semantic_validation is True
    assert agent.current_semantic_evaluator is True

    assert not isinstance(agent.policy, SemanticContextualPolicy)
    assert not isinstance(agent.base_neural_prophecy, OnlineGRUProphecy)
    assert not isinstance(agent.planner.scorer, GoalStateScorer)
    assert agent.core.config.use_effect_composition is False
    assert agent.legacy_components_active == ()

    diagnostics = agent.diagnostics()
    assert diagnostics["current_generation"] is True
    assert diagnostics["canonical_runtime"] == "current_standalone"
    assert diagnostics["calibration_same_transition_frozen"] is True
    assert diagnostics["prophecy_state_input_relational"] is True
    assert diagnostics["prophecy_action_input_relational"] is True
    assert diagnostics["legacy_components_active"] == []
    assert diagnostics["current_components"] == dict(CURRENT_COMPONENTS)
    repairs = diagnostics["current_repairs"]
    assert repairs["relational_input_output_contract"] is True
    assert repairs["relational_imagination_state_key"] is True
    assert repairs["multi_outcome_ensemble"] is True
    assert repairs["reliability_outcome_probability_separated"] is True
    assert repairs["stochastic_skill_rollout"] is True
    assert repairs["semantic_information_evaluator"] is True
    assert repairs["critic_sparse_return_aligned"] is True
    assert repairs["critic_root_scale_values"] is True
    assert repairs["deeper_structural_branching"] is True
    assert diagnostics["identity_contracts"]["aseq_cycle_detection"].startswith(
        "concrete"
    )
    assert diagnostics["identity_contracts"]["policy_transfer"].startswith(
        "relational"
    )


def test_current_prophecy_input_is_rename_invariant_even_when_raw_slots_move() -> None:
    agent = build_current_pentest_aassr_core(
        seed=11,
        train_transitions=128,
        use_imagination=True,
        device="cpu",
    )
    left, left_action = _renamed_state(
        route="route-28",
        profile="profile-14",
        object_id="object-07",
        raw_slot=2,
    )
    right, right_action = _renamed_state(
        route="route-05",
        profile="profile-02",
        object_id="object-31",
        raw_slot=29,
    )

    assert left.vector != right.vector
    assert agent.base_neural_prophecy._input(left, left_action) == agent.base_neural_prophecy._input(
        right,
        right_action,
    )


def test_current_agent_runs_one_real_v3_transition_and_learns_only_real_data() -> None:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0])
    agent = build_current_pentest_aassr_core(
        seed=42,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )

    agent.begin_episode()
    step = agent.step(
        world,
        episode=0,
        training=True,
        primitive_budget=1,
    )
    assert len(step.traces) == 1
    assert step.traces[0].before.metadata["observation_contract"] == OBSERVATION_CONTRACT
    assert step.decision.core_decision.used_imagination is False
    assert agent.calibrated_prophecy.freeze_count == 1
    assert agent.calibrated_prophecy._frozen_holdout is None

    agent.finish_episode(final_return=0.0, training=True)
    assert agent.dqn.environment_steps == 1
    assert agent.base_neural_prophecy.observations == 1
    assert agent.critic.stats().episodes == 1


def test_relational_policy_scores_renamed_actions_with_same_structural_input() -> None:
    left, left_action = _renamed_state(
        route="route-28",
        profile="profile-14",
        object_id="object-07",
        raw_slot=6,
    )
    right, right_action = _renamed_state(
        route="route-05",
        profile="profile-02",
        object_id="object-31",
        raw_slot=21,
    )
    dqn = RelationalInvariantDQN(123, train_transitions=128)

    left_input = dqn.encode_state(left) + relational_action_key(left, left_action)
    right_input = dqn.encode_state(right) + relational_action_key(right, right_action)
    assert left_input == right_input
