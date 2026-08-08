from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.current_generation import (
    CURRENT_COMPONENTS,
    CURRENT_GENERATION_VERSION,
    LEGACY_COMPONENTS_ACTIVE,
    CurrentNeuralDeltaProphecy,
    CurrentPentestAASSRAgent,
    CurrentRelationalPolicy,
    RelationalGRUBranchCritic,
    RelationalInvariantDQN,
    RelationalSkillLibrary,
    relational_action_key,
    relational_state_key,
)
from aassr_v2.current_performance import CurrentDepthBatchedProphecyView
from aassr_v2.current_runtime import (
    CurrentPentestRuntimeAgent,
    FrozenReplayRelationalCalibratedProphecy,
    FullyRelationalNeuralDeltaProphecy,
)
from aassr_v2.goals import GoalStateScorer
from aassr_v2.gru_prophecy import OnlineGRUProphecy
from aassr_v2.native_batching import DepthBatchedImaginationTree
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
    # Make the raw vector intentionally identifier-sensitive while preserving the
    # same public relational situation. A current transfer learner must ignore
    # this synthetic slot permutation; the legacy Http codec would not.
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
    assert CURRENT_GENERATION_VERSION == "aassr-current-generation-v1"
    assert LEGACY_COMPONENTS_ACTIVE == ()
    assert CURRENT_COMPONENTS["policy"].startswith("relational-invariant-dqn")
    assert CURRENT_COMPONENTS["prophecy"].startswith("neural-delta-ensemble")
    assert CURRENT_COMPONENTS["critic"].startswith("relational-gru-branch-critic")
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

    # ASEQ/cycle detection must still distinguish concrete episode-local entities.
    assert semantic_fingerprint(left) != semantic_fingerprint(right)


def test_public_current_builder_instantiates_current_runtime_not_legacy_ones() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        use_imagination=True,
        device="cpu",
    )

    assert isinstance(agent, CurrentPentestRuntimeAgent)
    assert isinstance(agent, CurrentPentestAASSRAgent)
    assert isinstance(agent.policy, CurrentRelationalPolicy)
    assert isinstance(agent.dqn, RelationalInvariantDQN)
    assert isinstance(agent.base_neural_prophecy, FullyRelationalNeuralDeltaProphecy)
    assert isinstance(agent.base_neural_prophecy, CurrentNeuralDeltaProphecy)
    assert isinstance(agent.calibrated_prophecy, FrozenReplayRelationalCalibratedProphecy)
    assert isinstance(agent.critic, RelationalGRUBranchCritic)
    assert isinstance(agent.skills, RelationalSkillLibrary)
    assert isinstance(agent.core.planner, DepthBatchedImaginationTree)
    assert isinstance(agent.current_batched_prophecy, CurrentDepthBatchedProphecyView)
    assert agent.current_depth_batching is True

    assert not isinstance(agent.policy, SemanticContextualPolicy)
    assert not isinstance(agent.base_neural_prophecy, OnlineGRUProphecy)
    assert not isinstance(agent.core.planner.scorer, GoalStateScorer)
    assert agent.core.config.use_effect_composition is False
    assert agent.legacy_components_active == ()

    diagnostics = agent.diagnostics()
    assert diagnostics["current_generation"] is True
    assert diagnostics["canonical_runtime"] == "current_runtime"
    assert diagnostics["calibration_same_transition_frozen"] is True
    assert diagnostics["prophecy_state_input_relational"] is True
    assert diagnostics["prophecy_action_input_relational"] is True
    assert diagnostics["legacy_components_active"] == []
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

    # The old concrete codec sees the permutation; the active Prophecy input must
    # not. This catches the exact half-relational regression that caused v0.4
    # transfer to collapse under seed-renamed identifiers.
    assert agent.base_neural_prophecy.codec.encode(left) != agent.base_neural_prophecy.codec.encode(right)
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

    # Training-time imagination is intentionally off until a frozen checkpoint is
    # evaluated with a trained critic. This is the latest same-checkpoint rule,
    # not the old hand-scored intervention path.
    assert step.decision.core_decision.used_imagination is False

    # Calibration was frozen exactly around the real transition and released
    # afterwards, so a just-created holdout item cannot calibrate itself.
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
