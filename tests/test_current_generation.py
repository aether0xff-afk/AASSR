from __future__ import annotations

import pytest

pytest.importorskip("torch")

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
    build_current_pentest_aassr_core,
    relational_action_key,
    relational_state_key,
)
from aassr_v2.goals import GoalStateScorer
from aassr_v2.gru_prophecy import OnlineGRUProphecy
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
    # The transfer representation deliberately ignores identifier-index channels.
    # Keep only public control channels equal here so the test isolates renaming.
    vector = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0)
    return (
        StateSnapshot(
            vector=vector,
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
    )
    right, right_action = _renamed_state(
        route="route-05",
        profile="profile-02",
        object_id="object-31",
    )

    assert relational_state_key(left) == relational_state_key(right)
    assert relational_action_key(left, left_action) == relational_action_key(
        right,
        right_action,
    )

    # ASEQ/cycle detection must still distinguish concrete episode-local entities.
    assert semantic_fingerprint(left) != semantic_fingerprint(right)


def test_current_builder_instantiates_current_components_not_legacy_ones() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=256,
        use_imagination=True,
        device="cpu",
    )

    assert isinstance(agent, CurrentPentestAASSRAgent)
    assert isinstance(agent.policy, CurrentRelationalPolicy)
    assert isinstance(agent.dqn, RelationalInvariantDQN)
    assert isinstance(agent.base_neural_prophecy, CurrentNeuralDeltaProphecy)
    assert isinstance(agent.critic, RelationalGRUBranchCritic)
    assert isinstance(agent.skills, RelationalSkillLibrary)

    assert not isinstance(agent.policy, SemanticContextualPolicy)
    assert not isinstance(agent.base_neural_prophecy, OnlineGRUProphecy)
    assert not isinstance(agent.core.planner.scorer, GoalStateScorer)
    assert agent.core.config.use_effect_composition is False
    assert agent.legacy_components_active == ()

    diagnostics = agent.diagnostics()
    assert diagnostics["current_generation"] is True
    assert diagnostics["legacy_components_active"] == []
    assert diagnostics["identity_contracts"]["aseq_cycle_detection"].startswith(
        "concrete"
    )
    assert diagnostics["identity_contracts"]["policy_transfer"].startswith(
        "relational"
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

    agent.finish_episode(final_return=0.0, training=True)
    assert agent.dqn.environment_steps == 1
    assert agent.base_neural_prophecy.observations == 1
    assert agent.critic.stats().episodes == 1


def test_relational_policy_scores_renamed_actions_with_same_structural_input() -> None:
    left, left_action = _renamed_state(
        route="route-28",
        profile="profile-14",
        object_id="object-07",
    )
    right, right_action = _renamed_state(
        route="route-05",
        profile="profile-02",
        object_id="object-31",
    )
    dqn = RelationalInvariantDQN(123, train_transitions=128)

    left_input = dqn.encode_state(left) + relational_action_key(left, left_action)
    right_input = dqn.encode_state(right) + relational_action_key(right, right_action)
    assert left_input == right_input
