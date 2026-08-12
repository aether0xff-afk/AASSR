from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_checkpoint import (
    restore_current_frozen_checkpoint,
    save_current_frozen_checkpoint,
)
from aassr_v2.current_critic_support import _action_key
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_generation import relational_action_key
from aassr_v2.current_relational_state_v3 import (
    latest_status_code,
    relational_state_descriptor_v3,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld
from aassr_v2.replay import ReplayTransition
from aassr_v2.skills import Skill


def test_current_frozen_checkpoint_restores_learned_and_gate_state(tmp_path) -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=128,
        use_imagination=True,
        device="cpu",
        allow_tf32=False,
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    action = state.available_actions[0]

    with agent.dqn.torch.no_grad():
        next(agent.dqn.online.parameters()).add_(0.123)
        next(agent.base_neural_prophecy.models[0].parameters()).add_(0.234)
    agent.dqn.target.load_state_dict(agent.dqn.online.state_dict())
    agent.dqn.environment_steps = 17
    agent.dqn.gradient_updates = 3
    agent.base_neural_prophecy.observations = 19
    agent.base_neural_prophecy.gradient_updates = 5
    agent._critic_counts.update(
        {
            "episodes": 32,
            "successes": 4,
            "non_successes": 4,
            "positive_returns": 4,
            "negative_returns": 4,
            "zero_returns": 24,
            # Frozen-evaluation checkpoints intentionally persist compact recent
            # readiness counts instead of the full Critic training replay.
            "recent_return_window": 128,
            "recent_positive_returns": 4,
            "recent_negative_returns": 4,
            "recent_zero_returns": 0,
        }
    )
    agent.critic.episodes = 32
    agent.critic.transitions = 64
    agent.critic.gradient_updates = 2
    key = _action_key(state, action)
    agent._critic_support_rows[key].append(
        (
            tuple(relational_state_descriptor_v3(state)),
            latest_status_code(state),
        )
    )

    # Reproduce the exact object-rich state from a real run. Action.parameters
    # and StateSnapshot.metadata are MappingProxyType instances, so a raw pickle
    # of ReplayTransition would fail even though an empty-checkpoint test passes.
    for index in range(5):
        agent.evaluator.replay.add(
            ReplayTransition(state, action, state, f"portable-{index}")
        )
    assert len(agent.evaluator.replay.train()) == 4
    assert len(agent.evaluator.replay.holdout()) == 1

    skill = Skill(
        skill_id="skill-0001",
        primitive_actions=(action,),
        achieved_goal_ids=("external:success",),
        required_facts=frozenset(),
        added_facts=frozenset(),
        removed_facts=frozenset(),
        successes=2,
        failures=0,
    )
    agent.skills._skills[skill.skill_id] = skill
    agent.skills._templates[skill.skill_id] = (relational_action_key(state, action),)
    agent.skills._next_id = 2

    before_q = agent.dqn.score_actions(state, (action,))[0]
    before_prophecy = tuple(
        parameter.detach().clone()
        for parameter in agent.base_neural_prophecy.models[0].parameters()
    )

    path = tmp_path / "aassr_frozen.pt"
    save_current_frozen_checkpoint(
        agent,
        path,
        research_seed=7,
        transition_budget=128,
        git_commit="checkpoint-test-sha",
    )
    restored = restore_current_frozen_checkpoint(
        path,
        device="cpu",
        allow_tf32=False,
        expected_git_commit="checkpoint-test-sha",
    )

    assert restored.dqn.score_actions(state, (action,))[0] == pytest.approx(
        before_q, abs=1e-7, rel=1e-7
    )
    for left, right in zip(
        before_prophecy,
        restored.base_neural_prophecy.models[0].parameters(),
        strict=True,
    ):
        assert left.equal(right.detach())
    assert restored.dqn.environment_steps == 17
    assert restored.dqn.gradient_updates == 3
    assert restored.base_neural_prophecy.observations == 19
    assert restored.base_neural_prophecy.gradient_updates == 5
    assert restored._critic_counts["non_successes"] == 4
    assert restored._critic_counts["recent_negative_returns"] == 4
    assert restored.critic.support_confidence(state, action) > 0.0
    assert restored.critic_ready is True
    assert restored.critic_reliably_ready() is True
    # Frozen prediction/calibration uses holdout only. The training partition is
    # recorded by count in the manifest, not duplicated into the checkpoint.
    assert len(restored.evaluator.replay.train()) == 0
    assert len(restored.evaluator.replay.holdout()) == 1
    assert restored.skills.get("skill-0001").primitive_actions[0].signature == action.signature
    assert restored.skills.template_length("skill-0001") == 1
    assert restored.config.imagination_intervention_margin == pytest.approx(0.05)
