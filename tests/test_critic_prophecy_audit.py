from __future__ import annotations

import math

import pytest

from aassr_v2.baseline_efficiency_benchmark import (
    CHOICE_ACTIONS,
    encode_gridpush_state,
)
from aassr_v2.benchmark_neural_prophecy import BenchmarkGridPushCodec
from aassr_v2.branch_critic import GRUBranchCritic, ParentTransitionCritic
from aassr_v2.critic_prophecy_audit import AuditConfig, run_audit
from aassr_v2.critic_prophecy_common import collect_random_episodes
from aassr_v2.prophecy_one_step_audit import evaluate_one_step, train_prophecy
from aassr_v2.transition_prefix_prophecy import (
    TransitionPrefixConfig,
    TransitionPrefixProphecy,
)


def test_parent_and_gru_critics_keep_branch_memory_separate() -> None:
    pytest.importorskip("torch")
    episodes = collect_random_episodes((10001, 10002), episodes=12, seed=3)
    parent = ParentTransitionCritic(
        encode_gridpush_state,
        25,
        batch_size=2,
        gradient_steps_per_episode=1,
        seed=3,
    )
    gru = GRUBranchCritic(
        encode_gridpush_state,
        25,
        batch_size=2,
        gradient_steps_per_episode=1,
        seed=3,
    )
    for episode in episodes:
        parent.observe_episode(episode.transitions, success=episode.success)
        gru.observe_episode(episode.transitions, success=episode.success)

    transition = episodes[0].transitions[0]
    parent_step = parent.score_step(
        transition.before,
        transition.action,
        transition.after,
    )
    root_memory = gru.initial_memory()
    left = gru.score_step(
        transition.before,
        transition.action,
        transition.after,
        memory=root_memory,
    )
    right = gru.score_step(
        transition.before,
        CHOICE_ACTIONS[-1],
        transition.after,
        memory=root_memory,
    )
    assert 0.0 <= parent_step.value <= 1.0
    assert 0.0 <= left.value <= 1.0
    assert 0.0 <= right.value <= 1.0
    assert left.memory is not root_memory
    assert right.memory is not root_memory
    assert left.memory.data_ptr() != right.memory.data_ptr()


def test_transition_prefix_prophecy_trains_only_from_real_transition_tuple() -> None:
    pytest.importorskip("torch")
    codec = BenchmarkGridPushCodec()
    model = TransitionPrefixProphecy(
        codec,
        config=TransitionPrefixConfig(
            model_dim=16,
            attention_heads=2,
            layers=1,
            feedforward_dim=32,
            replay_capacity=32,
            batch_size=2,
            warmup_steps=2,
            gradient_steps_per_observation=1,
        ),
        seed=5,
    )
    episodes = collect_random_episodes((10001, 10002), episodes=8, seed=5)
    train_prophecy(model, episodes, epochs=1)
    result = evaluate_one_step(model, episodes, codec)
    assert result["count"] > 0
    assert math.isfinite(result["vector_mae"])
    assert 0.0 <= result["terminal_accuracy"] <= 1.0
    assert model.stats().gradient_updates > 0


def test_tiny_end_to_end_audit_writes_separate_sections(tmp_path) -> None:
    pytest.importorskip("torch")
    payload = run_audit(
        tmp_path,
        AuditConfig(
            seed=2,
            train_map_count=2,
            unseen_map_count=2,
            behavior_train_episodes=8,
            critic_train_episodes=8,
            critic_eval_episodes=4,
            prophecy_train_episodes=12,
            prophecy_eval_episodes=4,
            prophecy_epochs=1,
            pruning_depth=2,
            pruning_beam_width=2,
        ),
    )
    assert (tmp_path / "summary.json").exists()
    assert set(payload["critic"]["pruning"]) == {"hand_scorer", "parent", "gru"}
    models = payload["prophecy_one_step"]["models"]
    assert {"legacy_gru", "neural_delta", "transition_prefix"} <= set(models)
    assert payload["identity_constraints"]["imagination_used_in_prophecy_audit"] is False
