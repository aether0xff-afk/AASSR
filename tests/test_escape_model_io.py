from __future__ import annotations

from pathlib import Path

import pytest

from aassr_v2.escape_gridworld import EscapeGridWorld, generate_escape_grid, oracle_plan
from aassr_v2.escape_reporting import serialize_agent_checkpoint
from aassr_v2.escape_training import EscapeTrainingConfig, _make_agent
from aassr_v2.model_io import (
    ModelCompatibilityError,
    ModelManagedAgent,
    load_agent_model,
    save_agent_model,
)


def _trained_agent(config: EscapeTrainingConfig) -> ModelManagedAgent:
    managed = ModelManagedAgent(_make_agent(config))
    environment = EscapeGridWorld(
        generate_escape_grid(
            config.seed,
            color_count=config.color_count,
            distractor_boxes=config.distractor_boxes,
        )
    )
    for action in oracle_plan(environment.spec):
        before = environment.snapshot()
        outcome = environment.step(action)
        managed.observe(before, action, outcome)
    managed.finish_episode(final_return=2.0)
    return managed


def test_portable_model_round_trip_restores_learning_state(tmp_path: Path) -> None:
    config = EscapeTrainingConfig(
        episodes=1,
        seed=7,
        color_count=1,
        distractor_boxes=1,
        use_imagination=False,
    )
    original = _trained_agent(config)
    model_path = original.save_model(
        tmp_path / "roundtrip",
        training_config=config,
        label="round trip",
    )

    restored_base = _make_agent(config)
    info = load_agent_model(
        restored_base,
        model_path,
        expected_training_config=config,
    )

    before = serialize_agent_checkpoint(original._agent, episode=1)
    after = serialize_agent_checkpoint(restored_base, episode=1)
    for key in (
        "transition_index",
        "decision_index",
        "random_state",
        "policy",
        "prophecy",
        "holdout",
    ):
        assert after[key] == before[key]
    assert info.completed_episodes == 1
    assert info.label == "round trip"
    assert model_path.name.endswith(".aassr-model.gz")


def test_loaded_episode_offset_continues_epsilon_schedule(tmp_path: Path) -> None:
    config = EscapeTrainingConfig(
        episodes=1,
        seed=7,
        color_count=1,
        distractor_boxes=1,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay_episodes=100,
        use_imagination=False,
    )
    original = _trained_agent(config)
    model_path = original.save_model(tmp_path / "offset", training_config=config)

    restored_base = _make_agent(config)
    info = load_agent_model(restored_base, model_path, expected_training_config=config)
    managed = ModelManagedAgent(restored_base, base_episode_offset=info.completed_episodes)

    assert managed.epsilon(0) == pytest.approx(restored_base.epsilon(1))
    assert managed.completed_episodes == 1


def test_model_rejects_incompatible_grid_shape(tmp_path: Path) -> None:
    source_config = EscapeTrainingConfig(
        episodes=1,
        seed=7,
        color_count=1,
        distractor_boxes=1,
        use_imagination=False,
    )
    model_path = save_agent_model(
        _make_agent(source_config),
        tmp_path / "incompatible",
        completed_episodes=0,
        training_config=source_config,
    )
    target_config = EscapeTrainingConfig(
        episodes=1,
        seed=9,
        color_count=2,
        distractor_boxes=2,
        use_imagination=False,
    )

    with pytest.raises(ModelCompatibilityError):
        load_agent_model(
            _make_agent(target_config),
            model_path,
            expected_training_config=target_config,
        )
