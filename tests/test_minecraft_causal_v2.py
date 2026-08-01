from __future__ import annotations

import json
from pathlib import Path

from aassr_v2.minecraft_causal_world import (
    MINECRAFT_CAUSAL_LAW_SHA256,
    MinecraftAdapter,
    MinecraftCausalWorld,
    MinecraftSkillTrack,
    MockMinecraftAdapter,
    certify_minecraft_world,
)
from aassr_v2.paper_v2_protocol import validate_v2_config


def test_semantic_and_opaque_tracks_share_law_but_not_action_tokens() -> None:
    semantic = MinecraftCausalWorld(world_seed=92001, track="semantic_skill")
    opaque = MinecraftCausalWorld(world_seed=92001, track="opaque_skill")
    assert semantic.causal_law_sha256 == opaque.causal_law_sha256 == MINECRAFT_CAUSAL_LAW_SHA256
    assert semantic.action_token_sha256 != opaque.action_token_sha256
    assert any(action.startswith("COLLECT(") for action in semantic.observe().available_actions)
    assert all(action.startswith("skill_") for action in opaque.observe().available_actions)


def test_opaque_affordances_are_explicitly_separate_ablation() -> None:
    core = MinecraftCausalWorld(world_seed=92001, track="opaque_skill")
    ablation = MinecraftCausalWorld(
        world_seed=92001,
        track="opaque_skill",
        expose_opaque_affordances=True,
    )
    assert not core.observe().action_affordances
    assert ablation.observe().action_affordances
    assert core.observe().available_actions == ablation.observe().available_actions


def test_minecraft_raw_observation_and_step_hide_private_labels() -> None:
    world = MinecraftCausalWorld(world_seed=92001, track="opaque_skill")
    payload = repr(world.observe().to_dict()).lower()
    step_payload = repr(world.step(world.observe().available_actions[0])).lower()
    for forbidden in (
        "solution_family", "viability", "latent_risk", "optimal_plan",
        "true_graph", "oracle", "goal_progress", "effect_for_analysis",
    ):
        assert forbidden not in payload
        assert forbidden not in step_payload


def test_minecraft_solver_certifies_both_tracks() -> None:
    results = [
        certify_minecraft_world(
            MinecraftCausalWorld(world_seed=92001, track=track),
            random_rollouts=400,
        )
        for track in MinecraftSkillTrack
    ]
    assert all(result.adequate for result in results)
    assert all(result.minimum_plan_length == 4 for result in results)
    assert all(result.causal_family_count >= 3 for result in results)
    assert all(result.random_policy_success_estimate <= 0.10 for result in results)


def test_mock_adapter_executes_semantic_bridge_route() -> None:
    adapter = MockMinecraftAdapter(track=MinecraftSkillTrack.SEMANTIC)
    assert isinstance(adapter, MinecraftAdapter)
    adapter.reset(seed=92001)
    reward = 0.0
    for action in (
        "COLLECT(log)", "CRAFT(planks)", "PLACE(bridge)", "USE(goal)",
    ):
        result = adapter.step(action)
        reward = result.reward
    assert reward == 1.0
    assert adapter.observe().terminal
    adapter.close()


def test_tracks_are_declared_as_distinct_claim_spaces() -> None:
    assert MinecraftSkillTrack.SEMANTIC.value != MinecraftSkillTrack.OPAQUE.value


def test_development_configs_keep_tracks_separate() -> None:
    tracks = set()
    for name in (
        "paper_minecraft_semantic_diagnostic_v2.json",
        "paper_minecraft_opaque_diagnostic_v2.json",
    ):
        config = json.loads((Path("configs") / name).read_text(encoding="utf-8"))
        resolved = validate_v2_config(config)
        tracks.add(resolved["minecraft"]["track"])
        assert resolved["minecraft"]["real_runtime_enabled"] is False
    assert tracks == {"semantic_skill", "opaque_skill"}
