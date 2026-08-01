from __future__ import annotations

import json

import pytest

from aassr_v2.causal_dependency_world import CausalDependencyWorldV2
from aassr_v2.open_creativity_v2 import (
    creativity_environment_adequacy,
    freeze_baseline_reference,
    graph_from_analysis_path,
    load_frozen_reference,
)


def _route(world: CausalDependencyWorldV2, keys: list[str]):
    path = []
    for key in keys:
        action = next(
            item
            for item in world.observe().available_actions
            if world.private_action_key(item) == key
        )
        path.append(action)
        world.step(action)
    return graph_from_analysis_path(world, path)


def test_action_token_rename_does_not_create_novel_graph() -> None:
    first = CausalDependencyWorldV2(world_seed=1, token_seed=11)
    second = CausalDependencyWorldV2(world_seed=1, token_seed=12)
    keys = ["stabilize", "safe_traverse", "claim_goal"]
    assert _route(first, keys) == _route(second, keys)


def test_structurally_different_routes_have_different_graphs() -> None:
    first = CausalDependencyWorldV2(world_seed=1)
    second = CausalDependencyWorldV2(world_seed=1)
    safe = _route(first, ["stabilize", "safe_traverse", "claim_goal"])
    information = _route(second, ["scan", "bind", "open_gate", "claim_goal"])
    assert safe != information


def test_frozen_reference_detects_tampering(tmp_path) -> None:
    path = tmp_path / "reference.json"
    freeze_baseline_reference(path, world_seeds=[86001, 86002], interaction_budget=40)
    references = load_frozen_reference(path)
    assert isinstance(references, tuple)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["interaction_budget"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_frozen_reference(path)


def test_enumerator_certifies_open_space_outside_small_reference() -> None:
    world = CausalDependencyWorldV2(
        world_seed=86001, composition_template="open_creativity_v1"
    )
    reference = [_route(world, ["stabilize", "scan", "safe_traverse", "claim_goal"])]
    adequacy = creativity_environment_adequacy(
        world_seed=86001, references=reference
    )
    assert adequacy["outside_reference_count"] >= 1
    assert adequacy["causal_family_count"] >= 3
    assert adequacy["adequate"]
