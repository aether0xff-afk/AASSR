from __future__ import annotations

import json

import pytest

from aassr_v2.creativity import (
    MultiSolutionDependencyWorld,
    canonicalize_effect_trace,
    strategy_distance_components,
    strategy_record_from_trace,
)
from aassr_v2.human_study import (
    HumanStudyService,
    HumanStudyStore,
    inter_rater_agreement,
)
from aassr_v2.safe_application import (
    SafeLocalApplicationWorld,
    validate_compose_safety,
)
from aassr_v2.paper_runner import run_paper_suite
from aassr_v2.types import Action


def _execute(world: MultiSolutionDependencyWorld, keys: list[str]):
    events = []
    for key in keys:
        outcome = world.step(world._operations[key].action)
        events.extend(outcome.effect_events)
    return events


def test_multi_solution_world_has_five_hidden_solution_families() -> None:
    routes = {
        "information_route": ["inspect", "decode"],
        "resource_route": ["gather", "unlock"],
        "bypass_route": ["reroute", "traverse"],
        "tool_route": ["assemble", "remove"],
        "emergent_combination": ["inspect", "gather", "synthesize"],
    }
    observed = set()
    for index, (family, route) in enumerate(routes.items()):
        world = MultiSolutionDependencyWorld(seed=10 + index)
        visible = json.dumps(
            {
                "facts": sorted(world.snapshot().facts),
                "actions": [
                    item.signature for item in world.snapshot().available_actions
                ],
            }
        )
        assert family not in visible
        _execute(world, route)
        assert world.terminal
        observed.add(world.analysis_solution_family)
    assert observed == set(routes)
    assert MultiSolutionDependencyWorld.FAMILY_COUNT >= 4


def test_effect_graph_is_action_name_independent() -> None:
    first = canonicalize_effect_trace(
        [
            {"effect": "information_acquisition"},
            {
                "effect": "goal_achievement",
                "prerequisites": ["information_acquisition"],
                "relation": "parameter_dependency",
            },
        ]
    )
    second = canonicalize_effect_trace(
        [
            {"effect": "information_acquisition", "action": "renamed-A"},
            {
                "effect": "goal_achievement",
                "prerequisites": ["information_acquisition"],
                "relation": "parameter_dependency",
                "action": "renamed-B",
            },
        ]
    )
    assert first == second
    assert all(
        value == 0.0
        for value in strategy_distance_components(first, second).values()
    )


def _record(identifier: str, source: str, seed: int):
    return strategy_record_from_trace(
        strategy_id=identifier,
        source_kind=source,
        research_seed=seed,
        world_seed=seed + 100,
        success=True,
        primitive_steps=2,
        errors=0,
        resources_used=1.0,
        risk_entries=0,
        events=[
            {"effect": "information_acquisition"},
            {
                "effect": "goal_achievement",
                "prerequisites": ["information_acquisition"],
            },
        ],
        solution_family="information_route",
    )


def test_human_store_is_anonymous_blind_and_duplicate_safe(tmp_path) -> None:
    store = HumanStudyStore(
        tmp_path / "human.sqlite3", dataset_version="pilot-v1"
    )
    owner = store.create_participant()
    evaluator = store.create_participant()
    second_evaluator = store.create_participant()
    store.add_strategy(_record("full_aassr_secret", "aassr", 1), participant_id=owner)
    assignment = store.next_assignment(evaluator)
    assert assignment is not None
    assert "strategy_id" not in assignment
    assert "full_aassr" not in json.dumps(assignment)
    assert "information_route" not in json.dumps(assignment)
    assert "solution_family" not in assignment["graph"]
    blind_id = assignment["blind_id"]
    scores = {
        "novelty": 4,
        "utility": 5,
        "coherence": 4,
        "surprise": 3,
    }
    store.add_rating(evaluator, blind_id, scores)
    with pytest.raises(ValueError, match="duplicate"):
        store.add_rating(evaluator, blind_id, scores)
    second_assignment = store.next_assignment(second_evaluator)
    assert second_assignment is not None
    store.add_rating(second_evaluator, second_assignment["blind_id"], scores)
    agreement = inter_rater_agreement(store.ratings())
    assert agreement["evaluator_count"] == 2
    exported = store.export(tmp_path / "export")
    assert all(path.exists() for path in exported)
    assert json.loads(exported[2].read_text(encoding="utf-8"))[
        "contains_direct_identifiers"
    ] is False


def test_human_live_world_records_complete_path(tmp_path) -> None:
    store = HumanStudyStore(
        tmp_path / "human.sqlite3", dataset_version="pilot-v1"
    )
    participant = store.create_participant()
    service = HumanStudyService(store)
    state = service.create_world(participant, seed=7)
    inspect = next(
        item
        for item in state["actions"]
        if item["description"] == "inspect one observable feature"
    )
    state = service.step_world(state["session_id"], inspect["action"])
    decode = next(
        item
        for item in state["actions"]
        if item["description"] == "apply collected information"
    )
    state = service.step_world(state["session_id"], decode["action"])
    assert state["completed"]
    assert state["strategy_id"].startswith("human_")
    exported = store.export(tmp_path / "export")
    record = json.loads(
        exported[0].read_text(encoding="utf-8").splitlines()[0]
    )
    assert len(record["trace"]) == 2
    assert record["novelty_components"]


def test_human_world_resumes_after_service_restart(tmp_path) -> None:
    store = HumanStudyStore(
        tmp_path / "human.sqlite3", dataset_version="pilot-v1"
    )
    participant = store.create_participant()
    first_service = HumanStudyService(store)
    state = first_service.create_world(participant, seed=19)
    inspect = next(
        item
        for item in state["actions"]
        if item["description"] == "inspect one observable feature"
    )
    state = first_service.step_world(
        state["session_id"],
        inspect["action"],
        participant_id=participant,
    )
    second_service = HumanStudyService(store)
    resumed = second_service.resume_world(
        state["session_id"], participant_id=participant
    )
    decode = next(
        item
        for item in resumed["actions"]
        if item["description"] == "apply collected information"
    )
    completed = second_service.step_world(
        resumed["session_id"],
        decode["action"],
        participant_id=participant,
    )
    assert completed["completed"]


def test_safe_application_rejects_external_targets_and_compose_is_isolated() -> None:
    world = SafeLocalApplicationWorld(seed=2, allowed_hosts=("paper-target",))
    outcome = world.step(
        Action(
            "request_local_flag",
            parameters={"target": "example.com", "token": "x"},
        )
    )
    assert outcome.error_code == "target_not_allowlisted"
    validate_compose_safety(
        "docker/paper_safe_application/docker-compose.yml"
    )
    first = SafeLocalApplicationWorld(
        seed=2, allowed_hosts=("paper-target",)
    )
    second = SafeLocalApplicationWorld(
        seed=3, allowed_hosts=("paper-target",)
    )
    assert (
        first.target_port,
        first.service_name,
        first.route_paths,
    ) != (
        second.target_port,
        second.service_name,
        second.route_paths,
    )


def test_approved_human_dataset_merges_into_paper_artifacts(tmp_path) -> None:
    store = HumanStudyStore(
        tmp_path / "study.sqlite3",
        dataset_version="approved-v1",
        approval_id="approved-protocol-1",
    )
    owner = store.create_participant()
    first = store.create_participant()
    second = store.create_participant()
    store.add_strategy(_record("human-reference", "human", 1), participant_id=owner)
    first_assignment = store.next_assignment(first)
    second_assignment = store.next_assignment(second)
    assert first_assignment and second_assignment
    scores = {
        "novelty": 4,
        "utility": 4,
        "coherence": 4,
        "surprise": 4,
    }
    store.add_rating(first, first_assignment["blind_id"], scores)
    store.add_rating(second, second_assignment["blind_id"], scores)
    dataset_dir = tmp_path / "human-export"
    store.export(dataset_dir)
    config = {
        "schema_version": 1,
        "name": "human_merge_smoke",
        "runner": "paper_suite",
        "protocol_version": "human-merge-smoke-v1",
        "study_stage": "pilot",
        "research_seeds": [1, 2, 3, 4, 5],
        "world_seeds": {
            "train": [101],
            "seen": [151],
            "unseen": [201],
        },
        "budgets": {
            "train_episodes": 1,
            "eval_episodes": 1,
            "real_transitions_per_episode": 8,
            "adaptation_episodes": [0, 1, 4, 16, 64],
        },
        "phases": [
            "training",
            "evaluation_seen",
            "evaluation_unseen_zero_shot",
            "adaptation",
            "evaluation_unseen_adaptation",
        ],
        "phase_learning": {
            "training": True,
            "evaluation_seen": False,
            "evaluation_unseen_zero_shot": False,
            "adaptation": True,
            "evaluation_unseen_adaptation": False,
        },
        "execution": {
            "workers": 1,
            "cuda_workers": 1,
            "device": "cpu",
        },
        "human_study": {
            "merge_enabled": True,
            "approval_id": "approved-protocol-1",
            "dataset_version": "approved-v1",
            "dataset_dir": str(dataset_dir),
            "minimum_raters": 2,
        },
        "safe_application": {
            "opt_in": True,
            "internal_network": True,
            "allowed_hosts": ["paper-target"],
            "compose_file": "docker/paper_safe_application/docker-compose.yml",
        },
        "suites": [{"kind": "safe_application", "episodes": 1}],
    }
    artifacts = run_paper_suite(
        config, output_dir=tmp_path / "paper", overwrite=True
    )
    assert (artifacts.output_dir / "raw" / "human_paths.jsonl").exists()
    analysis = json.loads(
        (
            artifacts.output_dir
            / "statistics"
            / "analysis_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert analysis["human_agreement"]["evaluator_count"] == 2
