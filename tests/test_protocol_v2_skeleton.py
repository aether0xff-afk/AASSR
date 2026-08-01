from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from aassr_v2.paper_types import ExperimentPhase
from aassr_v2.paper_v2_protocol import (
    V2ArtifactWriter,
    checkpoint_fingerprint,
    clone_agent_from_checkpoint,
    create_protocol_lock,
    replay_gzip_trace,
    reserve_confirmation_once,
    reserve_run,
    seed_commitment,
    validate_v2_config,
    v2_run_directory,
)
from aassr_v2.paper_v2_types import (
    FullAgentCheckpoint,
    V2RunIdentity,
    V2StudyStage,
)


def base_config(stage: str = "development_diagnostic") -> dict:
    phases = [
        "training",
        "evaluation_train_world_frozen",
        "evaluation_isomorphic_world_zero_shot",
        "evaluation_unseen_composition_zero_shot",
        "adaptation",
        "evaluation_unseen_after_adaptation",
    ]
    return {
        "protocol_version": "paper-causal-diagnostic-v2.0",
        "study_stage": stage,
        "phases": phases,
        "phase_learning": {phase: phase in {"training", "adaptation"} for phase in phases},
        "research_seeds": [2003, 2011, 2027],
        "excluded_research_seeds": [131, 173],
        "world_seeds": {
            "train": [82001, 82002],
            "isomorphic": [83001, 83002],
            "unseen_composition": [84001, 84002],
            "adaptation": [85001, 85002],
        },
        "output_root": "paper_results_v2",
        "reward_mode": "strict_sparse",
        "causal_law_sha256": "a" * 64,
    }


def test_v2_phase_names_are_additive() -> None:
    assert ExperimentPhase.EVALUATION_SEEN.value == "evaluation_seen"
    assert (
        ExperimentPhase.EVALUATION_ISOMORPHIC_WORLD_ZERO_SHOT.value
        == "evaluation_isomorphic_world_zero_shot"
    )


def test_seed_and_world_sets_are_strictly_separated() -> None:
    config = base_config()
    assert validate_v2_config(config)["seed_commitment_sha256"] == seed_commitment(
        config["research_seeds"]
    )
    config["world_seeds"]["isomorphic"][0] = 82001
    with pytest.raises(ValueError, match="world seed overlap"):
        validate_v2_config(config)


class CheckpointAgent:
    def __init__(self) -> None:
        self.value = 0
        self.rng = random.Random(7)

    def export_full_checkpoint(self) -> FullAgentCheckpoint:
        return FullAgentCheckpoint(
            policy={"value": self.value},
            rng=repr(self.rng.getstate()),
            planner_cache={"root": 1},
            counters={"decision": 3},
            replay_buffer=({"reward": 1.0},),
            normalization_state={"mean": 0.2},
            calibration_buffer=({"p": 0.8, "y": 1},),
            relational_representation={"motif": [1, 2]},
        )

    def import_full_checkpoint(self, checkpoint: FullAgentCheckpoint) -> None:
        self.value = int(checkpoint.policy["value"])
        self._checkpoint = checkpoint

    def export_full_checkpoint(self) -> FullAgentCheckpoint:  # type: ignore[no-redef]
        if hasattr(self, "_checkpoint"):
            return self._checkpoint
        return FullAgentCheckpoint(policy={"value": self.value}, rng="initial")


def test_frozen_evaluation_uses_a_checkpoint_clone() -> None:
    source = CheckpointAgent()
    source.value = 9
    clone, fingerprint = clone_agent_from_checkpoint(source, CheckpointAgent)
    assert clone is not source
    assert clone.value == 9
    assert checkpoint_fingerprint(clone.export_full_checkpoint()) == fingerprint


def test_gzip_trace_round_trip_and_corruption(tmp_path: Path) -> None:
    with V2ArtifactWriter(tmp_path / "run", ["seed", "success"]) as writer:
        writer.write_episode({"seed": 1, "success": 1})
        writer.write_trace({"seed": 1, "private": False})
    assert replay_gzip_trace(tmp_path / "run" / "trace.jsonl.gz") == [
        {"seed": 1, "private": False}
    ]
    (tmp_path / "broken.gz").write_bytes(b"not-gzip")
    with pytest.raises(OSError):
        replay_gzip_trace(tmp_path / "broken.gz")


def test_confirmation_is_one_shot_but_exact_resume_is_allowed(tmp_path: Path) -> None:
    identity = V2RunIdentity(
        "paper-causal-confirmation-v2.0",
        V2StudyStage.LOCKED_CONFIRMATION,
        "first",
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "commit",
    )
    reserve_confirmation_once(tmp_path, identity)
    with pytest.raises(FileExistsError):
        reserve_confirmation_once(tmp_path, identity)
    assert reserve_confirmation_once(tmp_path, identity, resume=True)


def test_run_claim_rejects_identity_change(tmp_path: Path) -> None:
    identity = V2RunIdentity(
        "paper-causal-diagnostic-v2.0",
        V2StudyStage.DEVELOPMENT_DIAGNOSTIC,
        "run-1",
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "commit",
    )
    directory = v2_run_directory(identity, repository_root=tmp_path)
    reserve_run(directory, identity)
    with pytest.raises(FileExistsError):
        reserve_run(directory, identity)
    assert reserve_run(directory, identity, resume=True).is_file()


def test_protocol_lock_commits_seed_and_thresholds() -> None:
    config = base_config("locked_confirmation")
    config["seed_commitment_sha256"] = seed_commitment(config["research_seeds"])
    config["thresholds"] = {"calibration": 0.8}
    lock = create_protocol_lock(config)
    assert lock["status"] == "locked"
    assert lock["seed_commitment_sha256"] == config["seed_commitment_sha256"]


def test_representation_comparison_requires_equal_capacity() -> None:
    config = base_config()
    config["representation_comparison"] = {
        "raw_observation_schema": "raw-v2",
        "identity_model_capacity": 10,
        "relational_model_capacity": 11,
        "identity_update_budget": 1,
        "relational_update_budget": 1,
    }
    with pytest.raises(ValueError, match="model capacity"):
        validate_v2_config(config)
