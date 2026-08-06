from __future__ import annotations

import csv

import pytest

from aassr_v2.toolgrid_matched_factorial import run_toolgrid_matched_hybrid


PAIR_KEYS = (
    "seed",
    "grid_size",
    "action_count",
    "checkpoint_transition_target",
    "phase",
    "episode",
    "map_seed",
)
SHARED_CHECKPOINT_FIELDS = (
    "actual_training_transitions",
    "training_episode_count",
    "training_wall_seconds",
    "model_units",
    "model_bytes",
    "gradient_updates",
)


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_matched_hybrid_uses_one_training_stream_and_one_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    payload = run_toolgrid_matched_hybrid(
        tmp_path,
        seed=7,
        grid_size=3,
        action_count=8,
        transition_budget=64,
        train_map_count=4,
        evaluation_map_count=3,
        checkpoints=(0, 64),
    )

    assert payload["config"]["shared_checkpoint"] is True
    assert payload["config"]["single_training_stream"] is True
    assert payload["final"]["actual_training_transitions"] == 64

    training = _read(tmp_path / "training_episodes.csv")
    policy_training = [
        row for row in training if row["condition"] == "neural_policy_only"
    ]
    imagination_training = [
        row for row in training if row["condition"] == "imagination_v2"
    ]
    assert len(policy_training) == len(imagination_training) > 0
    assert sum(int(row["imagination_runs"]) for row in imagination_training) == 0
    for left, right in zip(policy_training, imagination_training, strict=True):
        assert {
            key: value for key, value in left.items() if key != "condition"
        } == {
            key: value for key, value in right.items() if key != "condition"
        }

    evaluation = _read(tmp_path / "evaluation_episodes.csv")
    policy_keys = sorted(
        tuple(row[key] for key in PAIR_KEYS)
        for row in evaluation
        if row["condition"] == "neural_policy_only"
    )
    imagination_keys = sorted(
        tuple(row[key] for key in PAIR_KEYS)
        for row in evaluation
        if row["condition"] == "imagination_v2"
    )
    assert policy_keys == imagination_keys

    checkpoints = _read(tmp_path / "checkpoints.csv")
    policy = {
        tuple(row[key] for key in PAIR_KEYS[:4]): row
        for row in checkpoints
        if row["condition"] == "neural_policy_only"
    }
    imagined = {
        tuple(row[key] for key in PAIR_KEYS[:4]): row
        for row in checkpoints
        if row["condition"] == "imagination_v2"
    }
    assert policy.keys() == imagined.keys()
    for key in policy:
        for field in SHARED_CHECKPOINT_FIELDS:
            assert policy[key][field] == imagined[key][field]
