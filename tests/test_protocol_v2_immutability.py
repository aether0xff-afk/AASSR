from __future__ import annotations

from pathlib import Path

import pytest

from aassr_v2.v2_immutability import (
    assert_v2_output_path,
    verify_preservation_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_final_v1_preservation_manifest_matches_repository() -> None:
    issues = verify_preservation_manifest(
        ROOT / "configs" / "final_v1_preservation_manifest.json",
        repository_root=ROOT,
    )
    assert issues == []


@pytest.mark.parametrize(
    "target",
    (
        "paper_results/paper-autonomy-final-v1",
        "paper_results_v2",
        "runs/causal-v2",
        "configs/paper_autonomy_final_v1.json",
    ),
)
def test_v2_output_guard_rejects_v1_or_foreign_paths(target: str) -> None:
    with pytest.raises(ValueError):
        assert_v2_output_path(target, repository_root=ROOT)


def test_v2_output_guard_accepts_scoped_run() -> None:
    target = assert_v2_output_path(
        "paper_results_v2/development/paper-causal-diagnostic-v2/run-001",
        repository_root=ROOT,
    )
    assert target == (
        ROOT
        / "paper_results_v2"
        / "development"
        / "paper-causal-diagnostic-v2"
        / "run-001"
    ).resolve()
