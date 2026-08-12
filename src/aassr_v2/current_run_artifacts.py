from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .current_checkpoint import (
    checkpoint_manifest,
    current_frozen_checkpoint_payload,
)
from .current_entrypoint import CURRENT_INTERVENTION_MARGIN
from .current_manifest import CURRENT_COMPONENTS, CURRENT_GENERATION_VERSION
from .current_protocol import CurrentEpisodeRow, write_current_csv


RUN_MANIFEST_VERSION = "aassr-current-run-manifest-v1"


def resolve_git_commit() -> str:
    """Return the exact source revision without requiring GitHub/network access."""
    explicit = os.environ.get("GITHUB_SHA") or os.environ.get("AASSR_GIT_COMMIT")
    if explicit:
        return str(explicit).strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip()
        return value or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_run_manifest(
    output: Path,
    *,
    research_seed: int,
    transition_budget: int,
    block_target: int,
    device: str,
    allow_tf32: bool,
    git_commit: str,
) -> Path:
    path = output / "run_manifest.json"
    payload = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "architecture_version": CURRENT_GENERATION_VERSION,
        "git_commit": str(git_commit),
        "research_seed": int(research_seed),
        "transition_budget_per_training_condition": int(transition_budget),
        "block_target": int(block_target),
        "device": str(device),
        "allow_tf32": bool(allow_tf32),
        "imagination_intervention_margin": CURRENT_INTERVENTION_MARGIN,
        "exploration_scaling_contract": "budget-normalized-explicitly-reported",
        "current_components": dict(CURRENT_COMPONENTS),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_aassr_training_artifacts(
    output: Path,
    *,
    training_rows: Sequence[CurrentEpisodeRow],
    validation_rows: Sequence[CurrentEpisodeRow],
    curriculum_trace: Sequence[dict[str, Any]],
    partial: bool,
) -> None:
    suffix = ".partial" if partial else ""
    write_current_csv(output / f"training_aassr{suffix}.csv", training_rows)
    write_current_csv(
        output / f"curriculum_validation_aassr{suffix}.csv",
        validation_rows,
    )
    (output / f"curriculum_trace_aassr{suffix}.json").write_text(
        json.dumps(list(curriculum_trace), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_training_checkpoint(
    agent: object,
    output: Path,
    *,
    research_seed: int,
    transition_budget: int,
    git_commit: str,
) -> tuple[Path, Path]:
    """Save before diagnostics so evaluation failures never force retraining."""
    checkpoint_path = output / "aassr_frozen_training_checkpoint.pt"
    payload = current_frozen_checkpoint_payload(
        agent,
        research_seed=research_seed,
        transition_budget=transition_budget,
        git_commit=git_commit,
    )
    agent.dqn.torch.save(payload, checkpoint_path)
    manifest_path = output / "aassr_frozen_training_checkpoint.json"
    manifest_path.write_text(
        json.dumps(checkpoint_manifest(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint_path, manifest_path
