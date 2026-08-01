from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .paper_types import ExperimentPhase
from .paper_v2_types import FullAgentCheckpoint, V2RunIdentity, V2StudyStage
from .v2_immutability import assert_v2_output_path


V2_PHASES = (
    ExperimentPhase.TRAINING.value,
    ExperimentPhase.EVALUATION_TRAIN_WORLD_FROZEN.value,
    ExperimentPhase.EVALUATION_ISOMORPHIC_WORLD_ZERO_SHOT.value,
    ExperimentPhase.EVALUATION_UNSEEN_COMPOSITION_ZERO_SHOT.value,
    ExperimentPhase.ADAPTATION.value,
    ExperimentPhase.EVALUATION_UNSEEN_AFTER_ADAPTATION.value,
)
WORLD_SET_NAMES = (
    "train",
    "isomorphic",
    "unseen_composition",
    "adaptation",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seed_commitment(seeds: Sequence[int]) -> str:
    values = [int(seed) for seed in seeds]
    if len(values) != len(set(values)):
        raise ValueError("research seeds must be unique")
    return sha256_json(values)


def current_git_commit(repository_root: str | Path | None = None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def implementation_tree_sha256(
    repository_root: str | Path | None = None,
) -> str:
    root = Path(repository_root or Path.cwd()).resolve()
    sources = list((root / "src" / "aassr_v2").rglob("*.py"))
    sources.extend(
        path
        for path in (
            root / "scripts" / "run_paper_suite_v2.py",
            root / "scripts" / "lock_paper_v2_protocol.py",
            root / "scripts" / "run_minecraft_causal_suite.py",
            root / "scripts" / "freeze_creativity_reference_v2.py",
        )
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_v2_config(
    config: Mapping[str, Any], *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    protocol = str(resolved.get("protocol_version", ""))
    if "v2" not in protocol.lower() or "final-v1" in protocol.lower():
        raise ValueError("protocol_version must identify Protocol v2")
    try:
        stage = V2StudyStage(str(resolved["study_stage"]))
    except (KeyError, ValueError) as error:
        raise ValueError("invalid Protocol v2 study_stage") from error
    phases = tuple(str(item) for item in resolved.get("phases", ()))
    if phases != V2_PHASES:
        raise ValueError(f"phases must exactly match {V2_PHASES}")
    learning = resolved.get("phase_learning")
    expected_learning = {
        phase: phase in {
            ExperimentPhase.TRAINING.value,
            ExperimentPhase.ADAPTATION.value,
        }
        for phase in V2_PHASES
    }
    if learning != expected_learning:
        raise ValueError("phase_learning does not match Protocol v2 semantics")
    seeds = [int(item) for item in resolved.get("research_seeds", ())]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("research_seeds must be a non-empty unique list")
    excluded = {int(item) for item in resolved.get("excluded_research_seeds", ())}
    if set(seeds) & excluded:
        raise ValueError("research seeds overlap a prior stage")
    declared_commitment = str(resolved.get("seed_commitment_sha256", ""))
    actual_commitment = seed_commitment(seeds)
    if stage is not V2StudyStage.DEVELOPMENT_DIAGNOSTIC:
        if declared_commitment != actual_commitment:
            raise ValueError("seed commitment does not match research_seeds")
    elif declared_commitment and declared_commitment != actual_commitment:
        raise ValueError("development seed commitment is incorrect")
    worlds = resolved.get("world_seeds")
    if not isinstance(worlds, Mapping):
        raise ValueError("world_seeds must be an object")
    materialized: dict[str, list[int]] = {}
    for name in WORLD_SET_NAMES:
        values = [int(item) for item in worlds.get(name, ())]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"world_seeds.{name} must be non-empty and unique")
        materialized[name] = values
    for index, left in enumerate(WORLD_SET_NAMES):
        for right in WORLD_SET_NAMES[index + 1 :]:
            if set(materialized[left]) & set(materialized[right]):
                raise ValueError(f"world seed overlap: {left} and {right}")
    output = str(resolved.get("output_root", "paper_results_v2"))
    root = Path(repository_root or Path.cwd()).resolve()
    if Path(output).name != "paper_results_v2":
        raise ValueError("Protocol v2 output_root must be paper_results_v2")
    if str(resolved.get("reward_mode", "strict_sparse")) not in {
        "strict_sparse",
        "observable_progress",
    }:
        raise ValueError("invalid reward_mode")
    comparison = resolved.get("representation_comparison", {})
    if comparison:
        if not isinstance(comparison, Mapping):
            raise ValueError("representation_comparison must be an object")
        if int(comparison.get("identity_model_capacity", -1)) != int(
            comparison.get("relational_model_capacity", -2)
        ):
            raise ValueError("identity and relational model capacity must match")
        if int(comparison.get("identity_update_budget", -1)) != int(
            comparison.get("relational_update_budget", -2)
        ):
            raise ValueError("identity and relational update budgets must match")
        if not str(comparison.get("raw_observation_schema", "")).strip():
            raise ValueError("representation comparison needs a shared raw schema")
    if stage is not V2StudyStage.DEVELOPMENT_DIAGNOSTIC:
        lock_value = str(resolved.get("protocol_lock", "")).strip()
        if not lock_value:
            raise ValueError("confirmation and pilot configs require protocol_lock")
        lock_path = Path(lock_value)
        if not lock_path.is_absolute():
            lock_path = root / lock_path
        if not lock_path.is_file():
            raise FileNotFoundError(f"protocol lock not found: {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "locked":
            raise ValueError("protocol lock is not locked")
        if lock.get("seed_commitment_sha256") != actual_commitment:
            raise ValueError("protocol lock uses a different seed commitment")
        if lock.get("config_sha256") != sha256_json(
            {key: value for key, value in resolved.items() if key != "protocol_lock"}
        ):
            raise ValueError("protocol lock config hash mismatch")
        if lock.get("implementation_tree_sha256") != implementation_tree_sha256(root):
            raise ValueError("protocol lock implementation tree changed")
    if stage is V2StudyStage.PILOT:
        evidence_value = str(resolved.get("confirmation_artifact", "")).strip()
        evidence_hash = str(resolved.get("confirmation_artifact_sha256", ""))
        if not evidence_value or not evidence_hash:
            raise ValueError("Pilot requires locked confirmation evidence")
        evidence_path = Path(evidence_value)
        if not evidence_path.is_absolute():
            evidence_path = root / evidence_path
        if not evidence_path.is_file():
            raise FileNotFoundError(f"confirmation artifact not found: {evidence_path}")
        actual_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if actual_hash != evidence_hash:
            raise ValueError("confirmation artifact hash mismatch")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("stage") != V2StudyStage.LOCKED_CONFIRMATION.value:
            raise ValueError("Pilot evidence is not a Locked Confirmation")
        if not all(evidence.get("engineering_integrity_gates", {}).values()):
            raise ValueError("Locked Confirmation engineering gates failed")
        if not all(evidence.get("benchmark_adequacy_gates", {}).values()):
            raise ValueError("Locked Confirmation adequacy gates failed")
    resolved["research_seeds"] = seeds
    resolved["world_seeds"] = materialized
    resolved["seed_commitment_sha256"] = actual_commitment
    resolved["study_stage"] = stage.value
    return resolved


def build_run_identity(
    config: Mapping[str, Any], *, run_id: str, repository_root: str | Path | None = None
) -> V2RunIdentity:
    resolved = validate_v2_config(config, repository_root=repository_root)
    return V2RunIdentity(
        protocol_version=str(resolved["protocol_version"]),
        stage=V2StudyStage(str(resolved["study_stage"])),
        run_id=run_id,
        config_sha256=sha256_json(resolved),
        seed_commitment_sha256=str(resolved["seed_commitment_sha256"]),
        causal_law_sha256=str(resolved["causal_law_sha256"]),
        implementation_commit=current_git_commit(repository_root),
    )


def v2_run_directory(
    identity: V2RunIdentity, *, repository_root: str | Path | None = None
) -> Path:
    stage_dir = {
        V2StudyStage.DEVELOPMENT_DIAGNOSTIC: "development",
        V2StudyStage.LOCKED_CONFIRMATION: "locked_confirmation",
        V2StudyStage.PILOT: "pilot",
    }[identity.stage]
    relative = (
        Path("paper_results_v2")
        / stage_dir
        / identity.protocol_version
        / identity.run_id
    )
    return assert_v2_output_path(relative, repository_root=repository_root)


def reserve_run(
    directory: Path, identity: V2RunIdentity, *, resume: bool = False
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / "run_claim.json"
    payload = identity.to_dict()
    if claim.exists():
        existing = json.loads(claim.read_text(encoding="utf-8"))
        if not resume or existing != payload:
            raise FileExistsError("run is already claimed or identity changed")
        return claim
    if resume:
        raise FileNotFoundError("cannot resume a run without a claim")
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return claim


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_execution_lock(directory: Path, *, resume: bool) -> tuple[Path, str]:
    """Prevent two identical resume processes from writing the same run."""
    path = directory / "run_active.json"
    nonce = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "nonce": nonce,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        return path, nonce
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _pid_is_running(int(existing.get("pid", -1))):
            raise RuntimeError("an identical run is already executing")
        if not resume:
            raise FileExistsError("stale run lease requires --resume")
        failed = directory / "failed_attempts"
        failed.mkdir(parents=True, exist_ok=True)
        archived = failed / f"stale_run_active_{uuid.uuid4().hex}.json"
        path.replace(archived)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        return path, nonce


def release_run_execution_lock(path: Path, nonce: str) -> None:
    if not path.exists():
        raise RuntimeError("run execution lock disappeared")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nonce") != nonce:
        raise RuntimeError("run execution lock ownership changed")
    path.unlink()


def preserve_partial_directory(directory: Path, *, label: str) -> Path:
    """Move a failed partial artifact under failed_attempts without deleting it."""
    if not directory.exists():
        raise FileNotFoundError(directory)
    failed = directory.parent / "failed_attempts"
    failed.mkdir(parents=True, exist_ok=True)
    target = failed / f"{label}_{uuid.uuid4().hex}"
    directory.replace(target)
    return target


def reserve_confirmation_once(
    output_root: Path, identity: V2RunIdentity, *, resume: bool = False
) -> Path | None:
    if identity.stage is not V2StudyStage.LOCKED_CONFIRMATION:
        return None
    registry = output_root / "confirmation_claims"
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / (
        f"{identity.protocol_version}-{identity.seed_commitment_sha256}.json"
    )
    payload = identity.to_dict()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume or existing != payload:
            raise FileExistsError("locked confirmation was already attempted")
        return path
    if resume:
        raise FileNotFoundError("confirmation claim does not exist")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path


def create_protocol_lock(
    config: Mapping[str, Any], *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    if str(config.get("study_stage")) == V2StudyStage.DEVELOPMENT_DIAGNOSTIC.value:
        raise ValueError("development diagnostics are not locked evidence")
    without_reference = {
        key: value for key, value in copy.deepcopy(dict(config)).items()
        if key != "protocol_lock"
    }
    seeds = [int(item) for item in without_reference["research_seeds"]]
    return {
        "schema_version": 1,
        "status": "locked",
        "protocol_version": str(without_reference["protocol_version"]),
        "study_stage": str(without_reference["study_stage"]),
        "config_sha256": sha256_json(without_reference),
        "seed_commitment_sha256": seed_commitment(seeds),
        "causal_law_sha256": str(without_reference["causal_law_sha256"]),
        "thresholds_sha256": sha256_json(without_reference.get("thresholds", {})),
        "implementation_tree_sha256": implementation_tree_sha256(repository_root),
        "implementation_commit": current_git_commit(repository_root),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def checkpoint_fingerprint(checkpoint: FullAgentCheckpoint) -> str:
    return sha256_json(checkpoint.to_dict())


AgentT = TypeVar("AgentT")


def clone_agent_from_checkpoint(
    agent: AgentT, factory: Callable[[], AgentT]
) -> tuple[AgentT, str]:
    exporter = getattr(agent, "export_full_checkpoint", None)
    if not callable(exporter):
        raise TypeError("agent must implement export_full_checkpoint")
    checkpoint = exporter()
    if not isinstance(checkpoint, FullAgentCheckpoint):
        raise TypeError("export_full_checkpoint returned the wrong type")
    clone = factory()
    importer = getattr(clone, "import_full_checkpoint", None)
    if not callable(importer):
        raise TypeError("agent clone must implement import_full_checkpoint")
    importer(checkpoint)
    restored = clone.export_full_checkpoint()
    before = checkpoint_fingerprint(checkpoint)
    if checkpoint_fingerprint(restored) != before:
        raise RuntimeError("frozen clone did not restore the complete checkpoint")
    return clone, before


class V2ArtifactWriter:
    def __init__(self, directory: Path, episode_columns: Sequence[str]) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.episodes_path = directory / "episodes.csv"
        self.trace_path = directory / "trace.jsonl.gz"
        self._episode_stream = self.episodes_path.open(
            "x", newline="", encoding="utf-8"
        )
        self._trace_stream = gzip.open(self.trace_path, "xt", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._episode_stream, fieldnames=list(episode_columns)
        )
        self._writer.writeheader()
        self.episode_rows = 0
        self.trace_rows = 0

    def write_episode(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow(dict(row))
        self.episode_rows += 1

    def write_trace(self, row: Mapping[str, Any]) -> None:
        self._trace_stream.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.trace_rows += 1

    def close(self) -> None:
        self._episode_stream.close()
        self._trace_stream.close()

    def __enter__(self) -> "V2ArtifactWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def replay_gzip_trace(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid trace JSON at line {line_number}") from error
    return records
