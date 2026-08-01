from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


V2_OUTPUT_ROOT_NAME = "paper_results_v2"
V1_OUTPUT_ROOT_NAME = "paper_results"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_v2_output_path(
    output: str | Path, *, repository_root: str | Path | None = None
) -> Path:
    """Return a resolved v2 output path or reject any v1/foreign target.

    Protocol v2 deliberately has no compatibility escape hatch for writing to
    ``paper_results``.  Callers may choose any run directory below the v2 root,
    but may never point the new runner at an existing v1 artifact.
    """

    root = Path(repository_root or Path.cwd()).resolve()
    target = Path(output)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    v2_root = (root / V2_OUTPUT_ROOT_NAME).resolve()
    try:
        target.relative_to(v2_root)
    except ValueError as error:
        raise ValueError(
            f"protocol v2 output must be below {v2_root}; got {target}"
        ) from error
    if target == v2_root:
        raise ValueError("protocol v2 output must identify a stage and run")
    lowered = {part.lower() for part in target.parts}
    if V1_OUTPUT_ROOT_NAME in lowered or any(
        part.endswith("-final-v1") or part.endswith("_final_v1")
        for part in lowered
    ):
        raise ValueError("protocol v2 may not target Final v1 artifacts")
    return target


def load_preservation_manifest(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "read_only_baseline":
        raise ValueError("Final v1 preservation manifest is not read-only")
    return payload


def verify_preservation_manifest(
    path: str | Path, *, repository_root: str | Path | None = None
) -> list[str]:
    root = Path(repository_root or Path.cwd()).resolve()
    payload = load_preservation_manifest(path)
    issues: list[str] = []
    for item in payload.get("files", ()):
        relative = Path(str(item["path"]))
        source = root / relative
        if not source.is_file():
            issues.append(f"missing: {relative.as_posix()}")
            continue
        actual = sha256_file(source)
        expected = str(item["sha256"])
        if actual != expected:
            issues.append(
                f"hash mismatch: {relative.as_posix()} expected={expected} actual={actual}"
            )
    return issues
