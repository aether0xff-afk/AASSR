from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aassr_v2.paper_artifacts import validate_paper_artifacts


def _evidence(path: str) -> dict[str, str]:
    root = Path(path)
    issues = validate_paper_artifacts(root)
    if issues:
        raise SystemExit(
            f"{root} is not a valid gate artifact: " + "; ".join(issues)
        )
    manifest = root / "manifests" / "protocol_manifest.json"
    return {
        "path": str(root),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lock reviewed P0-P3 pilot evidence before any Final suite runs."
        )
    )
    parser.add_argument("--p0-results", required=True)
    parser.add_argument("--p1-results", required=True)
    parser.add_argument("--p2-results", required=True)
    parser.add_argument("--p3-results", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = {
        "status": "accepted",
        "p0": True,
        "p1": True,
        "p2": True,
        "p3": True,
        "reviewer": args.reviewer,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "evidence": {
            "p0": _evidence(args.p0_results),
            "p1": _evidence(args.p1_results),
            "p2": _evidence(args.p2_results),
            "p3": _evidence(args.p3_results),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
