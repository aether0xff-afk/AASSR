from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deliberately freeze a reviewed pilot creativity-threshold candidate "
            "before any Final result is run."
        )
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    source = Path(args.candidate)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "candidate":
        raise SystemExit("input is not a pilot threshold candidate")
    payload.update(
        {
            "status": "frozen",
            "reviewer": args.reviewer,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "candidate_sha256": __import__("hashlib").sha256(
                source.read_bytes()
            ).hexdigest(),
        }
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
