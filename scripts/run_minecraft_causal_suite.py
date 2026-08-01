from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.minecraft_diagnostic_v2 import run_minecraft_diagnostic
from aassr_v2.paper_v2_protocol import validate_v2_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Protocol v2 Minecraft-like mock suite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    resolved = validate_v2_config(config)
    print(f"Protocol: {resolved['protocol_version']}")
    print(f"Stage: {resolved['study_stage']}")
    print(f"Track: {resolved['minecraft']['track']}")
    if args.dry_run:
        return 0
    artifacts = run_minecraft_diagnostic(resolved, run_id=args.run_id, resume=args.resume)
    print(f"Output: {artifacts.output_dir.resolve()}")
    print(f"Rows: {artifacts.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
