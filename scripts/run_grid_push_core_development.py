from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.grid_push_core_development import run_grid_push_core_development
from aassr_v2.paper_v2_protocol import validate_v2_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the GridPush full-AASSRCore Development Diagnostic"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    resolved = validate_v2_config(config)
    settings = resolved["grid_push_core"]
    print(f"Protocol: {resolved['protocol_version']}")
    print(f"Stage: {resolved['study_stage']}")
    print(f"Worlds to certify: {len(settings['certification_world_seeds'])}")
    print(f"Research seeds: {len(resolved['research_seeds'])}")
    print(f"Training episodes per seed: {settings['training_episodes']}")
    if args.dry_run:
        return 0
    artifacts = run_grid_push_core_development(resolved, run_id=args.run_id)
    print(f"Output: {artifacts.output_dir.resolve()}")
    print(f"Certified worlds: {artifacts.certified_worlds}")
    print(f"Episode rows: {artifacts.episode_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
