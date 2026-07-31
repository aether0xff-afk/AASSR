from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aassr_v2.paper_protocol import load_paper_config
from aassr_v2.safe_application import (
    run_safe_compose,
    verify_safe_compose_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate or operate the isolated local paper environment."
    )
    parser.add_argument("--config")
    parser.add_argument("--compose")
    parser.add_argument(
        "action",
        choices=("config", "up", "verify", "down", "smoke"),
        nargs="?",
        default="config",
    )
    parser.add_argument("--project-name", default="aassr-paper-safe")
    args = parser.parse_args()
    compose = args.compose
    if args.config:
        config = load_paper_config(args.config)
        safe = config.get("safe_application", {})
        if not isinstance(safe, dict):
            parser.error("config has no safe_application settings")
        compose = str(safe.get("compose_file", ""))
        os.environ["WORLD_SEED"] = str(
            safe.get(
                "world_seed",
                config["world_seeds"]["unseen"][0],
            )
        )
    if not compose:
        parser.error("--config or --compose is required")
    compose_path = Path(compose)
    if args.action == "verify":
        print(
            json.dumps(
                verify_safe_compose_runtime(
                    compose_path,
                    project_name=args.project_name,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.action == "smoke":
        try:
            run_safe_compose(
                compose_path,
                action="up",
                project_name=args.project_name,
            )
            print(
                json.dumps(
                    verify_safe_compose_runtime(
                        compose_path,
                        project_name=args.project_name,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            run_safe_compose(
                compose_path,
                action="down",
                project_name=args.project_name,
            )
        return 0
    result = run_safe_compose(
        compose_path,
        action=args.action,
        project_name=args.project_name,
    )
    if result.stdout:
        print(result.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
