from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.open_creativity_v2 import freeze_baseline_reference


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an independent equal-budget Protocol v2 creativity reference"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    creativity = config.get("creativity", {})
    if creativity.get("reference_source") != "independent_equal_budget_baseline":
        raise ValueError("config does not request an independent baseline")
    payload = freeze_baseline_reference(
        args.output,
        world_seeds=config["world_seeds"]["unseen_composition"],
        interaction_budget=int(creativity["interaction_budget"]),
    )
    print(f"Reference: {Path(args.output).resolve()}")
    print(f"SHA-256: {payload['reference_sha256']}")
    print(f"Graphs: {len(payload['graphs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
