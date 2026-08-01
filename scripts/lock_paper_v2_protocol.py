from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.paper_v2_protocol import create_protocol_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a Protocol v2 run config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = create_protocol_lock(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
