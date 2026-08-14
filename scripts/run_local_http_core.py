from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aassr_v2.core import build_aassr_core
from aassr_v2.plugins.local_http import LocalHttpConfig, LocalHttpPlugin


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AASSR Core against a real loopback HTTP service."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    plugin = LocalHttpPlugin(LocalHttpConfig(args.base_url))
    core = build_aassr_core(
        plugin,
        seed=int(args.seed),
        device=str(args.device),
        use_imagination=True,
        train_transitions=max(
            1,
            int(args.episodes) * int(args.max_steps),
        ),
    )

    returns = []
    for episode in range(int(args.episodes)):
        returns.append(
            core.run_episode(
                episode=episode,
                max_steps=int(args.max_steps),
                training=True,
            )
        )
        print(
            f"episode={episode + 1}/{args.episodes} return={returns[-1]:+.3f}",
            flush=True,
        )

    result = {
        "base_url": args.base_url,
        "episodes": len(returns),
        "positive": sum(value > 0.0 for value in returns),
        "negative": sum(value < 0.0 for value in returns),
        "zero": sum(value == 0.0 for value in returns),
        "mean_return": sum(returns) / max(1, len(returns)),
        "returns": returns,
        "diagnostics": core.diagnostics(),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
