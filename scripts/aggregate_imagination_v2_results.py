from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.input)
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        if path.resolve() == Path(args.output).resolve():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "final" in payload and "config" in payload:
            summaries.append(payload)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in summaries:
        grouped[str(payload["config"]["condition"])].append(payload)

    conditions: dict[str, Any] = {}
    for condition, payloads in sorted(grouped.items()):
        seen = [float(item["final"]["seen_success_rate"]) for item in payloads]
        unseen = [float(item["final"]["unseen_success_rate"]) for item in payloads]
        efficiency = [
            float(item["final"]["unseen_path_efficiency"])
            for item in payloads
        ]
        select_seconds = [
            float(item["final"]["selection_seconds_total"])
            for item in payloads
        ]
        imagined_nodes = []
        change_rates = []
        critic_ready = []
        for item in payloads:
            diagnostics = item.get("agent_diagnostics", {})
            imagination = diagnostics.get("imagination", {})
            if isinstance(imagination, dict):
                runs = float(imagination.get("runs", 0))
                changed = float(imagination.get("changed_actions", 0))
                if runs:
                    change_rates.append(changed / runs)
            critic = diagnostics.get("critic", {})
            if isinstance(critic, dict):
                critic_ready.append(float(bool(critic.get("ready", False))))
            checkpoint = int(item["final"]["checkpoint_episode"])
            rows_path = None
            # Raw episode CSVs remain in each artifact. Their paths are not
            # required for the core success comparison, so this field stays
            # optional in the aggregate.
            del checkpoint, rows_path
        conditions[condition] = {
            "seeds": sorted(int(item["config"]["seed"]) for item in payloads),
            "seen_success_rate": _mean_std(seen),
            "unseen_success_rate": _mean_std(unseen),
            "unseen_path_efficiency": _mean_std(efficiency),
            "selection_seconds_total": _mean_std(select_seconds),
            "imagination_change_rate": _mean_std(change_rates),
            "critic_ready_rate": _mean_std(critic_ready),
        }

    output = {
        "experiment": "imagination_v2_five_seed_pilot",
        "summary_count": len(summaries),
        "conditions": conditions,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
