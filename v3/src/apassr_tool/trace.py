from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def build_trace_report(data: dict[str, Any], *, window: int = 8) -> str:
    lines: list[str] = ["# APASSR Trace Report", ""]
    episodes = data.get("episodes", [])
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- episodes: {len(episodes)}")
    lines.append(f"- successes: {sum(1 for row in episodes if row.get('success'))}")
    lines.append(f"- solved events: {sum(_solved_event_count(row) for row in episodes)}")
    novelty = data.get("novelty", {})
    if novelty:
        lines.append(f"- distinct action signatures: {novelty.get('signature_count', 0)}")
        lines.append(f"- distinct action chains: {novelty.get('chain_count', 0)}")
        lines.append(f"- distinct response transitions: {novelty.get('response_count', 0)}")
    lines.append("")

    solved_windows = list(_solved_windows(episodes, window=window))
    if solved_windows:
        lines.append("## Solved Windows")
        lines.append("")
        for item in solved_windows:
            lines.append(f"### Episode {item['episode']} Step {item['step']}")
            lines.append("")
            lines.append(f"- solved_delta: {item['solved_delta']}")
            lines.append(f"- solved_total: {item['solved_total']}")
            lines.append("")
            lines.extend(_record_table(item["records"]))
            lines.append("")
    else:
        lines.append("## No Solved Event Yet")
        lines.append("")
        lines.append(
            "No `solved_delta > 0` event was found in this run. The sections below summarize "
            "the most unusual traces observed so far."
        )
        lines.append("")

    lines.append("## High-Novelty Steps")
    lines.append("")
    high_novelty = _high_novelty_steps(episodes, limit=20)
    if high_novelty:
        lines.extend(_record_table(high_novelty))
    else:
        lines.append("No per-step records were included. Re-run with `--include-records`.")
    lines.append("")

    lines.append("## Action Mix")
    lines.append("")
    mix = _action_mix(episodes)
    if mix:
        for key, count in mix.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("No per-step records were included.")
    lines.append("")

    return "\n".join(lines)


def _solved_event_count(episode: dict[str, Any]) -> int:
    return sum(1 for record in episode.get("records", []) if int(record.get("solved_delta", 0)) > 0)


def _solved_windows(episodes: list[dict[str, Any]], *, window: int):
    for episode in episodes:
        records = episode.get("records", [])
        for index, record in enumerate(records):
            if int(record.get("solved_delta", 0)) <= 0:
                continue
            start = max(0, index - window)
            end = min(len(records), index + window + 1)
            yield {
                "episode": episode.get("episode"),
                "step": record.get("step"),
                "solved_delta": record.get("solved_delta"),
                "solved_total": record.get("solved_total"),
                "records": records[start:end],
            }


def _high_novelty_steps(episodes: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = episode.get("episode")
        for record in episode.get("records", []):
            row = dict(record)
            row["episode"] = episode_id
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("novelty_score", 0.0)) + float(row.get("novelty_bonus", 0.0)),
            float(row.get("reward", 0.0)),
        ),
        reverse=True,
    )
    return rows[:limit]


def _action_mix(episodes: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for episode in episodes:
        for record in episode.get("records", []):
            counter[str(record.get("template", ""))] += 1
    return counter


def _record_table(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| ep | step | template | status | new_kv | solved | novelty | reward | action |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        action = str(record.get("action", "")).replace("|", "\\|")
        if len(action) > 90:
            action = action[:87] + "..."
        lines.append(
            "| {ep} | {step} | {template} | {status} | {new_kv} | {solved} | {novelty:.3f} | {reward:.3f} | {action} |".format(
                ep=record.get("episode", ""),
                step=record.get("step", ""),
                template=record.get("template", ""),
                status=record.get("status", ""),
                new_kv=record.get("new_kv", ""),
                solved=record.get("solved_delta", 0),
                novelty=float(record.get("novelty_score", 0.0)),
                reward=float(record.get("reward", 0.0)),
                action=action,
            )
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract APASSR solved and novelty traces.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args(argv)

    with Path(args.input).open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    report = build_trace_report(data, window=args.window)
    Path(args.output).write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
