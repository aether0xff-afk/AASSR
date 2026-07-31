from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import csv
import gzip
import html
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .types import Action, StateSnapshot


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def serialize_action(action: Action) -> dict[str, Any]:
    return {
        "signature": action.signature,
        "verb": action.verb_name,
        "target": action.target,
        "tool": action.tool,
        "destination": action.destination,
        "metadata": _json_safe(action.metadata),
        "parameters": _json_safe(action.parameters),
    }


def serialize_snapshot(snapshot: StateSnapshot) -> dict[str, Any]:
    return {
        "vector": list(snapshot.vector),
        "facts": sorted(snapshot.facts),
        "available_actions": [serialize_action(action) for action in snapshot.available_actions],
        "goal_progress": snapshot.goal_progress,
        "metadata": _json_safe(snapshot.metadata),
    }


def serialize_world_spec(spec: object) -> dict[str, Any]:
    data = _json_safe(spec)
    if not isinstance(data, dict):
        raise TypeError("world spec must serialize to a mapping")
    return data


@dataclass(frozen=True, slots=True)
class EscapeEpisodeRecord:
    session_id: str
    episode: int
    success: bool
    interrupted: bool
    started_at_utc: str
    ended_at_utc: str
    duration_seconds: float
    session_elapsed_seconds: float
    steps: int
    optimal_steps: int
    efficiency: float
    score: float
    rolling_score: float
    epsilon: float
    move_actions: int
    interaction_actions: int
    errors: int
    repeated_actions: int
    blocked_moves: int
    found_keys: int
    opened_doors: int
    empty_boxes: int
    imagination_decisions: int
    imagined_nodes: int
    maximum_imagination_depth: int
    mean_prediction_score: float
    mean_holdout_before: float
    mean_holdout_after: float
    mean_holdout_gain: float
    positive_holdout_gain_total: float
    mean_intrinsic_value: float
    intrinsic_value_total: float
    live_seconds: float
    fast_seconds: float
    policy_entries: int
    prophecy_exact_entries: int
    holdout_size: int
    action_counts: Mapping[str, int]
    event_counts: Mapping[str, int]

    def csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["action_counts"] = json.dumps(self.action_counts, sort_keys=True)
        row["event_counts"] = json.dumps(self.event_counts, sort_keys=True)
        return row


EPISODE_COLUMNS = tuple(EscapeEpisodeRecord.__dataclass_fields__.keys())


def describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "p25": percentile(0.25),
        "p75": percentile(0.75),
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def rolling_mean(values: Sequence[float], window: int = 100) -> list[float]:
    result: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(float(value))
        total += float(value)
        if len(queue) > window:
            total -= queue.pop(0)
        result.append(total / len(queue))
    return result


def _svg_line_chart(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[float]]],
) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 90, 30, 70, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    lengths = [len(values) for _, values in series if values]
    count = max(lengths, default=0)
    all_values = [float(value) for _, values in series for value in values]
    minimum = min(all_values, default=0.0)
    maximum = max(all_values, default=1.0)
    if math.isclose(minimum, maximum):
        padding = max(1.0, abs(minimum) * 0.1)
        minimum -= padding
        maximum += padding
    else:
        padding = (maximum - minimum) * 0.08
        minimum -= padding
        maximum += padding

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")

    def x_position(index: int) -> float:
        return left + (plot_width * index / max(1, count - 1))

    def y_position(value: float) -> float:
        return top + plot_height * (maximum - value) / (maximum - minimum)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="24" font-family="sans-serif">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * fraction
        value = maximum - (maximum - minimum) * fraction
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-size="13" font-family="sans-serif">{value:.4g}</text>'
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
            f'<text x="22" y="{height/2}" text-anchor="middle" transform="rotate(-90 22 {height/2})" font-size="15" font-family="sans-serif">{html.escape(y_label)}</text>',
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="15" font-family="sans-serif">Episode</text>',
        ]
    )
    if count:
        for tick in range(6):
            index = round((count - 1) * tick / 5)
            x = x_position(index)
            lines.append(
                f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="13" font-family="sans-serif">{index + 1}</text>'
            )
    for series_index, (label, values) in enumerate(series):
        if not values:
            continue
        color = colors[series_index % len(colors)]
        points = " ".join(
            f"{x_position(index):.2f},{y_position(float(value)):.2f}"
            for index, value in enumerate(values)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_x = left + 20 + series_index * 210
        lines.append(
            f'<line x1="{legend_x}" y1="54" x2="{legend_x+28}" y2="54" stroke="{color}" stroke-width="4"/>'
        )
        lines.append(
            f'<text x="{legend_x+36}" y="59" font-size="14" font-family="sans-serif">{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _svg_bar_chart(
    path: Path,
    *,
    title: str,
    items: Sequence[tuple[str, int]],
) -> None:
    selected = list(items[:20])
    width, height = 1200, max(500, 100 + 34 * len(selected))
    left, right, top, bottom = 260, 40, 70, 40
    plot_width = width - left - right
    maximum = max((value for _, value in selected), default=1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="24" font-family="sans-serif">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(selected):
        y = top + index * 34
        bar_width = plot_width * value / maximum if maximum else 0
        lines.append(
            f'<text x="{left-10}" y="{y+20}" text-anchor="end" font-size="13" font-family="sans-serif">{html.escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y+5}" width="{bar_width:.2f}" height="20" fill="#2563eb" rx="3"/>'
        )
        lines.append(
            f'<text x="{left+bar_width+8:.2f}" y="{y+20}" font-size="13" font-family="sans-serif">{value}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_statistics(
    records: Sequence[EscapeEpisodeRecord],
    *,
    action_counts: Mapping[str, int],
    event_counts: Mapping[str, int],
) -> dict[str, Any]:
    steps = [float(record.steps) for record in records]
    scores = [record.score for record in records]
    durations = [record.duration_seconds for record in records]
    predictions = [record.mean_prediction_score for record in records]
    holdout_gains = [record.mean_holdout_gain for record in records]
    intrinsic = [record.intrinsic_value_total for record in records]
    imagined_nodes = [float(record.imagined_nodes) for record in records]
    errors = [float(record.errors) for record in records]
    repeats = [float(record.repeated_actions) for record in records]
    efficiencies = [record.efficiency for record in records]
    return {
        "episode_count": len(records),
        "steps": describe(steps),
        "scores": describe(scores),
        "durations_seconds": describe(durations),
        "efficiency": describe(efficiencies),
        "mean_prediction_score": describe(predictions),
        "mean_holdout_gain": describe(holdout_gains),
        "intrinsic_value_total": describe(intrinsic),
        "imagined_nodes": describe(imagined_nodes),
        "errors": describe(errors),
        "repeated_actions": describe(repeats),
        "correlations": {
            "episode_vs_steps": pearson(list(range(1, len(records) + 1)), steps),
            "steps_vs_score": pearson(steps, scores),
            "steps_vs_duration": pearson(steps, durations),
            "imagination_nodes_vs_score": pearson(imagined_nodes, scores),
            "prediction_vs_score": pearson(predictions, scores),
        },
        "action_counts": dict(sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))),
        "event_counts": dict(sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def write_charts(
    charts_dir: Path,
    records: Sequence[EscapeEpisodeRecord],
    *,
    action_counts: Mapping[str, int],
    event_counts: Mapping[str, int],
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    episode_steps = [float(record.steps) for record in records]
    scores = [record.score for record in records]
    durations = [record.duration_seconds for record in records]
    predictions = [record.mean_prediction_score for record in records]
    holdout = [record.mean_holdout_gain for record in records]
    intrinsic = [record.intrinsic_value_total for record in records]
    imagined_nodes = [float(record.imagined_nodes) for record in records]
    imagination_decisions = [float(record.imagination_decisions) for record in records]
    errors = [float(record.errors) for record in records]
    repeats = [float(record.repeated_actions) for record in records]
    _svg_line_chart(
        charts_dir / "episode_steps.svg",
        title="Steps per episode",
        y_label="Steps",
        series=(("Steps", episode_steps), ("Rolling mean (100)", rolling_mean(episode_steps))),
    )
    _svg_line_chart(
        charts_dir / "episode_scores.svg",
        title="Success score per episode",
        y_label="Score multiplier",
        series=(("Score", scores), ("Rolling mean (100)", rolling_mean(scores))),
    )
    _svg_line_chart(
        charts_dir / "episode_duration.svg",
        title="Episode duration",
        y_label="Seconds",
        series=(("Duration", durations), ("Rolling mean (100)", rolling_mean(durations))),
    )
    _svg_line_chart(
        charts_dir / "prediction_and_holdout.svg",
        title="Prediction and holdout metrics",
        y_label="Value",
        series=(("Prediction score", predictions), ("Holdout gain", holdout)),
    )
    _svg_line_chart(
        charts_dir / "intrinsic_value.svg",
        title="Intrinsic value by episode",
        y_label="Total intrinsic value",
        series=(("Intrinsic total", intrinsic), ("Rolling mean (100)", rolling_mean(intrinsic))),
    )
    _svg_line_chart(
        charts_dir / "imagination_usage.svg",
        title="Imagination usage by episode",
        y_label="Count",
        series=(("Imagined nodes", imagined_nodes), ("Imagination decisions", imagination_decisions)),
    )
    _svg_line_chart(
        charts_dir / "errors_and_repeats.svg",
        title="Errors and repeated actions",
        y_label="Count",
        series=(("Errors", errors), ("Repeated actions", repeats)),
    )
    _svg_bar_chart(
        charts_dir / "action_distribution.svg",
        title="Action distribution",
        items=sorted(action_counts.items(), key=lambda item: (-item[1], item[0])),
    )
    _svg_bar_chart(
        charts_dir / "event_distribution.svg",
        title="Event distribution",
        items=sorted(event_counts.items(), key=lambda item: (-item[1], item[0])),
    )


def _state_key_json(key: Any) -> Any:
    return _json_safe(key)


def serialize_agent_checkpoint(agent: object, *, episode: int) -> dict[str, Any]:
    policy = getattr(agent, "policy")
    prophecy = getattr(agent, "prophecy")
    holdout = getattr(agent, "holdout")

    local_policy = []
    for (state, action_signature), value in getattr(policy, "_local", {}).items():
        local_policy.append(
            {
                "state": _state_key_json(state),
                "action_signature": action_signature,
                "count": value.count,
                "mean": value.mean,
            }
        )
    global_policy = {
        signature: {"count": value.count, "mean": value.mean}
        for signature, value in getattr(policy, "_global", {}).items()
    }
    state_visits = [
        {"state": _state_key_json(state), "visits": visits}
        for state, visits in getattr(policy, "_state_visits", {}).items()
    ]

    exact_prophecy = []
    for (state, action_signature), counter in getattr(prophecy, "_exact", {}).items():
        exact_prophecy.append(
            {
                "state": _state_key_json(state),
                "action_signature": action_signature,
                "next_states": [
                    {"state": _state_key_json(next_state), "count": count}
                    for next_state, count in counter.items()
                ],
            }
        )
    global_prophecy = {
        verb: [
            {"state": _state_key_json(state), "count": count}
            for state, count in counter.items()
        ]
        for verb, counter in getattr(prophecy, "_global", {}).items()
    }
    prophecy_states = [
        {"fingerprint": _state_key_json(fingerprint), "snapshot": serialize_snapshot(snapshot)}
        for fingerprint, snapshot in getattr(prophecy, "_states", {}).items()
    ]
    holdout_items = [
        {
            "before": serialize_snapshot(item.before),
            "action": serialize_action(item.action),
            "after": serialize_snapshot(item.after),
        }
        for item in getattr(holdout, "_items", ())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "episode": episode,
        "saved_at_utc": utc_now_iso(),
        "agent_config": _json_safe(getattr(agent, "config", {})),
        "transition_index": getattr(agent, "_transition_index", 0),
        "decision_index": getattr(agent, "_decision_index", 0),
        "random_state": _json_safe(getattr(agent, "randomizer").getstate()),
        "policy": {
            "learning_rate": getattr(policy, "learning_rate", None),
            "local": local_policy,
            "global": global_policy,
            "state_visits": state_visits,
        },
        "prophecy": {
            "name": getattr(prophecy, "name", "unknown"),
            "exact": exact_prophecy,
            "global": global_prophecy,
            "states": prophecy_states,
        },
        "holdout": {
            "seen": getattr(holdout, "_seen", 0),
            "capacity": getattr(holdout, "capacity", None),
            "items": holdout_items,
            "random_state": _json_safe(getattr(holdout, "_randomizer").getstate()),
        },
        "effect_novelty_motifs": [
            _json_safe(item)
            for item in sorted(
                getattr(agent, "_seen_effect_motifs", ()),
                key=repr,
            )
        ],
    }


class EscapeSessionRecorder:
    """Durable, flush-on-write recorder for every escape session event."""

    def __init__(
        self,
        *,
        config: object,
        spec: object,
        oracle_steps: int,
        initial_mode: str,
        output_dir: str | Path | None = None,
    ) -> None:
        self.session_id = uuid4().hex
        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            seed = getattr(config, "seed", "unknown")
            output_dir = Path("runs") / "escape_gridworld" / f"{stamp}_seed{seed}_{self.session_id[:8]}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.charts_dir = self.output_dir / "charts"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        self.started_at_utc = utc_now_iso()
        self.records: list[EscapeEpisodeRecord] = []
        self.action_counts: Counter[str] = Counter()
        self.event_counts: Counter[str] = Counter()
        self.mode_counts: Counter[str] = Counter({initial_mode: 1})
        self._closed = False

        self._steps_file = (self.output_dir / "steps.jsonl").open("w", encoding="utf-8", buffering=1)
        self._episodes_jsonl = (self.output_dir / "episodes.jsonl").open("w", encoding="utf-8", buffering=1)
        self._episodes_csv_handle = (self.output_dir / "episodes.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self._episodes_csv = csv.DictWriter(self._episodes_csv_handle, fieldnames=EPISODE_COLUMNS)
        self._episodes_csv.writeheader()
        self._modes_file = (self.output_dir / "mode_switches.jsonl").open("w", encoding="utf-8", buffering=1)
        self._log_file = (self.output_dir / "session.log").open("w", encoding="utf-8", buffering=1)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "status": "running",
            "initial_mode": initial_mode,
            "oracle_steps": oracle_steps,
            "config": _json_safe(config),
            "world": serialize_world_spec(spec),
            "files": {
                "steps": "steps.jsonl",
                "episodes_csv": "episodes.csv",
                "episodes_jsonl": "episodes.jsonl",
                "mode_switches": "mode_switches.jsonl",
                "summary": "summary.json",
                "checkpoint_directory": "checkpoints",
                "chart_directory": "charts",
            },
        }
        self._manifest = manifest
        self._write_json(self.output_dir / "session.json", manifest)
        self._write_json(self.output_dir / "world.json", serialize_world_spec(spec))
        self.log(f"session started: {self.session_id}")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log(self, message: str) -> None:
        self._log_file.write(f"{utc_now_iso()} {message.rstrip()}\n")
        self._log_file.flush()

    def record_mode_switch(
        self,
        *,
        previous_mode: str,
        new_mode: str,
        session_elapsed_seconds: float,
        episode: int,
        step: int,
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "timestamp_utc": utc_now_iso(),
            "session_elapsed_seconds": session_elapsed_seconds,
            "episode": episode,
            "step": step,
            "previous_mode": previous_mode,
            "new_mode": new_mode,
        }
        self._modes_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._modes_file.flush()
        self.mode_counts[new_mode] += 1
        self.log(f"mode switch {previous_mode} -> {new_mode} at E{episode} step={step}")

    def record_step(self, payload: Mapping[str, Any]) -> None:
        enriched = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            **payload,
        }
        action = payload.get("action")
        if isinstance(action, Mapping):
            signature = str(action.get("signature", "unknown"))
            self.action_counts[signature] += 1
        event = str(payload.get("event", ""))
        if event:
            self.event_counts[event] += 1
        self._steps_file.write(json.dumps(_json_safe(enriched), ensure_ascii=False) + "\n")
        self._steps_file.flush()

    def record_episode(self, record: EscapeEpisodeRecord) -> None:
        self.records.append(record)
        payload = asdict(record)
        self._episodes_jsonl.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
        self._episodes_jsonl.flush()
        self._episodes_csv.writerow(record.csv_row())
        self._episodes_csv_handle.flush()
        self.log(
            f"episode={record.episode} success={int(record.success)} steps={record.steps} "
            f"score={record.score:.6f} duration={record.duration_seconds:.6f}s"
        )

    def write_checkpoint(self, agent: object, *, episode: int, final: bool = False) -> Path:
        payload = serialize_agent_checkpoint(agent, episode=episode)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        path = self.checkpoints_dir / (
            "final.json.gz" if final else f"episode_{episode:06d}.json.gz"
        )
        with gzip.open(path, "wb", compresslevel=6) as handle:
            handle.write(encoded)
        latest = self.checkpoints_dir / "latest.json.gz"
        with gzip.open(latest, "wb", compresslevel=6) as handle:
            handle.write(encoded)
        return path

    def finalize(
        self,
        *,
        summary: Mapping[str, Any],
        agent: object,
        stopped: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        statistics_payload = build_statistics(
            self.records,
            action_counts=self.action_counts,
            event_counts=self.event_counts,
        )
        final_payload = {
            **_json_safe(summary),
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": utc_now_iso(),
            "status": "error" if error else ("stopped" if stopped else "completed"),
            "output_dir": str(self.output_dir),
            "statistics": statistics_payload,
            "mode_selection_counts": dict(self.mode_counts),
            "error": error,
        }
        self._write_json(self.output_dir / "summary.json", final_payload)
        self._write_json(self.output_dir / "statistics.json", statistics_payload)
        summary_lines = [
            f"session_id: {self.session_id}",
            f"status: {final_payload['status']}",
            f"episodes: {len(self.records)}",
            f"total_steps: {sum(record.steps for record in self.records)}",
            f"mean_steps: {statistics_payload['steps']['mean']:.6f}",
            f"median_steps: {statistics_payload['steps']['median']:.6f}",
            f"mean_score: {statistics_payload['scores']['mean']:.6f}",
            f"mean_duration_seconds: {statistics_payload['durations_seconds']['mean']:.6f}",
            f"output_dir: {self.output_dir}",
        ]
        (self.output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        write_charts(
            self.charts_dir,
            self.records,
            action_counts=self.action_counts,
            event_counts=self.event_counts,
        )
        self.write_checkpoint(agent, episode=len(self.records), final=True)
        self._manifest.update(
            {
                "ended_at_utc": final_payload["ended_at_utc"],
                "status": final_payload["status"],
                "summary_file": "summary.json",
            }
        )
        self._write_json(self.output_dir / "session.json", self._manifest)
        if error:
            (self.output_dir / "error.txt").write_text(error, encoding="utf-8")
        self.log(f"session finalized: {final_payload['status']}")
        self.close()
        return final_payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in (
            self._steps_file,
            self._episodes_jsonl,
            self._episodes_csv_handle,
            self._modes_file,
            self._log_file,
        ):
            handle.flush()
            handle.close()

    def __enter__(self) -> "EscapeSessionRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
