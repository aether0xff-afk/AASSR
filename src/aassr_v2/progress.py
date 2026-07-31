from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, TextIO


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0.0:
        return "--:--:--"
    rounded = int(seconds + 0.5)
    days, remainder = divmod(rounded, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Persistent progress, speed and ETA reporting for long experiment runs."""

    _CONTEXT_ORDER = (
        "job",
        "seed",
        "environment",
        "condition",
        "phase",
        "episode",
        "recent_success",
    )

    def __init__(
        self,
        total: int,
        output_dir: str | Path,
        *,
        every_items: int = 100,
        every_seconds: float = 10.0,
        console: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        if total <= 0:
            raise ValueError("total must be positive")
        if every_items <= 0:
            raise ValueError("every_items must be positive")
        if every_seconds <= 0.0:
            raise ValueError("every_seconds must be positive")
        self.total = total
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every_items = every_items
        self.every_seconds = every_seconds
        self.console = console
        self.stream = stream or sys.stdout
        self.log_path = self.output_dir / "progress.log"
        self.jsonl_path = self.output_dir / "progress.jsonl"
        self.status_path = self.output_dir / "progress.json"
        self.completed = 0
        self.started_at = time.perf_counter()
        self.started_wall = datetime.now().astimezone()
        self._last_emit_at = self.started_at
        self._last_emit_completed = 0
        self._last_context: dict[str, Any] = {}

        self.log_path.write_text("", encoding="utf-8")
        self.jsonl_path.write_text("", encoding="utf-8")

    def start(self, context: Mapping[str, Any] | None = None) -> None:
        self._emit("start", context or {}, force=True)

    def stage(self, event: str, context: Mapping[str, Any]) -> None:
        self._emit(event, context, force=True)

    def advance(
        self,
        context: Mapping[str, Any],
        *,
        count: int = 1,
        force: bool = False,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self.completed = min(self.total, self.completed + count)
        now = time.perf_counter()
        due_items = self.completed - self._last_emit_completed >= self.every_items
        due_time = now - self._last_emit_at >= self.every_seconds
        finished = self.completed >= self.total
        if force or due_items or due_time or finished:
            self._emit("progress", context, force=True)

    def finish(self, context: Mapping[str, Any] | None = None) -> None:
        self.completed = self.total
        self._emit("finish", context or self._last_context, force=True)

    def fail(self, error: BaseException, context: Mapping[str, Any] | None = None) -> None:
        merged = dict(context or self._last_context)
        merged["error"] = f"{type(error).__name__}: {error}"
        self._emit("failed", merged, force=True)

    def _snapshot(self, event: str, context: Mapping[str, Any]) -> dict[str, Any]:
        now = time.perf_counter()
        elapsed = max(0.0, now - self.started_at)
        rate = self.completed / elapsed if self.completed and elapsed > 0.0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta_seconds = remaining / rate if rate > 0.0 else None
        now_wall = datetime.now().astimezone()
        eta_at = now_wall + timedelta(seconds=eta_seconds) if eta_seconds is not None else None
        return {
            "event": event,
            "timestamp": now_wall.isoformat(timespec="seconds"),
            "started_at": self.started_wall.isoformat(timespec="seconds"),
            "completed": self.completed,
            "total": self.total,
            "percent": 100.0 * self.completed / self.total,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": eta_seconds,
            "eta_at": eta_at.isoformat(timespec="seconds") if eta_at else None,
            "context": dict(context),
        }

    def _context_text(self, context: Mapping[str, Any]) -> str:
        parts = []
        used = set()
        for key in self._CONTEXT_ORDER:
            value = context.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={value}")
                used.add(key)
        for key in sorted(context):
            if key not in used and context[key] not in (None, ""):
                parts.append(f"{key}={context[key]}")
        return " | ".join(parts)

    def _line(self, snapshot: Mapping[str, Any]) -> str:
        eta = format_duration(snapshot["eta_seconds"])
        elapsed = format_duration(float(snapshot["elapsed_seconds"]))
        rate = float(snapshot["rate_per_second"])
        context = self._context_text(snapshot["context"])
        prefix = (
            f"[AASSR:{snapshot['event']}] "
            f"{float(snapshot['percent']):6.2f}% "
            f"{int(snapshot['completed']):,}/{int(snapshot['total']):,} "
            f"| {rate:8.2f} ep/s | elapsed {elapsed} | ETA {eta}"
        )
        return f"{prefix} | {context}" if context else prefix

    def _write_status(self, snapshot: Mapping[str, Any]) -> None:
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # On Windows, a short-lived reader can prevent os.replace from
        # replacing the destination. Progress telemetry must never abort a
        # multi-hour experiment, so retry the atomic replace and fall back to
        # the append-only log/JSONL records if a reader keeps it open.
        for attempt in range(20):
            try:
                temporary.replace(self.status_path)
                return
            except PermissionError:
                if attempt == 19:
                    temporary.unlink(missing_ok=True)
                    return
                time.sleep(0.05)

    def _emit(
        self,
        event: str,
        context: Mapping[str, Any],
        *,
        force: bool,
    ) -> None:
        del force
        self._last_context = dict(context)
        snapshot = self._snapshot(event, context)
        line = self._line(snapshot)
        if self.console:
            print(line, file=self.stream, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        self._write_status(snapshot)
        self._last_emit_at = time.perf_counter()
        self._last_emit_completed = self.completed
