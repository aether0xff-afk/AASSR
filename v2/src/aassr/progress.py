from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    label: str
    total: int
    enabled: bool = True
    stream: object = sys.stderr
    started_at: float = field(default_factory=time.monotonic)
    completed: int = 0

    def advance(self, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.completed += amount
        self.render()

    def render(self) -> None:
        if self.total <= 0:
            return
        elapsed = max(0.0, time.monotonic() - self.started_at)
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else 0.0
        percent = 100.0 * self.completed / self.total
        message = (
            f"\r[{self.label}] {self.completed}/{self.total} "
            f"({percent:5.1f}%) elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        )
        print(message, end="", file=self.stream, flush=True)
        if self.completed >= self.total:
            print(file=self.stream, flush=True)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minute:02d}m{second:02d}s"
    if minutes:
        return f"{minutes:d}m{second:02d}s"
    return f"{second:d}s"
