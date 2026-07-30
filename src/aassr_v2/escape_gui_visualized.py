from __future__ import annotations

import queue
import threading
import traceback
from typing import Any

from .escape_gui import EscapeTrainingApp
from .escape_imagination_capture import EscapeImaginationEvent
from .escape_imagination_gui import EscapeImaginationViewer
from .escape_training import EscapeRenderFrame, EscapeTrainingSummary, TrainingMode, TrainingRuntime
from .escape_training_visualized import train_escape_agent


class VisualizedEscapeTrainingApp(EscapeTrainingApp):
    """Normal escape trainer plus a second live Imagination window."""

    def __init__(self, root: Any) -> None:
        self._imagination_lock = threading.Lock()
        self._pending_fast_imagination: EscapeImaginationEvent | None = None
        super().__init__(root)
        self.imagination_viewer = EscapeImaginationViewer(root)

    def _start(self, mode: TrainingMode) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            self.status_var.set(f"설정을 확인하세요: {exc}")
            return

        self.mode = mode
        self.runtime = TrainingRuntime(mode)
        self.latest_frame = None
        self.latest_summary = None
        self.history.clear()
        self.stop_event = threading.Event()
        self.progress_var.set(0.0)
        self._clear_log()
        self.statistics_button.configure(state="disabled")
        self._refresh_mode_buttons(running=True)
        with self._imagination_lock:
            self._pending_fast_imagination = None
        self.imagination_viewer.reset()

        if mode is TrainingMode.LIVE:
            self.status_var.set(
                "실시간 모드: GridWorld와 모든 실제 Imagination 트리를 함께 표시·저장합니다."
            )
            self._idle("실시간 렌더링과 Imagination 창 준비 중…")
        else:
            self.status_var.set(
                "최대 속도 모드: 모든 상상은 저장하고 Imagination 창은 최신 트리로 갱신합니다."
            )
            self._idle(
                "최대 속도 모드\nGrid 렌더링 비활성화\n"
                "모든 Imagination 트리 저장 중"
            )

        def worker() -> None:
            try:
                train_escape_agent(
                    config,
                    mode=mode,
                    runtime=self.runtime,
                    on_frame=lambda frame: self.events.put(("frame", frame)),
                    on_imagination=self._receive_imagination,
                    on_complete=lambda summary: self.events.put(("complete", summary)),
                    stop_event=self.stop_event,
                )
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _receive_imagination(self, event: EscapeImaginationEvent) -> None:
        runtime = self.runtime
        if runtime is not None and runtime.mode is TrainingMode.LIVE:
            self.events.put(("imagination", event))
            return
        # Maximum-speed mode must not accumulate thousands of Tk events.
        # The complete event was already flushed to imaginations.jsonl.
        with self._imagination_lock:
            self._pending_fast_imagination = event

    def _take_pending_fast_imagination(self) -> EscapeImaginationEvent | None:
        with self._imagination_lock:
            event = self._pending_fast_imagination
            self._pending_fast_imagination = None
            return event

    def _poll(self) -> None:
        for _ in range(300):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "frame" and isinstance(payload, EscapeRenderFrame):
                self._frame(payload)
            elif kind == "imagination" and isinstance(payload, EscapeImaginationEvent):
                self.imagination_viewer.add_event(payload)
            elif kind == "complete" and isinstance(payload, EscapeTrainingSummary):
                pending = self._take_pending_fast_imagination()
                if pending is not None:
                    self.imagination_viewer.add_event(pending)
                self._complete(payload)
            elif kind == "error":
                self._error(str(payload))

        pending = self._take_pending_fast_imagination()
        if pending is not None:
            self.imagination_viewer.add_event(pending)
        self.root.after(30, self._poll)

    def _close(self) -> None:
        self.stop_event.set()
        self.imagination_viewer.close()
        self.root.destroy()


def launch_escape_gui() -> None:
    import tkinter as tk

    root = tk.Tk()
    VisualizedEscapeTrainingApp(root)
    root.mainloop()
