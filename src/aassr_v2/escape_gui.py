from __future__ import annotations

import queue
import threading
import traceback
from typing import Any

from .escape_gridworld import EscapeGridSpec
from .escape_training import (
    EscapeRenderFrame,
    EscapeTrainingConfig,
    EscapeTrainingSummary,
    TrainingMode,
    train_escape_agent,
)


DOOR_COLORS = {
    "red": "#ef5350",
    "blue": "#42a5f5",
    "green": "#66bb6a",
}


class EscapeTrainingApp:
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("AASSR Escape GridWorld Trainer")
        self.root.geometry("1120x760")
        self.root.minsize(900, 650)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.current_mode: TrainingMode | None = None
        self.latest_frame: EscapeRenderFrame | None = None
        self.success_history: list[float] = []

        self.episode_var = tk.StringVar(value="2000")
        self.seed_var = tk.StringVar(value="7")
        self.color_var = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value="실행 방식을 선택하세요.")
        self.episode_status_var = tk.StringVar(value="Episode 0 / 0")
        self.success_var = tk.StringVar(value="성공률 0.0%")
        self.rolling_var = tk.StringVar(value="최근 100회 0.0%")
        self.epsilon_var = tk.StringVar(value="ε 0.000")
        self.time_var = tk.StringVar(value="경과 0.0초")
        self.inventory_var = tk.StringVar(value="보유 열쇠: 없음")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_layout()
        self.root.after(30, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(outer, text="실험 설정", padding=12)
        controls.pack(fill="x")

        fields = ttk.Frame(controls)
        fields.pack(side="left", fill="x", expand=True)
        ttk.Label(fields, text="Episodes").grid(row=0, column=0, sticky="w")
        ttk.Entry(fields, textvariable=self.episode_var, width=10).grid(
            row=1, column=0, padx=(0, 12), sticky="w"
        )
        ttk.Label(fields, text="Seed").grid(row=0, column=1, sticky="w")
        ttk.Entry(fields, textvariable=self.seed_var, width=10).grid(
            row=1, column=1, padx=(0, 12), sticky="w"
        )
        ttk.Label(fields, text="색 수").grid(row=0, column=2, sticky="w")
        color_box = ttk.Combobox(
            fields,
            textvariable=self.color_var,
            values=("1", "2", "3"),
            width=7,
            state="readonly",
        )
        color_box.grid(row=1, column=2, padx=(0, 12), sticky="w")

        buttons = ttk.Frame(controls)
        buttons.pack(side="right")
        self.live_button = ttk.Button(
            buttons,
            text="실시간으로 보기",
            command=lambda: self._start(TrainingMode.LIVE),
        )
        self.live_button.grid(row=0, column=0, padx=5)
        self.fast_button = ttk.Button(
            buttons,
            text="안 보고 최대 속도",
            command=lambda: self._start(TrainingMode.FAST),
        )
        self.fast_button.grid(row=0, column=1, padx=5)
        self.stop_button = ttk.Button(
            buttons,
            text="중지",
            command=self._stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=2, padx=(12, 0))

        ttk.Label(outer, textvariable=self.status_var).pack(
            fill="x", pady=(10, 4)
        )
        ttk.Progressbar(
            outer,
            variable=self.progress_var,
            maximum=100.0,
        ).pack(fill="x", pady=(0, 12))

        metrics = ttk.Frame(outer)
        metrics.pack(fill="x", pady=(0, 12))
        for index, variable in enumerate(
            (
                self.episode_status_var,
                self.success_var,
                self.rolling_var,
                self.epsilon_var,
                self.time_var,
            )
        ):
            label = ttk.Label(metrics, textvariable=variable, anchor="center")
            label.grid(row=0, column=index, sticky="ew", padx=4)
            metrics.columnconfigure(index, weight=1)

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        grid_panel = ttk.LabelFrame(body, text="현재 GridWorld", padding=10)
        side_panel = ttk.Frame(body, padding=(10, 0, 0, 0))
        body.add(grid_panel, weight=3)
        body.add(side_panel, weight=2)

        self.canvas = tk.Canvas(
            grid_panel,
            background="#f8f6f1",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw_latest())

        ttk.Label(side_panel, textvariable=self.inventory_var).pack(
            fill="x", pady=(0, 8)
        )
        ttk.Label(side_panel, text="학습 이벤트").pack(anchor="w")
        self.log = tk.Text(
            side_panel,
            height=18,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
        )
        self.log.pack(fill="both", expand=True, pady=(4, 10))

        ttk.Label(side_panel, text="최근 성공률").pack(anchor="w")
        self.chart = tk.Canvas(
            side_panel,
            height=150,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.chart.pack(fill="x", pady=(4, 0))
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())

        self._draw_idle_canvas()

    def _config(self) -> EscapeTrainingConfig:
        episodes = int(self.episode_var.get())
        seed = int(self.seed_var.get())
        color_count = int(self.color_var.get())
        return EscapeTrainingConfig(
            episodes=episodes,
            seed=seed,
            color_count=color_count,
            distractor_boxes=max(1, color_count),
            max_steps=120 + 30 * color_count,
            epsilon_decay_episodes=max(100, int(episodes * 0.75)),
        )

    def _start(self, mode: TrainingMode) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            config = self._config()
        except ValueError as exc:
            self.status_var.set(f"설정을 확인하세요: {exc}")
            return

        self.current_mode = mode
        self.stop_event = threading.Event()
        self.latest_frame = None
        self.success_history.clear()
        self.progress_var.set(0.0)
        self._clear_log()
        self._draw_idle_canvas(
            "실시간 렌더링 준비 중…"
            if mode is TrainingMode.LIVE
            else "최대 속도 모드 — Grid 렌더링을 생략합니다."
        )
        self.status_var.set(
            "실시간 모드: 실제 step 속도로 렌더링합니다."
            if mode is TrainingMode.LIVE
            else "최대 속도 모드: sleep과 step 렌더링 없이 학습합니다."
        )
        self._set_running(True)

        def worker() -> None:
            try:
                train_escape_agent(
                    config,
                    mode=mode,
                    on_frame=lambda frame: self.events.put(("frame", frame)),
                    on_complete=lambda summary: self.events.put(
                        ("complete", summary)
                    ),
                    stop_event=self.stop_event,
                )
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("중지 요청을 처리하고 있습니다…")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.live_button.configure(state=state)
        self.fast_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _poll_events(self) -> None:
        handled = 0
        while handled < 200:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            handled += 1
            if kind == "frame":
                self._handle_frame(payload)
            elif kind == "complete":
                self._handle_complete(payload)
            elif kind == "error":
                self._handle_error(str(payload))
        self.root.after(30, self._poll_events)

    def _handle_frame(self, payload: object) -> None:
        if not isinstance(payload, EscapeRenderFrame):
            return
        frame = payload
        self.latest_frame = frame
        percent = 100.0 * frame.episode / max(1, frame.total_episodes)
        self.progress_var.set(percent)
        self.episode_status_var.set(
            f"Episode {frame.episode:,} / {frame.total_episodes:,}"
        )
        success_rate = frame.total_successes / max(1, frame.episode)
        self.success_var.set(f"성공률 {success_rate:.1%}")
        self.rolling_var.set(f"최근 100회 {frame.rolling_success:.1%}")
        self.epsilon_var.set(f"ε {frame.epsilon:.3f}")
        self.time_var.set(f"경과 {frame.elapsed_seconds:.1f}초")
        inventory = ", ".join(frame.inventory) if frame.inventory else "없음"
        self.inventory_var.set(f"보유 열쇠: {inventory}")

        if frame.episode_finished:
            self.success_history.append(frame.rolling_success)
            self._draw_chart()
            self._append_log(
                f"E{frame.episode:04d}  step={frame.step:03d}  "
                f"{'SUCCESS' if frame.success else 'timeout'}  "
                f"rolling={frame.rolling_success:.3f}  ε={frame.epsilon:.3f}"
            )
        elif frame.event not in {"", "moved", "episode_started"}:
            self._append_log(
                f"E{frame.episode:04d}:{frame.step:03d}  {frame.event}"
            )

        if frame.mode is TrainingMode.LIVE:
            self._draw_grid(frame)

    def _handle_complete(self, payload: object) -> None:
        if not isinstance(payload, EscapeTrainingSummary):
            return
        self._set_running(False)
        label = "중지됨" if payload.stopped else "완료"
        self.status_var.set(
            f"{label}: {payload.episodes:,} episodes, "
            f"성공 {payload.successes:,}회, {payload.elapsed_seconds:.2f}초, "
            f"Q entries {payload.q_entries:,}, oracle {payload.oracle_steps} steps"
        )
        self._append_log(
            f"[{label}] success={payload.success_rate:.3f}, "
            f"rolling={payload.rolling_success:.3f}, "
            f"elapsed={payload.elapsed_seconds:.3f}s"
        )
        if self.current_mode is TrainingMode.FAST:
            self._draw_idle_canvas(
                "최대 속도 학습 완료\n"
                f"성공률 {payload.success_rate:.1%}\n"
                f"최근 성공률 {payload.rolling_success:.1%}\n"
                f"경과 {payload.elapsed_seconds:.2f}초"
            )

    def _handle_error(self, message: str) -> None:
        self._set_running(False)
        self.status_var.set("실행 중 오류가 발생했습니다.")
        self._append_log(message)

    def _draw_idle_canvas(self, text: str = "실행 방식을 선택하세요.") -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_text(
            width / 2,
            height / 2,
            text=text,
            fill="#555555",
            justify="center",
            font=("Segoe UI", 15, "bold"),
        )

    def _redraw_latest(self) -> None:
        if self.latest_frame is not None and self.current_mode is TrainingMode.LIVE:
            self._draw_grid(self.latest_frame)
        elif self.current_mode is None:
            self._draw_idle_canvas()

    def _draw_grid(self, frame: EscapeRenderFrame) -> None:
        canvas = self.canvas
        canvas.delete("all")
        spec = frame.spec
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        cell = min((width - 30) / spec.width, (height - 30) / spec.height)
        offset_x = (width - cell * spec.width) / 2
        offset_y = (height - cell * spec.height) / 2

        def bounds(position: tuple[int, int]) -> tuple[float, float, float, float]:
            x, y = position
            left = offset_x + x * cell
            top = offset_y + y * cell
            return left, top, left + cell, top + cell

        for y in range(spec.height):
            for x in range(spec.width):
                canvas.create_rectangle(
                    *bounds((x, y)),
                    fill="#f5f2ea",
                    outline="#dedbd2",
                )

        for position in spec.walls:
            canvas.create_rectangle(
                *bounds(position), fill="#333842", outline="#20242b"
            )

        for position, color in spec.doors:
            left, top, right, bottom = bounds(position)
            door_color = DOOR_COLORS.get(color, "#8d6e63")
            if position in frame.open_doors:
                canvas.create_rectangle(
                    left + cell * 0.12,
                    top + cell * 0.12,
                    right - cell * 0.12,
                    bottom - cell * 0.12,
                    outline=door_color,
                    width=max(2, int(cell * 0.08)),
                )
            else:
                canvas.create_rectangle(
                    left + cell * 0.08,
                    top,
                    right - cell * 0.08,
                    bottom,
                    fill=door_color,
                    outline="#222222",
                )
                canvas.create_oval(
                    right - cell * 0.28,
                    top + cell * 0.45,
                    right - cell * 0.18,
                    top + cell * 0.55,
                    fill="#f7df82",
                    outline="",
                )

        for box in spec.boxes:
            left, top, right, bottom = bounds(box.position)
            opened = box.box_id in frame.open_boxes
            fill = "#c7aa82" if opened else "#8d6e4f"
            canvas.create_rectangle(
                left + cell * 0.18,
                top + cell * 0.22,
                right - cell * 0.18,
                bottom - cell * 0.18,
                fill=fill,
                outline="#4e342e",
                width=2,
            )
            canvas.create_line(
                left + cell * 0.18,
                top + cell * 0.42,
                right - cell * 0.18,
                top + cell * 0.42,
                fill="#4e342e",
                width=2,
            )
            if opened and box.key_color in frame.inventory:
                canvas.create_text(
                    (left + right) / 2,
                    (top + bottom) / 2,
                    text="✓",
                    fill=DOOR_COLORS.get(box.key_color or "", "#333333"),
                    font=("Segoe UI", max(9, int(cell * 0.35)), "bold"),
                )

        left, top, right, bottom = bounds(spec.goal)
        canvas.create_rectangle(
            left + cell * 0.12,
            top + cell * 0.12,
            right - cell * 0.12,
            bottom - cell * 0.12,
            fill="#fff4b5",
            outline="#b58b00",
            width=2,
        )
        canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text="★",
            fill="#9a7200",
            font=("Segoe UI Symbol", max(10, int(cell * 0.45))),
        )

        left, top, right, bottom = bounds(frame.position)
        canvas.create_oval(
            left + cell * 0.18,
            top + cell * 0.18,
            right - cell * 0.18,
            bottom - cell * 0.18,
            fill="#202124",
            outline="#ffffff",
            width=2,
        )
        canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text="A",
            fill="#ffffff",
            font=("Segoe UI", max(8, int(cell * 0.3)), "bold"),
        )

    def _draw_chart(self) -> None:
        canvas = self.chart
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        margin = 12
        canvas.create_line(margin, height - margin, width - margin, height - margin)
        canvas.create_line(margin, margin, margin, height - margin)
        if len(self.success_history) < 2:
            return
        values = self.success_history[-300:]
        points: list[float] = []
        for index, value in enumerate(values):
            x = margin + (width - 2 * margin) * index / max(1, len(values) - 1)
            y = height - margin - (height - 2 * margin) * value
            points.extend((x, y))
        canvas.create_line(*points, fill="#3f6fba", width=2, smooth=True)
        canvas.create_text(
            width - margin,
            margin,
            text="100%",
            anchor="ne",
            fill="#666666",
        )

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > 500:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def launch_escape_gui() -> None:
    import tkinter as tk

    root = tk.Tk()
    EscapeTrainingApp(root)
    root.mainloop()
