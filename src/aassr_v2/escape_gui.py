from __future__ import annotations

import queue
import threading
import traceback
from typing import Any

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
        self.mode: TrainingMode | None = None
        self.latest_frame: EscapeRenderFrame | None = None
        self.history: list[float] = []

        self.episodes_var = tk.StringVar(value="2000")
        self.seed_var = tk.StringVar(value="7")
        self.colors_var = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value="실행 방식을 선택하세요.")
        self.episode_var = tk.StringVar(value="Episode 0 / 0")
        self.success_var = tk.StringVar(value="성공률 0.0%")
        self.rolling_var = tk.StringVar(value="최근 100회 0.0%")
        self.epsilon_var = tk.StringVar(value="ε 0.000")
        self.model_var = tk.StringVar(value="Imagination 대기")
        self.inventory_var = tk.StringVar(value="보유 열쇠: 없음")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build()
        self.root.after(30, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        tk = self.tk
        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(outer, text="Full AASSR 학습 설정", padding=12)
        controls.pack(fill="x")
        fields = ttk.Frame(controls)
        fields.pack(side="left", fill="x", expand=True)
        for column, (label, variable, width) in enumerate(
            (
                ("Episodes", self.episodes_var, 10),
                ("Seed", self.seed_var, 10),
                ("색 수", self.colors_var, 7),
            )
        ):
            ttk.Label(fields, text=label).grid(row=0, column=column, sticky="w")
            if label == "색 수":
                widget = ttk.Combobox(
                    fields,
                    textvariable=variable,
                    values=("1", "2", "3"),
                    width=width,
                    state="readonly",
                )
            else:
                widget = ttk.Entry(fields, textvariable=variable, width=width)
            widget.grid(row=1, column=column, padx=(0, 12), sticky="w")

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

        ttk.Label(outer, textvariable=self.status_var).pack(fill="x", pady=(10, 4))
        ttk.Progressbar(
            outer,
            variable=self.progress_var,
            maximum=100.0,
        ).pack(fill="x", pady=(0, 12))

        metrics = ttk.Frame(outer)
        metrics.pack(fill="x", pady=(0, 12))
        for index, variable in enumerate(
            (
                self.episode_var,
                self.success_var,
                self.rolling_var,
                self.epsilon_var,
                self.model_var,
            )
        ):
            ttk.Label(metrics, textvariable=variable, anchor="center").grid(
                row=0, column=index, sticky="ew", padx=4
            )
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
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        ttk.Label(side_panel, textvariable=self.inventory_var).pack(
            fill="x", pady=(0, 8)
        )
        ttk.Label(side_panel, text="학습 이벤트").pack(anchor="w")
        self.log = tk.Text(
            side_panel,
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
        self._idle()

    def _config(self) -> EscapeTrainingConfig:
        episodes = int(self.episodes_var.get())
        colors = int(self.colors_var.get())
        return EscapeTrainingConfig(
            episodes=episodes,
            seed=int(self.seed_var.get()),
            color_count=colors,
            distractor_boxes=max(1, colors),
            max_steps=120 + 30 * colors,
            epsilon_decay_episodes=max(100, int(episodes * 0.75)),
            imagination_depth=4 + colors,
            imagination_beam_width=16,
        )

    def _start(self, mode: TrainingMode) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            config = self._config()
        except ValueError as exc:
            self.status_var.set(f"설정을 확인하세요: {exc}")
            return

        self.mode = mode
        self.latest_frame = None
        self.history.clear()
        self.stop_event = threading.Event()
        self.progress_var.set(0.0)
        self._clear_log()
        self._set_running(True)
        if mode is TrainingMode.LIVE:
            self.status_var.set("실시간 모드: 모든 primitive step을 렌더링합니다.")
            self._idle("실시간 렌더링 준비 중…")
        else:
            self.status_var.set("최대 속도 모드: sleep과 step 렌더링을 제거했습니다.")
            self._idle("최대 속도 모드\nGrid 렌더링 비활성화")

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
        self.status_var.set("현재 episode를 중단하고 있습니다…")

    def _set_running(self, running: bool) -> None:
        self.live_button.configure(state="disabled" if running else "normal")
        self.fast_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _poll(self) -> None:
        for _ in range(200):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "frame" and isinstance(payload, EscapeRenderFrame):
                self._frame(payload)
            elif kind == "complete" and isinstance(payload, EscapeTrainingSummary):
                self._complete(payload)
            elif kind == "error":
                self._error(str(payload))
        self.root.after(30, self._poll)

    def _frame(self, frame: EscapeRenderFrame) -> None:
        self.latest_frame = frame
        self.progress_var.set(100.0 * frame.episode / max(1, frame.total_episodes))
        self.episode_var.set(f"Episode {frame.episode:,} / {frame.total_episodes:,}")
        self.success_var.set(
            f"성공률 {frame.total_successes / max(1, frame.episode):.1%}"
        )
        self.rolling_var.set(f"최근 100회 {frame.rolling_success:.1%}")
        self.epsilon_var.set(f"ε {frame.epsilon:.3f}")
        self.model_var.set(
            f"Imagination {frame.imagined_nodes} nodes"
            if frame.used_imagination
            else f"Prediction {frame.prediction_score:.3f}"
        )
        inventory = ", ".join(frame.inventory) if frame.inventory else "없음"
        self.inventory_var.set(f"보유 열쇠: {inventory}")

        if frame.episode_finished:
            self.history.append(frame.rolling_success)
            self._draw_chart()
            self._append(
                f"E{frame.episode:04d} step={frame.step:03d} "
                f"{'SUCCESS' if frame.success else 'timeout'} "
                f"rolling={frame.rolling_success:.3f} "
                f"imagined={frame.imagined_nodes} "
                f"gain={frame.holdout_gain:.6f}"
            )
        elif frame.event not in {"", "moved", "episode_started"}:
            imagination = " [IMAGINE]" if frame.used_imagination else ""
            self._append(
                f"E{frame.episode:04d}:{frame.step:03d} "
                f"{frame.event}{imagination}"
            )

        if frame.mode is TrainingMode.LIVE:
            self._draw_grid(frame)

    def _complete(self, summary: EscapeTrainingSummary) -> None:
        self._set_running(False)
        state = "중지됨" if summary.stopped else "완료"
        self.status_var.set(
            f"{state}: {summary.episodes:,} episodes, 성공 {summary.successes:,}회, "
            f"{summary.elapsed_seconds:.2f}초, policy entries {summary.policy_entries:,}, "
            f"Imagination {summary.imagination_decisions:,}회"
        )
        self._append(
            f"[{state}] success={summary.success_rate:.3f} "
            f"rolling={summary.rolling_success:.3f} "
            f"imagined_nodes={summary.imagined_nodes:,} "
            f"oracle_steps={summary.oracle_steps}"
        )
        if self.mode is TrainingMode.FAST:
            self._idle(
                "최대 속도 학습 완료\n"
                f"성공률 {summary.success_rate:.1%}\n"
                f"최근 성공률 {summary.rolling_success:.1%}\n"
                f"경과 {summary.elapsed_seconds:.2f}초"
            )

    def _error(self, message: str) -> None:
        self._set_running(False)
        self.status_var.set("실행 중 오류가 발생했습니다.")
        self._append(message)

    def _draw_grid(self, frame: EscapeRenderFrame) -> None:
        canvas = self.canvas
        canvas.delete("all")
        spec = frame.spec
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        cell = min((width - 30) / spec.width, (height - 30) / spec.height)
        ox = (width - cell * spec.width) / 2
        oy = (height - cell * spec.height) / 2

        def box(position: tuple[int, int]) -> tuple[float, float, float, float]:
            x, y = position
            left = ox + x * cell
            top = oy + y * cell
            return left, top, left + cell, top + cell

        for y in range(spec.height):
            for x in range(spec.width):
                canvas.create_rectangle(
                    *box((x, y)), fill="#f5f2ea", outline="#dedbd2"
                )
        for position in spec.walls:
            canvas.create_rectangle(*box(position), fill="#333842", outline="#20242b")
        for position, color in spec.doors:
            left, top, right, bottom = box(position)
            fill = DOOR_COLORS.get(color, "#8d6e63")
            if position in frame.open_doors:
                canvas.create_rectangle(
                    left + cell * 0.12,
                    top + cell * 0.12,
                    right - cell * 0.12,
                    bottom - cell * 0.12,
                    outline=fill,
                    width=max(2, int(cell * 0.08)),
                )
            else:
                canvas.create_rectangle(
                    left + cell * 0.08,
                    top,
                    right - cell * 0.08,
                    bottom,
                    fill=fill,
                    outline="#222222",
                )
        for item in spec.boxes:
            left, top, right, bottom = box(item.position)
            opened = item.box_id in frame.open_boxes
            canvas.create_rectangle(
                left + cell * 0.18,
                top + cell * 0.22,
                right - cell * 0.18,
                bottom - cell * 0.18,
                fill="#c7aa82" if opened else "#8d6e4f",
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
            if opened:
                canvas.create_text(
                    (left + right) / 2,
                    (top + bottom) / 2,
                    text="✓",
                    fill=DOOR_COLORS.get(item.key_color or "", "#555555"),
                    font=("Segoe UI", max(9, int(cell * 0.35)), "bold"),
                )
        left, top, right, bottom = box(spec.goal)
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
        left, top, right, bottom = box(frame.position)
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
        if len(self.history) < 2:
            return
        values = self.history[-300:]
        points: list[float] = []
        for index, value in enumerate(values):
            points.extend(
                (
                    margin + (width - 2 * margin) * index / (len(values) - 1),
                    height - margin - (height - 2 * margin) * value,
                )
            )
        canvas.create_line(*points, fill="#3f6fba", width=2, smooth=True)

    def _redraw(self) -> None:
        if self.latest_frame is not None and self.mode is TrainingMode.LIVE:
            self._draw_grid(self.latest_frame)
        elif self.mode is None:
            self._idle()

    def _idle(self, text: str = "실행 방식을 선택하세요.") -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            max(1, self.canvas.winfo_width()) / 2,
            max(1, self.canvas.winfo_height()) / 2,
            text=text,
            fill="#555555",
            justify="center",
            font=("Segoe UI", 15, "bold"),
        )

    def _append(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 500:
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
