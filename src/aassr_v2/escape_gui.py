from __future__ import annotations

import queue
import threading
import traceback
from typing import Any

from .escape_stats_gui import show_statistics_window
from .escape_training import (
    EscapeRenderFrame,
    EscapeTrainingConfig,
    EscapeTrainingSummary,
    TrainingMode,
    TrainingRuntime,
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
        self.root.geometry("1160x780")
        self.root.minsize(920, 660)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.runtime: TrainingRuntime | None = None
        self.mode: TrainingMode | None = None
        self.latest_frame: EscapeRenderFrame | None = None
        self.latest_summary: EscapeTrainingSummary | None = None
        self.history: list[float] = []

        self.episodes_var = tk.StringVar(value="2000")
        self.seed_var = tk.StringVar(value="7")
        self.colors_var = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value="실행 방식을 선택하세요.")
        self.episode_var = tk.StringVar(value="Episode 0 / 0")
        self.tick_var = tk.StringVar(value="현재 tick 0")
        self.score_var = tk.StringVar(value="최근 점수 0.000x")
        self.rolling_score_var = tk.StringVar(value="최근 100 평균 0.000x")
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
            command=lambda: self._select_mode(TrainingMode.LIVE),
        )
        self.live_button.grid(row=0, column=0, padx=5)
        self.fast_button = ttk.Button(
            buttons,
            text="안 보고 최대 속도",
            command=lambda: self._select_mode(TrainingMode.FAST),
        )
        self.fast_button.grid(row=0, column=1, padx=5)
        self.stop_button = ttk.Button(
            buttons,
            text="중지",
            command=self._stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=2, padx=5)
        self.statistics_button = ttk.Button(
            buttons,
            text="통계 창",
            command=self._show_statistics,
            state="disabled",
        )
        self.statistics_button.grid(row=0, column=3, padx=(12, 0))

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
                self.tick_var,
                self.score_var,
                self.rolling_score_var,
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
        ttk.Label(side_panel, text="최근 성공 점수").pack(anchor="w")
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
            epsilon_decay_episodes=max(100, int(episodes * 0.75)),
            imagination_depth=4 + colors,
            imagination_beam_width=16,
            save_episode_checkpoints=True,
        )

    def _select_mode(self, mode: TrainingMode) -> None:
        if self.worker is not None and self.worker.is_alive():
            if self.runtime is None:
                return
            self.runtime.set_mode(mode)
            self.mode = mode
            self._refresh_mode_buttons(running=True)
            if mode is TrainingMode.LIVE:
                self.status_var.set(
                    "실시간 모드로 전환됨: 현재 episode를 이어서 렌더링합니다."
                )
                if self.latest_frame is not None:
                    self._draw_grid(self.latest_frame)
            else:
                self.status_var.set(
                    "최대 속도 모드로 전환됨: 같은 episode와 학습 상태를 그대로 이어갑니다."
                )
                self._idle("최대 속도 모드\n학습과 전체 기록 저장은 계속 진행 중")
            return
        self._start(mode)

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
        if mode is TrainingMode.LIVE:
            self.status_var.set("실시간 모드: 모든 primitive step을 렌더링하고 저장합니다.")
            self._idle("실시간 렌더링 준비 중…")
        else:
            self.status_var.set("최대 속도 모드: 렌더링 없이 모든 step을 파일에 저장합니다.")
            self._idle("최대 속도 모드\nGrid 렌더링 비활성화\n전체 step 기록 저장 중")

        def worker() -> None:
            try:
                train_escape_agent(
                    config,
                    mode=mode,
                    runtime=self.runtime,
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
        self.status_var.set("현재 episode와 전체 세션을 중단하고 기록을 마무리합니다…")

    def _refresh_mode_buttons(self, *, running: bool) -> None:
        if not running:
            self.live_button.configure(state="normal")
            self.fast_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return
        self.live_button.configure(
            state="disabled" if self.mode is TrainingMode.LIVE else "normal"
        )
        self.fast_button.configure(
            state="disabled" if self.mode is TrainingMode.FAST else "normal"
        )
        self.stop_button.configure(state="normal")

    def _poll(self) -> None:
        for _ in range(300):
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
        self.mode = frame.mode
        self.progress_var.set(100.0 * frame.episode / max(1, frame.total_episodes))
        self.episode_var.set(f"Episode {frame.episode:,} / {frame.total_episodes:,}")
        self.tick_var.set(f"현재 tick {frame.step:,}")
        self.score_var.set(f"최근 점수 {frame.episode_score:.3f}x")
        self.rolling_score_var.set(f"최근 100 평균 {frame.rolling_score:.3f}x")
        self.epsilon_var.set(f"ε {frame.epsilon:.3f}")
        self.model_var.set(
            f"Imagination {frame.imagined_nodes} nodes"
            if frame.used_imagination
            else f"Prediction {frame.prediction_score:.3f}"
        )
        inventory = ", ".join(frame.inventory) if frame.inventory else "없음"
        self.inventory_var.set(f"보유 열쇠: {inventory}")

        if frame.episode_finished and frame.episode_score > 0.0:
            self.history.append(frame.episode_score)
            self._draw_chart()
            self._append(
                f"E{frame.episode:04d} step={frame.step:,} SUCCESS "
                f"score={frame.episode_score:.4f}x "
                f"rolling_score={frame.rolling_score:.4f}x "
                f"imagined={frame.imagined_nodes} "
                f"gain={frame.holdout_gain:.6f}"
            )
        elif frame.event not in {"", "moved", "episode_started"}:
            imagination = " [IMAGINE]" if frame.used_imagination else ""
            self._append(
                f"E{frame.episode:04d}:{frame.step:,} "
                f"{frame.event}{imagination}"
            )

        if frame.mode is TrainingMode.LIVE:
            self._draw_grid(frame)

    def _complete(self, summary: EscapeTrainingSummary) -> None:
        self.latest_summary = summary
        self._refresh_mode_buttons(running=False)
        self.statistics_button.configure(state="normal")
        state = "중지됨" if summary.stopped else "완료"
        self.status_var.set(
            f"{state}: {summary.episodes:,} episodes, {summary.total_steps:,} ticks, "
            f"평균 점수 {summary.mean_score:.4f}x, {summary.elapsed_seconds:.2f}초"
        )
        self._append(
            f"[{state}] success={summary.success_rate:.3f} "
            f"mean_score={summary.mean_score:.4f}x "
            f"rolling_score={summary.rolling_score:.4f}x "
            f"imagined_nodes={summary.imagined_nodes:,} "
            f"oracle_steps={summary.oracle_steps}"
        )
        self._append(f"[저장 완료] {summary.output_dir}")
        if self.mode is TrainingMode.FAST:
            self._idle(
                "최대 속도 학습 완료\n"
                f"총 {summary.total_steps:,} ticks\n"
                f"평균 점수 {summary.mean_score:.4f}x\n"
                f"경과 {summary.elapsed_seconds:.2f}초\n\n"
                "통계 창이 자동으로 열립니다."
            )
        self.root.after(100, self._show_statistics)

    def _show_statistics(self) -> None:
        if self.latest_summary is None:
            return
        show_statistics_window(self.root, self.latest_summary)

    def _error(self, message: str) -> None:
        self._refresh_mode_buttons(running=False)
        self.status_var.set("실행 중 오류가 발생했습니다. 오류 기록도 결과 폴더에 저장됩니다.")
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
        minimum = 1.0
        maximum = max(2.0, max(values))
        span = max(0.001, maximum - minimum)
        points: list[float] = []
        for index, value in enumerate(values):
            points.extend(
                (
                    margin + (width - 2 * margin) * index / (len(values) - 1),
                    height
                    - margin
                    - (height - 2 * margin) * (value - minimum) / span,
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
