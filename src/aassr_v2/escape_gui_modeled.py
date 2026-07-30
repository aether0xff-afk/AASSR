from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import threading
import traceback
from typing import Any

from .escape_gui_visualized import VisualizedEscapeTrainingApp
from .escape_imagination_capture import EscapeImaginationEvent
from .escape_training import EscapeRenderFrame, EscapeTrainingSummary, TrainingMode, TrainingRuntime
from .escape_training_modeled import train_escape_agent
from .model_io import MODEL_EXTENSION, ModelLoadInfo, ModelManagedAgent


class ModeledEscapeTrainingApp(VisualizedEscapeTrainingApp):
    """GridWorld + Imagination GUI with portable model save/load controls."""

    def __init__(self, root: Any) -> None:
        self.managed_agent: ModelManagedAgent | None = None
        self.selected_model_path: Path | None = None
        self.active_config: object | None = None
        self.model_path_var = None
        self.load_model_button = None
        self.save_model_button = None
        super().__init__(root)
        self.root.title("AASSR Escape GridWorld Trainer · Model Save/Load")

    def _build(self) -> None:
        super()._build()
        tk = self.tk
        ttk = self.ttk
        self.model_path_var = tk.StringVar(value="불러올 모델: 새 모델")
        button_parent = self.live_button.master
        self.load_model_button = ttk.Button(
            button_parent,
            text="모델 불러오기",
            command=self._choose_model,
        )
        self.load_model_button.grid(row=1, column=0, padx=5, pady=(9, 0), sticky="ew")
        self.save_model_button = ttk.Button(
            button_parent,
            text="현재 모델 저장",
            command=self._save_current_model,
            state="disabled",
        )
        self.save_model_button.grid(row=1, column=1, padx=5, pady=(9, 0), sticky="ew")
        ttk.Button(
            button_parent,
            text="새 모델로 시작",
            command=self._clear_selected_model,
        ).grid(row=1, column=2, padx=5, pady=(9, 0), sticky="ew")
        ttk.Label(
            self.root,
            textvariable=self.model_path_var,
            anchor="w",
        ).pack_forget()
        # Put the model source directly below the main control box.
        controls = button_parent.master
        ttk.Label(controls, textvariable=self.model_path_var, anchor="e").pack(
            side="bottom", fill="x", pady=(8, 0)
        )

    def _choose_model(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="AASSR 모델 불러오기",
            filetypes=(("AASSR model", f"*{MODEL_EXTENSION}"), ("All files", "*.*")),
        )
        if not selected:
            return
        self.selected_model_path = Path(selected)
        self.model_path_var.set(f"불러올 모델: {self.selected_model_path}")
        self.status_var.set("모델을 선택했습니다. 학습 시작 시 새 세션에 복원합니다.")

    def _clear_selected_model(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("학습 중에는 불러올 모델을 바꿀 수 없습니다.")
            return
        self.selected_model_path = None
        self.model_path_var.set("불러올 모델: 새 모델")
        self.status_var.set("새 모델로 시작하도록 설정했습니다.")

    def _save_current_model(self) -> None:
        from tkinter import filedialog

        agent = self.managed_agent
        if agent is None:
            self.status_var.set("저장할 모델이 아직 준비되지 않았습니다.")
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="현재 AASSR 모델 저장",
            initialfile=f"aassr_escape_{stamp}{MODEL_EXTENSION}",
            defaultextension=MODEL_EXTENSION,
            filetypes=(("AASSR model", f"*{MODEL_EXTENSION}"), ("All files", "*.*")),
        )
        if not selected:
            return
        destination = Path(selected)
        self.save_model_button.configure(state="disabled")
        self.status_var.set("현재 학습 스텝이 끝나는 즉시 일관된 모델을 저장합니다…")

        def save_worker() -> None:
            try:
                path = agent.save_model(
                    destination,
                    training_config=self.active_config,
                    label="manual GUI save",
                    notes="Saved from the live Escape GridWorld GUI",
                )
                self.events.put(("model_saved", str(path)))
            except Exception:
                self.events.put(("model_save_error", traceback.format_exc()))

        threading.Thread(target=save_worker, daemon=True).start()

    def _start(self, mode: TrainingMode) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            self.status_var.set(f"설정을 확인하세요: {exc}")
            return

        self.active_config = config
        self.managed_agent = None
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

        source = (
            f"저장 모델 {self.selected_model_path.name}에서 이어 학습"
            if self.selected_model_path is not None
            else "새 모델 학습"
        )
        if mode is TrainingMode.LIVE:
            self.status_var.set(f"실시간 모드 · {source}")
            self._idle("실시간 렌더링·Imagination·모델 준비 중…")
        else:
            self.status_var.set(f"최대 속도 모드 · {source}")
            self._idle(
                "최대 속도 모드\nGrid 렌더링 비활성화\n"
                "상상·학습 기록과 모델 저장 준비 중"
            )

        def agent_ready(agent: ModelManagedAgent, info: ModelLoadInfo | None) -> None:
            self.events.put(("agent_ready", (agent, info)))

        def worker() -> None:
            try:
                summary = train_escape_agent(
                    config,
                    mode=mode,
                    runtime=self.runtime,
                    on_frame=lambda frame: self.events.put(("frame", frame)),
                    on_imagination=self._receive_imagination,
                    on_complete=lambda result: self.events.put(("complete", result)),
                    on_agent_ready=agent_ready,
                    stop_event=self.stop_event,
                    load_model_path=self.selected_model_path,
                    auto_save_final_model=True,
                )
                automatic = Path(summary.output_dir) / "models" / "final.aassr-model.gz"
                self.events.put(("automatic_model_saved", str(automatic)))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _refresh_mode_buttons(self, *, running: bool) -> None:
        super()._refresh_mode_buttons(running=running)
        if self.load_model_button is None or self.save_model_button is None:
            return
        self.load_model_button.configure(state="disabled" if running else "normal")
        self.save_model_button.configure(
            state="normal" if self.managed_agent is not None else "disabled"
        )

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
            elif kind == "agent_ready" and isinstance(payload, tuple):
                agent, info = payload
                self.managed_agent = agent
                self.save_model_button.configure(state="normal")
                if isinstance(info, ModelLoadInfo):
                    self._append(
                        f"[모델 복원] {info.path} · 누적 {info.completed_episodes:,} episodes"
                    )
                    self.status_var.set(
                        f"모델 복원 완료: 누적 {info.completed_episodes:,} episodes부터 이어 학습"
                    )
                else:
                    self._append("[모델] 새 모델이 준비되었습니다.")
            elif kind == "model_saved":
                self.save_model_button.configure(state="normal")
                self.status_var.set(f"모델 저장 완료: {payload}")
                self._append(f"[모델 저장] {payload}")
            elif kind == "automatic_model_saved":
                self._append(f"[최종 모델 자동 저장] {payload}")
            elif kind == "model_save_error":
                self.save_model_button.configure(state="normal")
                self.status_var.set("모델 저장 중 오류가 발생했습니다.")
                self._append(str(payload))
            elif kind == "complete" and isinstance(payload, EscapeTrainingSummary):
                pending = self._take_pending_fast_imagination()
                if pending is not None:
                    self.imagination_viewer.add_event(pending)
                self._complete(payload)
                self._refresh_mode_buttons(running=False)
            elif kind == "error":
                self._error(str(payload))
                self._refresh_mode_buttons(running=False)

        pending = self._take_pending_fast_imagination()
        if pending is not None:
            self.imagination_viewer.add_event(pending)
        self.root.after(30, self._poll)


def launch_escape_gui() -> None:
    import tkinter as tk

    root = tk.Tk()
    ModeledEscapeTrainingApp(root)
    root.mainloop()
