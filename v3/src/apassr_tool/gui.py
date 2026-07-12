from __future__ import annotations

from dataclasses import asdict
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Any

from .dmp import APASSRToolDMP, StepRecord
from .experiment import _objective_settings, _top_prophecy
from .novelty import NoveltyMemory
from .plugins import available_plugins, get_plugin, plugin_manifest_rows
from .policy import PolicyABC
from .prophecy import TableProphecyModel
from .reward import JuiceShopChallengeObserver
from .tools import ToolExecutor


class GuiObserver:
    def __init__(self, events: queue.Queue[tuple[str, object]], *, top_k: int = 20) -> None:
        self.events = events
        self.top_k = top_k

    def on_candidates_scored(self, *, scored, store, policy, prophecy, dmp) -> None:
        self.events.put(("module", {"active": "Imagination", "message": "후보 생성 -> PolicyABC/Prophecy/Imagination 평가"}))
        rows = []
        for candidate, score in scored[: self.top_k]:
            bindings = {kk.value: value for kk, value in candidate.bindings.items()}
            rows.append(
                {
                    "label": candidate.label,
                    "template": candidate.template.value,
                    "what": candidate.policy.what.value,
                    "how": candidate.policy.how.value,
                    "where": candidate.policy.where.value,
                    "tool": candidate.tool_call.tool.value,
                    "bindings": bindings,
                    **score,
                }
            )
        self.events.put(("candidates", rows))

    def on_step(self, *, record: StepRecord, result, store, policy, prophecy, dmp) -> None:
        self.events.put(("module", {"active": "Knowledge Update", "message": "행동 실행 -> 관측 파싱 -> 지식/정책/예언 갱신"}))
        self.events.put(
            (
                "step",
                {
                    "record": asdict(record),
                    "knowledge": store.rows(),
                    "policy": _policy_rows(policy),
                    "policy_table": _policy_table_rows(policy),
                    "prophecy": _prophecy_snapshot(prophecy),
                    "solved": {
                        "count": len(dmp.solved_challenges),
                        "items": list(dmp.solved_challenges),
                    },
                    "stdout": (result.stdout[:2000] if result is not None else ""),
                    "stderr": (result.stderr[:1000] if result is not None else ""),
                },
            )
        )


class TrainerThread(threading.Thread):
    def __init__(
        self,
        *,
        events: queue.Queue[tuple[str, object]],
        base_url: str,
        plugin_name: str,
        objective: str,
        episodes: int,
        step_limit: int,
        prefer_curl: bool,
        backend: str,
        reward_observer_name: str,
        pause_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.events = events
        self.base_url = base_url
        self.plugin_name = plugin_name
        self.objective = objective
        self.episodes = episodes
        self.step_limit = step_limit
        self.prefer_curl = prefer_curl
        self.backend = backend
        self.reward_observer_name = reward_observer_name
        self.pause_event = pause_event
        self.stop_event = stop_event

    def run(self) -> None:
        started_at = time.time()
        policy = PolicyABC()
        prophecy = TableProphecyModel()
        novelty = NoveltyMemory()
        plugin = get_plugin(self.plugin_name)
        reward_observer = plugin.reward_observer(self.reward_observer_name, self.base_url)
        config = _objective_settings(self.objective)
        observer = GuiObserver(self.events)

        try:
            for episode in range(self.episodes):
                if self.stop_event.is_set():
                    break
                executor = ToolExecutor(prefer_curl=self.prefer_curl, backend=self.backend)
                dmp = APASSRToolDMP(
                    base_url=self.base_url,
                    plugin=plugin,
                    executor=executor,
                    policy=policy,
                    reward_observer=reward_observer,
                    prophecy_model=prophecy,
                    novelty_memory=novelty,
                    novelty_reward=config["novelty_reward"],
                    novelty_score_weight=config["novelty_score_weight"],
                    knowledge_reward_cap=int(config["knowledge_reward_cap"]),
                    knowledge_reward_scale=config["knowledge_reward_scale"],
                    step_limit=self.step_limit,
                    observer=observer,
                )
                self.events.put(("episode_start", {"episode": episode, "episodes": self.episodes}))
                for step in range(self.step_limit):
                    if self.stop_event.is_set():
                        break
                    while self.pause_event.is_set() and not self.stop_event.is_set():
                        time.sleep(0.1)
                    candidate = dmp.choose_candidate()
                    if candidate is None:
                        break
                    record, _ = dmp.execute_candidate(step, candidate)
                    dmp.records.append(record)
                solved_count = 0
                challenge_total = 0
                solved_items: list[str] = []
                if isinstance(reward_observer, JuiceShopChallengeObserver):
                    solved_count = len(reward_observer.solved_keys)
                    challenge_total = reward_observer.challenge_total
                    solved_items = list(reward_observer.solved_keys)
                self.events.put(
                    (
                        "episode_end",
                        {
                            "episode": episode,
                            "steps": len(dmp.records),
                            "reward": sum(record.reward for record in dmp.records),
                            "new_kv": sum(record.new_kv for record in dmp.records),
                            "errors": sum(1 for record in dmp.records if record.status == 0 or record.status >= 400),
                            "solved_count": solved_count,
                            "challenge_total": challenge_total,
                            "solved_items": solved_items,
                            "prophecy_stats": len(prophecy.stats),
                            "elapsed_s": time.time() - started_at,
                        },
                    )
                )
        except Exception as exc:  # pragma: no cover - GUI safety net
            self.events.put(("error", str(exc)))
        finally:
            self.events.put(("done", {"elapsed_s": time.time() - started_at}))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("APASSR v3 Learning Monitor")
        self.geometry("1680x940")
        self.minsize(1320, 780)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: TrainerThread | None = None
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.episode_count = 0
        self.started_at = 0.0
        self._configure_style()
        self._build_ui()
        self.after(100, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f8fb")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f6f8fb", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#1b2a41", background="#f6f8fb")
        style.configure("ColumnBlue.TLabel", font=("Segoe UI", 14, "bold"), foreground="#0b57d0", background="#f6f8fb")
        style.configure("ColumnGreen.TLabel", font=("Segoe UI", 14, "bold"), foreground="#137333", background="#f6f8fb")
        style.configure("Section.TLabelframe", background="#ffffff", bordercolor="#c8d3e1", relief="solid")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#1b2a41", background="#ffffff")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#188038")
        style.map("Primary.TButton", background=[("active", "#137333")])
        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#d93025")
        style.map("Danger.TButton", background=[("active", "#b3261e")])
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25, fieldbackground="#ffffff", background="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), foreground="#1b2a41")

    def _build_ui(self) -> None:
        self.configure(bg="#f6f8fb")
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(18, 10, 18, 4))
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Label(header, text="APASSR v3 Learning Monitor", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="Knowledge -> Candidate Branches -> Intent -> Action -> Learning",
            foreground="#5f6b7a",
        ).pack(side="right")

        left = ttk.Frame(self, padding=8)
        left.grid(row=1, column=0, sticky="ns")
        left.configure(width=390)
        center = ttk.Frame(self, padding=8)
        center.grid(row=1, column=1, sticky="nsew")
        right = ttk.Frame(self, padding=8)
        right.grid(row=1, column=2, sticky="ns")
        right.configure(width=440)

        ttk.Label(left, text="왼쪽: 실행 제어 / 진행률 / 플러그인", style="ColumnBlue.TLabel").pack(anchor="center", pady=(0, 8))
        ttk.Label(center, text="중앙: 판단 과정", style="ColumnBlue.TLabel").pack(anchor="center", pady=(0, 8))
        ttk.Label(right, text="오른쪽: 현재 의도 / 지식 / 학습 상태", style="ColumnGreen.TLabel").pack(anchor="center", pady=(0, 8))

        self._build_controls(left)
        self._build_progress(left)
        self._build_plugin_view(left)
        self._build_module_flow(center)
        self._build_candidate_view(center)
        self._build_timeline(center)
        self._build_current_action(right)
        self._build_decision_breakdown(right)
        self._build_knowledge_view(right)
        self._build_policy_table(right)
        self._build_learning_view(right)
        self._build_flow_footer()

    def _build_flow_footer(self) -> None:
        footer = ttk.Frame(self, padding=(18, 4, 18, 10))
        footer.grid(row=2, column=0, columnspan=3, sticky="ew")
        ttk.Label(
            footer,
            text="흐름: 후보 생성 -> Imagination/Prophecy 평가 -> 행동 실행 -> 관측 파싱 -> Knowledge/Policy/Prophecy 업데이트",
            foreground="#345",
        ).pack(side="left")

    def _build_controls(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="1. Run Control: 실행 설정", padding=10, style="Section.TLabelframe")
        box.pack(fill="x", pady=(0, 8))
        self.base_url = tk.StringVar(value="http://127.0.0.1:3000")
        self.plugin = tk.StringVar(value="juice-shop-full" if "juice-shop-full" in available_plugins() else "web")
        self.objective = tk.StringVar(value="balanced")
        self.episodes = tk.IntVar(value=20)
        self.step_limit = tk.IntVar(value=80)
        self.reward_observer = tk.StringVar(value="juice-shop")
        self.prefer_curl = tk.BooleanVar(value=False)
        self.backend = tk.StringVar(value="local")

        _entry(box, "Target URL", self.base_url)
        _combo(box, "Plugin", self.plugin, available_plugins())
        _combo(box, "Objective", self.objective, ("balanced", "novelty", "weird"))
        _combo(box, "Reward", self.reward_observer, ("juice-shop", "none"))
        _entry(box, "Episodes", self.episodes)
        _entry(box, "Step limit", self.step_limit)
        _combo(box, "Backend", self.backend, ("local", "wsl"))
        ttk.Checkbutton(box, text="Prefer curl", variable=self.prefer_curl).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="▶ Start", command=self._start, style="Primary.TButton").pack(side="left", padx=(0, 4))
        ttk.Button(row, text="⏸ Pause", command=self._toggle_pause).pack(side="left", padx=4)
        ttk.Button(row, text="■ Stop", command=self._stop, style="Danger.TButton").pack(side="left", padx=4)

    def _build_progress(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="2. Progress: 학습 진행률", padding=10, style="Section.TLabelframe")
        box.pack(fill="x", pady=8)
        self.progress_text = tk.StringVar(value="idle")
        self.progress_episode = tk.StringVar(value="-")
        self.progress_solved = tk.StringVar(value="-")
        self.progress_reward = tk.StringVar(value="-")
        self.progress_kv = tk.StringVar(value="-")
        self.progress_eta = tk.StringVar(value="-")
        for label, variable in [
            ("Status", self.progress_text),
            ("Episode", self.progress_episode),
            ("Solved", self.progress_solved),
            ("Reward", self.progress_reward),
            ("New KV", self.progress_kv),
            ("ETA", self.progress_eta),
        ]:
            row = ttk.Frame(box)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=10, foreground="#5f6b7a").pack(side="left")
            ttk.Label(row, textvariable=variable, font=("Segoe UI", 10, "bold")).pack(side="left")

    def _build_plugin_view(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="3. Plugin Manifest: 사용 중인 능력", padding=10, style="Section.TLabelframe")
        box.pack(fill="both", expand=True, pady=8)
        self.plugin_text = tk.Text(box, width=44, height=16, wrap="word", bg="#ffffff", relief="flat", font=("Consolas", 9))
        self.plugin_text.pack(fill="both", expand=True)
        self._refresh_plugin_text()

    def _build_module_flow(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="0. Module Flow: 현재 실행 중인 APASSR 모듈", padding=10, style="Section.TLabelframe")
        box.pack(fill="x", pady=(0, 8))
        self.module_message = tk.StringVar(value="대기 중")
        self.module_canvas = tk.Canvas(box, height=105, bg="#ffffff", highlightthickness=0)
        self.module_canvas.pack(fill="x")
        ttk.Label(box, textvariable=self.module_message, foreground="#345").pack(anchor="w", pady=(4, 0))
        self._draw_module_flow(active="")

    def _build_candidate_view(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="1. Imagination Top-K Candidate Branches: 실행 전 후보 비교", padding=10, style="Section.TLabelframe")
        box.pack(fill="both", expand=True)
        columns = ("rank", "score", "policy", "imag", "support", "p_reward", "p_knowledge", "p_error", "template", "action")
        self.candidate_tree = ttk.Treeview(box, columns=columns, show="headings", height=17)
        widths = {
            "rank": 48,
            "score": 78,
            "policy": 78,
            "imag": 72,
            "support": 64,
            "p_reward": 78,
            "p_knowledge": 78,
            "p_error": 68,
            "template": 150,
            "action": 430,
        }
        headings = {
            "rank": "#",
            "score": "score",
            "policy": "policy",
            "imag": "imag",
            "support": "n",
            "p_reward": "예상 보상",
            "p_knowledge": "예상 지식",
            "p_error": "오류",
            "template": "template",
            "action": "action",
        }
        for column in columns:
            self.candidate_tree.heading(column, text=headings[column])
            self.candidate_tree.column(column, width=widths[column], anchor="w")
        self.candidate_tree.tag_configure("selected", background="#d1e7dd", foreground="#0f5132")
        self.candidate_tree.tag_configure("explore", background="#e8f0fe", foreground="#174ea6")
        self.candidate_tree.tag_configure("risk", background="#fce8e6", foreground="#a50e0e")
        self.candidate_tree.tag_configure("repeat", background="#fff4ce", foreground="#7a4d00")
        self.candidate_tree.pack(fill="both", expand=True)

    def _build_timeline(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="2. Action Timeline: 실제 실행 로그", padding=10, style="Section.TLabelframe")
        box.pack(fill="both", expand=True, pady=(8, 0))
        columns = ("ep", "step", "status", "reward", "kv", "solved", "template", "action")
        self.timeline = ttk.Treeview(box, columns=columns, show="headings", height=12)
        for column, width in {
            "ep": 45,
            "step": 55,
            "status": 65,
            "reward": 70,
            "kv": 50,
            "solved": 65,
            "template": 155,
            "action": 520,
        }.items():
            self.timeline.heading(column, text=column)
            self.timeline.column(column, width=width, anchor="w")
        self.timeline.tag_configure("success", background="#d1e7dd", foreground="#0f5132")
        self.timeline.tag_configure("knowledge", background="#e8f0fe", foreground="#174ea6")
        self.timeline.tag_configure("error", background="#fce8e6", foreground="#a50e0e")
        self.timeline.tag_configure("neutral", background="#ffffff", foreground="#1b2a41")
        self.timeline.pack(fill="both", expand=True)

    def _build_current_action(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="1. Current Intent: 현재 선택 의도", padding=10, style="Section.TLabelframe")
        box.pack(fill="x", pady=(0, 8))
        self.current_text = tk.Text(box, width=50, height=11, wrap="word", bg="#ffffff", relief="flat", font=("Consolas", 9))
        self.current_text.pack(fill="x")

    def _build_decision_breakdown(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="1-1. Decision Breakdown: 점수 분해", padding=10, style="Section.TLabelframe")
        box.pack(fill="x", pady=(0, 8))
        columns = ("factor", "value")
        self.breakdown_tree = ttk.Treeview(box, columns=columns, show="headings", height=8)
        self.breakdown_tree.heading("factor", text="factor")
        self.breakdown_tree.heading("value", text="value")
        self.breakdown_tree.column("factor", width=170, anchor="w")
        self.breakdown_tree.column("value", width=110, anchor="w")
        self.breakdown_tree.tag_configure("strong", background="#d1e7dd", foreground="#0f5132")
        self.breakdown_tree.tag_configure("medium", background="#e8f0fe", foreground="#174ea6")
        self.breakdown_tree.tag_configure("warning", background="#fff4ce", foreground="#7a4d00")
        self.breakdown_tree.tag_configure("danger", background="#fce8e6", foreground="#a50e0e")
        self.breakdown_tree.pack(fill="x")

    def _build_knowledge_view(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="2. Knowledge Store: KK별 발견 값", padding=10, style="Section.TLabelframe")
        box.pack(fill="both", expand=True, pady=8)
        columns = ("kk", "count", "latest")
        self.knowledge_tree = ttk.Treeview(box, columns=columns, show="headings", height=12)
        for column in columns:
            self.knowledge_tree.heading(column, text=column)
            self.knowledge_tree.column(column, width=120 if column != "latest" else 230, anchor="w")
        self.knowledge_tree.tag_configure("rich", background="#d1e7dd", foreground="#0f5132")
        self.knowledge_tree.tag_configure("new", background="#e8f0fe", foreground="#174ea6")
        self.knowledge_tree.pack(fill="both", expand=True)

    def _build_policy_table(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="2-1. PolicyABC Table: WHAT / HOW / WHERE 확률", padding=10, style="Section.TLabelframe")
        box.pack(fill="both", expand=True, pady=(0, 8))
        columns = ("axis", "key", "probability")
        self.policy_tree = ttk.Treeview(box, columns=columns, show="headings", height=9)
        for column, width in {"axis": 70, "key": 190, "probability": 95}.items():
            self.policy_tree.heading(column, text=column)
            self.policy_tree.column(column, width=width, anchor="w")
        self.policy_tree.tag_configure("high", background="#d1e7dd", foreground="#0f5132")
        self.policy_tree.tag_configure("mid", background="#e8f0fe", foreground="#174ea6")
        self.policy_tree.tag_configure("low", background="#f8f9fa", foreground="#5f6b7a")
        self.policy_tree.pack(fill="both", expand=True)

    def _build_learning_view(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="3. Policy / Prophecy: 학습 상태", padding=10, style="Section.TLabelframe")
        box.pack(fill="both", expand=True)
        self.learning_text = tk.Text(box, width=50, height=15, wrap="word", bg="#ffffff", relief="flat", font=("Consolas", 9))
        self.learning_text.pack(fill="both", expand=True)

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.episode_count = 0
        self._clear_tree(self.candidate_tree)
        self._clear_tree(self.timeline)
        self.worker = TrainerThread(
            events=self.events,
            base_url=self.base_url.get(),
            plugin_name=self.plugin.get(),
            objective=self.objective.get(),
            episodes=int(self.episodes.get()),
            step_limit=int(self.step_limit.get()),
            prefer_curl=bool(self.prefer_curl.get()),
            backend=self.backend.get(),
            reward_observer_name=self.reward_observer.get(),
            pause_event=self.pause_event,
            stop_event=self.stop_event,
        )
        self.worker.start()
        self.started_at = time.time()
        self.progress_text.set("running")
        self.progress_episode.set(f"0/{int(self.episodes.get())}")
        self.progress_solved.set("-")
        self.progress_reward.set("0.00")
        self.progress_kv.set("0")
        self.progress_eta.set("calculating")

    def _toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.progress_text.set("running")
        else:
            self.pause_event.set()
            self.progress_text.set("paused")

    def _stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.progress_text.set("stopping...")

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(kind, payload)
        self.after(100, self._poll_events)

    def _handle_event(self, kind: str, payload: object) -> None:
        if kind == "episode_start":
            data = payload  # type: ignore[assignment]
            self.episode_count = int(data["episode"])  # type: ignore[index]
            self.progress_text.set(f"episode {int(data['episode']) + 1}/{data['episodes']}")  # type: ignore[index]
            self.progress_episode.set(f"{int(data['episode']) + 1}/{data['episodes']}")  # type: ignore[index]
        elif kind == "candidates":
            self._update_candidates(payload)  # type: ignore[arg-type]
        elif kind == "step":
            self._update_step(payload)  # type: ignore[arg-type]
        elif kind == "module":
            data = payload  # type: ignore[assignment]
            self.module_message.set(str(data["message"]))  # type: ignore[index]
            self._draw_module_flow(active=str(data["active"]))  # type: ignore[index]
        elif kind == "episode_end":
            data = payload  # type: ignore[assignment]
            self.progress_text.set(
                "episode={episode} steps={steps} reward={reward:.2f} kv={new_kv} "
                "errors={errors} solved={solved_count}/{challenge_total} prophecy={prophecy_stats}".format(**data)  # type: ignore[arg-type]
            )
            self.progress_episode.set(f"{int(data['episode']) + 1}/{int(self.episodes.get())}")  # type: ignore[index]
            total = data["challenge_total"] or "?"  # type: ignore[index]
            self.progress_solved.set(f"{data['solved_count']}/{total}")  # type: ignore[index]
            self.progress_reward.set(f"{data['reward']:.2f}")  # type: ignore[index]
            self.progress_kv.set(str(data["new_kv"]))  # type: ignore[index]
            remaining = max(int(self.episodes.get()) - (int(data["episode"]) + 1), 0)  # type: ignore[index]
            elapsed = max(float(data["elapsed_s"]), 0.001)  # type: ignore[index]
            completed = int(data["episode"]) + 1  # type: ignore[index]
            eta = elapsed / completed * remaining
            self.progress_eta.set(f"{eta:.1f}s")
        elif kind == "error":
            self.progress_text.set(f"error: {payload}")
        elif kind == "done":
            self.progress_text.set(f"done elapsed={payload['elapsed_s']:.1f}s")  # type: ignore[index]
            self._draw_module_flow(active="")

    def _update_candidates(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.candidate_tree)
        for rank, row in enumerate(rows, start=1):
            tags = (self._candidate_tag(rank, row),)
            self.candidate_tree.insert(
                "",
                "end",
                values=(
                    rank,
                    f"{float(row['final_score']):.4f}",
                    f"{float(row['policy_score']):.4f}",
                    f"{float(row['imagination_score']):.3f}",
                    int(row["imagination_support"]),
                    f"{float(row['predicted_reward']):.3f}",
                    f"{float(row['predicted_knowledge']):.3f}",
                    f"{float(row['predicted_error_rate']):.3f}",
                    row["template"],
                    row["label"],
                ),
                tags=tags,
            )
        if rows:
            self._update_breakdown(rows[0])
            self._set_text(
                self.current_text,
                "\n".join(
                    [
                        f"candidate: {rows[0]['label']}",
                        f"template: {rows[0]['template']}",
                        f"WHAT/HOW/WHERE: {rows[0]['what']} / {rows[0]['how']} / {rows[0]['where']}",
                        f"tool: {rows[0]['tool']}",
                        f"score: {float(rows[0]['final_score']):.5f}",
                        f"predicted reward: {float(rows[0]['predicted_reward']):.3f}",
                        f"predicted knowledge: {float(rows[0]['predicted_knowledge']):.3f}",
                        f"predicted error: {float(rows[0]['predicted_error_rate']):.3f}",
                        f"bindings: {rows[0]['bindings']}",
                    ]
                ),
            )

    def _update_step(self, data: dict[str, Any]) -> None:
        record = data["record"]
        tags = (self._timeline_tag(record),)
        self.timeline.insert(
            "",
            "end",
            values=(
                self.episode_count,
                record["step"],
                record["status"],
                f"{record['reward']:.2f}",
                record["new_kv"],
                record["solved_delta"],
                record["template"],
                record["action"],
            ),
            tags=tags,
        )
        self.timeline.yview_moveto(1.0)
        self._update_knowledge(data["knowledge"])
        self._update_policy_table(data["policy_table"])
        self._update_learning(data["policy"], data["prophecy"])

    def _update_knowledge(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.knowledge_tree)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["kk"]), []).append(row)
        for kk, items in sorted(grouped.items()):
            latest = items[-1]["value"] if items else ""
            tag = "rich" if len(items) >= 5 else "new"
            self.knowledge_tree.insert("", "end", values=(kk, len(items), latest), tags=(tag,))

    def _update_learning(self, policy_rows: list[str], prophecy: dict[str, Any]) -> None:
        lines = ["PolicyABC", *policy_rows, "", f"Prophecy stats: {prophecy['stat_count']}", "Top reward:"]
        for row in prophecy["top_reward"][:5]:
            lines.append(f"- {row['key']} r={row['reward_mean']:.2f} k={row['knowledge_mean']:.2f} n={row['count']}")
        lines.append("")
        lines.append("Top solved:")
        for row in prophecy["top_solved"][:5]:
            lines.append(f"- {row['key']} solved={row['solved_rate']:.2f} n={row['count']}")
        self._set_text(self.learning_text, "\n".join(lines))

    def _update_breakdown(self, row: dict[str, Any]) -> None:
        self._clear_tree(self.breakdown_tree)
        for label, key in [
            ("final_score", "final_score"),
            ("policy_score", "policy_score"),
            ("breadth", "breadth"),
            ("axis_breadth", "axis_breadth"),
            ("endpoint_breadth", "endpoint_breadth"),
            ("imagination_score", "imagination_score"),
            ("novelty_multiplier", "novelty_multiplier"),
            ("tried_count", "tried_count"),
        ]:
            value = row.get(key, "")
            if isinstance(value, float):
                text = f"{value:.5f}"
            else:
                text = str(value)
            self.breakdown_tree.insert("", "end", values=(label, text), tags=(self._breakdown_tag(label, value),))

    def _update_policy_table(self, rows: list[dict[str, Any]]) -> None:
        self._clear_tree(self.policy_tree)
        for row in rows:
            tag = self._policy_tag(float(row["probability"]))
            self.policy_tree.insert("", "end", values=(row["axis"], row["key"], f"{row['probability']:.4f}"), tags=(tag,))

    def _draw_module_flow(self, *, active: str) -> None:
        if not hasattr(self, "module_canvas"):
            return
        canvas = self.module_canvas
        canvas.delete("all")
        modules = [
            "Knowledge",
            "Candidate\nBinding",
            "PolicyABC",
            "Prophecy",
            "Imagination",
            "Execution",
            "Observation",
            "Knowledge\nUpdate",
        ]
        width = max(canvas.winfo_width(), 900)
        step = max((width - 60) // max(len(modules) - 1, 1), 95)
        y = 48
        for index, label in enumerate(modules):
            x = 30 + index * step
            normalized = label.replace("\n", " ")
            is_active = active.lower() in normalized.lower() or normalized.lower() in active.lower()
            fill, outline = _module_color(normalized, is_active)
            canvas.create_oval(x - 22, y - 22, x + 22, y + 22, fill=fill, outline=outline, width=2)
            canvas.create_text(x, y, text=str(index + 1), fill=outline, font=("Segoe UI", 10, "bold"))
            canvas.create_text(x, y + 39, text=label, fill="#1b2a41", font=("Segoe UI", 8), justify="center")
            if index < len(modules) - 1:
                canvas.create_line(x + 24, y, x + step - 24, y, arrow=tk.LAST, fill="#7b8794", width=2)

    def _candidate_tag(self, rank: int, row: dict[str, Any]) -> str:
        if rank == 1:
            return "selected"
        if float(row.get("predicted_error_rate", 0.0)) >= 0.5:
            return "risk"
        if int(row.get("tried_count", 0)) >= 3:
            return "repeat"
        if float(row.get("predicted_knowledge", 0.0)) > 0.0 or float(row.get("novelty_score", 0.0)) > 0.0:
            return "explore"
        return ""

    def _timeline_tag(self, record: dict[str, Any]) -> str:
        if int(record["status"]) == 0 or int(record["status"]) >= 400:
            return "error"
        if int(record["solved_delta"]) > 0:
            return "success"
        if int(record["new_kv"]) > 0:
            return "knowledge"
        return "neutral"

    def _breakdown_tag(self, label: str, value: object) -> str:
        numeric = float(value) if isinstance(value, int | float) else 0.0
        if label in {"final_score", "imagination_score", "policy_score"} and numeric >= 1.0:
            return "strong"
        if label in {"breadth", "axis_breadth", "endpoint_breadth"} and numeric > 0:
            return "medium"
        if label == "tried_count" and numeric >= 3:
            return "warning"
        if label == "predicted_error_rate" and numeric >= 0.5:
            return "danger"
        return ""

    def _policy_tag(self, probability: float) -> str:
        if probability >= 0.30:
            return "high"
        if probability >= 0.12:
            return "mid"
        return "low"

    def _refresh_plugin_text(self) -> None:
        lines: list[str] = []
        for row in plugin_manifest_rows():
            lines.append(f"{row['name']} [{row['domain']}]")
            lines.append(f"  {row['description']}")
            lines.append(f"  capabilities: {', '.join(row['capabilities'])}")
            lines.append(f"  dependencies: {', '.join(row['dependencies'])}")
            lines.append("")
        self._set_text(self.plugin_text, "\n".join(lines))

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")


def _policy_rows(policy: PolicyABC) -> list[str]:
    rows: list[str] = []
    rows.append("WHAT: " + ", ".join(f"{key.value}={value:.3f}" for key, value in policy.what_probs.items()))
    rows.append("HOW: " + ", ".join(f"{key.value}={value:.3f}" for key, value in policy.how_probs.items()))
    rows.append("WHERE: " + ", ".join(f"{key.value}={value:.3f}" for key, value in policy.where_probs.items()))
    return rows


def _policy_table_rows(policy: PolicyABC) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis, table in [("WHAT", policy.what_probs), ("HOW", policy.how_probs), ("WHERE", policy.where_probs)]:
        for key, value in sorted(table.items(), key=lambda item: item[1], reverse=True):
            rows.append({"axis": axis, "key": key.value, "probability": value})
    return rows


def _prophecy_snapshot(prophecy: TableProphecyModel) -> dict[str, Any]:
    return {
        "stat_count": len(prophecy.stats),
        "top_reward": _top_prophecy(prophecy, by="reward"),
        "top_solved": _top_prophecy(prophecy, by="solved"),
    }


def _module_color(module: str, is_active: bool) -> tuple[str, str]:
    palette = {
        "Knowledge": ("#e8f0fe", "#174ea6"),
        "Candidate Binding": ("#e6f4ea", "#137333"),
        "PolicyABC": ("#fef7e0", "#b06000"),
        "Prophecy": ("#f3e8fd", "#8430ce"),
        "Imagination": ("#e0f2f1", "#00796b"),
        "Execution": ("#fce8e6", "#a50e0e"),
        "Observation": ("#e8f0fe", "#0b57d0"),
        "Knowledge Update": ("#d1e7dd", "#0f5132"),
    }
    fill, outline = palette.get(module, ("#f8f9fa", "#5f6b7a"))
    if is_active:
        return fill, outline
    return "#ffffff", outline


def _entry(parent: ttk.Frame, label: str, variable: tk.Variable) -> None:
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=label, width=12).pack(side="left")
    ttk.Entry(row, textvariable=variable, width=28).pack(side="left", fill="x", expand=True)


def _combo(parent: ttk.Frame, label: str, variable: tk.StringVar, values: tuple[str, ...]) -> None:
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text=label, width=12).pack(side="left")
    ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=26).pack(side="left", fill="x", expand=True)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
