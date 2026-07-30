from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .escape_imagination_capture import EscapeImaginationEvent


def _action_signature(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("signature", ""))
    return ""


def _short(text: str, limit: int = 25) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class EscapeImaginationViewer:
    """A separate, navigable window for every captured imagination tree."""

    def __init__(self, parent: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.window = tk.Toplevel(parent)
        self.window.title("AASSR Imagination Viewer")
        self.window.geometry("1280x820")
        self.window.minsize(980, 650)
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        self.history: list[EscapeImaginationEvent] = []
        self.index = -1
        self.maximum_history = 2_000
        self.auto_follow_var = tk.BooleanVar(value=True)
        self.title_var = tk.StringVar(value="상상 대기 중")
        self.meta_var = tk.StringVar(
            value="Prophecy 커버리지가 충분한 비랜덤 스텝에서 트리가 생성됩니다."
        )
        self.counter_var = tk.StringVar(value="0 / 0")
        self._current_payload: Mapping[str, Any] | None = None

        self._build()
        self.window.after(50, self._redraw)

    def _build(self) -> None:
        tk = self.tk
        ttk = self.ttk
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, textvariable=self.title_var, font=("Segoe UI", 13, "bold")).pack(
            side="left"
        )
        ttk.Label(header, textvariable=self.meta_var).pack(side="left", padx=(18, 0))

        controls = ttk.Frame(header)
        controls.pack(side="right")
        ttk.Button(controls, text="◀ 이전", command=self.previous).grid(row=0, column=0, padx=3)
        ttk.Label(controls, textvariable=self.counter_var, width=12, anchor="center").grid(
            row=0, column=1, padx=3
        )
        ttk.Button(controls, text="다음 ▶", command=self.next).grid(row=0, column=2, padx=3)
        ttk.Checkbutton(
            controls,
            text="최신 상상 자동 추적",
            variable=self.auto_follow_var,
            command=self._auto_follow_changed,
        ).grid(row=0, column=3, padx=(12, 0))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        tree_tab = ttk.Frame(notebook, padding=6)
        table_tab = ttk.Frame(notebook, padding=6)
        raw_tab = ttk.Frame(notebook, padding=6)
        notebook.add(tree_tab, text="상상 트리")
        notebook.add(table_tab, text="전체 노드 표")
        notebook.add(raw_tab, text="선택 노드 상세")

        body = ttk.Panedwindow(tree_tab, orient="horizontal")
        body.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(body)
        side = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(canvas_frame, weight=4)
        body.add(side, weight=2)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#fbfbfb",
            highlightthickness=1,
            highlightbackground="#d0d0d0",
        )
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        ttk.Label(side, text="루트 행동 평가", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self.evaluations = ttk.Treeview(
            side,
            columns=("chosen", "value", "leaves", "path"),
            show="headings",
            height=11,
        )
        for column, label, width in (
            ("chosen", "선택", 48),
            ("value", "집계 가치", 90),
            ("leaves", "leaf 수", 65),
            ("path", "최선 경로", 260),
        ):
            self.evaluations.heading(column, text=label)
            self.evaluations.column(column, width=width, anchor="center" if column != "path" else "w")
        self.evaluations.pack(fill="x", pady=(4, 10))

        ttk.Label(side, text="범례", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        legend = (
            "금색: 최종 선택된 최선 경로\n"
            "파랑: 계속 확장된 예측 상태\n"
            "회색: 깊이/신뢰도/반복으로 종료된 상태\n"
            "V: 누적 가치, C: 누적 신뢰도"
        )
        ttk.Label(side, text=legend, justify="left").pack(anchor="w", pady=(4, 10))

        ttk.Label(side, text="현재 선택 노드", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self.node_summary = tk.Text(side, height=15, wrap="word", state="disabled")
        self.node_summary.pack(fill="both", expand=True, pady=(4, 0))

        self.nodes_table = ttk.Treeview(
            table_tab,
            columns=("id", "parent", "depth", "action", "value", "confidence", "terminal", "position"),
            show="headings",
        )
        for column, label, width in (
            ("id", "ID", 55),
            ("parent", "부모", 55),
            ("depth", "깊이", 55),
            ("action", "부모에서 온 행동", 270),
            ("value", "누적 가치", 95),
            ("confidence", "누적 신뢰도", 95),
            ("terminal", "종료 이유", 120),
            ("position", "예측 위치", 100),
        ):
            self.nodes_table.heading(column, text=label)
            self.nodes_table.column(column, width=width, anchor="center" if column != "action" else "w")
        table_scroll = ttk.Scrollbar(table_tab, orient="vertical", command=self.nodes_table.yview)
        self.nodes_table.configure(yscrollcommand=table_scroll.set)
        self.nodes_table.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")
        self.nodes_table.bind("<<TreeviewSelect>>", self._table_selected)

        self.raw_text = tk.Text(raw_tab, wrap="none", font=("Consolas", 10), state="disabled")
        raw_y = ttk.Scrollbar(raw_tab, orient="vertical", command=self.raw_text.yview)
        raw_x = ttk.Scrollbar(raw_tab, orient="horizontal", command=self.raw_text.xview)
        self.raw_text.configure(yscrollcommand=raw_y.set, xscrollcommand=raw_x.set)
        self.raw_text.grid(row=0, column=0, sticky="nsew")
        raw_y.grid(row=0, column=1, sticky="ns")
        raw_x.grid(row=1, column=0, sticky="ew")
        raw_tab.rowconfigure(0, weight=1)
        raw_tab.columnconfigure(0, weight=1)

    def reset(self) -> None:
        self.history.clear()
        self.index = -1
        self._current_payload = None
        self.title_var.set("상상 대기 중")
        self.meta_var.set("조건을 만족한 첫 Imagination 트리를 기다리고 있습니다.")
        self.counter_var.set("0 / 0")
        self._clear_views()
        self.window.deiconify()
        self.window.lift()

    def add_event(self, event: EscapeImaginationEvent) -> None:
        self.history.append(event)
        if len(self.history) > self.maximum_history:
            overflow = len(self.history) - self.maximum_history
            del self.history[:overflow]
            self.index = max(-1, self.index - overflow)
        if self.auto_follow_var.get() or self.index < 0:
            self.index = len(self.history) - 1
            self._display_current()

    def previous(self) -> None:
        if not self.history:
            return
        self.auto_follow_var.set(False)
        self.index = max(0, self.index - 1)
        self._display_current()

    def next(self) -> None:
        if not self.history:
            return
        self.index = min(len(self.history) - 1, self.index + 1)
        self._display_current()

    def _auto_follow_changed(self) -> None:
        if self.auto_follow_var.get() and self.history:
            self.index = len(self.history) - 1
            self._display_current()

    def _display_current(self) -> None:
        if not (0 <= self.index < len(self.history)):
            return
        event = self.history[self.index]
        self._current_payload = event.payload
        position = event.root_position if event.root_position is not None else "?"
        self.title_var.set(
            f"Imagination #{event.sequence:,} · root tick {event.root_step:,} · 위치 {position}"
        )
        self.meta_var.set(
            f"선택 {event.chosen_action} · 노드 {event.node_count:,} · "
            f"확장 {event.expanded_nodes:,} · 최대 깊이 {event.maximum_depth}"
        )
        self.counter_var.set(f"{self.index + 1:,} / {len(self.history):,}")
        self._fill_evaluations(event.payload)
        self._fill_nodes(event.payload)
        self._redraw()

    def _clear_views(self) -> None:
        self.canvas.delete("all")
        for item in self.evaluations.get_children():
            self.evaluations.delete(item)
        for item in self.nodes_table.get_children():
            self.nodes_table.delete(item)
        self._set_text(self.node_summary, "")
        self._set_text(self.raw_text, "")

    def _fill_evaluations(self, payload: Mapping[str, Any]) -> None:
        for item in self.evaluations.get_children():
            self.evaluations.delete(item)
        for evaluation in payload.get("root_evaluations", []):
            if not isinstance(evaluation, Mapping):
                continue
            action = _action_signature(evaluation.get("action"))
            chosen = "✓" if evaluation.get("chosen") else ""
            path = " → ".join(str(part) for part in evaluation.get("best_path", []))
            self.evaluations.insert(
                "",
                "end",
                values=(
                    chosen,
                    f"{float(evaluation.get('aggregate_value', 0.0)):.4f}",
                    len(evaluation.get("leaf_values", [])),
                    _short(path, 70),
                ),
                text=action,
            )

    def _fill_nodes(self, payload: Mapping[str, Any]) -> None:
        for item in self.nodes_table.get_children():
            self.nodes_table.delete(item)
        for node in payload.get("nodes", []):
            if not isinstance(node, Mapping):
                continue
            state = node.get("state", {})
            metadata = state.get("metadata", {}) if isinstance(state, Mapping) else {}
            position = metadata.get("position", "") if isinstance(metadata, Mapping) else ""
            action = _action_signature(node.get("action_from_parent"))
            self.nodes_table.insert(
                "",
                "end",
                iid=str(node.get("node_id")),
                values=(
                    node.get("node_id"),
                    node.get("parent_id"),
                    node.get("depth"),
                    action,
                    f"{float(node.get('cumulative_value', 0.0)):.4f}",
                    f"{float(node.get('cumulative_confidence', 0.0)):.4f}",
                    node.get("terminal_reason") or "계속",
                    position,
                ),
            )
        if self.nodes_table.exists("0"):
            self.nodes_table.selection_set("0")
            self._show_node(0)

    def _table_selected(self, _event: object) -> None:
        selected = self.nodes_table.selection()
        if selected:
            self._show_node(int(selected[0]))

    def _show_node(self, node_id: int) -> None:
        payload = self._current_payload
        if payload is None:
            return
        nodes = payload.get("nodes", [])
        node = next(
            (
                item
                for item in nodes
                if isinstance(item, Mapping) and int(item.get("node_id", -1)) == node_id
            ),
            None,
        )
        if node is None:
            return
        import json

        summary = (
            f"node={node.get('node_id')} parent={node.get('parent_id')} depth={node.get('depth')}\n"
            f"action={_action_signature(node.get('action_from_parent')) or '(root)'}\n"
            f"root_action={_action_signature(node.get('root_action')) or '(none)'}\n"
            f"value={float(node.get('cumulative_value', 0.0)):.6f}\n"
            f"step_confidence={float(node.get('step_confidence', 0.0)):.6f}\n"
            f"cumulative_confidence={float(node.get('cumulative_confidence', 0.0)):.6f}\n"
            f"terminal={node.get('terminal_reason') or '계속 확장'}\n"
            f"action_path={' → '.join(str(part) for part in node.get('action_path', []))}"
        )
        self._set_text(self.node_summary, summary)
        self._set_text(
            self.raw_text,
            json.dumps(node, ensure_ascii=False, indent=2, sort_keys=True),
        )

    def _best_path_node_ids(self, payload: Mapping[str, Any]) -> set[int]:
        evaluations = payload.get("root_evaluations", [])
        chosen = next(
            (
                item
                for item in evaluations
                if isinstance(item, Mapping) and bool(item.get("chosen"))
            ),
            None,
        )
        if chosen is None:
            return {0}
        nodes = {
            int(item.get("node_id")): item
            for item in payload.get("nodes", [])
            if isinstance(item, Mapping)
        }
        current = int(chosen.get("best_leaf_id", 0))
        path: set[int] = set()
        while current in nodes:
            path.add(current)
            parent = nodes[current].get("parent_id")
            if parent is None:
                break
            current = int(parent)
        path.add(0)
        return path

    def _redraw(self) -> None:
        payload = self._current_payload
        if payload is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                max(1, self.canvas.winfo_width()) / 2,
                max(1, self.canvas.winfo_height()) / 2,
                text="아직 실행된 Imagination이 없습니다.",
                fill="#666666",
                font=("Segoe UI", 14, "bold"),
            )
            return
        nodes = [item for item in payload.get("nodes", []) if isinstance(item, Mapping)]
        if not nodes:
            return
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for node in nodes:
            grouped[int(node.get("depth", 0))].append(node)
        for values in grouped.values():
            values.sort(key=lambda item: int(item.get("node_id", 0)))

        x_gap = 190
        y_gap = 115
        margin_x = 90
        margin_y = 65
        maximum_width = max(len(values) for values in grouped.values()) * x_gap + margin_x * 2
        maximum_depth = max(grouped)
        total_height = (maximum_depth + 1) * y_gap + margin_y * 2
        positions: dict[int, tuple[float, float]] = {}
        for depth, values in grouped.items():
            row_width = (len(values) - 1) * x_gap
            start_x = max(margin_x, (maximum_width - row_width) / 2)
            for index, node in enumerate(values):
                positions[int(node.get("node_id", 0))] = (
                    start_x + index * x_gap,
                    margin_y + depth * y_gap,
                )

        best_path = self._best_path_node_ids(payload)
        canvas = self.canvas
        canvas.delete("all")
        by_id = {int(node.get("node_id", 0)): node for node in nodes}
        for node_id, node in by_id.items():
            parent = node.get("parent_id")
            if parent is None or int(parent) not in positions:
                continue
            x1, y1 = positions[int(parent)]
            x2, y2 = positions[node_id]
            canvas.create_line(
                x1,
                y1 + 25,
                x2,
                y2 - 25,
                fill="#c39100" if node_id in best_path and int(parent) in best_path else "#9aa0a6",
                width=3 if node_id in best_path and int(parent) in best_path else 1,
                arrow="last",
            )

        for node_id, node in by_id.items():
            x, y = positions[node_id]
            terminal = node.get("terminal_reason")
            if node_id in best_path:
                fill = "#fff0b3"
                outline = "#bd8700"
                width = 3
            elif terminal:
                fill = "#eeeeee"
                outline = "#888888"
                width = 1
            else:
                fill = "#dcecff"
                outline = "#4775a8"
                width = 2
            tag = f"node_{node_id}"
            canvas.create_rectangle(
                x - 78,
                y - 29,
                x + 78,
                y + 29,
                fill=fill,
                outline=outline,
                width=width,
                tags=(tag,),
            )
            action = _short(_action_signature(node.get("action_from_parent")) or "ROOT", 20)
            label = (
                f"#{node_id} d={node.get('depth')} {action}\n"
                f"V={float(node.get('cumulative_value', 0.0)):.3f}  "
                f"C={float(node.get('cumulative_confidence', 0.0)):.3f}"
            )
            canvas.create_text(x, y, text=label, font=("Consolas", 8), tags=(tag,))
            canvas.tag_bind(tag, "<Button-1>", lambda _event, value=node_id: self._show_node(value))

        canvas.configure(scrollregion=(0, 0, maximum_width, total_height))

    @staticmethod
    def _set_text(widget: Any, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def close(self) -> None:
        try:
            self.window.destroy()
        except Exception:
            pass
