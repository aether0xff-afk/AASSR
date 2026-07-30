from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .escape_reporting import EscapeEpisodeRecord, rolling_mean


PALETTE = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")


def _open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class LineChart:
    def __init__(
        self,
        parent: Any,
        *,
        title: str,
        y_label: str,
        series: Sequence[tuple[str, Sequence[float]]],
    ) -> None:
        import tkinter as tk

        self.title = title
        self.y_label = y_label
        self.series = [(label, [float(value) for value in values]) for label, values in series]
        self.canvas = tk.Canvas(parent, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(400, canvas.winfo_width())
        height = max(280, canvas.winfo_height())
        left, right, top, bottom = 76, 24, 56, 58
        plot_width = width - left - right
        plot_height = height - top - bottom
        all_values = [value for _, values in self.series for value in values]
        count = max((len(values) for _, values in self.series), default=0)
        if not all_values:
            canvas.create_text(width / 2, height / 2, text="표시할 데이터가 없습니다.")
            return
        minimum = min(all_values)
        maximum = max(all_values)
        if math.isclose(minimum, maximum):
            padding = max(1.0, abs(minimum) * 0.1)
            minimum -= padding
            maximum += padding
        else:
            padding = (maximum - minimum) * 0.08
            minimum -= padding
            maximum += padding

        def x_position(index: int) -> float:
            return left + plot_width * index / max(1, count - 1)

        def y_position(value: float) -> float:
            return top + plot_height * (maximum - value) / (maximum - minimum)

        canvas.create_text(
            width / 2,
            24,
            text=self.title,
            font=("Segoe UI", 16, "bold"),
        )
        for tick in range(6):
            fraction = tick / 5
            y = top + plot_height * fraction
            value = maximum - (maximum - minimum) * fraction
            canvas.create_line(left, y, width - right, y, fill="#e5e7eb")
            canvas.create_text(
                left - 8,
                y,
                text=f"{value:.4g}",
                anchor="e",
                font=("Segoe UI", 9),
            )
        canvas.create_line(left, top, left, height - bottom, fill="#111827")
        canvas.create_line(left, height - bottom, width - right, height - bottom, fill="#111827")
        canvas.create_text(
            18,
            height / 2,
            text=self.y_label,
            angle=90,
            font=("Segoe UI", 10),
        )
        canvas.create_text(
            width / 2,
            height - 16,
            text="Episode",
            font=("Segoe UI", 10),
        )
        if count:
            for tick in range(6):
                index = round((count - 1) * tick / 5)
                x = x_position(index)
                canvas.create_text(
                    x,
                    height - bottom + 18,
                    text=str(index + 1),
                    font=("Segoe UI", 9),
                )
        for series_index, (label, values) in enumerate(self.series):
            if not values:
                continue
            color = PALETTE[series_index % len(PALETTE)]
            points: list[float] = []
            for index, value in enumerate(values):
                points.extend((x_position(index), y_position(value)))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, smooth=False)
            elif len(points) == 2:
                x, y = points
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
            legend_x = left + series_index * 190
            canvas.create_line(legend_x, 43, legend_x + 24, 43, fill=color, width=4)
            canvas.create_text(
                legend_x + 30,
                43,
                text=label,
                anchor="w",
                font=("Segoe UI", 9),
            )


class BarChart:
    def __init__(
        self,
        parent: Any,
        *,
        title: str,
        items: Sequence[tuple[str, int]],
    ) -> None:
        import tkinter as tk

        self.title = title
        self.items = list(items[:25])
        self.canvas = tk.Canvas(parent, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(500, canvas.winfo_width())
        height = max(320, canvas.winfo_height())
        canvas.create_text(
            width / 2,
            24,
            text=self.title,
            font=("Segoe UI", 16, "bold"),
        )
        if not self.items:
            canvas.create_text(width / 2, height / 2, text="표시할 데이터가 없습니다.")
            return
        left, right, top, bottom = 230, 60, 56, 24
        row_height = max(18, min(32, (height - top - bottom) / len(self.items)))
        maximum = max(value for _, value in self.items) or 1
        plot_width = width - left - right
        for index, (label, value) in enumerate(self.items):
            y = top + index * row_height
            canvas.create_text(
                left - 8,
                y + row_height * 0.5,
                text=label[:36],
                anchor="e",
                font=("Consolas", 9),
            )
            bar_width = plot_width * value / maximum
            canvas.create_rectangle(
                left,
                y + row_height * 0.16,
                left + bar_width,
                y + row_height * 0.84,
                fill="#2563eb",
                outline="",
            )
            canvas.create_text(
                left + bar_width + 7,
                y + row_height * 0.5,
                text=f"{value:,}",
                anchor="w",
                font=("Segoe UI", 9),
            )


def _add_line_tab(
    notebook: Any,
    *,
    title: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[float]]],
) -> None:
    from tkinter import ttk

    frame = ttk.Frame(notebook, padding=8)
    notebook.add(frame, text=title)
    LineChart(frame, title=title, y_label=y_label, series=series)


def _add_bar_tab(
    notebook: Any,
    *,
    title: str,
    items: Sequence[tuple[str, int]],
) -> None:
    from tkinter import ttk

    frame = ttk.Frame(notebook, padding=8)
    notebook.add(frame, text=title)
    BarChart(frame, title=title, items=items)


def show_statistics_window(root: Any, summary: Any) -> Any:
    import tkinter as tk
    from tkinter import ttk

    records: tuple[EscapeEpisodeRecord, ...] = tuple(summary.episode_records)
    output_dir = Path(summary.output_dir)
    window = tk.Toplevel(root)
    window.title("AASSR Escape GridWorld — 세션 통계")
    window.geometry("1280x820")
    window.minsize(960, 640)

    header = ttk.Frame(window, padding=(12, 10))
    header.pack(fill="x")
    ttk.Label(
        header,
        text=f"세션 통계 · {summary.episodes:,} episodes · 총 {summary.total_steps:,} ticks",
        font=("Segoe UI", 15, "bold"),
    ).pack(side="left")
    ttk.Button(
        header,
        text="결과 폴더 열기",
        command=lambda: _open_path(output_dir),
    ).pack(side="right")

    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    overview = ttk.Frame(notebook, padding=14)
    notebook.add(overview, text="요약")
    summary_text = tk.Text(
        overview,
        wrap="word",
        state="normal",
        font=("Consolas", 10),
    )
    summary_text.pack(fill="both", expand=True)
    stats: Mapping[str, Any] = summary.statistics
    lines = [
        f"output_dir                : {output_dir}",
        f"status                    : {'stopped' if summary.stopped else 'completed'}",
        f"episodes                  : {summary.episodes:,}",
        f"successes                 : {summary.successes:,}",
        f"total ticks               : {summary.total_steps:,}",
        f"oracle shortest ticks     : {summary.oracle_steps:,}",
        f"total elapsed seconds     : {summary.elapsed_seconds:.6f}",
        f"mean episode ticks        : {stats['steps']['mean']:.6f}",
        f"median episode ticks      : {stats['steps']['median']:.6f}",
        f"min / max episode ticks   : {stats['steps']['minimum']:.0f} / {stats['steps']['maximum']:.0f}",
        f"mean score                : {summary.mean_score:.6f}x",
        f"rolling score             : {summary.rolling_score:.6f}x",
        f"mean episode duration     : {stats['durations_seconds']['mean']:.6f}s",
        f"policy entries            : {summary.policy_entries:,}",
        f"imagination decisions     : {summary.imagination_decisions:,}",
        f"imagined nodes            : {summary.imagined_nodes:,}",
        "",
        "correlations",
    ]
    for key, value in stats.get("correlations", {}).items():
        lines.append(f"  {key:30s}: {float(value): .6f}")
    lines.extend(
        [
            "",
            "saved files",
            "  session.json / world.json",
            "  steps.jsonl",
            "  episodes.csv / episodes.jsonl",
            "  mode_switches.jsonl",
            "  summary.json / summary.txt / statistics.json",
            "  checkpoints/episode_*.json.gz / latest.json.gz / final.json.gz",
            "  charts/*.svg",
            "  session.log",
        ]
    )
    summary_text.insert("1.0", "\n".join(lines))
    summary_text.configure(state="disabled")

    steps = [float(record.steps) for record in records]
    scores = [record.score for record in records]
    durations = [record.duration_seconds for record in records]
    efficiencies = [record.efficiency for record in records]
    predictions = [record.mean_prediction_score for record in records]
    holdout_before = [record.mean_holdout_before for record in records]
    holdout_after = [record.mean_holdout_after for record in records]
    holdout_gain = [record.mean_holdout_gain for record in records]
    intrinsic = [record.intrinsic_value_total for record in records]
    imagined_nodes = [float(record.imagined_nodes) for record in records]
    imagination_decisions = [float(record.imagination_decisions) for record in records]
    errors = [float(record.errors) for record in records]
    repeats = [float(record.repeated_actions) for record in records]
    live_seconds = [record.live_seconds for record in records]
    fast_seconds = [record.fast_seconds for record in records]

    _add_line_tab(
        notebook,
        title="에피소드별 스텝 수",
        y_label="Ticks",
        series=(("Steps", steps), ("Rolling mean (100)", rolling_mean(steps))),
    )
    _add_line_tab(
        notebook,
        title="점수",
        y_label="Multiplier",
        series=(("Score", scores), ("Rolling mean (100)", rolling_mean(scores))),
    )
    _add_line_tab(
        notebook,
        title="효율",
        y_label="Oracle / actual",
        series=(("Efficiency", efficiencies), ("Rolling mean (100)", rolling_mean(efficiencies))),
    )
    _add_line_tab(
        notebook,
        title="에피소드 시간",
        y_label="Seconds",
        series=(("Duration", durations), ("Rolling mean (100)", rolling_mean(durations))),
    )
    _add_line_tab(
        notebook,
        title="Prediction",
        y_label="Prediction score",
        series=(("Mean prediction", predictions), ("Rolling mean (100)", rolling_mean(predictions))),
    )
    _add_line_tab(
        notebook,
        title="Holdout",
        y_label="Value",
        series=(
            ("Before", holdout_before),
            ("After", holdout_after),
            ("Gain", holdout_gain),
        ),
    )
    _add_line_tab(
        notebook,
        title="Intrinsic value",
        y_label="Total value",
        series=(("Intrinsic total", intrinsic), ("Rolling mean (100)", rolling_mean(intrinsic))),
    )
    _add_line_tab(
        notebook,
        title="Imagination",
        y_label="Count",
        series=(("Nodes", imagined_nodes), ("Decisions", imagination_decisions)),
    )
    _add_line_tab(
        notebook,
        title="오류·반복",
        y_label="Count",
        series=(("Errors", errors), ("Repeated", repeats)),
    )
    _add_line_tab(
        notebook,
        title="모드별 시간",
        y_label="Seconds",
        series=(("Live", live_seconds), ("Fast", fast_seconds)),
    )

    action_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    for record in records:
        action_counts.update(record.action_counts)
        event_counts.update(record.event_counts)
    _add_bar_tab(
        notebook,
        title="행동 분포",
        items=sorted(action_counts.items(), key=lambda item: (-item[1], item[0])),
    )
    _add_bar_tab(
        notebook,
        title="이벤트 분포",
        items=sorted(event_counts.items(), key=lambda item: (-item[1], item[0])),
    )

    table_frame = ttk.Frame(notebook, padding=8)
    notebook.add(table_frame, text="에피소드 표")
    columns = (
        "episode",
        "steps",
        "score",
        "duration",
        "epsilon",
        "errors",
        "repeats",
        "keys",
        "doors",
        "imaginations",
        "nodes",
        "prediction",
        "holdout_gain",
        "intrinsic",
    )
    tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    labels = {
        "episode": "Episode",
        "steps": "Ticks",
        "score": "Score",
        "duration": "Seconds",
        "epsilon": "Epsilon",
        "errors": "Errors",
        "repeats": "Repeats",
        "keys": "Keys",
        "doors": "Doors",
        "imaginations": "Imagine calls",
        "nodes": "Nodes",
        "prediction": "Prediction",
        "holdout_gain": "Holdout gain",
        "intrinsic": "Intrinsic total",
    }
    for column in columns:
        tree.heading(column, text=labels[column])
        tree.column(column, width=95, anchor="center", stretch=True)
    vertical = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)
    for record in records:
        tree.insert(
            "",
            "end",
            values=(
                record.episode,
                record.steps,
                f"{record.score:.6f}",
                f"{record.duration_seconds:.6f}",
                f"{record.epsilon:.4f}",
                record.errors,
                record.repeated_actions,
                record.found_keys,
                record.opened_doors,
                record.imagination_decisions,
                record.imagined_nodes,
                f"{record.mean_prediction_score:.6f}",
                f"{record.mean_holdout_gain:.8f}",
                f"{record.intrinsic_value_total:.6f}",
            ),
        )
    return window
