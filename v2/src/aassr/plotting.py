from __future__ import annotations

from pathlib import Path
from typing import Any

from .labels import condition_label


CONDITION_COLORS = {
    "C0": "#6b7280",
    "C1": "#2563eb",
    "C2": "#7c3aed",
    "C3": "#16a34a",
    "C4": "#14b8a6",
    "C5": "#0f766e",
    "APASSR_FULL": "#db2777",
    "APASSR_FULL_CAL": "#9333ea",
    "QLEARN": "#0ea5e9",
    "DQN_PARTIAL": "#ef4444",
    "ORACLE_MDP": "#111827",
}


def write_analysis_plots(
    *,
    summary_rows: list[Any],
    condition_stats: list[Any],
    learning_curve: list[Any],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()

    _bar_chart(
        plt,
        summary_rows,
        field="success_rate_mean",
        title="Success Rate by Condition",
        ylabel="Success rate",
        path=output_path / "figure_success_rate.png",
    )
    _bar_chart(
        plt,
        summary_rows,
        field="steps_to_flag_mean",
        title="Steps to FLAG by Condition",
        ylabel="Steps to FLAG (successful episodes)",
        path=output_path / "figure_steps_to_flag.png",
    )
    _bar_chart(
        plt,
        summary_rows,
        field="semantic_gain_mean",
        title="Semantic Delta-K per Episode",
        ylabel="Semantic Delta-K",
        path=output_path / "figure_semantic_gain.png",
    )
    _repeat_error_chart(
        plt,
        summary_rows,
        path=output_path / "figure_repeat_error_rate.png",
    )
    _learning_curve_chart(
        plt,
        learning_curve,
        path=output_path / "figure_learning_curve.png",
    )


def write_diagnostic_plots(*, diagnostic_rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    _dict_bar_chart(
        plt,
        diagnostic_rows,
        field="imagined_trajectory_depth_mean",
        title="Mean Imagined Trajectory Depth",
        ylabel="Depth",
        path=output_path / "figure_imagined_depth.png",
    )
    _dict_bar_chart(
        plt,
        diagnostic_rows,
        field="newly_unlocked_action_mean",
        title="Newly Unlocked Actions per Episode",
        ylabel="Actions",
        path=output_path / "figure_unlocked_actions.png",
    )
    _dict_bar_chart(
        plt,
        diagnostic_rows,
        field="future_dependency_selection_rate_mean",
        title="Future Dependency Selection Rate",
        ylabel="Rate",
        path=output_path / "figure_future_dependency_rate.png",
    )
    _dict_bar_chart(
        plt,
        diagnostic_rows,
        field="predicted_kk_f1_mean",
        title="Predicted KK F1 by Condition",
        ylabel="F1",
        path=output_path / "figure_prophecy_kk_alignment.png",
    )
    _dict_bar_chart(
        plt,
        diagnostic_rows,
        field="imagined_action_execution_match_rate_mean",
        title="Imagined Next-Action Exact Match",
        ylabel="Rate",
        path=output_path / "figure_imagined_action_match.png",
    )


def _bar_chart(plt: Any, rows: list[Any], *, field: str, title: str, ylabel: str, path: Path) -> None:
    conditions = [row.condition for row in rows]
    labels = [condition_label(condition) for condition in conditions]
    values = [getattr(row, field) for row in rows]
    colors = [CONDITION_COLORS.get(condition, "#475569") for condition in conditions]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dict_bar_chart(plt: Any, rows: list[dict[str, Any]], *, field: str, title: str, ylabel: str, path: Path) -> None:
    conditions = [str(row["condition"]) for row in rows]
    labels = [condition_label(condition) for condition in conditions]
    values = [_safe_float(row.get(field)) for row in rows]
    colors = [CONDITION_COLORS.get(condition, "#475569") for condition in conditions]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _repeat_error_chart(plt: Any, rows: list[Any], *, path: Path) -> None:
    conditions = [row.condition for row in rows]
    labels = [condition_label(condition) for condition in conditions]
    repeat_values = [row.repeat_rate_mean for row in rows]
    error_values = [row.error_rate_mean for row in rows]
    x = list(range(len(conditions)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar([item - width / 2 for item in x], repeat_values, width=width, label="Repeat rate", color="#f59e0b")
    ax.bar([item + width / 2 for item in x], error_values, width=width, label="Error rate", color="#dc2626")
    ax.set_xticks(x, labels)
    ax.set_title("Repeat/Error Rate by Condition")
    ax.set_ylabel("Rate")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _learning_curve_chart(plt: Any, rows: list[Any], *, path: Path) -> None:
    by_condition: dict[str, list[Any]] = {}
    for row in rows:
        by_condition.setdefault(row.condition, []).append(row)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for condition, condition_rows in sorted(by_condition.items()):
        ordered = sorted(condition_rows, key=lambda row: row.window_start)
        x = [row.window_start for row in ordered]
        y = [row.success_rate for row in ordered]
        ax.plot(
            x,
            y,
            marker="o",
            label=condition_label(condition),
            color=CONDITION_COLORS.get(condition, None),
        )
    ax.set_title("Learning Curve")
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Episode window start")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
