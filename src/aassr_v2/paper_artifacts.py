from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .paper_protocol import PaperPaths, sha256_json
from .paper_statistics import (
    bootstrap_confidence_interval,
    iter_csv_rows,
    read_csv_rows,
    write_csv_rows,
)


def _format(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def make_paper_tables(root: str | Path) -> tuple[Path, ...]:
    paths = PaperPaths.create(root)
    manifest = json.loads(
        (paths.manifests / "protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary_path = paths.statistics / "cross_seed_summary.csv"
    summaries = read_csv_rows(summary_path) if summary_path.exists() else []
    comparisons_path = paths.statistics / "condition_comparisons.csv"
    comparisons = (
        read_csv_rows(comparisons_path) if comparisons_path.exists() else []
    )
    adaptation_path = paths.statistics / "adaptation_summary.csv"
    adaptation = (
        read_csv_rows(adaptation_path) if adaptation_path.exists() else []
    )
    learning_path = paths.statistics / "learning_summary.csv"
    learning = (
        read_csv_rows(learning_path) if learning_path.exists() else []
    )
    strategies_path = paths.raw / "strategies.jsonl"
    strategies = []
    if strategies_path.exists():
        strategies = [
            json.loads(line)
            for line in strategies_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    creativity_summary_path = (
        paths.statistics / "creativity_summary.csv"
    )
    creativity_summary = (
        read_csv_rows(creativity_summary_path)
        if creativity_summary_path.exists()
        else []
    )
    creativity_by_strategy = {
        row.get("strategy_id", ""): row for row in creativity_summary
    }

    table1 = paths.tables / "table1_protocol.md"
    table1.write_text(
        "\n".join(
            (
                "# Table 1. Experiment protocol",
                "",
                "| Protocol | Stage | Research seeds | Train worlds | Seen worlds | Unseen worlds |",
                "|---|---|---:|---:|---:|---:|",
                "| "
                + " | ".join(
                    (
                        str(manifest["protocol_version"]),
                        str(manifest["study_stage"]),
                        str(len(manifest["research_seeds"])),
                        str(len(manifest["world_seeds"]["train"])),
                        str(len(manifest["world_seeds"]["seen"])),
                        str(len(manifest["world_seeds"]["unseen"])),
                    )
                )
                + " |",
                "",
            )
        ),
        encoding="utf-8",
    )

    table2 = paths.tables / "table2_autonomy.md"
    learning_groups: dict[tuple[str, str], list[Mapping[str, str]]] = (
        defaultdict(list)
    )
    for row in learning:
        learning_groups[
            (row.get("condition", ""), row.get("environment", ""))
        ].append(row)
    lines = [
        "# Table 2. Autonomy results",
        "",
        "| Condition | Environment | Phase | Seeds | Success | Learning AUC | First-success transitions | Final 10% | Errors | Repeats | Runtime |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row.get("suite") not in {"autonomy", "ablation", "autonomous_discovery"}:
            continue
        curve = learning_groups.get(
            (row.get("condition", ""), row.get("environment", "")), []
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    row.get("condition", ""),
                    row.get("environment", ""),
                    row.get("phase", ""),
                    row.get("seed_count", ""),
                    (
                        f"{_format(row.get('success_mean'))} "
                        f"[{_format(row.get('success_ci95_low'))}, "
                        f"{_format(row.get('success_ci95_high'))}]"
                    ),
                    _format(
                        fmean(
                            float(item["learning_auc"]) for item in curve
                        )
                        if curve
                        else None
                    ),
                    _format(
                        fmean(
                            float(item["first_success_real_transitions"])
                            for item in curve
                            if item.get("first_success_real_transitions")
                        )
                        if any(
                            item.get("first_success_real_transitions")
                            for item in curve
                        )
                        else None
                    ),
                    _format(
                        fmean(
                            float(item["final_10_percent_success"])
                            for item in curve
                        )
                        if curve
                        else None
                    ),
                    _format(row.get("errors_mean")),
                    _format(row.get("repeats_mean")),
                    _format(row.get("runtime_seconds_mean")),
                )
            )
            + " |"
        )
    table2.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table3 = paths.tables / "table3_transfer.md"
    lines = [
        "# Table 3. Structural transfer",
        "",
        "| Condition | Environment | Seed | Adaptation AUC | Transfer gain | Calibration error | Episodes to 50% | Save to 50% | Episodes to 80% | Save to 80% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in adaptation:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.get("condition", ""),
                    row.get("environment", ""),
                    row.get("seed", ""),
                    _format(row.get("adaptation_auc")),
                    _format(row.get("transfer_gain_vs_from_scratch")),
                    _format(
                        row.get("unseen_prediction_calibration_error")
                    ),
                    _format(row.get("episodes_to_50")),
                    _format(row.get("sample_saving_to_50")),
                    _format(row.get("episodes_to_80")),
                    _format(row.get("sample_saving_to_80")),
                )
            )
            + " |"
        )
    table3.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table4 = paths.tables / "table4_creativity.md"
    ratings_path = paths.raw / "human_ratings.csv"
    ratings = (
        read_csv_rows(ratings_path) if ratings_path.exists() else []
    )
    ratings_by_strategy: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in ratings:
        ratings_by_strategy[row.get("strategy_id", "")].append(row)
    lines = [
        "# Table 4. Creativity strategies",
        "",
        "| Strategy | Source | Valid | Novelty | Graph edit | Motif | Prerequisite | Family | Sequence | Utility qualified | Reproducible | Creative candidate | Steps | Resources | Risk | Reuse | Blind novelty | Blind utility | Coherence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategies:
        strategy_ratings = ratings_by_strategy.get(
            str(row.get("strategy_id", "")), []
        )
        analyzed = creativity_by_strategy.get(
            str(row.get("strategy_id", "")), {}
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("strategy_id", "")),
                    str(row.get("source_kind", "")),
                    str(int(bool(row.get("valid", False)))),
                    _format(row.get("novelty_score")),
                    _format(analyzed.get("graph_edit_distance")),
                    _format(analyzed.get("motif_jaccard_distance")),
                    _format(analyzed.get("prerequisite_edge_distance")),
                    _format(analyzed.get("solution_family_distance")),
                    _format(analyzed.get("effect_sequence_distance")),
                    _format(analyzed.get("utility_qualified")),
                    _format(analyzed.get("reproducible")),
                    _format(analyzed.get("creative_candidate")),
                    _format(row.get("primitive_steps")),
                    _format(row.get("resources_used")),
                    _format(row.get("risk_entries")),
                    _format(row.get("reusable_success_rate")),
                    _format(
                        fmean(
                            float(item["novelty"])
                            for item in strategy_ratings
                        )
                        if strategy_ratings
                        else None
                    ),
                    _format(
                        fmean(
                            float(item["utility"])
                            for item in strategy_ratings
                        )
                        if strategy_ratings
                        else None
                    ),
                    _format(
                        fmean(
                            float(item["coherence"])
                            for item in strategy_ratings
                        )
                        if strategy_ratings
                        else None
                    ),
                )
            )
            + " |"
        )
    table4.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table1, table2, table3, table4


def _svg_chart(
    path: Path,
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
) -> None:
    width, height = 900, 480
    left, top, bottom = 80, 60, 70
    chart_width = width - left - 30
    chart_height = height - top - bottom
    maximum = max(values, default=1.0) or 1.0
    count = max(1, len(values))
    bar_width = chart_width / count * 0.65
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+chart_height}" x2="{left+chart_width}" y2="{top+chart_height}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = left + (index + 0.5) * chart_width / count
        bar_height = max(0.0, value) / maximum * chart_height
        y = top + chart_height - bar_height
        elements.extend(
            (
                f'<rect x="{x-bar_width/2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#31688e"/>',
                f'<text x="{x:.2f}" y="{top+chart_height+18}" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(label[:18])}</text>',
                f'<text x="{x:.2f}" y="{max(top+12, y-4):.2f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.3f}</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_flow(path: Path, *, title: str, nodes: Sequence[str]) -> None:
    width, height = 1100, 220
    node_width, node_height = 135, 54
    gap = (width - 60 - len(nodes) * node_width) / max(1, len(nodes) - 1)
    y = 85
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="21">{html.escape(title)}</text>',
    ]
    for index, label in enumerate(nodes):
        x = 30 + index * (node_width + gap)
        if index:
            previous_x = x - gap
            elements.append(
                f'<line x1="{previous_x:.1f}" y1="{y+node_height/2}" x2="{x-8:.1f}" y2="{y+node_height/2}" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>'
            )
        elements.extend(
            (
                f'<rect x="{x:.1f}" y="{y}" width="{node_width}" height="{node_height}" rx="8" fill="#e5f5f9" stroke="#2c7fb8"/>',
                f'<text x="{x+node_width/2:.1f}" y="{y+node_height/2+5}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_learning_curves(
    path: Path,
    series: Mapping[str, Sequence[tuple[float, float, float, float]]],
) -> None:
    width, height = 1100, 620
    left, top, bottom, right = 80, 60, 70, 260
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum_x = max(
        (point[0] for values in series.values() for point in values),
        default=1.0,
    ) or 1.0
    palette = (
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#1f78b4",
    )

    def x_coord(value: float) -> float:
        return left + value / maximum_x * chart_width

    def y_coord(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * chart_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left+chart_width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Autonomy training success with seed 95% intervals</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+chart_height}" x2="{left+chart_width}" y2="{top+chart_height}" stroke="#333"/>',
        f'<text x="{left+chart_width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">training episode</text>',
        f'<text transform="translate(20 {top+chart_height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">success rate</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = y_coord(value)
        elements.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_width}" y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>',
            )
        )
    for index, (label, values) in enumerate(sorted(series.items())):
        if not values:
            continue
        color = palette[index % len(palette)]
        upper = " ".join(
            f"{x_coord(x):.1f},{y_coord(high):.1f}"
            for x, _, _, high in values
        )
        lower = " ".join(
            f"{x_coord(x):.1f},{y_coord(low):.1f}"
            for x, _, low, _ in reversed(values)
        )
        points = " ".join(
            f"{x_coord(x):.1f},{y_coord(mean):.1f}"
            for x, mean, _, _ in values
        )
        elements.extend(
            (
                f'<polygon points="{upper} {lower}" fill="{color}" opacity="0.10"/>',
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{left+chart_width+20}" y1="{top+18*index}" x2="{left+chart_width+42}" y2="{top+18*index}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{left+chart_width+48}" y="{top+18*index+4}" font-family="sans-serif" font-size="10">{html.escape(label[:30])}</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_forest(
    path: Path, comparisons: Sequence[Mapping[str, str]]
) -> None:
    width = 1000
    height = max(300, 90 + 34 * max(1, len(comparisons)))
    left, right, top = 300, 80, 55
    chart_width = width - left - right
    intervals = [
        (
            float(row.get("paired_mean_difference") or 0.0),
            float(row.get("ci95_low") or 0.0),
            float(row.get("ci95_high") or 0.0),
        )
        for row in comparisons
    ]
    extent = max(
        [abs(value) for triple in intervals for value in triple] + [0.1]
    )

    def x_coord(value: float) -> float:
        return left + (value + extent) / (2 * extent) * chart_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Ablation effects versus Full AASSR</text>',
        f'<line x1="{x_coord(0):.1f}" y1="{top}" x2="{x_coord(0):.1f}" y2="{height-40}" stroke="#777" stroke-dasharray="4 4"/>',
    ]
    for index, (row, (mean, low, high)) in enumerate(
        zip(comparisons, intervals, strict=True)
    ):
        y = top + 28 + index * 34
        label = f"{row.get('suite','')} / {row.get('baseline','')}"
        elements.extend(
            (
                f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{html.escape(label[:42])}</text>',
                f'<line x1="{x_coord(low):.1f}" y1="{y}" x2="{x_coord(high):.1f}" y2="{y}" stroke="#31688e" stroke-width="2"/>',
                f'<circle cx="{x_coord(mean):.1f}" cy="{y}" r="5" fill="#31688e"/>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _svg_creativity_scatter(
    path: Path, strategies: Sequence[Mapping[str, Any]]
) -> None:
    width, height = 900, 540
    left, top, chart_width, chart_height = 80, 55, 700, 400
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left+chart_width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Strategy novelty and reuse</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+chart_height}" x2="{left+chart_width}" y2="{top+chart_height}" stroke="#333"/>',
        f'<text x="{left+chart_width/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="13">structural novelty</text>',
        f'<text transform="translate(20 {top+chart_height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">reuse success rate</text>',
    ]
    for row in strategies:
        x_value = float(row.get("novelty_score") or 0.0)
        y_value = float(row.get("reusable_success_rate") or 0.0)
        x = left + max(0.0, min(1.0, x_value)) * chart_width
        y = top + (1.0 - max(0.0, min(1.0, y_value))) * chart_height
        color = "#d95f02" if row.get("source_kind") == "aassr" else "#1b9e77"
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="0.75"><title>{html.escape(str(row.get("strategy_id", "")))}</title></circle>'
        )
    elements.extend(
        (
            f'<circle cx="{left+chart_width+30}" cy="{top+20}" r="5" fill="#d95f02"/><text x="{left+chart_width+42}" y="{top+24}" font-family="sans-serif" font-size="11">AASSR</text>',
            f'<circle cx="{left+chart_width+30}" cy="{top+42}" r="5" fill="#1b9e77"/><text x="{left+chart_width+42}" y="{top+46}" font-family="sans-serif" font-size="11">reference</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def make_paper_figures(root: str | Path) -> tuple[Path, ...]:
    paths = PaperPaths.create(root)
    summaries = read_csv_rows(paths.statistics / "cross_seed_summary.csv")
    comparisons_path = paths.statistics / "condition_comparisons.csv"
    comparisons = (
        read_csv_rows(comparisons_path) if comparisons_path.exists() else []
    )
    strategies_path = paths.raw / "strategies.jsonl"
    strategies = (
        [
            json.loads(line)
            for line in strategies_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if strategies_path.exists()
        else []
    )
    outputs = tuple(
        paths.figures / filename
        for filename in (
            "figure1_closed_loop.svg",
            "figure2_protocol.svg",
            "figure3_autonomy.svg",
            "figure4_transfer.svg",
            "figure5_ablation.svg",
            "figure6_creativity.svg",
        )
    )
    _svg_flow(
        outputs[0],
        title="AASSR closed-loop architecture",
        nodes=(
            "Observation",
            "Knowledge",
            "Policy / Prophecy",
            "Imagination",
            "Action",
            "Actual transition",
            "Holdout update",
        ),
    )
    _svg_flow(
        outputs[1],
        title="Paper experiment protocol",
        nodes=(
            "Pilot",
            "Protocol freeze",
            "Training",
            "Seen evaluation",
            "Unseen zero-shot",
            "Adaptation",
            "Blind creativity",
        ),
    )
    curve_values: dict[
        tuple[str, int], list[float]
    ] = defaultdict(list)
    for row in iter_csv_rows(paths.raw / "episodes.csv"):
        if row.get("suite") not in {"autonomy", "ablation"}:
            continue
        if row.get("phase") != "training":
            continue
        label = f"{row.get('condition','')}@{row.get('environment','')}"
        curve_values[(label, int(float(row.get("episode") or 0)))].append(
            float(row.get("success") or 0.0)
        )
    curves: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for (label, episode), values in sorted(curve_values.items()):
        low, high = bootstrap_confidence_interval(
            values, samples=500, seed=episode + len(label)
        )
        curves[label].append((float(episode), fmean(values), low, high))
    _svg_learning_curves(outputs[2], curves)
    adaptation_path = paths.statistics / "adaptation_summary.csv"
    adaptation = (
        read_csv_rows(adaptation_path) if adaptation_path.exists() else []
    )
    seen = [
        row
        for row in summaries
        if row.get("phase") == "evaluation_seen"
    ]
    adaptation_groups: dict[str, list[float]] = defaultdict(list)
    for row in adaptation:
        adaptation_groups[row.get("condition", "")].append(
            float(row.get("adaptation_auc") or 0.0)
        )
    labels = [
        f"seen:{row.get('condition','')}" for row in seen
    ] + [
        f"adapt:{condition}" for condition in sorted(adaptation_groups)
    ]
    values = [
        float(row.get("success_mean") or 0.0) for row in seen
    ] + [
        fmean(adaptation_groups[condition])
        for condition in sorted(adaptation_groups)
    ]
    _svg_chart(
        outputs[3],
        title="Seen success and unseen adaptation AUC",
        labels=labels,
        values=values,
        y_label="rate / normalized AUC",
    )
    _svg_forest(outputs[4], comparisons)
    _svg_creativity_scatter(outputs[5], strategies)
    return outputs


def write_paper_report(root: str | Path) -> Path:
    paths = PaperPaths.create(root)
    analysis = json.loads(
        (paths.statistics / "analysis_summary.json").read_text(
            encoding="utf-8"
        )
    )
    comparisons = read_csv_rows(
        paths.statistics / "condition_comparisons.csv"
    )
    for filename in (
        "learning_auc_comparisons.csv",
        "transfer_comparisons.csv",
    ):
        path = paths.statistics / filename
        if path.exists():
            comparisons.extend(read_csv_rows(path))
    lines = [
        "# AASSR paper experiment report",
        "",
        f"- Episode rows: {analysis['episode_rows']}",
        f"- Seed-level rows: {analysis['seed_rows']}",
        f"- Paired comparisons: {analysis['comparison_rows']}",
        f"- Learning-AUC comparisons: {analysis['learning_auc_comparison_rows']}",
        f"- Transfer comparisons: {analysis['transfer_comparison_rows']}",
        f"- Adaptation summaries: {analysis['adaptation_rows']}",
        f"- Learning-curve seed summaries: {analysis['learning_curve_rows']}",
        f"- Creativity strategies: {analysis['creativity_rows']}",
        "",
        "## Statistical method",
        "",
        "Episodes are aggregated inside each research seed before inference. "
        "Reported comparisons use paired seed differences, seed bootstrap 95% "
        "confidence intervals, paired permutation tests, and Holm correction. "
        "The privileged oracle condition is excluded from baseline inference.",
        "",
        "## Paired results",
        "",
        "| Suite | Target | Baseline | Difference | 95% CI | Holm p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.get("suite", ""),
                    row.get("target", ""),
                    row.get("baseline", ""),
                    _format(row.get("paired_mean_difference")),
                    f"[{_format(row.get('ci95_low'))}, {_format(row.get('ci95_high'))}]",
                    _format(row.get("p_value_holm")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Interpretation guardrail",
            "",
            "Claims must follow the protocol decision table. A successful run "
            "does not by itself establish autonomy, transfer, or creativity; "
            "each claim additionally requires its paired, novelty, utility, "
            "reproducibility, and blind-rating criteria.",
            "",
        )
    )
    report = paths.root / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def validate_paper_artifacts(
    root: str | Path, *, require_human_merge: bool = False
) -> list[str]:
    paths = PaperPaths.create(root)
    required = [
        paths.raw / "episodes.csv",
        paths.raw / "transitions.jsonl",
        paths.raw / "strategies.jsonl",
        paths.seed_level / "seed_summary.csv",
        paths.statistics / "analysis_summary.json",
        paths.statistics / "cross_seed_summary.csv",
        paths.statistics / "condition_comparisons.csv",
        paths.statistics / "adaptation_summary.csv",
        paths.statistics / "learning_summary.csv",
        paths.statistics / "learning_auc_comparisons.csv",
        paths.statistics / "transfer_comparisons.csv",
        paths.statistics / "creativity_summary.csv",
        paths.manifests / "resolved_config.json",
        paths.manifests / "protocol_manifest.json",
        paths.manifests / "protocol_manifest.json.sha256",
        paths.root / "report.md",
        *(paths.tables / f"table{index}_{name}.md" for index, name in (
            (1, "protocol"),
            (2, "autonomy"),
            (3, "transfer"),
            (4, "creativity"),
        )),
        *(paths.figures / f"figure{index}_{name}.svg" for index, name in (
            (1, "closed_loop"),
            (2, "protocol"),
            (3, "autonomy"),
            (4, "transfer"),
            (5, "ablation"),
            (6, "creativity"),
        )),
    ]
    issues = [
        f"missing artifact: {path.relative_to(paths.root)}"
        for path in required
        if not path.exists()
    ]
    manifest_path = paths.manifests / "protocol_manifest.json"
    digest_path = paths.manifests / "protocol_manifest.json.sha256"
    config_path = paths.manifests / "resolved_config.json"
    config: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if any(
            str(item.get("kind", "")) == "safe_application"
            for item in config.get("suites", ())
        ) and not (
            paths.manifests / "safe_application_world.json"
        ).exists():
            issues.append(
                "missing artifact: manifests/safe_application_world.json"
            )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.exists() and digest_path.exists():
        expected = digest_path.read_text(encoding="ascii").strip()
        actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if expected != actual:
            issues.append("protocol manifest SHA256 mismatch")
    if config is not None and manifest is not None:
        if manifest.get("config_sha256") != sha256_json(config):
            issues.append("resolved config SHA256 does not match manifest")
        if manifest.get("protocol_version") != config.get(
            "protocol_version"
        ):
            issues.append("protocol version does not match manifest")
        if list(manifest.get("research_seeds", ())) != list(
            config.get("research_seeds", ())
        ):
            issues.append("research seed list does not match manifest")
        for name in ("train", "seen", "unseen"):
            if list(
                manifest.get("world_seeds", {}).get(name, ())
            ) != list(config.get("world_seeds", {}).get(name, ())):
                issues.append(
                    f"{name} world seed list does not match manifest"
                )
        for lock_name, expected_hash in manifest.get(
            "protocol_locks", {}
        ).items():
            filename = {
                "acceptance_gate_manifest": (
                    "acceptance_gate_manifest.json"
                ),
                "frozen_creativity_rules": (
                    "frozen_creativity_rules.json"
                ),
            }.get(lock_name)
            if filename:
                copied = paths.manifests / filename
                if (
                    not copied.exists()
                    or hashlib.sha256(copied.read_bytes()).hexdigest()
                    != expected_hash
                ):
                    issues.append(
                        f"protocol lock hash mismatch: {lock_name}"
                    )
    if require_human_merge:
        for name in (
            "human_paths.jsonl",
            "human_ratings.csv",
        ):
            if not (paths.raw / name).exists():
                issues.append(f"human merge requires raw/{name}")
        metadata_path = paths.manifests / "human_dataset.json"
        if not metadata_path.exists():
            issues.append(
                "human merge requires manifests/human_dataset.json"
            )
        elif config is not None:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            settings = config.get("human_study", {})
            if metadata.get("dataset_version") != settings.get(
                "dataset_version"
            ):
                issues.append("human dataset version mismatch")
            if metadata.get("approval_id") != settings.get("approval_id"):
                issues.append("human dataset approval ID mismatch")
    if (paths.raw / "episodes.csv").exists():
        branch_origins: dict[
            tuple[str, str, str], set[str]
        ] = defaultdict(set)
        branch_budgets: dict[
            tuple[str, str, str], set[int]
        ] = defaultdict(set)
        fair_groups: dict[
            tuple[str, str, str, str], dict[str, float]
        ] = {}
        frozen_found = False
        missing_fingerprints_found = False
        budget_exceeded = False
        limit = (
            int(
                config.get("budgets", {}).get(
                    "real_transitions_per_episode", 0
                )
            )
            if config is not None
            else 0
        )
        for row in iter_csv_rows(paths.raw / "episodes.csv"):
            phase = str(row.get("phase", ""))
            evaluating = phase.startswith("evaluation")
            before = row.get("checkpoint_fingerprint_before")
            after = row.get("checkpoint_fingerprint_after")
            if evaluating and before and after and before != after:
                frozen_found = True
            if (
                evaluating
                and row.get("suite")
                in {"autonomy", "ablation", "transfer", "creativity"}
                and (not before or not after)
            ):
                missing_fingerprints_found = True
            actual = float(
                row.get("real_transitions") or row.get("steps") or 0
            )
            if limit > 0 and actual > limit:
                budget_exceeded = True
            if row.get("suite") == "transfer":
                origin = str(row.get("branch_start_fingerprint", ""))
                if origin:
                    branch_key = (
                        str(row.get("condition", "")),
                        str(row.get("seed", "")),
                        str(row.get("world_seed", "")),
                    )
                    branch_origins[branch_key].add(origin)
                    budget = row.get("adaptation_budget")
                    if budget not in (None, ""):
                        branch_budgets[branch_key].add(
                            int(float(budget))
                        )
            if row.get("suite") in {"autonomy", "ablation"}:
                fair_key = (
                    row.get("suite", ""),
                    row.get("environment", ""),
                    phase,
                    row.get("seed", ""),
                )
                condition_totals = fair_groups.setdefault(fair_key, {})
                condition = row.get("condition", "")
                condition_totals[condition] = condition_totals.get(
                    condition, 0.0
                ) + actual
        if frozen_found:
            issues.append("learning-state mutation found in evaluation rows")
        if missing_fingerprints_found:
            issues.append(
                "evaluation rows are missing learning-state fingerprints"
            )
        if budget_exceeded:
            issues.append("episode exceeds real transition budget")
        if any(len(values) != 1 for values in branch_origins.values()):
            issues.append(
                "adaptation budgets do not share one branch checkpoint"
            )
        if config is not None and branch_budgets:
            expected_budgets = {
                int(item)
                for item in config.get("budgets", {}).get(
                    "adaptation_episodes", ()
                )
            }
            if any(
                values != expected_budgets
                for values in branch_budgets.values()
            ):
                issues.append(
                    "transfer branch is missing an adaptation budget"
                )
        if any(
            len(values) > 1
            and max(values.values()) != min(values.values())
            for values in fair_groups.values()
        ):
            issues.append(
                "autonomy/ablation conditions use unequal real transitions"
            )
        if config is not None and config.get("study_stage") == "final":
            if not (paths.manifests / "acceptance_gate_manifest.json").exists():
                issues.append(
                    "Final run is missing manifests/acceptance_gate_manifest.json"
                )
            if any(
                item.get("kind") == "creativity"
                for item in config.get("suites", ())
            ) and not (
                paths.manifests / "frozen_creativity_rules.json"
            ).exists():
                issues.append(
                    "Final creativity run is missing frozen rules"
                )
    transitions_path = paths.raw / "transitions.jsonl"
    if transitions_path.exists():
        private_labels = {
            "information_route",
            "resource_route",
            "bypass_route",
            "tool_route",
            "emergent_combination",
            "viable_branch",
            "solution_family",
        }
        with transitions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                phase = str(payload.get("phase", ""))
                if (
                    phase.startswith("evaluation")
                    and payload.get("learning_enabled") is not False
                ):
                    issues.append(
                        "evaluation transition is not explicitly frozen"
                    )
                    break
                if payload.get("suite") != "creativity":
                    continue
                visible = json.dumps(
                    {
                        "before": payload.get("before"),
                        "action": payload.get("action"),
                        "after": payload.get("after"),
                    }
                )
                if any(label in visible for label in private_labels):
                    issues.append(
                        "creative agent-visible trace contains a private label"
                    )
                    break
    strategies_path = paths.raw / "strategies.jsonl"
    if strategies_path.exists():
        with strategies_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                strategy = json.loads(line)
                if (
                    "trace" not in strategy
                    or "novelty_components" not in strategy
                ):
                    issues.append(
                        "strategy record is missing trace or distance components"
                    )
                    break
    return issues
