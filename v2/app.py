from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from aassr.dashboard import (
    action_template_rows,
    binding_rows,
    candidate_label,
    candidate_rows,
    format_table_value,
    imagination_score_rows,
    implementation_status_rows,
    knowledge_rows,
    paper_project_comparison_rows,
    policy_probability_rows,
    trace_row,
)
from aassr.gridworld import ActionCandidate, CellKind, DMPConfig, GridWorld, GridWorldDMP
from aassr.imagination import ImaginationCycle
from aassr.knowledge import KK
from aassr.policy import KnowledgeGainScorer, PolicyABC, RandomScorer
from aassr.prophecy import TableProphecyModel


CELL_STYLES = {
    "A": ("Agent", "#2563eb", "#eff6ff"),
    "?": ("Frontier", "#ca8a04", "#fefce8"),
    ".": ("Known", "#475569", "#f8fafc"),
    "#": ("Wall", "#334155", "#e2e8f0"),
    "H": ("Hint", "#7c3aed", "#f5f3ff"),
    "K": ("Key", "#16a34a", "#f0fdf4"),
    "D": ("Door", "#ea580c", "#fff7ed"),
    "F": ("Flag", "#dc2626", "#fef2f2"),
    " ": ("Unknown", "#94a3b8", "#ffffff"),
}


def main() -> None:
    st.set_page_config(page_title="AASSR GridWorld", layout="wide")
    st.title("AASSR Knowledge-Driven GridWorld")
    st.caption("Knowledge Storage = memory + action parameter supplier")

    ensure_state()
    dmp: GridWorldDMP = st.session_state.dmp

    with st.sidebar:
        st.header("Controls")
        selector_mode = st.selectbox(
            "Selector",
            [
                "PolicyABC C1",
                "PolicyABC + Prophecy C2",
                "PolicyABC + Prophecy + Imagination C3",
                "Random C0",
                "KnowledgeGain debug",
            ],
        )
        if st.button("Reset With Selector", width="stretch"):
            reset_demo(selector_mode)
            st.rerun()
        if st.button("Reset Demo", width="stretch"):
            reset_demo(selector_mode)
            st.rerun()

        strategy = st.selectbox(
            "Policy strategy",
            ["scorer", "random", "nearest", "least_tried", "high_uncertainty"],
        )
        if st.button("Run Policy Step", width="stretch"):
            run_policy_step(strategy)
            st.rerun()

        max_steps = st.slider("Auto-run steps", 1, 20, 5)
        if st.button("Auto-run", width="stretch"):
            for _ in range(max_steps):
                if not run_policy_step(strategy):
                    break
            st.rerun()

    runtime_tab, comparison_tab = st.tabs(["DMP Runtime", "Paper vs Project"])
    with runtime_tab:
        render_runtime_page(dmp)
    with comparison_tab:
        render_comparison_page()


def render_runtime_page(dmp: GridWorldDMP) -> None:
    metrics = dmp.metrics()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Step", dmp.step_index)
    metric_cols[1].metric("Position", str(dmp.position))
    metric_cols[2].metric("Binding success", f"{metrics['slot_binding_success_rate']:.0%}")
    metric_cols[3].metric("Knowledge reuse", int(metrics["knowledge_reuse_count"]))

    left, right = st.columns([1.05, 1.4], gap="large")
    with left:
        st.subheader("GridWorld")
        st.markdown(render_grid_html(dmp), unsafe_allow_html=True)
        st.markdown(render_legend_html(), unsafe_allow_html=True)

        st.subheader("Last Result")
        st.json(st.session_state.last_result or {"status": "No action executed yet."})
        imagination_rows = imagination_score_rows(st.session_state.last_step_result)
        if imagination_rows:
            st.subheader("Imagination Scores")
            st.dataframe(imagination_rows, width="stretch", hide_index=True)

    with right:
        st.subheader("KK Slot Binding")
        st.dataframe(binding_rows(dmp), width="stretch", hide_index=True)

        st.subheader("Executable Action Candidates")
        render_candidate_controls(dmp)

    st.subheader("Policy WHAT / HOW / WHERE")
    st.dataframe(candidate_rows(dmp), width="stretch", hide_index=True)

    st.subheader("PolicyABC Probability Tables")
    policy_rows = policy_probability_rows(dmp)
    if policy_rows:
        st.dataframe(policy_rows, width="stretch", hide_index=True)
    else:
        st.info("Current selector has no PolicyABC probability table.")

    st.subheader("Knowledge Storage")
    st.dataframe(knowledge_rows(dmp), width="stretch", hide_index=True)

    st.subheader("DMP Trace")
    st.dataframe(st.session_state.trace, width="stretch", hide_index=True)

    st.subheader("Action Template Library")
    st.dataframe(action_template_rows(), width="stretch", hide_index=True)

    st.subheader("Core Loop")
    st.markdown(
        """
```mermaid
flowchart LR
    A["Action"] --> B["Observation"]
    B --> C["KV Extraction"]
    C --> D["Knowledge Storage"]
    D --> E["KK Slot Binding"]
    E --> F["Candidate Actions"]
    F --> G["Policy / Prophecy / Imagination"]
    G --> A
```
"""
    )


def render_comparison_page() -> None:
    st.subheader("Original APASSR vs This GridWorld Project")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Original / prior setting**")
        st.code(
            "nmap {WHAT_OPTION} {HOW_OPTION} {PORT} {TARGET_IP}\n"
            "-> observation\n"
            "-> new KV such as PORT, SERVICE, PATH, TARGET_IP",
            language="text",
        )
    with right:
        st.markdown("**This project**")
        st.code(
            "MOVE_TOWARD {KK_FRONTIER_CELL}\n"
            "INSPECT_CELL {KK_UNKNOWN_NEIGHBOR}\n"
            "USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}",
            language="text",
        )

    st.subheader("Parallel Comparison")
    st.dataframe(paper_project_comparison_rows(), width="stretch", hide_index=True)

    st.subheader("Implementation Status")
    st.dataframe(implementation_status_rows(), width="stretch", hide_index=True)

    st.subheader("Condition Mapping")
    st.markdown(
        """
```text
C0 = RandomScorer
C1 = PolicyABC
C2 = PolicyABC + Prophecy reward
C3 = PolicyABC + Prophecy + Imagination
```
"""
    )


def ensure_state() -> None:
    if "dmp" not in st.session_state:
        reset_demo()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_step_result" not in st.session_state:
        st.session_state.last_step_result = None
    if "trace" not in st.session_state:
        st.session_state.trace = []


def reset_demo(selector_mode: str = "PolicyABC C1") -> None:
    world = GridWorld(
        width=6,
        height=4,
        start=(1, 1),
        cells={
            (2, 0): CellKind.WALL,
            (3, 1): CellKind.HINT,
            (1, 2): CellKind.KEY,
            (4, 2): CellKind.DOOR,
            (5, 2): CellKind.FLAG,
        },
        hints={(3, 1): (5, 2)},
    )
    prophecy = make_prophecy(selector_mode)
    st.session_state.dmp = GridWorldDMP(
        world,
        scorer=make_scorer(selector_mode),
        prophecy=prophecy,
        imagination=make_imagination(selector_mode, prophecy),
        config=make_config(selector_mode),
    )
    st.session_state.last_result = None
    st.session_state.last_step_result = None
    st.session_state.trace = []


def make_scorer(selector_mode: str) -> Any:
    if selector_mode == "Random C0":
        return RandomScorer(seed=0)
    if selector_mode == "KnowledgeGain debug":
        return KnowledgeGainScorer()
    return PolicyABC.uniform_gridworld(seed=0)


def make_prophecy(selector_mode: str) -> TableProphecyModel | None:
    if selector_mode in {"PolicyABC + Prophecy C2", "PolicyABC + Prophecy + Imagination C3"}:
        return TableProphecyModel()
    return None


def make_imagination(selector_mode: str, prophecy: TableProphecyModel | None) -> ImaginationCycle | None:
    if selector_mode == "PolicyABC + Prophecy + Imagination C3" and prophecy is not None:
        return ImaginationCycle(prophecy)
    return None


def make_config(selector_mode: str) -> DMPConfig:
    return DMPConfig(
        use_prophecy=selector_mode in {"PolicyABC + Prophecy C2", "PolicyABC + Prophecy + Imagination C3"},
        use_imagination=selector_mode == "PolicyABC + Prophecy + Imagination C3",
    )


def run_policy_step(strategy: str) -> bool:
    dmp: GridWorldDMP = st.session_state.dmp
    candidate = dmp.choose_candidate(strategy)
    if candidate is None:
        st.session_state.last_result = {"status": "No executable candidates."}
        return False
    execute_candidate(dmp, candidate, selected_by=f"policy:{strategy}")
    return True


def render_candidate_controls(dmp: GridWorldDMP) -> None:
    candidates = dmp.generate_candidates()
    if not candidates:
        st.info("No executable candidates.")
        return

    for index, candidate in enumerate(candidates, start=1):
        cols = st.columns([0.6, 2.5, 1])
        cols[0].markdown(f"**{index}**")
        cols[1].code(candidate_label(candidate), language="text")
        if cols[2].button("Execute", key=f"candidate-{index}", width="stretch"):
            execute_candidate(dmp, candidate, selected_by="manual")
            st.rerun()


def execute_candidate(dmp: GridWorldDMP, candidate: ActionCandidate, *, selected_by: str) -> None:
    before_pos = dmp.position
    result = dmp.execute(candidate)
    st.session_state.last_step_result = result
    st.session_state.last_result = result.to_dict()
    st.session_state.trace.append(
        trace_row(
            result=result,
            candidate=candidate,
            selected_by=selected_by,
            pos_before=before_pos,
            pos_after=dmp.position,
        )
    )


def render_grid_html(dmp: GridWorldDMP) -> str:
    cells = observed_cells(dmp)
    html = [
        "<style>",
        ".gridworld-board{display:grid;gap:6px;width:min(100%,520px);",
        f"grid-template-columns:repeat({dmp.world.width}, minmax(42px,1fr));}}",
        ".gridworld-cell{aspect-ratio:1/1;border:1px solid #cbd5e1;",
        "display:flex;align-items:center;justify-content:center;font-weight:700;",
        "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:20px;}",
        ".gridworld-legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}",
        ".gridworld-chip{display:inline-flex;align-items:center;gap:6px;border:1px solid #cbd5e1;",
        "padding:4px 8px;border-radius:6px;font-size:13px;}",
        "</style>",
        '<div class="gridworld-board">',
    ]
    for y in range(dmp.world.height):
        for x in range(dmp.world.width):
            symbol = cells[(x, y)]
            label, color, background = CELL_STYLES[symbol]
            html.append(
                '<div class="gridworld-cell" '
                f'title="{escape(label)} {(x, y)}" '
                f'style="color:{color};background:{background};">{escape(symbol)}</div>'
            )
    html.append("</div>")
    return "".join(html)


def render_legend_html() -> str:
    chips = ['<div class="gridworld-legend">']
    for symbol, (label, color, background) in CELL_STYLES.items():
        display = symbol if symbol != " " else "blank"
        chips.append(
            '<span class="gridworld-chip" '
            f'style="color:{color};background:{background};">'
            f"<strong>{escape(display)}</strong>{escape(label)}</span>"
        )
    chips.append("</div>")
    return "".join(chips)


def observed_cells(dmp: GridWorldDMP) -> dict[tuple[int, int], str]:
    cells = {(x, y): " " for y in range(dmp.world.height) for x in range(dmp.world.width)}
    for cell in values(dmp, KK.KNOWN_CELL, include_inactive=True):
        cells[cell] = "."
    for cell in values(dmp, KK.FRONTIER_CELL):
        cells[cell] = "?"
    for cell in values(dmp, KK.WALL_CELL, include_inactive=True):
        cells[cell] = "#"
    for cell in values(dmp, KK.HINT_CELL):
        cells[cell] = "H"
    for cell in values(dmp, KK.KEY_CELL, include_inactive=True):
        cells[cell] = "K"
    for cell in values(dmp, KK.DOOR_CELL, include_inactive=True):
        cells[cell] = "D"
    for cell in values(dmp, KK.FLAG_CELL):
        cells[cell] = "F"
    cells[dmp.position] = "A"
    return cells


def values(dmp: GridWorldDMP, kk: KK, *, include_inactive: bool = False) -> set[tuple[int, int]]:
    results: set[tuple[int, int]] = set()
    for kv in dmp.store.values(kk, include_inactive=include_inactive):
        if isinstance(kv.value, tuple) and len(kv.value) == 2:
            results.add(kv.value)
    return results


if __name__ == "__main__":
    main()
