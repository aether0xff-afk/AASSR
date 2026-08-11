from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import run_imagination_gate_ablation as gate
import run_imagination_intervention_trace as detail


DEFAULT_OUTPUT = "runs/imagination_intervention_trace"


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(bool(getattr(stream, "isatty", lambda: False)()) for stream in self.streams)


def _arg_value(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def _short(value: Any, width: int = 42) -> str:
    text = str(value or "-")
    return text if len(text) <= width else text[: width - 3] + "..."


def _learning_snapshot(agent: object) -> dict[str, Any]:
    try:
        return dict(gate._learning_counters(agent))
    except Exception as exc:  # pragma: no cover - diagnostics must never stop a run
        return {"diagnostic_error": repr(exc)}


def _print_learning(prefix: str, agent: object) -> None:
    counters = _learning_snapshot(agent)
    critic_ready = bool(getattr(agent, "critic_ready", False))
    print(
        f"{prefix} critic_ready={critic_ready} learning={counters}",
        flush=True,
    )


def _install_runtime_logging() -> None:
    original_builder = gate.build_current_pentest_aassr_core
    original_episode = gate.run_current_episode
    original_frozen_eval = gate._run_aassr_frozen_eval

    def verbose_builder(*args: Any, **kwargs: Any) -> object:
        print("\n[AASSR] building current-generation agent", flush=True)
        print(
            "[AASSR] "
            f"seed={kwargs.get('seed', '?')} "
            f"train_transitions={kwargs.get('train_transitions', '?')} "
            f"device={kwargs.get('device', '?')} "
            f"use_imagination={kwargs.get('use_imagination', '?')}",
            flush=True,
        )
        started = time.perf_counter()
        agent = original_builder(*args, **kwargs)
        print(f"[AASSR] agent ready in {time.perf_counter() - started:.2f}s", flush=True)
        _print_learning("[AASSR] initial", agent)

        original_record = agent._record_decision
        decision_index = 0

        def verbose_record(decision: Any) -> Any:
            nonlocal decision_index
            decision_index += 1
            policy = _short(getattr(decision, "policy_action_signature", ""))
            preferred = _short(
                getattr(decision, "imagination_preferred_action_signature", "")
            )
            executed = _short(getattr(getattr(decision, "action", None), "signature", ""))
            used = bool(getattr(decision, "used_imagination", False))
            eligible = bool(getattr(decision, "imagination_eligible", False))
            changed = bool(getattr(decision, "imagination_changed_action", False))
            reason = getattr(decision, "imagination_gate_reason", "-")
            coverage = float(getattr(decision, "model_coverage", 0.0))
            advantage = float(getattr(decision, "imagination_advantage", 0.0))
            required = float(getattr(decision, "imagination_required_advantage", 0.0))
            nodes = int(getattr(decision, "imagined_nodes", 0))
            depth = int(getattr(decision, "imagination_depth", 0))
            marker = "INTERVENE" if changed else ("IMAGINE" if used else "POLICY")
            print(
                f"[DECISION {decision_index:05d}] {marker:<9} "
                f"gate={reason:<34} cov={coverage:0.3f} "
                f"adv={advantage:+0.4f}/{required:0.4f} "
                f"nodes={nodes:<4} depth={depth:<2} eligible={eligible} | "
                f"policy={policy} | preferred={preferred} | exec={executed}",
                flush=True,
            )
            return original_record(decision)

        agent._record_decision = verbose_record
        return agent

    def verbose_episode(*args: Any, **kwargs: Any) -> Any:
        agent = args[0] if args else kwargs.get("agent")
        phase = kwargs.get("phase", "?")
        block = kwargs.get("block", "?")
        episode = kwargs.get("episode", "?")
        stage_index = int(kwargs.get("stage_index", -1))
        scenario_seed = kwargs.get("scenario_seed", "?")
        transition_start = int(kwargs.get("transition_start", 0))
        transition_budget = int(kwargs.get("transition_budget", 0))
        transition_cap = kwargs.get("transition_cap", "?")
        stage_name = (
            gate.TRANSFER_STAGES[stage_index].name
            if 0 <= stage_index < len(gate.TRANSFER_STAGES)
            else "?"
        )
        percent = (
            100.0 * transition_start / transition_budget
            if transition_budget > 0
            else 0.0
        )
        print(
            "\n"
            f"[EP START] phase={phase} block={block} ep={episode} "
            f"stage=L{stage_index}:{stage_name} scenario={scenario_seed} "
            f"progress={transition_start}/{transition_budget} ({percent:5.1f}%) "
            f"cap={transition_cap}",
            flush=True,
        )
        started = time.perf_counter()
        row, consumed = original_episode(*args, **kwargs)
        elapsed = time.perf_counter() - started
        rate = consumed / elapsed if elapsed > 0 else 0.0
        print(
            f"[EP END  ] status={row.status:<10} success={row.success} "
            f"failure={row.failure} stalled={row.stalled} trunc={row.truncation} "
            f"consumed={consumed:<4} total={row.transition_total:<5} "
            f"reward={row.reward:+.2f} aseq={row.aseq_guard_events:<4} "
            f"img_runs={row.imagination_runs:<4} interventions={row.imagination_interventions:<3} "
            f"changed={row.imagination_changed_actions:<3} "
            f"time={elapsed:6.2f}s rate={rate:6.2f} tr/s",
            flush=True,
        )
        if agent is not None:
            _print_learning("[LEARN   ]", agent)
        return row, consumed

    def verbose_frozen_eval(*args: Any, **kwargs: Any) -> Any:
        stage_index = int(kwargs.get("stage_index", -1))
        seeds = tuple(kwargs.get("scenario_seeds", ()))
        phase = kwargs.get("phase", "?")
        use_imagination = bool(kwargs.get("use_imagination", False))
        print(
            "\n"
            f"[EVAL START] phase={phase} stage=L{stage_index} "
            f"seeds={list(seeds)} imagination={use_imagination}",
            flush=True,
        )
        started = time.perf_counter()
        rows = original_frozen_eval(*args, **kwargs)
        elapsed = time.perf_counter() - started
        successes = sum(int(row.success) for row in rows)
        failures = sum(int(row.failure) for row in rows)
        stalls = sum(int(row.stalled) for row in rows)
        truncations = sum(int(row.truncation) for row in rows)
        transitions = sum(int(row.primitive_transitions) for row in rows)
        img_runs = sum(int(row.imagination_runs) for row in rows)
        interventions = sum(int(row.imagination_interventions) for row in rows)
        changed = sum(int(row.imagination_changed_actions) for row in rows)
        print(
            f"[EVAL END  ] episodes={len(rows)} success={successes}/{len(rows)} "
            f"failure={failures} stalled={stalls} trunc={truncations} "
            f"transitions={transitions} img_runs={img_runs} "
            f"interventions={interventions} changed={changed} time={elapsed:.2f}s",
            flush=True,
        )
        return rows

    gate.build_current_pentest_aassr_core = verbose_builder
    gate.run_current_episode = verbose_episode
    gate._run_aassr_frozen_eval = verbose_frozen_eval


def _install_tee() -> Any:
    output = Path(_arg_value("--output-dir", DEFAULT_OUTPUT))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "live.log"
    handle = path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)
    print("=" * 100, flush=True)
    print(
        f"[RUN] pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"cwd={Path.cwd()}",
        flush=True,
    )
    print(f"[RUN] argv={' '.join(sys.argv)}", flush=True)
    print(f"[RUN] persistent live log: {path}", flush=True)
    return handle


def main() -> None:
    handle = _install_tee()
    started = time.perf_counter()
    try:
        _install_runtime_logging()
        detail.main()
    except BaseException:
        print(
            f"\n[RUN FAILED] elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )
        raise
    else:
        print(
            f"\n[RUN COMPLETE] elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )
    finally:
        handle.flush()


if __name__ == "__main__":
    main()
