from __future__ import annotations

import argparse
import json

from .dmp import APASSRToolDMP
from .novelty import NoveltyMemory
from .plugins import available_plugins, get_plugin
from .policy import PolicyABC
from .prophecy import TableProphecyModel
from .tools import ToolExecutor


def run_once(
    *,
    base_url: str,
    step_limit: int = 30,
    prefer_curl: bool = True,
    backend: str = "local",
    wsl_distro: str = "kali-linux",
    reward_observer: str = "none",
    objective: str = "balanced",
    plugin: str = "web",
    policy: PolicyABC | None = None,
    prophecy_model: TableProphecyModel | None = None,
    experience_memory: TableProphecyModel | None = None,
    novelty_memory: NoveltyMemory | None = None,
) -> dict[str, object]:
    executor = ToolExecutor(prefer_curl=prefer_curl, backend=backend, wsl_distro=wsl_distro)
    target_plugin = get_plugin(plugin)
    observer = target_plugin.reward_observer(reward_observer, base_url)
    objective_config = _objective_settings(objective)
    dmp = APASSRToolDMP(
        base_url=base_url,
        plugin=target_plugin,
        executor=executor,
        policy=policy,
        reward_observer=observer,
        prophecy_model=prophecy_model or experience_memory,
        novelty_memory=novelty_memory,
        novelty_reward=objective_config["novelty_reward"],
        novelty_score_weight=objective_config["novelty_score_weight"],
        knowledge_reward_cap=int(objective_config["knowledge_reward_cap"]),
        knowledge_reward_scale=objective_config["knowledge_reward_scale"],
        step_limit=step_limit,
    )
    result = dmp.run()
    return {
        "success": result.success,
        "steps": result.steps,
        "flag": result.flag,
        "solved_challenges": result.solved_challenges,
        "records": [record.__dict__ for record in result.records],
        "knowledge": result.knowledge_rows,
    }


def run_many(
    *,
    base_url: str,
    episodes: int,
    step_limit: int,
    prefer_curl: bool = True,
    backend: str = "local",
    wsl_distro: str = "kali-linux",
    reward_observer: str = "none",
    objective: str = "balanced",
    plugin: str = "web",
    include_records: bool = False,
    novelty_store: str | None = None,
    run_id: str = "default",
    reset_novelty: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    policy = PolicyABC(seed=seed)
    prophecy = TableProphecyModel(seed=seed)
    novelty = NoveltyMemory(persistence_path=novelty_store, run_id=run_id)
    if reset_novelty:
        novelty.reset(delete_persisted=True)
    episode_rows: list[dict[str, object]] = []
    for episode in range(episodes):
        output = run_once(
            base_url=base_url,
            step_limit=step_limit,
            prefer_curl=prefer_curl,
            backend=backend,
            wsl_distro=wsl_distro,
            reward_observer=reward_observer,
            objective=objective,
            plugin=plugin,
            policy=policy,
            prophecy_model=prophecy,
            novelty_memory=novelty,
        )
        row = {
            "episode": episode,
            "success": output["success"],
            "steps": output["steps"],
            "solved_count": len(output["solved_challenges"]),  # type: ignore[arg-type]
            "solved_challenges": output["solved_challenges"],
            "total_reward": sum(float(record["reward"]) for record in output["records"]),  # type: ignore[index]
            "novelty_bonus": sum(float(record.get("novelty_bonus", 0.0)) for record in output["records"]),  # type: ignore[union-attr]
            "challenge_progress_events": sum(1 for record in output["records"] if float(record.get("challenge_progress", 0.0)) > 0),  # type: ignore[union-attr]
            "semantic_novelty": sum(int(record.get("semantic_novelty", 0)) for record in output["records"]),  # type: ignore[union-attr]
            "repeated_action_rate": _record_rate(output["records"], "repeated_action"),  # type: ignore[arg-type]
            "repeated_response_rate": _record_rate(output["records"], "repeated_response"),  # type: ignore[arg-type]
            "no_progress_rate": _record_rate(output["records"], "penalty_no_progress"),  # type: ignore[arg-type]
            "predicted_progress_mean": _record_mean(output["records"], "predicted_progress_probability"),  # type: ignore[arg-type]
        }
        if include_records:
            row["records"] = output["records"]
        episode_rows.append(row)
    return {
        "episodes": episode_rows,
        "policy": {
            "what": {key.value: value for key, value in policy.what_probs.items()},
            "how": {key.value: value for key, value in policy.how_probs.items()},
            "where": {key.value: value for key, value in policy.where_probs.items()},
        },
        "prophecy": {
            "implementation": "TableProphecyModel",
            "stat_count": len(prophecy.stats),
            "top_reward": _top_prophecy(prophecy, by="reward"),
            "top_solved": _top_prophecy(prophecy, by="solved"),
        },
        "experience": {
            "stat_count": len(prophecy.stats),
            "top_reward": _top_prophecy(prophecy, by="reward"),
            "top_solved": _top_prophecy(prophecy, by="solved"),
        },
        "novelty": {
            "signature_count": len(novelty.signature_counts),
            "chain_count": len(novelty.chain_counts),
            "response_count": len(novelty.response_counts),
            "top_signatures": sorted(
                novelty.signature_counts.items(),
                key=lambda row: row[1],
                reverse=True,
            )[:10],
        },
        "plugin": plugin,
        "run_id": run_id,
        "seed": seed,
    }


def _record_rate(records: list[dict[str, object]], field: str) -> float:
    return sum(1 for row in records if row.get(field)) / len(records) if records else 0.0


def _record_mean(records: list[dict[str, object]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in records) / len(records) if records else 0.0


def _objective_settings(objective: str) -> dict[str, float]:
    if objective == "novelty":
        return {
            "novelty_reward": 1.0,
            "novelty_score_weight": 1.2,
            "knowledge_reward_cap": 3.0,
            "knowledge_reward_scale": 0.5,
        }
    if objective == "weird":
        return {
            "novelty_reward": 2.0,
            "novelty_score_weight": 2.5,
            "knowledge_reward_cap": 1.0,
            "knowledge_reward_scale": 0.2,
        }
    return {
        "novelty_reward": 0.0,
        "novelty_score_weight": 0.0,
        "knowledge_reward_cap": 5.0,
        "knowledge_reward_scale": 1.0,
    }


def _top_prophecy(prophecy: TableProphecyModel, *, by: str, limit: int = 10) -> list[dict[str, object]]:
    def sort_value(item):
        stat = item[1]
        if by == "solved":
            return stat.solved_rate  # type: ignore[attr-defined]
        return stat.reward_mean  # type: ignore[attr-defined]

    rows = sorted(prophecy.stats.items(), key=sort_value, reverse=True)[:limit]
    return [
        {
            "key": key,
            "count": stat.count,
            "reward_mean": stat.reward_mean,
            "knowledge_mean": stat.knowledge_mean,
            "solved_rate": stat.solved_rate,
            "error_rate": stat.error_rate,
        }
        for key, stat in rows
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local tool-backed APASSR prototype.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--condition", choices=["APASSR"], default="APASSR")
    parser.add_argument("--step-limit", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--backend", choices=["local", "wsl"], default="local")
    parser.add_argument("--wsl-distro", default="kali-linux")
    parser.add_argument("--reward-observer", choices=["none", "juice-shop"], default="none")
    parser.add_argument("--objective", choices=["balanced", "novelty", "weird"], default="balanced")
    parser.add_argument("--plugin", choices=available_plugins(), default="web")
    parser.add_argument("--include-records", action="store_true", help="Include per-step records in multi-episode JSON.")
    parser.add_argument("--novelty-store", help="Optional directory/file for persistent run-scoped novelty.")
    parser.add_argument("--run-id", default="default", help="Experiment id used to isolate persisted novelty.")
    parser.add_argument("--reset-novelty", action="store_true", help="Reset the selected persisted novelty ledger.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-curl", action="store_true", help="Use Python requests fallback instead of curl.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes > 1:
        output = run_many(
            base_url=args.base_url,
            episodes=args.episodes,
            step_limit=args.step_limit,
            prefer_curl=not args.no_curl,
            backend=args.backend,
            wsl_distro=args.wsl_distro,
            reward_observer=args.reward_observer,
            objective=args.objective,
            plugin=args.plugin,
            include_records=args.include_records,
            novelty_store=args.novelty_store,
            run_id=args.run_id,
            reset_novelty=args.reset_novelty,
            seed=args.seed,
        )
        if args.json:
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return
        for row in output["episodes"]:  # type: ignore[index]
            print(
                "episode={episode} success={success} steps={steps} solved_count={solved_count} total_reward={total_reward}".format(
                    **row
                )
            )
        return
    output = run_once(
        base_url=args.base_url,
        step_limit=args.step_limit,
        prefer_curl=not args.no_curl,
        backend=args.backend,
        wsl_distro=args.wsl_distro,
        reward_observer=args.reward_observer,
        objective=args.objective,
        plugin=args.plugin,
    )
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return
    print(
        f"success={output['success']} steps={output['steps']} flag={output['flag']} "
        f"solved={output['solved_challenges']}"
    )
    for row in output["records"]:  # type: ignore[index]
        print(
            "step={step} status={status} new_kv={new_kv} flag={flag_found} action={action}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
