from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from .dmp import APASSRToolDMP, StepRecord
from .experiment import _objective_settings, _top_prophecy
from .novelty import NoveltyMemory
from .policy import How, PolicyABC, What, Where
from .plugins import available_plugins, get_plugin
from .prophecy import TableProphecyModel
from .reward import JuiceShopChallengeObserver, RewardObserver
from .tools import ToolExecutor


def train_juice_shop(
    *,
    base_url: str,
    episodes: int,
    step_limit: int,
    output_dir: str | Path,
    objective: str = "balanced",
    prefer_curl: bool = True,
    backend: str = "local",
    wsl_distro: str = "kali-linux",
    plugin: str = "web",
    include_records: bool = False,
    checkpoint_every: int = 1,
    stop_when_all_solved: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    episodes_path = output_path / "juice_train_episodes.jsonl"
    records_path = output_path / "juice_train_records.jsonl"
    summary_path = output_path / "juice_train_summary.json"
    checkpoint_path = output_path / "checkpoint_latest.json"

    policy = PolicyABC()
    prophecy = TableProphecyModel()
    novelty = NoveltyMemory()
    target_plugin = get_plugin(plugin)
    observer = target_plugin.reward_observer("juice-shop", base_url)
    if observer is None:
        raise ValueError(f"plugin {plugin} does not provide a juice-shop reward observer")
    objective_config = _objective_settings(objective)
    episode_rows: list[dict[str, Any]] = []
    started_at = time.time()

    for episode in range(episodes):
        executor = ToolExecutor(prefer_curl=prefer_curl, backend=backend, wsl_distro=wsl_distro)
        dmp = APASSRToolDMP(
            base_url=base_url,
            plugin=target_plugin,
            executor=executor,
            policy=policy,
            reward_observer=observer,
            prophecy_model=prophecy,
            novelty_memory=novelty,
            novelty_reward=objective_config["novelty_reward"],
            novelty_score_weight=objective_config["novelty_score_weight"],
            knowledge_reward_cap=int(objective_config["knowledge_reward_cap"]),
            knowledge_reward_scale=objective_config["knowledge_reward_scale"],
            step_limit=step_limit,
        )
        result = dmp.run()
        row = _episode_row(
            episode=episode,
            result_records=result.records,
            success=result.success,
            steps=result.steps,
            new_solved=result.solved_challenges,
            observer=observer,
            prophecy=prophecy,
            novelty=novelty,
        )
        episode_rows.append(row)
        _append_jsonl(episodes_path, row)
        if include_records:
            for record in result.records:
                _append_jsonl(records_path, {"episode": episode, **asdict(record)})

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            _write_json(checkpoint_path, _checkpoint(policy, prophecy, novelty, observer, episode_rows))
        _print_progress(row, episode=episode, episodes=episodes, started_at=started_at)

        total = observer.challenge_total
        if stop_when_all_solved and total > 0 and len(observer.solved_keys) >= total:
            break

    summary = _checkpoint(policy, prophecy, novelty, observer, episode_rows)
    summary["run"] = {
        "base_url": base_url,
        "episodes_requested": episodes,
        "episodes_completed": len(episode_rows),
        "step_limit": step_limit,
        "objective": objective,
        "backend": backend,
        "plugin": plugin,
        "prefer_curl": prefer_curl,
        "elapsed_s": round(time.time() - started_at, 3),
    }
    _write_json(summary_path, summary)
    _write_json(checkpoint_path, summary)
    return summary


def _episode_row(
    *,
    episode: int,
    result_records: list[StepRecord],
    success: bool,
    steps: int,
    new_solved: list[str],
    observer: JuiceShopChallengeObserver,
    prophecy: TableProphecyModel,
    novelty: NoveltyMemory,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "success": success,
        "steps": steps,
        "new_solved_count": len(new_solved),
        "new_solved": new_solved,
        "cumulative_solved": len(observer.solved_keys),
        "challenge_total": observer.challenge_total,
        "total_reward": sum(record.reward for record in result_records),
        "policy_reward": sum(record.policy_reward for record in result_records),
        "novelty_bonus": sum(record.novelty_bonus for record in result_records),
        "new_kv_total": sum(record.new_kv for record in result_records),
        "error_count": sum(1 for record in result_records if record.status >= 400 or record.status == 0),
        "prophecy_stat_count": len(prophecy.stats),
        "novelty_signature_count": len(novelty.signature_counts),
        "last_action": result_records[-1].action if result_records else "",
    }


def _checkpoint(
    policy: PolicyABC,
    prophecy: TableProphecyModel,
    novelty: NoveltyMemory,
    observer: JuiceShopChallengeObserver,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "solved": {
            "count": len(observer.solved_keys),
            "challenge_total": observer.challenge_total,
            "keys": list(observer.solved_keys),
        },
        "episodes": episodes,
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
        "novelty": {
            "signature_count": len(novelty.signature_counts),
            "chain_count": len(novelty.chain_counts),
            "response_count": len(novelty.response_counts),
            "top_signatures": sorted(novelty.signature_counts.items(), key=lambda row: row[1], reverse=True)[:10],
        },
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_progress(row: dict[str, Any], *, episode: int, episodes: int, started_at: float) -> None:
    elapsed = time.time() - started_at
    completed = episode + 1
    eta = elapsed / completed * max(episodes - completed, 0) if completed else 0.0
    solved = row["cumulative_solved"]
    total = row["challenge_total"] or "?"
    values = dict(row)
    values["episode"] = completed
    print(
        "episode={episode}/{episodes} solved={solved}/{total} new={new_solved_count} "
        "steps={steps} reward={total_reward:.2f} kv={new_kv_total} "
        "prophecy_stats={prophecy_stat_count} elapsed={elapsed:.1f}s eta={eta:.1f}s".format(
            episodes=episodes,
            solved=solved,
            total=total,
            elapsed=elapsed,
            eta=eta,
            **values,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train APASSR continuously on a local Juice Shop target.")
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--output-dir", default="runs/juice_train")
    parser.add_argument("--objective", choices=["balanced", "novelty", "weird"], default="balanced")
    parser.add_argument("--backend", choices=["local", "wsl"], default="local")
    parser.add_argument("--wsl-distro", default="kali-linux")
    parser.add_argument("--plugin", choices=available_plugins(), default="web")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--stop-when-all-solved", action="store_true")
    parser.add_argument("--no-curl", action="store_true", help="Use Python requests fallback instead of curl.")
    args = parser.parse_args(argv)
    train_juice_shop(
        base_url=args.base_url,
        episodes=args.episodes,
        step_limit=args.step_limit,
        output_dir=args.output_dir,
        objective=args.objective,
        prefer_curl=not args.no_curl,
        backend=args.backend,
        wsl_distro=args.wsl_distro,
        plugin=args.plugin,
        include_records=args.include_records,
        checkpoint_every=args.checkpoint_every,
        stop_when_all_solved=args.stop_when_all_solved,
    )


if __name__ == "__main__":
    main()
