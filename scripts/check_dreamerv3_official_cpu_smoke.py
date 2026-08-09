from __future__ import annotations

import argparse
import json
from pathlib import Path

from aassr_v2.dreamerv3_external import (
    _DreamerPentestEnv,
    _DreamerTrainState,
    _load_official_dreamer,
    _make_agent,
    _official_config,
    _run_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the pinned official DreamerV3 Agent/Replay/Driver/train API "
            "against the current relational pentest adapter on CPU. This is a "
            "non-canonical debug smoke, not a benchmark result."
        )
    )
    parser.add_argument("--dreamer-root", required=True)
    parser.add_argument(
        "--output", default="runs/dreamerv3_official_cpu_smoke.json"
    )
    parser.add_argument("--real-transitions", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.real_transitions < 1:
        raise ValueError("real-transitions must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    upstream = _load_official_dreamer(
        args.dreamer_root,
        allow_upstream_mismatch=False,
    )
    config = _official_config(
        upstream,
        output_dir=output.parent,
        research_seed=args.seed,
        jax_platform="cpu",
        train_ratio=8.0,
        prealloc=False,
    )

    # Overlay the official upstream debug preset only for API/contract smoke cost.
    # Final suite assembly explicitly rejects CPU, train_ratio=8, and noncanonical
    # config artifacts, so this path cannot become a benchmark result.
    configs = upstream["yaml"].YAML(typ="safe").load(
        (upstream["root"] / "dreamerv3" / "configs.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = config.update(configs["debug"])
    config = config.update(
        logdir=str((output.parent / "official_dreamer_cpu_smoke").resolve()),
        seed=int(args.seed),
    )
    config = config.update(
        {
            "jax": {
                "platform": "cpu",
                "prealloc": False,
            },
            "run": {"train_ratio": 8.0},
        }
    )

    probe = _DreamerPentestEnv(
        elements=upstream["elements"],
        np=upstream["np"],
        research_seed=args.seed,
        stage_index=0,
        scenario_seed=90_001,
        transition_cap=1,
        phase="probe",
    )
    agent = _make_agent(upstream, config, probe)
    replay = upstream["main"].make_replay(config, "replay")
    train_stream = iter(
        agent.stream(upstream["main"].make_stream(config, replay, "train"))
    )
    train_carry = [agent.init_train(int(config.batch_size))]
    train_state = _DreamerTrainState()

    consumed = 0
    episodes = 0
    train_rows = []
    while consumed < args.real_transitions:
        cap = min(24, args.real_transitions - consumed)
        row = _run_episode(
            upstream=upstream,
            agent=agent,
            research_seed=args.seed,
            stage_index=0,
            scenario_seed=90_001 + (episodes % 16),
            transition_cap=cap,
            phase="cpu_smoke_train",
            mode="train",
            replay=replay,
            train_stream=train_stream,
            train_carry=train_carry,
            train_state=train_state,
            batch_size=int(config.batch_size),
            batch_length=int(config.batch_length),
            train_ratio=8.0,
        )
        if row.primitive_transitions <= 0:
            raise RuntimeError(
                "official DreamerV3 CPU smoke made no real progress"
            )
        consumed += row.primitive_transitions
        episodes += 1
        train_rows.append(row)

    if consumed != args.real_transitions:
        raise AssertionError((consumed, args.real_transitions))
    if train_state.real_transitions != consumed:
        raise AssertionError((train_state.real_transitions, consumed))
    if train_state.gradient_updates <= 0:
        raise AssertionError(
            "official DreamerV3 CPU smoke never reached agent.train(); increase "
            "--real-transitions if upstream replay requirements change"
        )

    updates_before_eval = train_state.gradient_updates
    eval_row = _run_episode(
        upstream=upstream,
        agent=agent,
        research_seed=args.seed,
        stage_index=0,
        scenario_seed=93_001,
        transition_cap=24,
        phase="cpu_smoke_eval",
        mode="eval",
        replay=None,
        train_stream=None,
        train_carry=None,
        train_state=None,
        batch_size=int(config.batch_size),
        batch_length=int(config.batch_length),
        train_ratio=8.0,
    )
    if train_state.gradient_updates != updates_before_eval:
        raise AssertionError("official DreamerV3 eval mutated training updates")

    result = {
        "status": "official_dreamerv3_cpu_api_verified",
        "canonical_benchmark": False,
        "upstream_commit": upstream["head"],
        "jax_platform": "cpu",
        "upstream_debug_preset": True,
        "real_transitions": consumed,
        "gradient_updates": train_state.gradient_updates,
        "training_episodes": episodes,
        "training_successes": sum(row.success for row in train_rows),
        "eval_status": eval_row.status,
        "eval_transitions": eval_row.primitive_transitions,
        "projection_max_squared_distance": max(
            [row.projection_max_squared_distance for row in train_rows]
            + [eval_row.projection_max_squared_distance]
        ),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
