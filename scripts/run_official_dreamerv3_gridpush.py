from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _has_flag(arguments: Sequence[str], name: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in arguments
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--benchmark-mode",
        choices=("train", "seen", "unseen"),
        required=True,
    )
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--episode-log", type=Path, required=True)
    parser.add_argument("--target-episodes", type=int, default=0)
    parser.add_argument("--train-map-count", type=int, default=64)
    parser.add_argument("--evaluation-map-count", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, dreamer_args = parser.parse_known_args(argv)

    import dreamerv3.main as dreamer_main

    from aassr_v2.dreamerv3_gridpush_env import (
        BenchmarkEvaluationComplete,
        DreamerV3GridPushEnv,
    )

    def make_env(config, index, **overrides):
        del overrides
        environment = DreamerV3GridPushEnv(
            args.benchmark_mode,
            seed=args.benchmark_seed,
            train_map_count=args.train_map_count,
            evaluation_map_count=args.evaluation_map_count,
            target_episodes=args.target_episodes,
            episode_log=args.episode_log,
            worker_index=index,
        )
        return dreamer_main.wrap_env(environment, config)

    dreamer_main.make_env = make_env
    forwarded = list(dreamer_args)
    if not _has_flag(forwarded, "--seed"):
        forwarded.extend(("--seed", str(args.benchmark_seed)))
    if not _has_flag(forwarded, "--task"):
        forwarded.extend(
            ("--task", f"aassr_{args.benchmark_mode}")
        )

    try:
        dreamer_main.main(forwarded)
    except BenchmarkEvaluationComplete as exc:
        if args.target_episodes <= 0:
            raise
        print(f"DreamerV3 benchmark evaluation complete: {exc}")


if __name__ == "__main__":
    main()
