from __future__ import annotations

import argparse
import json

from aassr_v2.critic_prophecy_audit import run_audit
from aassr_v2.critic_prophecy_common import AuditConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two branch critics and isolated one-step Prophecy models."
    )
    parser.add_argument("--output", default="runs/critic_prophecy_audit")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-map-count", type=int, default=32)
    parser.add_argument("--unseen-map-count", type=int, default=32)
    parser.add_argument("--behavior-train-episodes", type=int, default=800)
    parser.add_argument("--critic-train-episodes", type=int, default=600)
    parser.add_argument("--critic-eval-episodes", type=int, default=200)
    parser.add_argument("--prophecy-train-episodes", type=int, default=800)
    parser.add_argument("--prophecy-eval-episodes", type=int, default=300)
    parser.add_argument("--prophecy-epochs", type=int, default=1)
    parser.add_argument("--pruning-depth", type=int, default=5)
    parser.add_argument("--pruning-beam-width", type=int, default=4)
    args = parser.parse_args()
    payload = run_audit(
        args.output,
        AuditConfig(
            seed=args.seed,
            train_map_count=args.train_map_count,
            unseen_map_count=args.unseen_map_count,
            behavior_train_episodes=args.behavior_train_episodes,
            critic_train_episodes=args.critic_train_episodes,
            critic_eval_episodes=args.critic_eval_episodes,
            prophecy_train_episodes=args.prophecy_train_episodes,
            prophecy_eval_episodes=args.prophecy_eval_episodes,
            prophecy_epochs=args.prophecy_epochs,
            pruning_depth=args.pruning_depth,
            pruning_beam_width=args.pruning_beam_width,
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
