from __future__ import annotations

import argparse

from aassr_v2.escape_gui import launch_escape_gui
from aassr_v2.escape_training import (
    EscapeTrainingConfig,
    TrainingMode,
    train_escape_agent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the colored-key escape GridWorld agent."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the desktop GUI with live and maximum-speed buttons",
    )
    parser.add_argument(
        "--mode",
        choices=(TrainingMode.LIVE.value, TrainingMode.FAST.value),
        default=TrainingMode.FAST.value,
        help="headless display mode; live sleeps between primitive steps",
    )
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--colors", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gui:
        launch_escape_gui()
        return

    config = EscapeTrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        color_count=args.colors,
        distractor_boxes=args.distractors,
        max_steps=args.max_steps,
    )

    def print_frame(frame: object) -> None:
        episode = getattr(frame, "episode")
        total = getattr(frame, "total_episodes")
        finished = getattr(frame, "episode_finished")
        mode = getattr(frame, "mode")
        if finished or mode is TrainingMode.LIVE:
            print(
                f"episode={episode}/{total} "
                f"step={getattr(frame, 'step')} "
                f"success={int(getattr(frame, 'success'))} "
                f"rolling={getattr(frame, 'rolling_success'):.3f} "
                f"epsilon={getattr(frame, 'epsilon'):.3f} "
                f"event={getattr(frame, 'event')}"
            )

    summary = train_escape_agent(
        config,
        mode=TrainingMode(args.mode),
        on_frame=print_frame,
    )
    print(
        "completed "
        f"episodes={summary.episodes} "
        f"success_rate={summary.success_rate:.4f} "
        f"rolling_success={summary.rolling_success:.4f} "
        f"elapsed={summary.elapsed_seconds:.3f}s "
        f"q_entries={summary.q_entries} "
        f"oracle_steps={summary.oracle_steps}"
    )


if __name__ == "__main__":
    main()
