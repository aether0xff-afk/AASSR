from __future__ import annotations

import argparse

from aassr_v2.escape_gui_modeled import launch_escape_gui
from aassr_v2.escape_training import EscapeTrainingConfig, TrainingMode
from aassr_v2.escape_training_modeled import train_escape_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Full AASSR in the colored-key escape GridWorld."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the desktop GUI with GridWorld, Imagination and model controls",
    )
    parser.add_argument(
        "--mode",
        choices=(TrainingMode.LIVE.value, TrainingMode.FAST.value),
        default=TrainingMode.FAST.value,
        help="initial headless display mode; the environment has no tick timeout",
    )
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--colors", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--output",
        default=None,
        help="exact result directory; default creates runs/escape_gridworld/<timestamp>_...",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        help="load a .aassr-model.gz file and continue training from its learned state",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="also save the final portable model to this path",
    )
    parser.add_argument(
        "--no-auto-save-model",
        action="store_true",
        help="disable the automatic <output>/models/final.aassr-model.gz save",
    )
    parser.add_argument(
        "--no-imagination",
        action="store_true",
        help="run the contextual-policy ablation instead of Full AASSR",
    )
    parser.add_argument(
        "--no-episode-checkpoints",
        action="store_true",
        help="skip per-episode compressed recovery checkpoints; final checkpoint is still saved",
    )
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
        use_imagination=not args.no_imagination,
        save_episode_checkpoints=not args.no_episode_checkpoints,
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
                f"score={getattr(frame, 'episode_score'):.4f} "
                f"rolling_score={getattr(frame, 'rolling_score'):.4f} "
                f"epsilon={getattr(frame, 'epsilon'):.3f} "
                f"imagination={int(getattr(frame, 'used_imagination'))} "
                f"imagined_nodes={getattr(frame, 'imagined_nodes')} "
                f"event={getattr(frame, 'event')}"
            )

    summary = train_escape_agent(
        config,
        mode=TrainingMode(args.mode),
        on_frame=print_frame,
        output_dir=args.output,
        load_model_path=args.load_model,
        save_model_path=args.save_model,
        auto_save_final_model=not args.no_auto_save_model,
    )
    print(
        "completed "
        f"episodes={summary.episodes} "
        f"success_rate={summary.success_rate:.4f} "
        f"total_steps={summary.total_steps} "
        f"mean_score={summary.mean_score:.4f} "
        f"rolling_score={summary.rolling_score:.4f} "
        f"elapsed={summary.elapsed_seconds:.3f}s "
        f"policy_entries={summary.policy_entries} "
        f"imagination_decisions={summary.imagination_decisions} "
        f"imagined_nodes={summary.imagined_nodes} "
        f"oracle_steps={summary.oracle_steps} "
        f"output={summary.output_dir}"
    )


if __name__ == "__main__":
    main()
