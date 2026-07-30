from __future__ import annotations

from pathlib import Path
import threading

from .escape_imagination_capture import (
    EscapeImaginationEvent,
    ImaginationCallback,
    ImaginationEventStream,
    capture_imaginations,
    default_visualized_output_dir,
)
from .escape_training import (
    CompleteCallback,
    EscapeTrainingConfig,
    EscapeTrainingSummary,
    FrameCallback,
    TrainingMode,
    TrainingRuntime,
    train_escape_agent as _train_escape_agent,
)


def train_escape_agent(
    config: EscapeTrainingConfig,
    *,
    mode: TrainingMode = TrainingMode.FAST,
    runtime: TrainingRuntime | None = None,
    on_frame: FrameCallback | None = None,
    on_complete: CompleteCallback | None = None,
    on_imagination: ImaginationCallback | None = None,
    stop_event: threading.Event | None = None,
    output_dir: str | Path | None = None,
) -> EscapeTrainingSummary:
    """Run the normal trainer while capturing every complete imagination tree."""

    resolved_output = (
        Path(output_dir) if output_dir is not None else default_visualized_output_dir()
    )
    stream = ImaginationEventStream(resolved_output, callback=on_imagination)
    try:
        with capture_imaginations(stream.record):
            return _train_escape_agent(
                config,
                mode=mode,
                runtime=runtime,
                on_frame=on_frame,
                on_complete=on_complete,
                stop_event=stop_event,
                output_dir=resolved_output,
            )
    finally:
        stream.close()


__all__ = [
    "EscapeImaginationEvent",
    "EscapeTrainingConfig",
    "EscapeTrainingSummary",
    "TrainingMode",
    "TrainingRuntime",
    "train_escape_agent",
]
