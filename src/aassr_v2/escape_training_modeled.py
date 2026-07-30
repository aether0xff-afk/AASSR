from __future__ import annotations

from pathlib import Path
import threading
from typing import Callable

from . import escape_training as training_core
from .escape_imagination_capture import ImaginationCallback
from .escape_training import (
    CompleteCallback,
    EscapeTrainingConfig,
    EscapeTrainingSummary,
    FrameCallback,
    TrainingMode,
    TrainingRuntime,
)
from .escape_training_visualized import train_escape_agent as _train_visualized
from .model_io import ModelLoadInfo, ModelManagedAgent, load_agent_model


AgentReadyCallback = Callable[[ModelManagedAgent, ModelLoadInfo | None], None]
_MODEL_FACTORY_LOCK = threading.Lock()


def train_escape_agent(
    config: EscapeTrainingConfig,
    *,
    mode: TrainingMode = TrainingMode.FAST,
    runtime: TrainingRuntime | None = None,
    on_frame: FrameCallback | None = None,
    on_complete: CompleteCallback | None = None,
    on_imagination: ImaginationCallback | None = None,
    on_agent_ready: AgentReadyCallback | None = None,
    stop_event: threading.Event | None = None,
    output_dir: str | Path | None = None,
    load_model_path: str | Path | None = None,
    save_model_path: str | Path | None = None,
    auto_save_final_model: bool = True,
) -> EscapeTrainingSummary:
    """Run visualized training with portable model loading and final saving."""

    holder: dict[str, ModelManagedAgent] = {}
    with _MODEL_FACTORY_LOCK:
        original_factory = training_core._make_agent

        def model_factory(factory_config: EscapeTrainingConfig) -> ModelManagedAgent:
            base_agent = original_factory(factory_config)
            load_info = None
            if load_model_path is not None:
                load_info = load_agent_model(
                    base_agent,
                    load_model_path,
                    expected_training_config=factory_config,
                )
            managed = ModelManagedAgent(
                base_agent,
                base_episode_offset=(
                    load_info.completed_episodes if load_info is not None else 0
                ),
            )
            holder["agent"] = managed
            if on_agent_ready is not None:
                on_agent_ready(managed, load_info)
            return managed

        training_core._make_agent = model_factory
        try:
            summary = _train_visualized(
                config,
                mode=mode,
                runtime=runtime,
                on_frame=on_frame,
                on_complete=on_complete,
                on_imagination=on_imagination,
                stop_event=stop_event,
                output_dir=output_dir,
            )
        finally:
            training_core._make_agent = original_factory

    managed = holder.get("agent")
    if managed is not None:
        if auto_save_final_model:
            automatic = Path(summary.output_dir) / "models" / "final.aassr-model.gz"
            managed.save_model(
                automatic,
                training_config=config,
                label="automatic final model",
                notes=f"session_output={summary.output_dir}",
            )
        if save_model_path is not None:
            managed.save_model(
                save_model_path,
                training_config=config,
                label="requested final model",
                notes=f"session_output={summary.output_dir}",
            )
    return summary


__all__ = [
    "AgentReadyCallback",
    "EscapeTrainingConfig",
    "EscapeTrainingSummary",
    "ModelLoadInfo",
    "ModelManagedAgent",
    "TrainingMode",
    "TrainingRuntime",
    "train_escape_agent",
]
