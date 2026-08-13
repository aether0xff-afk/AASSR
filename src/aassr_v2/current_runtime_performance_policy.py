from __future__ import annotations

from .current_replay_performance import install_indexable_current_replays
from .current_runtime_performance import (
    PERFORMANCE_CONTRACT,
    _install_indexed_calibration,
    _install_status_mixture_fast_path,
)
from .current_runtime_performance_v2 import install_current_runtime_performance_v2


def install_current_runtime_performance_device_aware(agent: object) -> object:
    """Install only schedule- and semantics-preserving runtime fast paths.

    V1 removes replay/calibration Python work and avoidable CUDA synchronizations.
    V2 keeps independent ensemble parameters/optimizers but fuses CUDA *inference*
    across the ensemble dimension, and pre-packs Critic sequence tensors once per
    update instead of rebuilding them at every recurrent step.
    """

    if not getattr(agent, "current_runtime_performance", False):
        install_indexable_current_replays(agent)
        _install_indexed_calibration(agent.calibrated_prophecy)
        base = agent.base_neural_prophecy
        device = getattr(base, "device", None)
        device_type = str(getattr(device, "type", device or "cpu"))
        cuda_fast_path = device_type == "cuda"
        if cuda_fast_path:
            _install_status_mixture_fast_path(base)

        agent.current_runtime_performance = True
        agent.current_runtime_performance_contract = PERFORMANCE_CONTRACT
        agent.current_runtime_performance_device = device_type
        agent.current_runtime_cuda_fast_path = cuda_fast_path
        agent.current_runtime_calibration_index = True
        agent.current_runtime_indexable_replays = True
    else:
        device_type = str(
            getattr(agent, "current_runtime_performance_device", "cpu")
        )
        cuda_fast_path = bool(
            getattr(agent, "current_runtime_cuda_fast_path", device_type == "cuda")
        )

    # Batched CUDA GEMMs change the accumulation order of the independent Linear
    # modules and can violate the exact learning-counter contract.
    install_current_runtime_performance_v2(agent, enable_cuda_fusion=False)
    return agent
