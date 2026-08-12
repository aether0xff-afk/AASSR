from __future__ import annotations

from .current_replay_performance import install_indexable_current_replays
from .current_runtime_performance import (
    PERFORMANCE_CONTRACT,
    _install_indexed_calibration,
    _install_status_mixture_fast_path,
)


def install_current_runtime_performance_device_aware(agent: object) -> object:
    """Install only schedule- and semantics-preserving runtime fast paths.

    O(1) replay indexing and the calibration holdout index remove Python work on
    both CPU and CUDA. Tensor packing and deferred device synchronization target
    accelerator transfer latency; on CPU they only add tensor packing/copy
    overhead, so those changes are enabled on CUDA only.
    """

    if getattr(agent, "current_runtime_performance", False):
        return agent

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
    return agent
