from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2.current_entrypoint import build_current_pentest_aassr_core


def test_cpu_runtime_keeps_only_device_independent_performance_path() -> None:
    agent = build_current_pentest_aassr_core(
        seed=71,
        train_transitions=64,
        device="cpu",
        allow_tf32=False,
    )
    assert agent.current_runtime_performance is True
    assert agent.current_runtime_performance_device == "cpu"
    assert agent.current_runtime_calibration_index is True
    assert agent.current_runtime_cuda_fast_path is False
    assert hasattr(agent.calibrated_prophecy, "performance_holdout_index_rebuilds")
    assert not hasattr(agent.base_neural_prophecy, "performance_host_transfer_batches")
