from __future__ import annotations

from types import SimpleNamespace

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


def test_cuda_policy_does_not_enable_non_exact_ensemble_fusion(monkeypatch) -> None:
    from aassr_v2 import current_runtime_performance_policy as policy

    calls: list[bool] = []
    agent = SimpleNamespace(
        current_runtime_performance=False,
        base_neural_prophecy=SimpleNamespace(device=SimpleNamespace(type="cuda")),
        calibrated_prophecy=object(),
        critic=object(),
    )
    monkeypatch.setattr(policy, "install_indexable_current_replays", lambda value: value)
    monkeypatch.setattr(policy, "_install_indexed_calibration", lambda value: None)
    monkeypatch.setattr(policy, "_install_status_mixture_fast_path", lambda value: None)
    monkeypatch.setattr(
        policy,
        "install_current_runtime_performance_v2",
        lambda value, *, enable_cuda_fusion: calls.append(enable_cuda_fusion),
    )

    policy.install_current_runtime_performance_device_aware(agent)

    assert agent.current_runtime_cuda_fast_path is True
    assert calls == [False]
