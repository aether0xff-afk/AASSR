from __future__ import annotations

from .current_performance import enable_current_depth_batching
from .current_runtime import (
    CurrentPentestRuntimeAgent,
    build_current_pentest_aassr_core as _build_safe_current,
)


def build_current_pentest_aassr_core(
    *,
    seed: int = 0,
    train_transitions: int = 10_000,
    use_imagination: bool = True,
    device: str = "cpu",
    enable_batching: bool = True,
) -> CurrentPentestRuntimeAgent:
    """Build the active current-generation AASSR stack.

    Historical v0.4 builders remain importable for reproduction. New experiments
    should use this entrypoint, which installs methodology-safe calibration first
    and then the semantics-preserving current Neural-Delta depth batcher.
    """

    agent = _build_safe_current(
        seed=int(seed),
        train_transitions=int(train_transitions),
        use_imagination=bool(use_imagination),
        device=device,
    )
    if enable_batching:
        enable_current_depth_batching(agent)
    return agent
