from __future__ import annotations

import pytest

pytest.importorskip("torch")

from aassr_v2 import build_current_pentest_aassr_core
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def test_current_depth_batcher_executes_neural_delta_batch_path() -> None:
    agent = build_current_pentest_aassr_core(
        seed=7,
        train_transitions=64,
        use_imagination=True,
        device="cpu",
    )
    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions[:2])
    assert actions

    states = tuple(state for _ in actions)
    memories = tuple(None for _ in actions)
    rows = agent.current_batched_prophecy.predict_step_batch(
        states,
        actions,
        memories,
        samples=1,
    )

    assert len(rows) == len(actions)
    assert all(row.predictions for row in rows)
    diagnostics = agent.current_batched_prophecy.runtime_diagnostics()
    assert diagnostics["current_imagination_batch_calls"] == 1
    assert diagnostics["current_imagination_batch_rows"] == len(actions)
    assert diagnostics["current_imagination_skill_fallback_rows"] == 0
    assert agent.base_neural_prophecy.batch_prediction_calls >= 1
