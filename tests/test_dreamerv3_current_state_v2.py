from __future__ import annotations

from aassr_v2.current_relational_state import relational_state_vector_v2
from aassr_v2.dreamerv3_baseline import (
    dreamer_adapter_manifest,
    dreamer_observation_vector,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def test_dreamer_observation_is_exact_current_relational_v2_state() -> None:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[5])
    world.request_count = 7
    world.workflow_progress = 2
    world.audit_score = 2
    world.session_requests_remaining = 3
    state = world.snapshot()

    assert dreamer_observation_vector(state) == relational_state_vector_v2(state)
    # Hidden audit/session pressure must remain masked for the baseline too.
    assert dreamer_observation_vector(state)[7] == 0.0
    assert dreamer_observation_vector(state)[9] == 0.0


def test_dreamer_manifest_declares_current_public_state_v2() -> None:
    manifest = dreamer_adapter_manifest()
    assert manifest["adapter_version"] == "official-dreamerv3-relational-categorical-v3"
    assert manifest["state_representation"] == "current-relational-public-state-v2"
