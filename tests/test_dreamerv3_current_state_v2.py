from __future__ import annotations

from aassr_v2.current_relational_state_v3 import (
    latest_status_code,
    relational_state_vector_v3,
)
from aassr_v2.dreamerv3_baseline import (
    dreamer_adapter_manifest,
    dreamer_observation_vector,
)
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def test_dreamer_observation_is_exact_current_relational_v3_state() -> None:
    world = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[5])
    world.request_count = 7
    world.workflow_progress = 2
    world.audit_score = 2
    world.session_requests_remaining = 3
    state = world.snapshot()

    assert dreamer_observation_vector(state) == relational_state_vector_v3(state)
    # Hidden audit/session pressure must remain masked for the baseline too.
    assert dreamer_observation_vector(state)[7] == 0.0
    assert dreamer_observation_vector(state)[9] == 0.0
    assert latest_status_code(state) == latest_status_code(state)


def test_dreamer_manifest_declares_current_public_state_v3() -> None:
    manifest = dreamer_adapter_manifest()
    assert manifest["adapter_version"] == "official-dreamerv3-relational-categorical-v4"
    assert manifest["state_representation"] == (
        "current-relational-public-state-v3+latest-http-status"
    )
