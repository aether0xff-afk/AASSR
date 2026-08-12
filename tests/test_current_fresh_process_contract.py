from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_relational_dqn_uses_v3_without_aassr_import_side_effects() -> None:
    """The real CLI trains this baseline before constructing AASSR.

    Run in a brand-new interpreter so another pytest module cannot make the test
    pass by installing the process-global relational v3 contract during
    collection. The portable ``[dev]`` matrix intentionally omits Torch; the
    dedicated current-generation gate installs the Torch path and therefore owns
    this contract when Torch is unavailable here.
    """

    if importlib.util.find_spec("torch") is None:
        pytest.skip("fresh-process DQN contract requires the optional Torch path")

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = (
        src
        if not env.get("PYTHONPATH")
        else src + os.pathsep + env["PYTHONPATH"]
    )
    code = textwrap.dedent(
        """
        from aassr_v2.current_dqn_baseline import build_relational_dqn_agent
        from aassr_v2.current_relational_state_v3 import (
            STATUS_CODES_V3,
            STATUS_START_INDEX,
            relational_state_vector_v3,
        )
        from aassr_v2.pentest_agent_main_test import AGENT_STATE_SIZE
        from aassr_v2.types import StateSnapshot

        vector = [0.0] * AGENT_STATE_SIZE
        raw_start = AGENT_STATE_SIZE - len(STATUS_CODES_V3)
        vector[raw_start + STATUS_CODES_V3.index(404)] = 1.0
        state = StateSnapshot(
            vector=tuple(vector),
            facts=frozenset({"last_status:404"}),
            available_actions=(),
            metadata={},
        )
        agent = build_relational_dqn_agent(
            seed=7,
            train_transitions=64,
            device="cpu",
        )
        encoded = tuple(agent.dqn.encode_state(state))
        expected = tuple(relational_state_vector_v3(state))
        assert encoded == expected
        status = encoded[STATUS_START_INDEX:STATUS_START_INDEX + len(STATUS_CODES_V3)]
        assert status == tuple(float(code == 404) for code in STATUS_CODES_V3)
        assert agent.representation == "current-relational-public-state-v3+latest-http-status"
        """
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=True,
    )
