from __future__ import annotations

import argparse
import json

from aassr_v2.current_dqn_baseline import (
    build_raw_dqn_agent,
    build_relational_dqn_agent,
)
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_hardware import hardware_diagnostics
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify raw DQN, relational DQN and current AASSR on the requested "
            "torch device, including Policy/Prophecy/Critic batching."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    common = {
        "seed": args.seed,
        "train_transitions": 256,
        "device": args.device,
        "allow_tf32": not args.no_tf32,
    }
    aassr = build_current_pentest_aassr_core(
        **common,
        use_imagination=True,
    )
    relational = build_relational_dqn_agent(**common)
    raw = build_raw_dqn_agent(**common)

    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions[:16])
    if not actions:
        raise RuntimeError("hardware smoke found no actions")

    # Exercise actual tensors rather than accepting a device string in metadata.
    aassr.dqn.score_actions(state, actions)
    relational.dqn.score_actions(state, actions)
    raw.dqn.score_actions(state, actions)
    aassr.base_neural_prophecy.predict_batch(
        tuple(state for _ in actions),
        actions,
        samples=1,
    )
    # One planner call exercises frontier-wide Policy ranking, batched Prophecy,
    # and batched GRU Critic scoring together.
    aassr.planner.plan(state, maximum_depth=1)

    aassr_hw = hardware_diagnostics(aassr)
    relational_hw = relational.diagnostics()["hardware"]
    raw_hw = raw.diagnostics()["hardware"]
    requested = aassr_hw["resolved_device"]

    for label, actual in (
        ("AASSR DQN", aassr_hw["dqn"]["device"]),
        ("AASSR Prophecy", aassr_hw["prophecy"]["device"]),
        ("AASSR Critic", aassr_hw["critic"]["device"]),
        ("relational DQN", relational_hw["dqn"]["device"]),
        ("raw DQN", raw_hw["dqn"]["device"]),
    ):
        if actual != requested:
            raise AssertionError(f"{label} is not on the requested device")

    if not aassr_hw["depth_batching"]:
        raise AssertionError("current AASSR depth batching is disabled")
    for label, hardware in (
        ("AASSR DQN", aassr_hw["dqn"]),
        ("relational DQN", relational_hw["dqn"]),
        ("raw DQN", raw_hw["dqn"]),
    ):
        if hardware["per_row_target_item_syncs"] != 0:
            raise AssertionError(f"{label} reintroduced per-row target host sync")
        if hardware["fused_next_action_reduce"] != 1:
            raise AssertionError(f"{label} target reductions are not fused")

    if aassr_hw["planner"]["policy_batch_calls"] <= 0:
        raise AssertionError("current planner did not batch Policy ranking")
    if aassr_hw["planner"]["policy_scalar_fallback_rows"] != 0:
        raise AssertionError("current planner fell back to scalar Policy ranking")
    if aassr_hw["dqn"]["pair_score_batch_calls"] <= 0:
        raise AssertionError("current DQN did not execute frontier pair batching")
    if aassr_hw["planner"]["critic_batch_calls"] <= 0:
        raise AssertionError("current planner did not batch Critic scoring")
    if aassr_hw["planner"]["critic_scalar_fallback_rows"] != 0:
        raise AssertionError("current planner fell back to scalar Critic scoring")
    if aassr_hw["critic"]["scalar_score_calls"] != 0:
        raise AssertionError("current Critic performed per-branch scalar scoring")
    if aassr_hw["prophecy"]["per_row_batch_host_sync"] != 0:
        raise AssertionError("Neural Delta reintroduced per-row batch host sync")

    print(
        json.dumps(
            {
                "aassr": aassr_hw,
                "dqn_raw": raw_hw,
                "dqn_relational": relational_hw,
                "status": "hardware_path_verified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
