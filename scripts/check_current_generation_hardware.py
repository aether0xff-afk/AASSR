from __future__ import annotations

import argparse
import json

from aassr_v2.current_dqn_baseline import build_bare_dqn_agent
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
from aassr_v2.current_hardware import hardware_diagnostics
from aassr_v2.pentest_transfer_stages import TRANSFER_STAGES, TransferDiagnosticWorld


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that current AASSR and bare DQN use the requested torch "
            "device and that AASSR keeps Neural Delta depth batching enabled."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    aassr = build_current_pentest_aassr_core(
        seed=args.seed,
        train_transitions=256,
        use_imagination=True,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )
    bare = build_bare_dqn_agent(
        seed=args.seed,
        train_transitions=256,
        device=args.device,
        allow_tf32=not args.no_tf32,
    )

    state = TransferDiagnosticWorld(90_001, stage=TRANSFER_STAGES[0]).snapshot()
    actions = tuple(state.available_actions[:16])
    if not actions:
        raise RuntimeError("hardware smoke found no actions")

    # Exercise real tensor creation/forward on both DQN instances and the Neural
    # Delta batch path. This is intentionally a micro smoke, not a speed claim.
    aassr.dqn.score_actions(state, actions)
    bare.dqn.score_actions(state, actions)
    aassr.base_neural_prophecy.predict_batch(
        tuple(state for _ in actions),
        actions,
        samples=1,
    )

    aassr_hw = hardware_diagnostics(aassr)
    bare_hw = bare.diagnostics()["hardware"]
    requested = aassr_hw["resolved_device"]
    if aassr_hw["dqn"]["device"] != requested:
        raise AssertionError("AASSR DQN is not on the requested device")
    if aassr_hw["prophecy"]["device"] != requested:
        raise AssertionError("AASSR Prophecy is not on the requested device")
    if bare_hw["dqn"]["device"] != requested:
        raise AssertionError("bare DQN is not on the requested device")
    if not aassr_hw["depth_batching"]:
        raise AssertionError("current AASSR depth batching is disabled")
    if aassr_hw["dqn"]["per_row_target_item_syncs"] != 0:
        raise AssertionError("DQN target path reintroduced per-row host syncs")

    print(
        json.dumps(
            {
                "aassr": aassr_hw,
                "dqn_bare": bare_hw,
                "status": "hardware_path_verified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
