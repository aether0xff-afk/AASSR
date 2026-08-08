from __future__ import annotations

import argparse
import math
import time

from aassr_v2.effect_prophecy import EffectComposedProphecy
from aassr_v2.imagination_tree import ImaginationConfig, ImaginationTree, StateDeltaScorer
from aassr_v2.integrated_agent import ContextualSkillAwareProphecy, IntegratedProphecyView
from aassr_v2.local_acceleration import BatchedIntegratedProphecyView
from aassr_v2.native_batching import DepthBatchedImaginationTree, DepthBatchedProphecyView
from aassr_v2.policy import WeightedPolicy
from aassr_v2.skills import SkillLibrary
from aassr_v2.torch_gru_prophecy import TorchGRUProphecy
from aassr_v2.types import Action, StateSnapshot


def _sync(model: TorchGRUProphecy) -> None:
    if model.device.type == "cuda":
        model.torch.cuda.synchronize(model.device)


def _state(vector, actions, *, progress=0.0, fact="root"):
    return StateSnapshot(
        tuple(float(v) for v in vector),
        facts=frozenset((fact,)),
        available_actions=tuple(actions),
        goal_progress=float(progress),
    )


def _build(action_count: int, state_size: int, device: str, dtype: str):
    actions = tuple(Action("probe", target=f"target-{i:04d}") for i in range(action_count))
    root = _state([0.0] * state_size, actions)
    model = TorchGRUProphecy(
        state_size,
        seed=2026,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=False,
    )
    skills = SkillLibrary()
    contextual = ContextualSkillAwareProphecy(model, skills)
    effect = EffectComposedProphecy(contextual, minimum_samples=1)

    # Give every action empirical confidence and one structurally similar effect.
    for index, action in enumerate(actions):
        vector = [0.0] * state_size
        vector[index % state_size] = 0.25 + (index % 7) * 0.01
        target = _state(
            vector,
            actions,
            progress=min(0.9, 0.05 + (index % 9) * 0.01),
            fact=f"observed-{index % 5}",
        )
        model.reset_sequence()
        effect.learn(root, action, target)

    return root, model, contextual, effect


def _time(fn, *, iterations: int, model: TorchGRUProphecy) -> float:
    for _ in range(2):
        fn()
    _sync(model)
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(model)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--actions", type=int, default=128)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    root, model, contextual, effect = _build(
        args.actions,
        args.state_size,
        args.device,
        args.dtype,
    )
    config = ImaginationConfig(
        branching_factor=4,
        maximum_depth=2,
        beam_width=16,
        outcome_samples=1,
        minimum_path_confidence=0.0,
        update_policy=False,
        expand_all_root_actions=True,
    )
    scorer = StateDeltaScorer()

    scalar = ImaginationTree(
        WeightedPolicy(),
        IntegratedProphecyView(effect, contextual),
        config=config,
        scorer=scorer,
    )
    batch_view = DepthBatchedProphecyView(
        BatchedIntegratedProphecyView(effect, contextual)
    )
    batched = DepthBatchedImaginationTree(
        WeightedPolicy(),
        batch_view,
        config=config,
        scorer=scorer,
    )

    expected = scalar.plan(root)
    actual = batched.plan(root)
    if expected.chosen_action.signature != actual.chosen_action.signature:
        raise RuntimeError("scalar/batched chosen actions differ")
    if not math.isclose(
        expected.root_evaluations[0].aggregate_value,
        actual.root_evaluations[0].aggregate_value,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError("scalar/batched values differ")

    scalar_seconds = _time(lambda: scalar.plan(root), iterations=args.iterations, model=model)
    batch_seconds = _time(lambda: batched.plan(root), iterations=args.iterations, model=model)
    runtime = batch_view.runtime_diagnostics()

    print(f"device={model.device} dtype={args.dtype} actions={args.actions}")
    print(f"iterations={args.iterations} depth={config.maximum_depth} beam={config.beam_width}")
    print(f"scalar: {scalar_seconds:.4f}s ({scalar_seconds/args.iterations*1000:.2f} ms/plan)")
    print(f"batch : {batch_seconds:.4f}s ({batch_seconds/args.iterations*1000:.2f} ms/plan)")
    print(f"speedup: {scalar_seconds/max(batch_seconds, 1e-12):.2f}x")
    print(f"batch diagnostics: {runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
