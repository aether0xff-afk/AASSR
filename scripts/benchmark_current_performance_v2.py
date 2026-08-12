from __future__ import annotations

import argparse
import json
import time
from typing import Callable

from aassr_v2.current_relational_state_v3 import install_status_aware_relational_contract
from aassr_v2.current_runtime_performance_v2 import (
    _stack_ensemble_parameters,
    _stacked_linear_forward,
    _stacked_linear_forward_prepacked,
)
from aassr_v2.current_status_models import StatusAwareConditionalMixtureRelationalProphecy


def _sync(torch: object, device: object) -> None:
    if str(getattr(device, "type", device)) == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    torch: object,
    device: object,
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    _sync(torch, device)
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(torch, device)
    return (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark current Prophecy sequential ensemble inference versus "
            "performance-v2 ensemble-dimension batched GEMM."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be non-negative and iterations must be positive")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("benchmark requires PyTorch") from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")

    install_status_aware_relational_contract()
    prophecy = StatusAwareConditionalMixtureRelationalProphecy(
        seed=int(args.seed),
        device=str(device),
    )
    models = prophecy.models
    packed = _stack_ensemble_parameters(torch, models)
    batch_sizes = tuple(
        int(value.strip())
        for value in str(args.batch_sizes).split(",")
        if value.strip()
    )
    if not batch_sizes or min(batch_sizes) <= 0:
        raise SystemExit("batch sizes must be positive")

    rows = []
    for batch_size in batch_sizes:
        torch.manual_seed(int(args.seed) + batch_size)
        inputs = torch.randn(
            batch_size,
            prophecy.input_size,
            dtype=torch.float32,
            device=device,
        )

        def sequential() -> object:
            with torch.no_grad():
                return torch.stack([model(inputs) for model in models], dim=0)

        def fused_cached() -> object:
            with torch.no_grad():
                return _stacked_linear_forward_prepacked(torch, packed, inputs)

        def fused_fresh() -> object:
            with torch.no_grad():
                return _stacked_linear_forward(torch, models, inputs)

        with torch.no_grad():
            reference = sequential()
            optimized = fused_cached()
            max_abs = float((reference - optimized).abs().max().detach().cpu().item())

        sequential_seconds = _measure(
            torch,
            device,
            sequential,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        cached_seconds = _measure(
            torch,
            device,
            fused_cached,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        fresh_seconds = _measure(
            torch,
            device,
            fused_fresh,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        rows.append(
            {
                "batch_size": batch_size,
                "max_abs_difference": max_abs,
                "sequential_ms": sequential_seconds * 1000.0,
                "fused_cached_ms": cached_seconds * 1000.0,
                "fused_fresh_pack_ms": fresh_seconds * 1000.0,
                "cached_speedup_x": (
                    sequential_seconds / cached_seconds
                    if cached_seconds > 0.0
                    else 0.0
                ),
                "fresh_pack_speedup_x": (
                    sequential_seconds / fresh_seconds
                    if fresh_seconds > 0.0
                    else 0.0
                ),
            }
        )

    output = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "ensemble_size": len(models),
        "input_size": prophecy.input_size,
        "output_size": prophecy.output_size,
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "rows": rows,
    }
    print(json.dumps(output, indent=2, sort_keys=True))

    if any(row["max_abs_difference"] > 1e-5 for row in rows):
        raise SystemExit("fused inference exceeded the 1e-5 numerical contract")


if __name__ == "__main__":
    main()
