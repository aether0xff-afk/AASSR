# Reliability, train/test separation, and acceleration

## What changed

### Pure noisy observations

`NoisyInformationWrapper.snapshot()` no longer samples random facts. Noise is sampled once per real `step()` and cached until the next transition. Logging, GUI rendering, and repeated observation calls therefore cannot change the experiment RNG stream.

### Calibrated GRU confidence

The pure-Python and PyTorch GRU models reserve `1-confidence` probability mass for an uncertain self-state outcome. Confidence is no longer multiplied into all outcomes and normalized away. Learned predictions use coverage-compatible source suffixes and expose explicit `confidence()` and `coverage()` methods.

### Seen and unseen evaluation

Autonomous experiments support:

- `evaluation_seen`: frozen-agent evaluation on the worlds used during training.
- `evaluation_unseen`: frozen-agent evaluation on disjoint world seeds.

Each environment can set `train_worlds_per_seed`, `eval_worlds_per_seed`, `seed_offset`, and `eval_seed_offset`. The unseen seed family is disjoint from the training seed family.

### Hybrid parallel execution

Independent seed/environment/condition jobs are process-parallelized with the `spawn` start method for Windows and CUDA safety.

- Tabular and pure-Python jobs use the CPU worker pool.
- `torch_gru` jobs use the CUDA worker pool.
- `cuda_workers` should normally equal the number of GPUs. For one RTX 4090, use `1`.

This is intentional: the environment and Tabular model are branch-heavy Python workloads and do not become faster merely by moving them to CUDA.

## Installation

Install development tools and a CUDA-enabled PyTorch build appropriate for the machine, then install the optional extras:

```powershell
python -m pip install -e ".[dev,gpu]"
```

Confirm CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Parallel tests

CPU-parallel regression suite:

```powershell
pytest -q -n auto
```

CUDA-specific test:

```powershell
pytest -q -n 1 -m cuda
```

The cross-platform matrix tests the dependency-free path, and a separate Ubuntu job installs PyTorch and executes the Torch GRU on CPU. The CUDA test is skipped when a visible GPU is unavailable and should be run on the RTX 4090 before interpreting CUDA experiment results.

## Main tabular experiment

The main experiment now reports both memorization and zero-shot transfer:

```powershell
python scripts/run_experiment.py --config configs/autonomous_main.json --output runs/autonomous_main --overwrite --workers 0
```

`--workers 0` resolves to `max(1, cpu_count - 1)`.

## CUDA GRU experiment

```powershell
python scripts/run_experiment.py --config configs/autonomous_cuda.json --output runs/autonomous_cuda --overwrite --workers 0 --cuda-workers 1 --device cuda
```

CPU and CUDA jobs run concurrently. One CUDA process owns the single GPU while the CPU pool executes tabular baselines.

## Interpretation

A high `evaluation_seen` score with a low `evaluation_unseen` score means the agent learned the training puzzles but did not transfer their structure. Generalization claims must be based on `evaluation_unseen`, not on the seen-world score.
