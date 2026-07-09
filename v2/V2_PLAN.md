# AASSR v2

v2 starts from the v1 snapshot and adds a more complex GridWorld comparison
environment.

## Implemented In v2

- `WorldKind.V2_COMPLEX`
- `WorldKind.LOCKED_BOTTLENECK`
- 9x6 randomized maps
- 10 walls
- 2 key cells
- 2 door cells
- 2 hint cells pointing to the flag
- 1 far flag cell
- C0/C1/C2/C3/C4 comparison
- QLEARN, DQN_PARTIAL, and ORACLE_MDP baseline comparison
- Automated analysis and plots
- Paper-facing ablation suites:
  - ablation_1: TableProphecyModel vs TransformerProphecyModel
  - ablation_2: Prophecy prediction-error reward on vs off
  - ablation_3: Imagination rollout depth and branching-factor sweep
- Paper-facing Korean summary: `docs/research_summary_ko.md`

`ORACLE_MDP` is a full-map shortest-path oracle upper bound. It is useful as a
ceiling reference, but it is not a same-information-condition baseline for the
APASSR family because it knows the complete map before acting.

`DQN_PARTIAL` is a same-information-condition deep RL baseline. It uses a small
numpy MLP Q-network over Knowledge Storage masks and candidate-action features;
it does not receive the hidden full map.

`LOCKED_BOTTLENECK` is a structured dependency stress environment: the flag is
behind mandatory door bottlenecks, so agents must discover and reuse key/door
knowledge rather than only wander toward local frontier cells.

C3 remains the main paper-aligned framework condition:

```text
C3 = PolicyABC + Prophecy Module + Imagination Cycle
```

C4 is only an optional implementation variant:

```text
C4 = PolicyABC + sequence-based Prophecy implementation + Imagination Cycle
```

The framework should not be renamed around C4. `TableProphecyModel`,
`SequenceProphecyModel`, `TransformerProphecyModel`, and future neural
implementations are alternatives for the Prophecy Module, not replacements for
the KK/KV-DMP-Imagination framework.

Environment changes such as `v2_complex` and `locked_bottleneck` are not
ablations. They are controlled environment sweeps used to check whether the
knowledge-action dependency mechanism is robust across task structures.

## Commands

```powershell
cd X:\Dev\AASSR\v2
$env:PYTHONPATH='src'

python -m aassr.v2_compare --episodes 8 --seeds 3 --step-limit 100 --output-dir runs\v2_complex_quick
python -m aassr.v2_compare --episodes 30 --seeds 10 --step-limit 120 --workers 6 --output-dir runs\v2_complex_parallel
python -m aassr.ablation --suite all --world v2_complex --episodes 30 --seeds 10 --step-limit 120 --workers 6 --output-dir runs\ablations
python -m aassr.ablation --suite all --world all --episodes 30 --seeds 10 --step-limit 120 --workers 6 --output-dir runs\ablation_env_sweep
python -m unittest discover -s tests -v
```

## Quick Result

Quick smoke run:

```text
world=v2_complex
episodes=8
seeds=3
step_limit=100
```

Summary:

```text
ORACLE_MDP success=1.000, steps=9.29
C3     success=0.708, steps=65.12
QLEARN success=0.542, steps=60.96
C2     success=0.417, steps=66.40
C0     success=0.333, steps=66.19
C1     success=0.333, steps=52.03
```

Generated figures:

```text
runs/v2_complex_quick/analysis/figure_success_rate.png
runs/v2_complex_quick/analysis/figure_steps_to_flag.png
runs/v2_complex_quick/analysis/figure_semantic_gain.png
runs/v2_complex_quick/analysis/figure_repeat_error_rate.png
runs/v2_complex_quick/analysis/figure_learning_curve.png
```

Validation:

```text
Ran 61 tests
OK
```
