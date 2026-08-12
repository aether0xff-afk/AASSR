# AASSR current runtime

This document is the navigation page for the **active research runtime** on the current-generation branch.

Do not infer the active model from the package milestone in the top-level README, old experiment names, or historical workflow names. Those files intentionally remain in the repository for reproduction and can describe earlier generations.

## Sources of truth

Use these in this order:

1. `src/aassr_v2/current_manifest.py` — active component contract.
2. `src/aassr_v2/current_entrypoint.py` — sole active AASSR builder.
3. `src/aassr_v2/pentest_current_generation_main.py` — current training / frozen evaluation protocol.
4. `scripts/run_pentest_current_generation_main.py` — canonical full current-generation CLI.
5. current-generation regression tests and CI.

If prose documentation disagrees with those files, the code path above wins and the prose is stale.

## Canonical builder

```python
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
```

`build_current_pentest_aassr_core()` is the only active current-generation pentest builder.

The previously separate `current_mixture_entrypoint.py` is now a compatibility alias. It exists so repaired diagnostic scripts and historical commands keep working; it does not define a second runtime.

## Active model contract

The current builder installs, in order:

- response-causal public observation contract,
- relational public state v3 including the latest observed HTTP status,
- hardware DQN,
- fully batched Prophecy / Critic planning path,
- current semantic/runtime repairs,
- status-balanced conditional-mixture relational Prophecy,
- confidence-as-reliability-only gate,
- local real-training Critic support gate,
- structural root compute deduplication,
- current decision optimizations.

The status objective is generic class balancing. There are no rules such as “avoid 403”, no hidden curriculum level input, and no answer trajectory injection.

## Current experimental protocol

Training and evaluation remain separated.

- external sparse return: success `+1`, true failure `-1`, truncation/stall/rate-limit `0`;
- training Imagination is disabled for the same-checkpoint comparison;
- Policy-only and Full Imagination evaluation use the same frozen AASSR checkpoint;
- hidden audit pressure and exact hidden session countdown remain masked;
- Critic support is a reliability gate, not a reward/value bonus;
- an Imagination intervention is counted only if it survives every gate and changes the real executed action.

## Canonical runners

### Full current-generation comparison

```bash
python scripts/run_pentest_current_generation_main.py \
  --output-dir runs/current_generation/seed-7 \
  --research-seed 7 \
  --transition-budget 10000 \
  --block-target 512 \
  --device cuda
```

This trains the Raw DQN control, Relational DQN control, and one current AASSR checkpoint, then evaluates AASSR no-Imagination versus Full from that same frozen checkpoint.

### Detailed Imagination diagnostic compatibility runner

`scripts/run_repaired_imagination_final.py` is retained because it produces the detailed decision / prediction traces used by the recent repair audits. Its name is historical; after canonical-builder consolidation it resolves to the same current AASSR runtime.

### Rare-status diagnostic

`scripts/run_current_status_rare_holdout.py` is a development diagnostic, not the main performance runner. Its GitHub Actions workflow is manual-only after validation.

## Repository classification

### Active

Files named `current_*` that are imported by `current_entrypoint.py`, `pentest_current_generation_main.py`, the canonical CLI, or active current-generation tests/CI.

### Diagnostic / development

Focused audit runners such as repaired Imagination traces, rare-status holdout diagnostics, hardware profiling, and targeted ablations. These may inspect the active runtime but are not themselves the canonical model definition.

### Historical / reproduction

Older v0.4, Imagination-v2, GridPush, ToolGrid, prior Prophecy, previous pentest-mechanism, paper reproduction, and old one-off workflow files remain for reproducing prior evidence. Their existence does **not** make their model components active.

Historical code should not be imported into a new current-generation runner unless the import is explicitly a compatibility layer and the active builder is still `build_current_pentest_aassr_core()`.

## Before a large run

The large run should not start unless:

1. current-generation unit/contract tests are green;
2. current builder and compatibility builder resolve to the same implementation;
3. frozen evaluation does not mutate learning state;
4. intervention counters describe the final executed action after confidence and Critic-support gates;
5. the output summary records `CURRENT_COMPONENTS` from `current_manifest.py`.

The next scaling experiment should change the **transition budget**, not the curriculum rules or task-specific behavior, so it can test whether the 2k frontier bottleneck disappears naturally with more experience.
