# AASSR current-generation runtime

This document defines the active post-v0.4 research runtime. Historical modules and
entrypoints remain in the repository for exact reproduction, but they are not part
of the current-generation execution path.

## Public entrypoint

```python
from aassr_v2 import build_pentest_aassr_core
```

At package level, `build_pentest_aassr_core` means the current standalone generation.
Frozen v0.4 reproduction remains available only through explicit legacy builders or
historical modules. Current experiment runners do not import the frozen v0.4 main or
the historical 2x2 training-mechanism runner.

## Active AASSR stack

| Layer | Current implementation | Status |
|---|---|---|
| Observation | `response_causal_observation_v3` | active |
| ASEQ | empirical concrete-semantic `S -> A -> S` guard | active |
| Transfer state identity | rename-invariant relational descriptor | active |
| Transfer action identity | route/profile/object relation features | active |
| Policy | relational DQN + separate information-value residual | active |
| Prophecy | Neural Delta ensemble, relational state/action input | active |
| Confidence | frozen-holdout relational calibration | active |
| Knowledge | pre-existing episode-local response Knowledge | active |
| Imagination | multi-step parallel-universe tree | active at frozen evaluation |
| Branch scorer/pruning | relational GRU Branch Critic | active |
| Skill | relational ASeq template rebound to concrete actions | active |
| Hardware | DQN + Neural Delta + GRU Critic on one requested device | active |
| Imagination hardware | Policy + Prophecy + Critic depth-batched | active |
| Current experiment protocol | standalone current protocol | active |
| old effect composition | historical path | disabled |
| hand-written Goal/StateDelta scorer | historical path | disabled |
| `OnlineGRUProphecy` | historical v0.4 path | disabled |
| `SemanticContextualPolicy` | historical v0.4 path | disabled |

`current_manifest.py` is the executable source of truth for the active AASSR
component set. `LEGACY_COMPONENTS_ACTIVE` must remain empty.

## Hardware execution contract

`--device cuda:0` applies to the current AASSR Policy DQN, Neural Delta world model,
and GRU Branch Critic. Raw and relational DQN controls use the same corrected
hardware execution path.

The hot paths are accelerator-aware rather than merely moving small modules to GPU:

1. all Policy state/action pairs at one Imagination depth are scored in one batch;
2. primitive Neural Delta branches at one depth use one world-model batch;
3. predicted Neural Delta outputs are transferred to host in bulk;
4. all GRU Critic children at one depth are scored in one batch;
5. Critic replay training uses padded/masked episode batches while preserving the
   original equal-per-episode objective;
6. DQN Bellman next-action values use a flat batch and device-side
   `scatter_reduce(amax)` with no replay-row `.item()` synchronization;
7. frozen holdout calibration refresh keeps the same rows/equation but evaluates
   them with one `predict_batch` call.

Regression tests compare hardware-aware DQN/Critic outputs with their scalar
references and compare the fully batched planner against the scalar planner on the
same seed/state. TF32 is recorded and can be disabled with `--no-tf32`.
`torch.compile` is intentionally not active because dynamic action cardinality and
short online updates make recompilation overhead risky.

Target-GPU placement is checked locally with:

```powershell
python scripts/check_current_generation_hardware.py --device cuda:0
```

## Two identity contracts

### Concrete semantic identity

Used by ASEQ and concrete cycle detection. Different route/profile/object entities
inside one episode remain distinct.

### Relational transfer identity

Used by Policy, Prophecy input, Critic, Skill, the relational DQN control, and the
DreamerV3 environment adapter. Seed-renamed identifiers with the same observed
roles map to the same structural representation.

Tests deliberately move concrete identifier slots and require relational
Policy/Prophecy/Dreamer inputs to remain unchanged while the concrete ASEQ identity
changes.

## World model and Knowledge boundary

The current Neural Delta model receives relational state + relational action and
predicts a delta applied to the caller's current concrete scaffold. Raw identifier
slots are not transfer lookup keys. Holdout calibration is frozen before each real
transition, and Knowledge learned by a transition cannot be used to predict that
same transition.

The historical effect-composed snapshot model is not stacked on top of the current
Neural Delta model.

## Experiment controls

### Local PyTorch/current core

`run_pentest_current_generation_main.py` trains three local checkpoints and reports
four rows:

1. `dqn_raw`
2. `dqn_relational`
3. `aassr_current_no_imagination`
4. `aassr_current_full`

`dqn_raw` uses the exposed v3 state vector plus historical raw action-signature hash
features and contains no AASSR module. Known TD episode-boundary bugs are not
restored.

`dqn_relational` is DQN-only but uses the same relational state/action representation
as current AASSR. Therefore `dqn_raw -> dqn_relational` isolates representation
effects.

AASSR trains one checkpoint with training-time Imagination intervention disabled.
That same persistent checkpoint is frozen and evaluated twice with Imagination OFF
and ON. Evaluation mutation is a hard failure.

All three local training checkpoints receive the same real-transition budget,
sparse external reward, seed pools, response-causal environment, action surface,
and adaptive curriculum rule. Exploration is indexed by the fraction of the global
real-transition budget consumed and never resets at a curriculum block boundary.

### Official DreamerV3 control

The canonical final experiment additionally includes `dreamerv3_relational`.

The algorithm comes from a pinned, unmodified `danijar/dreamerv3` checkout. AASSR
provides only a dynamic-action environment adapter and the current experiment
orchestration. The full adapter/config contract is frozen in
`docs/dreamerv3_current_baseline.md`.

Dreamer receives the same relational state representation plus a 240-slot structural
availability mask. Its **official discrete categorical policy path** acts over a
fixed 240-way relational vocabulary. A currently legal slot is executed directly;
an unavailable slot is deterministically projected to the nearest currently legal
relational slot using only public structural information. Hidden scenario state,
correct-action labels, future state, and reward are never used by the projection.

The canonical Dreamer preset is upstream `dmc_proprio + size1m`: train ratio `1024`,
`bfloat16`, and the upstream imagination/actor/critic/world-model losses remain
unchanged. Dreamer's train-ratio scheduler follows the official Embodied Driver-step
clock, including `is_first` callbacks, while the scientific budget independently
counts only real primitive HTTP actions.

Dreamer runs in a separate process/environment because its official implementation
uses JAX. This prevents PyTorch and JAX from retaining competing CUDA allocators
inside one process and lets the final suite keep an exact upstream pin.

## Canonical five-condition suite

The final report is:

1. `dqn_raw`
2. `dqn_relational`
3. `dreamerv3_relational`
4. `aassr_current_no_imagination`
5. `aassr_current_full`

There are four trained checkpoints per research seed: raw DQN, relational DQN,
DreamerV3, and AASSR. The two AASSR rows share one checkpoint.

At 10,000 real training transitions per checkpoint, this is 40,000 nominal real
training transitions per research seed.

Interpretation:

- raw DQN -> relational DQN: representation effect
- relational DQN -> DreamerV3: official world-model + imagined actor-critic baseline
- relational DQN -> AASSR no-Imagination: AASSR stack beyond representation
- AASSR no-Imagination -> Full: AASSR Imagination marginal effect
- DreamerV3 <-> AASSR Full: model-based imagination-family comparison

The final suite is assembled with:

```text
scripts/assemble_pentest_current_generation_suite.py
```

The assembler rejects mismatched research seed, transition budget, train/validation/
diagnostic seed pools, stage manifest, final-blind status, Dreamer upstream commit,
Dreamer preset/train ratio/dtype/JAX platform, and Dreamer categorical action-adapter
contract.

## Current CI contract

The dedicated current-generation gate checks:

- standalone current runtime and no active legacy component;
- raw/relational DQN reference equivalence and absence of AASSR modules;
- fused DQN targets and no per-row host sync;
- relational Policy, Neural Delta, and GRU Critic batching;
- fully batched planner vs scalar planner result equivalence;
- holdout calibration batching;
- exact current training budget and frozen evaluation;
- 192-real-transition current AASSR learning smoke;
- Dreamer 240-way categorical action vocabulary uniqueness and rename invariance;
- every predeclared stage action surface maps into the Dreamer adapter;
- legal categorical slots map with zero distance and unavailable slots map only to
  current legal actions;
- Dreamer environment adapter reset/real-step/terminal budget semantics without JAX;
- five-condition suite contract and pinned/canonical Dreamer enforcement.

A separate `DreamerV3 current baseline smoke` workflow checks out the pinned official
upstream repository and actually executes upstream `Agent -> Replay -> Driver ->
agent.train()` on CPU with the upstream debug preset. That smoke verifies API
compatibility and the official Driver-step train-ratio cadence, but is explicitly not
a benchmark result. The canonical Dreamer result still requires JAX/CUDA.

## Before the next full main

Do not relaunch the old v0.4 main. Before freezing the new full experiment:

1. all current-generation and official-Dreamer contract CI must remain green;
2. run the current PyTorch hardware check on the target CUDA machine;
3. run a reduced current AASSR experiment until the learned Critic reaches readiness
   and Full records actual Imagination runs;
4. run a reduced pinned official DreamerV3 baseline from Linux/WSL with JAX/CUDA,
   categorical 240-way action space, and exact real-transition accounting;
5. assemble the reduced five-condition artifact and verify all contracts;
6. only then freeze and launch the full five-condition experiment.
