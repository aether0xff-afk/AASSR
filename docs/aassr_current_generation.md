# AASSR current-generation runtime

This document defines the active post-v0.4 research runtime. Historical modules and
entrypoints remain in the repository for exact reproduction, but they are not part
of the current-generation execution path.

## Public entrypoint

```python
from aassr_v2 import build_pentest_aassr_core
```

At package level, `build_pentest_aassr_core` means the current generation.
`build_current_pentest_aassr_core` is equivalent. Frozen v0.4 reproduction remains
available only through explicit legacy builders such as
`build_legacy_v040_pentest_aassr_core` or the historical module itself.

Current experiment runners do not import the frozen v0.4 full-system runner or the
historical 2x2 training-mechanism runner.

## Active stack

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
| Hardware | shared requested torch device + full depth batching | active |
| Current experiment protocol | standalone current protocol | active |
| Effect-composed snapshot model | historical path | disabled |
| hand-written Goal/StateDelta scorer | historical path | disabled |
| `OnlineGRUProphecy` | historical v0.4 path | disabled |
| `SemanticContextualPolicy` | historical v0.4 path | disabled |

`current_manifest.py` is the sole executable source of truth for the active
component set. `LEGACY_COMPONENTS_ACTIVE` must remain empty.

## Hardware execution contract

`--device cuda:0` now applies to the current AASSR Policy DQN, Neural Delta world
model and GRU Branch Critic. The two DQN controls use the same requested hardware
execution contract.

The important optimization is not merely moving tiny models to CUDA. The hot paths
were rewritten so accelerator work is actually batched:

1. **Policy frontier ranking** — every state/action pair at one Imagination depth is
   deduplicated by exact DQN input and scored in one DQN batch.
2. **Neural Delta prediction** — primitive branches at one depth use one batched
   world-model call. Predicted states, terminal classes and confidences are copied
   back to the host in bulk rather than one branch at a time.
3. **GRU Critic scoring** — all predicted child transitions at one depth are scored
   in one GRU batch; the current planner does not use scalar branch scoring.
4. **GRU Critic training** — replay episodes are padded/masked and trained as an
   episode batch while preserving the original equal-per-episode loss weighting.
5. **DQN Bellman targets** — all next actions in a replay minibatch are scored
   together and reduced with a device-side `scatter_reduce(amax)`. There is no
   `.item()` synchronization per replay row.
6. **Holdout calibration** — the same frozen holdout items and scoring equation are
   retained, but calibration refreshes call `predict_batch` once for the selected
   rows instead of one world-model call per item.

These changes are execution optimizations, not algorithm changes. Regression tests
compare the hardware-aware DQN/Critic against their scalar/current references and
compare the fully batched planner against the scalar Policy/Critic planner on the
same seed and state.

Artifacts expose diagnostics including:

- `per_row_target_item_syncs = 0`
- `fused_next_action_reduce = 1`
- Policy/Critic batch call and row counts
- `per_row_batch_host_sync = 0` for Neural Delta
- calibration batch refresh counts
- resolved device and TF32 setting

TF32 is a recorded CUDA float32 execution option and may be disabled with
`--no-tf32`. Deterministic-algorithm mode remains enabled. `torch.compile` is not
part of the active contract because dynamic action cardinality and short online
updates make compile/recompile overhead risky, especially on Windows.

Hosted CI verifies CPU numerical/protocol equivalence. Actual target-GPU placement
must be checked locally:

```powershell
python scripts/check_current_generation_hardware.py --device cuda:0
```

The check performs real forwards for raw DQN, relational DQN and AASSR and requires
AASSR DQN/Prophecy/Critic to resolve to the requested device. It also requires
Policy, Prophecy and Critic batching and rejects per-row DQN/Neural host-sync
regressions.

## Two identity contracts

AASSR deliberately uses two different equivalence relations.

### Concrete semantic identity

Used for ASEQ and Imagination cycle detection. Different concrete routes, profiles
or objects inside one episode remain distinct.

### Relational transfer identity

Used by Policy, Prophecy input, Critic and relational Skill promotion. Seed-renamed
identifiers with the same observed roles map to the same structural representation.
Regression tests move raw identifier-index slots and require Policy/Prophecy input
to remain unchanged while the concrete ASEQ key remains different.

## World model and Knowledge boundary

The current Neural Delta model receives relational state + relational action and
predicts a delta that is applied to the caller's current concrete scaffold. Raw
route/profile/object slot numbers are therefore not world-model lookup keys, while
planning remains connected to the real current action surface.

The old snapshot/effect-composition model is not stacked on top of Neural Delta.
Holdout calibration is frozen before each real transition, and Knowledge that is
learned by the transition cannot be used to predict that same transition.

## Current experiment conditions

The current main reports **four conditions**:

1. `dqn_raw`
2. `dqn_relational`
3. `aassr_current_no_imagination`
4. `aassr_current_full`

### `dqn_raw`

This is the true plain-DQN control requested for the experiment.

- state input: the exposed v3 observation vector directly
- action input: stable raw action-signature hash features
- no relational route/profile/object abstraction
- no ASEQ, Knowledge, Prophecy, Imagination, Skills, Critic, FeatureMemory or
  information-value residual

Known methodology bugs are **not** restored: episode boundaries correctly stop TD
bootstrap, and the hardware backend uses the same fused/sync-free Bellman target
execution contract.

### `dqn_relational`

This is a representation-controlled DQN ablation. It is still DQN-only, but its
state/action representation is exactly the current relational representation used
by AASSR Policy.

Comparing `dqn_raw` with `dqn_relational` isolates representation effects. Comparing
`dqn_relational` with AASSR helps test whether gains remain after that representation
advantage is controlled.

### AASSR conditions

AASSR trains **one** checkpoint with training-time Imagination intervention off.
The same persistent checkpoint is then evaluated twice:

- `aassr_current_no_imagination`
- `aassr_current_full`

This preserves the same-checkpoint estimate of Imagination's marginal effect.

The three training checkpoints — raw DQN, relational DQN and AASSR — each receive
the same real-transition budget, training seed pool, validation pool, diagnostic
pool, sparse external reward, response-causal environment, action surface and
adaptive-curriculum rule. Their curricula advance independently according to that
same rule.

Training exploration for all learners is indexed by the fraction of the **global
real-transition budget** consumed. It no longer resets at curriculum block
boundaries; the previous current-main per-block epsilon wiring was corrected before
this experiment was frozen.

For a standard 10,000-transition run, one research seed therefore uses three
training budgets (30,000 nominal real training transitions total) and reports four
evaluation conditions because the two AASSR modes share a checkpoint.

## Current CI status

The dedicated current-generation gate checks:

- standalone current runtime / no active legacy component
- raw DQN and relational DQN reference equivalence on CPU
- raw/relational controls contain no AASSR modules
- hardware DQN fused target reduction / no per-row host sync
- relational Policy frontier batching
- Neural Delta depth batching and bulk host transfer
- GRU Critic depth batching and batched training implementation
- fully batched planner versus scalar Policy/Critic result equivalence
- calibration refresh batching without changing selected holdout data
- four-condition tiny main artifact creation
- exact training budget and frozen evaluation
- 192-real-transition AASSR learning smoke

The latest hardware-aware 192-transition AASSR CPU smoke still produced 3 training
successes, 65 DQN updates and 27 Neural Delta updates; both frozen L0 diagnostic
modes were 2/2. The Critic was not ready in that small smoke, so it recorded no
actual Imagination run and is not evidence for an Imagination benefit.

## Before the next full main

Do not relaunch the old v0.4 main. Before a new full current-generation experiment:

1. current-generation CI must remain green;
2. run `check_current_generation_hardware.py` on the target CUDA machine;
3. run a larger reduced current experiment until the learned Critic reaches
   readiness and `aassr_current_full` records real Imagination runs;
4. inspect raw DQN, relational DQN and AASSR artifacts for exact budget/freeze and
   hardware diagnostics;
5. then freeze and launch the full current-generation experiment.
