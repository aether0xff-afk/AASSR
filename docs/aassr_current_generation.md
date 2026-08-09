# AASSR current-generation runtime

This document defines the active post-v0.4 research runtime. Historical modules and
entrypoints remain in the repository for exact reproduction, but they are not part
of the current-generation execution path.

## Public entrypoint

```python
from aassr_v2 import build_pentest_aassr_core
```

At package level, `build_pentest_aassr_core` now means the current generation.
The explicit spelling `build_current_pentest_aassr_core` is equivalent.

Frozen v0.4 reproduction remains available only through:

```python
from aassr_v2 import build_legacy_v040_pentest_aassr_core
```

or by importing the historical builder directly from `aassr_v2.integrated_agent`.
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
| Policy hardware | explicit shared torch device, sync-free Bellman target reduction | active |
| Prophecy | Neural Delta ensemble | active |
| Prophecy state input | rename-invariant relational state vector | active |
| Prophecy action input | relational action features | active |
| Prophecy output | learned delta applied to the current concrete scaffold | active |
| Confidence | frozen-holdout relational calibration | active |
| Knowledge | pre-existing episode-local response Knowledge | active |
| Imagination | multi-step parallel-universe tree | active at frozen evaluation |
| Branch scorer/pruning | GRU Branch Critic trained from final real outcome | active |
| Skill | relational ASeq template rebound to current concrete actions | active |
| GPU path | DQN + Neural Delta on one requested device; depth-wise Imagination batching | active |
| Current experiment protocol | standalone current protocol | active |
| Effect-composed snapshot model | historical path | disabled |
| hand-written Goal/StateDelta scorer | historical path | disabled |
| `OnlineGRUProphecy` | historical v0.4 path | disabled |
| `SemanticContextualPolicy` | historical v0.4 path | disabled |

`current_manifest.py` is the sole executable source of truth for the active
component set. `LEGACY_COMPONENTS_ACTIVE` must remain empty. CI checks both the
manifest and the actual instantiated object graph.

## Hardware execution contract

The current runtime no longer treats `--device cuda:0` as a Prophecy-only option.
The same requested torch device is used for the active relational DQN and the
Neural Delta world model.

The DQN Bellman target implementation also avoids the historical CPU-shaped
reduction pattern in which every replay row converted a next-action maximum to a
Python scalar. On CUDA, that would force repeated device/host synchronization.
The current hardware backend keeps those maxima as device tensors until the loss
is formed. Artifacts explicitly record `per_row_target_item_syncs = 0`.

Neural Delta primitive branches are evaluated depth-wise in batches during
Imagination. Relational Skill macros retain their variable-length scalar fallback.
This is a semantics-preserving batching optimization rather than a change to the
planning algorithm.

TF32 is a recorded execution option for CUDA float32 runs. It can be disabled with
`--no-tf32`. Deterministic-algorithm mode remains enabled. `torch.compile` is not
part of the active contract because the current workload has dynamic action-set
sizes and short online updates where compile/recompile overhead may dominate,
especially on the local Windows path.

To verify the actual local CUDA path rather than merely the CPU-equivalent code
path, run:

```powershell
python scripts/check_current_generation_hardware.py --device cuda:0
```

The check performs real DQN and Neural Delta forwards and requires AASSR DQN,
AASSR Prophecy and the bare-DQN control to resolve to the requested CUDA device,
with depth batching enabled and no per-row Bellman-target host sync.

## Two identity contracts

AASSR no longer forces one equivalence relation onto unrelated jobs.

### Concrete semantic identity

Used only for ASEQ and Imagination cycle detection. Different concrete routes,
profiles or objects inside one episode remain distinct. Volatile administrative
counters are still ignored according to the audited v3 semantic contract.

### Relational transfer identity

Used by Policy, Prophecy input, Critic and relational Skill promotion. Seed-renamed
identifiers with the same observed roles map to the same structural representation.
A regression test deliberately moves raw identifier-index slots and verifies that
Policy and Prophecy inputs remain identical while the concrete ASEQ key remains
different.

## World model and Knowledge boundary

The current world model is the Neural Delta ensemble developed in the Prophecy /
Imagination-v2 line. Its learned input is relational state + relational action; raw
route/profile/object slot numbers are not provided as world-model input. The
predicted delta is then applied to the caller's current concrete state scaffold so
that planning remains connected to the real current action surface.

The legacy snapshot/effect-composition model is not stacked on top of this world
model.

Holdout calibration is frozen at the start of each real transition. If that
transition is subsequently assigned to holdout, it cannot calibrate a prediction
of itself. Explicit Knowledge follows the same anti-hindsight rule: only Knowledge
that existed before the action may influence the action's contextual prediction.
Context-free holdout prediction never sees live episode Knowledge.

## Current experiment conditions

The current main has three reported conditions:

1. `dqn_bare`
2. `aassr_current_no_imagination`
3. `aassr_current_full`

`dqn_bare` is deliberately **DQN-only**, not an old raw-ID baseline. It shares the
same current relational state/action representation so that a representation
mismatch does not masquerade as an AASSR advantage. It also shares the same v3
observation contract, response-causal environment, action surface, sparse external
reward, adaptive-curriculum rule, seed pools, transition budget and hardware-aware
DQN backend.

It does **not** instantiate or use ASEQ, Knowledge, Prophecy, Imagination, Skills,
the branch Critic, FeatureMemory, or the information-value residual.

The bare DQN and AASSR train separate checkpoints and advance their adaptive
curricula independently under the same predeclared rule. Within AASSR,
`aassr_current_no_imagination` and `aassr_current_full` are evaluated from the
**same persistent AASSR checkpoint**.

Training exploration for both learners is indexed by the fraction of the global
real-transition budget consumed. It does not reset at curriculum block boundaries.
This corrects the earlier current-main wiring in which the AASSR epsilon index was
accidentally restarted from a small per-block episode number.

## Same-checkpoint Imagination protocol

Training-time Imagination intervention is disabled. Real interaction trains:

- sparse-reward relational DQN Policy,
- fully relational-input Neural Delta Prophecy,
- calibration evidence,
- information-value predictor,
- GRU Branch Critic from final episode outcome,
- Knowledge and relational Skill discovery.

After training, the *same persistent AASSR checkpoint* is evaluated twice:

1. Policy-only / no Imagination;
2. Full current AASSR with Imagination enabled when the learned Critic is ready.

This avoids comparing two separately trained stochastic AASSR agents when
estimating the marginal effect of Imagination. The independent `dqn_bare` checkpoint
provides the external DQN-only control.

## GOAL status

The old GridPush GOAL Maker/Executor is intentionally not activated here. That
experiment relied on hand-written state-delta / goal-distance scoring and predates
the learned GRU branch critic direction. The final external success goal remains,
and successful relational ASeq fragments may become reusable Skills. A future
learned GOAL controller must use the current relational/learned-value contracts
rather than silently restoring the old hand-scored implementation.

## Legacy policy

Historical files are not deleted. Reproduction scripts may still import old v0.4,
GridPush, GOAL, GRU and effect-composition implementations. New current-generation
scripts must import only the package current builder and `current_protocol`.

CI fails if:

- the package default pentest builder points back to v0.4;
- a current runner imports the frozen v0.4 full-system or historical 2x2 runner;
- old Policy, old GRU Prophecy or hand-written Goal scorer appears in the active
  current object graph;
- renamed structural states/actions produce different current Policy or Prophecy
  inputs;
- AASSR and bare DQN stop sharing the hardware-aware DQN backend;
- the DQN target path reintroduces per-row host synchronization;
- the bare DQN condition acquires an AASSR module;
- same-checkpoint evaluation mutates persistent learning state.

## Before another full main

Do not repeat the old 90,000-transition main. Before a new full current-generation
experiment, all of the following must be green:

1. current-generation component/rename-invariance/hardware gate;
2. local CUDA hardware-path smoke on the target machine;
3. real-transition learning smoke;
4. bare-DQN/current-AASSR condition smoke with frozen evaluation;
5. a larger reduced run in which the learned GRU Critic becomes ready and the
   Full condition records real Imagination runs on the same checkpoint.
