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
| Prophecy | Neural Delta ensemble | active |
| Prophecy state input | rename-invariant relational state vector | active |
| Prophecy action input | relational action features | active |
| Prophecy output | learned delta applied to the current concrete scaffold | active |
| Confidence | frozen-holdout relational calibration | active |
| Knowledge | pre-existing episode-local response Knowledge | active |
| Imagination | multi-step parallel-universe tree | active at frozen evaluation |
| Branch scorer/pruning | GRU Branch Critic trained from final real outcome | active |
| Skill | relational ASeq template rebound to current concrete actions | active |
| GPU path | Neural Delta batch + depth-wise Imagination batching | active |
| Current experiment protocol | standalone current protocol | active |
| Effect-composed snapshot model | historical path | disabled |
| hand-written Goal/StateDelta scorer | historical path | disabled |
| `OnlineGRUProphecy` | historical v0.4 path | disabled |
| `SemanticContextualPolicy` | historical v0.4 path | disabled |

`current_manifest.py` is the sole executable source of truth for the active
component set. `LEGACY_COMPONENTS_ACTIVE` must remain empty. CI checks both the
manifest and the actual instantiated object graph.

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

## Same-checkpoint Imagination protocol

Training-time Imagination intervention is disabled. Real interaction trains:

- sparse-reward relational DQN Policy,
- fully relational-input Neural Delta Prophecy,
- calibration evidence,
- information-value predictor,
- GRU Branch Critic from final episode outcome,
- Knowledge and relational Skill discovery.

After training, the *same persistent checkpoint* is evaluated twice:

1. Policy-only / no Imagination;
2. Full current AASSR with Imagination enabled when the learned Critic is ready.

This avoids comparing two separately trained stochastic agents when estimating the
marginal effect of Imagination.

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
- same-checkpoint evaluation mutates persistent learning state.

## Before another full main

Do not repeat the 90,000-transition main until all of the following are green:

1. current-generation component/rename-invariance gate;
2. real-transition smoke;
3. same-checkpoint evaluation freeze smoke;
4. reduced L0 learning smoke showing that the agent can discover at least one
   successful trajectory without Oracle/guided/correct-action intervention;
5. a larger reduced run in which the learned GRU Critic becomes ready and the
   Full condition records real Imagination runs on the same checkpoint.
