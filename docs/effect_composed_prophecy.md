# Effect-Composed Prophecy and Actionable Imagination

## Why this exists

The original online Prophecy implementations predict a numerical next-state
vector and recover symbolic facts and available actions from a previously
observed `StateSnapshot`.  That fallback is useful, but by itself it cannot
construct a state combination that has never been observed exactly.

For example, if the model has separately observed:

```text
key absent + door closed
key present + door closed
key absent + door open
```

nearest-snapshot retrieval cannot directly represent:

```text
key present + door open
```

`EffectComposedProphecy` therefore learns the transition *effect* caused by an
action and applies that effect to the current branch-local imagined state.

## Learned effect

Each real training transition records:

```text
Δstate vector
facts added
facts removed
actions unlocked
actions locked
Δgoal progress
```

No task-specific key, door, flag, or direction label is assigned a reward or
special importance.  The effect is derived only from the observed difference
between the real before and after states.

## Reuse hierarchy

Effects are reused conservatively in this order:

1. exact state context and exact action signature;
2. exact action signature in another compatible context;
3. the same action verb family.

Exact/signature reuse may transfer symbolic fact and action changes.  Verb-only
fallback transfers the numerical and progress effect but keeps the current
state's symbolic bindings.  This prevents an effect learned for one concrete
stage, coordinate, target, or parameter from injecting stale actions into a
new state.

## Relationship with the wrapped model

The adapter does not replace Tabular, Python GRU, or Torch GRU Prophecy.

The wrapped model still supplies:

- recurrent memory;
- uncertainty and confidence;
- learned numerical prediction;
- a fallback observed state;
- model-specific training.

The effect layer supplies compositional state changes.  Its probability mass
increases with repeated consistent observations but remains below one, so the
wrapped model keeps explicit fallback mass.

## Imagination gate diagnostics

Every `ActionDecision` now records:

```text
policy_action_signature
imagination_opportunity
imagination_eligible
imagination_gate_reason
model_coverage
imagination_changed_action
```

The agent also exposes cumulative counters through
`imagination_diagnostics()`:

```text
opportunities
eligible
runs
changed_actions
gate:disabled
gate:epsilon_random
gate:interval
gate:coverage
gate:eligible
change_rate_per_run
eligibility_rate
```

This separates four different cases that were previously conflated:

```text
Imagination was disabled
Imagination was considered but gated
Imagination ran but agreed with the policy
Imagination ran and changed the selected action
```

A dedicated regression test constructs a two-step dependency where the
myopic policy chooses a locally attractive dead end and Imagination changes the
choice to the setup action that reaches the final goal.

## Standard experiment output

The autonomous experiment runner writes the intervention counters into every
episode row and aggregates the numeric fields into the standard summary:

```text
imagination_opportunities
imagination_eligible
imagination_runs
imagination_changed_actions
imagination_change_rate
imagination_eligibility_rate
imagination_coverage_mean
imagination_gate_reasons
```

Each transition record also stores the policy's original action, the executed
action, coverage, gate reason and whether planning changed the action.  The
JSON condition fields `use_effect_composition` and `effect_minimum_samples`
allow the effect layer to be ablated without changing the wrapped model or the
interaction budget.

The matched configuration `configs/effect_imagination_diagnostic.json`
compares:

```text
contextual policy
effect Prophecy without Imagination
legacy whole-snapshot Imagination
full effect-composed Imagination
```

All four conditions use the same research seeds, world families, training and
evaluation episode counts, and real-transition protocol.

## Portable model persistence

For Tabular Prophecy, learned effects are stored as compact deduplicated records
inside the metadata of the already serialized next-state snapshots.  Loading a
`.aassr-model.gz` reconstructs the effect buckets lazily before the first
prediction.

The persistence test verifies that an effect learned twice can be saved,
loaded into a fresh agent, and applied to a previously unseen state while
preserving unrelated current facts.

This does **not** claim to serialize neural weights.  Python GRU and Torch GRU
weight persistence remains a separate model-specific requirement.  Symbolic
effect records alone are not a substitute for neural parameter checkpoints.

## CI separation

Portable unit tests exclude the optional Docker runtime marker.  Docker
isolation is validated by a dedicated workflow that runs when Docker-related
files change or when manually dispatched.  This prevents transient Compose
startup failures from hiding failures in the agent core while preserving the
actual runtime-isolation test.

## Validation snapshot

The post-change autonomous smoke artifact recorded the following for the
`full_aassr` condition:

```text
training episodes: 640
training Imagination opportunities: 581
training eligible/runs: 288 / 288
training changed actions: 19
training imagined nodes: 3236

evaluation episodes: 80
evaluation Imagination opportunities: 280
evaluation eligible/runs: 60 / 60
evaluation changed actions: 0
evaluation imagined nodes: 500
```

These numbers verify that the planner is active and can alter real training
decisions.  They do not establish a performance advantage.  In particular,
the zero evaluation action changes show why generated-node counts alone are not
a sufficient planning-effect metric.

## Current interpretation boundary

This implementation establishes that:

- novel imagined state combinations can be composed from observed effects;
- Imagination can alter a policy decision in a controlled dependency test;
- effect memory survives the supported portable Tabular model format;
- the autonomous smoke suite actually produces imagined nodes and records
  real action changes.

It does not by itself establish that Full AASSR is statistically better than
all no-Imagination baselines.  That remains an empirical result to be measured
with fixed seeds, equal real-transition budgets, and explicit action-change
and realized-advantage diagnostics.
