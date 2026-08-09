# Official DreamerV3 baseline for the current AASSR experiment

This document freezes the DreamerV3 comparison contract used by the post-v0.4
AASSR experiment. The Dreamer algorithm itself is not vendored or reimplemented in
this repository.

## Upstream algorithm

Canonical upstream:

- repository: `danijar/dreamerv3`
- pinned commit: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`
- AASSR condition name: `dreamerv3_relational`

The current baseline imports the official upstream `dreamerv3.agent.Agent`, replay,
stream, driver, RSSM, actor, critic, and optimizer implementation. AASSR code does
not patch Dreamer model equations, losses, imagined trajectories, or actor/critic
updates.

A canonical result refuses an upstream checkout whose Git HEAD differs from the
pinned commit. `--allow-upstream-mismatch` exists only for non-canonical diagnostic
runs and such an artifact cannot be assembled into the final five-condition suite.

## Why an action adapter is necessary

The pentest benchmark has a dynamic action surface: the concrete HTTP actions that
are legal change as routes, profiles, and objects become observable. DreamerV3 uses
a fixed action space.

The adapter does not add a success-aware policy, oracle trajectory, or correct-action
mapping. It exposes two relational interfaces built only from the public current
state:

1. a 240-slot availability mask over
   `(verb, route_role, profile_role, object_role)`; and
2. a fixed continuous vector with the same dimensionality as the current relational
   action features.

For each Dreamer action vector, the environment computes the same relational feature
vector for every currently legal primitive HTTP action and executes the nearest one
by squared Euclidean distance. Concrete action signature is used only as a final
stable tie-break among structurally identical candidates.

This mapping has three important properties:

- every Dreamer decision resolves to a currently legal primitive action;
- route/profile/object identifier renaming does not change its structural action
  representation;
- projection reads neither the hidden benchmark scenario nor future reward/success.

The action chosen by Dreamer is therefore the continuous action of the wrapped MDP;
the nearest-legal projection is part of the environment adapter, not an edited
Dreamer actor.

## Observation representation

Dreamer receives:

- `state`: exactly the current rename-invariant relational state vector used by the
  relational DQN/AASSR Policy;
- `action_mask`: the 240-slot relational availability surface;
- `reward`, `is_first`, `is_last`, `is_terminal`: the standard Embodied control
  fields.

The availability mask prevents Dreamer from losing information that DQN/AASSR have
through `StateSnapshot.available_actions`. It contains structural legality only and
no hidden correctness label.

## Dreamer configuration

The baseline is frozen before seeing benchmark results to the official
proprioceptive/vector preset and smallest published model size:

- config: upstream `dmc_proprio` + `size1m`
- batch size: upstream value
- batch length: upstream value
- train ratio: upstream `dmc_proprio` value unless an explicitly non-canonical
  diagnostic override is requested
- imagination length and actor/critic/world-model losses: upstream values
- JAX compute dtype: upstream value

The result artifact records the effective values.

This intentionally gives Dreamer its official high-update vector-observation
configuration rather than weakening it to match DQN optimizer compute. The primary
sample-efficiency budget in this experiment is real environment transitions; compute
and wall-clock are reported separately.

## Current sparse-reward protocol

Dreamer receives the same external reward as the current DQN and AASSR conditions:

- success: `+1`
- locked failure: `-1`
- all other real transitions and truncations: `0`

Every episode reset boundary cuts continuation/bootstrap, including stall,
rate-limit, and exact-budget truncation, matching the corrected current DQN/AASSR
episode-boundary contract.

Training uses the same independent adaptive curriculum:

- same training seed pool
- same validation seed pool
- same diagnostic seed pool
- same stage definitions
- same 512-transition nominal curriculum block target
- same focus-level validation rule
- exact same real-transition budget per trained checkpoint

Reset observations used internally by Embodied are not counted as real pentest
transitions. The runner independently counts executed primitive HTTP actions and
hard-fails if that count differs from the requested budget.

Frozen validation and diagnostic phases call Dreamer policy with `mode='eval'` and
never call replay insertion or `agent.train()`. The runner verifies that its gradient
update counter is unchanged across those phases.

## Final five-condition suite

The canonical current experiment reports:

1. `dqn_raw`
2. `dqn_relational`
3. `dreamerv3_relational`
4. `aassr_current_no_imagination`
5. `aassr_current_full`

There are four trained checkpoints per research seed: raw DQN, relational DQN,
DreamerV3, and AASSR. The two AASSR rows are frozen evaluations of one shared AASSR
checkpoint.

At a 10,000-transition budget this is 40,000 nominal real training transitions per
research seed.

Interpretation:

- raw DQN -> relational DQN: representation effect
- relational DQN -> DreamerV3: official world-model/imagination actor-critic baseline
- relational DQN -> AASSR no-Imagination: AASSR non-Imagination stack effect
- AASSR no-Imagination -> AASSR Full: same-checkpoint AASSR Imagination marginal
  effect
- DreamerV3 <-> AASSR Full: model-based imagination-family comparison

`assemble_pentest_current_generation_suite.py` refuses to combine artifacts if the
research seed, real-transition budget, seed pools, stage manifest, final-blind
status, or pinned Dreamer upstream commit differ.

## Execution

DreamerV3 upstream documents Linux and Mac as tested platforms. For the Windows RTX
machine used by the current AASSR work, run the official JAX/CUDA baseline from a
Linux/WSL environment rather than silently replacing it with a PyTorch clone.

Example after installing the pinned upstream checkout and the current AASSR package
in that environment:

```bash
python scripts/run_dreamerv3_current_baseline.py \
  --dreamer-root /path/to/dreamerv3 \
  --research-seed 7 \
  --transition-budget 10000 \
  --jax-platform cuda \
  --output-dir runs/aassr_current_generation_main/seed-7/dreamerv3
```

The PyTorch/AASSR conditions can run separately. Then assemble:

```bash
python scripts/assemble_pentest_current_generation_suite.py \
  --current-summary runs/aassr_current_generation_main/seed-7/summary.json \
  --dreamer-summary runs/aassr_current_generation_main/seed-7/dreamerv3/summary_dreamerv3_relational.json \
  --output runs/aassr_current_generation_main/seed-7/five_condition_summary.json
```

Do not use a reduced/debug Dreamer config for the final result. Reduced runs are for
adapter/runtime validation only.
