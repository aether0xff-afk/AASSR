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
not patch Dreamer model equations, losses, imagined trajectories, policy
distributions, or actor/critic updates.

A canonical result refuses an upstream checkout whose Git HEAD differs from the
pinned commit. `--allow-upstream-mismatch` exists only for non-canonical diagnostic
runs and such an artifact cannot be assembled into the final five-condition suite.

## Why an action adapter is necessary

The pentest benchmark has a dynamic action surface: the concrete HTTP actions that
are legal change as routes, profiles, and objects become observable. DreamerV3 uses
a fixed action space.

The adapter does not add a success-aware policy, oracle trajectory, or correct-action
mapping. It exposes a fixed relational interface built only from the public current
state:

1. a 240-slot availability mask over
   `(verb, route_role, profile_role, object_role)`; and
2. a 240-way discrete action space over that same structural vocabulary.

The official DreamerV3 code already supports discrete action spaces through its
categorical policy distribution. The environment therefore exposes
`Space(np.int32, (), 0, 240)` and does not replace or modify Dreamer's actor.

If Dreamer samples a slot that is currently legal, that structural action is
executed directly. If it samples a structurally unavailable slot, the environment
projects it to the nearest currently legal relational slot using a one-hot
structural embedding of verb, route role, profile role, and object role. Concrete
action signature is used only as a final deterministic tie-break among structurally
identical candidates.

This mapping has three important properties:

- every Dreamer decision resolves to a currently legal primitive action;
- route/profile/object identifier renaming does not change its structural action
  representation;
- projection reads neither the hidden benchmark scenario nor future reward/success.

The action chosen by Dreamer is therefore a native categorical action of the
wrapped fixed-action MDP; the nearest-legal mapping is part of the environment
adapter, not an edited Dreamer policy.

## Observation representation

Dreamer receives:

- `state`: exactly the current rename-invariant relational state vector used by the
  relational DQN/AASSR Policy;
- `action_mask`: the 240-slot relational availability surface;
- `reward`, `is_first`, `is_last`, `is_terminal`: the standard Embodied control
  fields.

The availability mask prevents Dreamer from losing information that DQN/AASSR have
through `StateSnapshot.available_actions`. It contains structural legality only and
no hidden correctness label. The upstream actor is not manually masked; the mask is
part of Dreamer's observation and unavailable categorical choices are resolved by
the environment projection above.

## Dreamer configuration

The baseline is frozen before seeing benchmark results to the official
proprioceptive/vector preset and smallest published model size:

- config: upstream `dmc_proprio` + `size1m`
- batch size: upstream value
- batch length: upstream value
- train ratio: upstream `dmc_proprio` value (`1024`)
- imagination length: upstream value (`15`)
- actor/critic/world-model losses: upstream values
- policy distribution for this discrete action space: upstream categorical path
- JAX compute dtype: upstream value (`bfloat16`)
- canonical JAX platform: `cuda`

The result artifact records the effective values. The final suite assembler rejects
CPU, retuned, non-pinned, continuous-adapter, or otherwise non-canonical Dreamer
artifacts.

This intentionally gives Dreamer its official high-update vector-observation
configuration rather than weakening it to match DQN optimizer compute. The primary
sample-efficiency budget in this experiment is real environment transitions; compute
and wall-clock are reported separately.

## Official training cadence versus real-transition budget

There are two deliberately separate counters.

The **scientific sample budget** counts only executed primitive HTTP actions. Reset
observations emitted by Embodied are not real pentest interactions and do not consume
the 10,000-transition budget.

Dreamer's **internal train-ratio clock** follows the official Embodied training loop.
That clock advances on every Driver callback, including the `is_first` reset
observation. The runner therefore tracks `dreamer_driver_steps` separately and uses
that count for the official train-ratio schedule while hard-checking the primitive
action count independently.

This prevents two opposite errors: reset observations cannot inflate the reported
sample budget, and Dreamer is not under-trained relative to its official update
schedule.

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

Frozen validation and diagnostic phases call Dreamer policy with `mode='eval'` and
never call replay insertion or `agent.train()`. The runner verifies that gradient
updates, Driver-step training clock, and real-transition training counter remain
unchanged across those phases.

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
status, pinned Dreamer upstream commit, canonical Dreamer preset, categorical action
space, legal-action adapter, train ratio, dtype, or JAX platform differ.

## Verification layers

Ordinary current-generation CI checks the pure adapter contract without installing
JAX: 240 unique structural slots, rename invariance, complete stage coverage, legal
projection, exact reset/terminal semantics, and canonical five-condition assembly.

A separate official DreamerV3 CPU smoke checks out the pinned upstream repository and
executes the real upstream `Agent -> Replay -> Driver -> agent.train()` path with the
upstream debug preset. It also verifies the official Driver-step train-ratio cadence.
That CPU/debug result is explicitly non-canonical and cannot be assembled into the
final suite.

The final benchmark still requires a reduced official JAX/CUDA run before the full
experiment is frozen.

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
