# Post-10k L2 Iteration Log

This file is the frozen experiment notebook for the post-10k L2 failure investigation.

## Rule for all following iterations

Change exactly one scientific variable at a time.

1. Record the current baseline and hypothesis.
2. Change one behavior only.
3. Re-run the same frozen checkpoint and the same diagnostic seeds when possible.
4. Record success, transition count, ASEQ pressure, and any mechanism-specific counters.
5. Keep the change only if the result supports it; otherwise revert/reject it before the next hypothesis.
6. Do not change environment difficulty, representation, Policy, Prophecy, Critic, ASEQ, Imagination gate, and curriculum together.

## Frozen baseline

- Checkpoint source commit: `500ff175ca089fdf6bd4e822a1fdabdef7b1b69e`
- Checkpoint: seed 7, 10,000 real transitions, frozen-evaluation-only.
- Diagnostic harness commit: `82416de3270a7fc2398bc6695bf89d40e662c047`
- Diagnostic seeds: `92001..92008`.
- Baseline L2 stage: `l2_profile_choices`, `extra_profile_count=4`.
- Frozen diagnostic wall time: 3006.212 s.

### Profile-count sweep

| Extra profile decoys | Success | Mean transitions | Mean ASEQ guards | Alias states |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3/8 | 72.125 | 48.875 | 173 |
| 1 | 0/8 | 97.25 | 74.375 | 405 |
| 2 | 0/8 | 98.25 | 75.5 | 413 |
| 4 | 0/8 | 101.0 | 78.25 | 435 |

Observed fact: L1->L2 changes only `extra_profile_count` from 0 to 4 in the transfer curriculum. Adding even one profile decoy collapses this frozen checkpoint from 3/8 to 0/8 on the development diagnostic seeds.

### Already-rejected/simple explanations

- **Simple repeated-candidate memory is not established as the primary cause.** Diagnostic without-replacement filtering fired 26 times but remained 0/8.
- **Imagination effectiveness was not evaluated by the original 10k ON/OFF result.** The restored Critic is basically ready but not reliably ready: recent signed support is positive=42, zero=17, negative=3, while the gate requires four recent negative episodes. In the L2 shadow condition the real gate recorded `critic_not_ready` 808 times.
- Shadow planning did produce alternative preferences: 15 shadow plans, 5 reliable disagreements with Policy. This is evidence that planning is not identically equal to Policy, but it does not prove that the alternative action is better.
- ASEQ semantics themselves remain the intended exact semantic self-loop guard. The very high L2 guard rate is treated as a symptom of poor upstream decisions unless a later experiment proves otherwise.

## Current system diagnosis before Iteration 1

The environment itself is not currently classified as broken: it exposes the read-profile candidate and decoy profile candidates together through response-causal observations and does not expose which candidate is correct.

The strongest current structural concern is the boundary between that environment and the rename-invariant action representation. Concrete profile candidates with the same observed role can map to identical structural action features. Policy value and information residual therefore tie for those concrete representatives. Current Policy ranking resolves an exact score tie using raw `action.signature` lexicographic order.

That raw-signature winner is inconsistent with the intended rename-invariant abstraction: the model declares the candidates structurally equivalent, then an arbitrary concrete identifier string decides which representative is executed.

This is a hypothesis, not yet the root-cause conclusion.

## Iteration 1 — structural tie representative only

### Single changed variable

When the frozen Policy has an exact top-score tie among concrete actions that also have the exact same structural action representation, do **not** let lexicographic raw `action.signature` choose the concrete representative. Uniformly choose one concrete representative within that one structural equivalence class using the already-restored deterministic agent RNG.

### Explicitly unchanged

- `src/aassr_v2` runtime code
- environment and L2 stage
- state representation
- action representation
- checkpoint weights
- Policy scores
- information residual values
- Prophecy
- Critic
- ASEQ
- Imagination (disabled for this experiment)
- seeds and episode cap

This first test is diagnostic-only. If it helps, a later iteration may implement a principled core fix. If it does not help, it is rejected before changing representation semantics.

### Interpretation

- Improvement from 0/8 and/or a material decrease in ASEQ pressure: raw-signature representative bias contributes to the L2 cliff.
- Near-baseline 0/8 with similar ASEQ pressure: reject raw-signature tie-breaking as the primary L2 cause and move to the next single-variable representation experiment.

Do not promote this diagnostic behavior to canonical AASSR before observing the result.
