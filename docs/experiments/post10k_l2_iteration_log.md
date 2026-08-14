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

## System diagnosis before Iteration 1

The environment itself is not currently classified as broken: it exposes the read-profile candidate and decoy profile candidates together through response-causal observations and does not expose which candidate is correct.

Concrete profile candidates with the same observed role can map to identical structural action features. Policy value and information residual can therefore tie for concrete representatives. Current Policy ranking resolves an exact score tie using raw `action.signature` lexicographic order.

This made raw-signature representative bias a reasonable first hypothesis, but not a root-cause conclusion.

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
- Imagination (disabled)
- seeds and episode cap

### Result

| Metric | Frozen baseline | Iteration 1 | Delta |
| --- | ---: | ---: | ---: |
| Success | 0/8 | 0/8 | 0 |
| Mean transitions | 101.0 | 101.0 | 0 |
| Mean ASEQ guards | 78.25 | 76.0 | -2.25 |
| Mean unknown attempts | 1.125 | 0.0 | -1.125 |
| Alias states | 435 | 343 | -92 |

Mechanism counters:

- structural tie events: 576
- randomized choices: 576
- concrete actions participating across tie events: 3496
- selections changed from the lexicographic winner: 457

### Decision

**REJECT as the primary L2 cause.**

The intervention was strong enough to change the concrete representative on 457 decisions, but success remained 0/8 and transition exhaustion remained unchanged. ASEQ pressure fell only slightly. More importantly, unknown-profile attempts fell to zero.

Therefore the main failure appears to occur **before concrete representative selection**: the frozen Policy often does not rank the unknown object-profile probe class highly enough to enter it at all. Raw-signature tie-breaking may still be undesirable implementation detail, but fixing it is not justified as the next scientific repair for the L2 cliff.

Do not promote Iteration-1 behavior to canonical AASSR.

## Iteration 2 — unknown object-profile probe class priority

### Hypothesis

At L2, the crucial new actions are `request_object` actions whose profile role is still unknown. The frozen Policy may be ranking this whole response-causal probe class below other available actions, so ASEQ spends most of the episode suppressing no-progress choices while the agent barely tests the new profile candidates.

### Single changed variable

During frozen evaluation only, when one or more **unknown-profile `request_object`** actions are available, choose the highest-scoring action **within that subset according to the existing frozen Policy ranking**, instead of choosing the global highest-scoring action.

This changes only the priority of the unknown object-profile probe class.

### Explicitly unchanged

- environment and L2 stage
- state representation
- action representation
- all learned weights
- Policy scores and ordering inside the unknown subset
- no candidate ID/answer is exposed
- no extra tried-candidate memory is added
- no randomization is added
- no reward or information bonus is added
- ASEQ unchanged
- Prophecy unchanged
- Critic unchanged
- Imagination disabled
- same seeds and transition cap

### Interpretation rule

- If success rises materially and unknown attempts rise, the main current bottleneck is the Policy's **class-level exploration/ranking of response-causal profile probes**, not concrete tie-breaking.
- If unknown attempts rise strongly but success remains 0/8, then merely entering the candidate class is insufficient; the next experiment should test within-candidate relational memory/identity or downstream response learning.
- If the intervention rarely fires, the failure lies earlier in action-surface/state construction and this hypothesis is rejected.

Iteration 2 is diagnostic-only and must not be promoted to canonical AASSR without a positive result and a subsequent principled implementation test.
