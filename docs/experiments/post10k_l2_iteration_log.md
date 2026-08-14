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

### Already-established context

- Simple exact-action without-replacement filtering fired 26 times but remained 0/8.
- The original 10k ON/OFF comparison did not evaluate effective Imagination intervention. The restored Critic had recent signed support positive=42, zero=17, negative=3, while the reliability gate requires four recent negative episodes.
- In the L2 shadow condition the real gate recorded `critic_not_ready` 808 times. Shadow planning produced 15 plans and 5 reliable disagreements with Policy, but this does not prove those alternatives were better.
- ASEQ remains the intended exact semantic self-loop guard. High guard pressure is treated as a symptom unless separately disproven.

## Iteration 1 — structural tie representative only

### Changed variable

For exact top-score ties whose concrete actions also share the exact same structural representation, uniformly choose the concrete representative instead of allowing raw `action.signature` lexicographic order to choose it.

### Result

| Metric | Baseline | Iteration 1 | Delta |
| --- | ---: | ---: | ---: |
| Success | 0/8 | 0/8 | 0 |
| Mean transitions | 101.0 | 101.0 | 0 |
| Mean ASEQ guards | 78.25 | 76.0 | -2.25 |
| Mean unknown attempts | 1.125 | 0.0 | -1.125 |
| Alias states | 435 | 343 | -92 |

Mechanism counters: 576 structural tie events, 576 randomized choices, 3496 tied concrete-action participations, and 457 decisions changed from the signature winner.

### Decision

**REJECT as the primary L2 cause.** The intervention substantially changed concrete representatives but success stayed 0/8 and exhaustion stayed unchanged. Do not promote this behavior to canonical AASSR.

## Iteration 2 — unknown object-profile probe class priority

### Changed variable

Whenever response-causal unknown-profile `request_object` actions exist, force selection into that class while preserving the frozen Policy's existing ranking inside the class.

### Result

| Metric | Baseline | Iteration 2 | Delta |
| --- | ---: | ---: | ---: |
| Success | 0/8 | 0/8 | 0 |
| Mean transitions | 101.0 | 59.0 | -42.0 |
| Mean ASEQ guards | 78.25 | 37.125 | -41.125 |
| Mean unknown attempts | 1.125 | 12.375 | +11.25 |
| Mean unique unknown concrete | 0.375 | 6.125 | +5.75 |
| Mean unique unknown structural | 0.125 | 2.0 | +1.875 |
| Alias states | 435 | 99 | -336 |

Mechanism counters: 99 probe opportunities, 99 forced probe choices, 1245 total probe candidates across those opportunities, maximum 15 candidates at one state.

The aggregate reports success=0, truncation=0, stall=0 while episodes ended after mean 59 transitions. Given the harness status categories, this strongly implies lockout failures; the next diagnostic records failures explicitly rather than relying only on this inference.

### Code-level explanation

The environment treats every non-browse profile as risky. If a route/profile combination is not applicable, it returns the public response fact `request_profile_not_applicable` and increments audit by one; reaching the lockout threshold terminates the episode as failure.

At L2, up to five profile candidates can be crossed with three object candidates, matching the observed maximum of 15 `request_object` candidates.

Critically, the current response-memory observation retains successful semantic roles, tried object IDs, and latest status, but does **not** preserve a route/profile-specific fact saying that a particular profile was observed to be not applicable. Thus the underlying HTTP response contains causal negative evidence that is discarded before the next Policy state.

### Decision

**REJECT forced class priority as a repair, but KEEP the diagnostic finding.** Entering the unknown probe class much more often does not solve L2 and likely causes lockout by repeatedly spending audit budget on negative profile evidence that the observation/memory layer fails to retain relationally.

This moves the main suspicion from generic Policy exploration to **lossy response-semantic memory at the environment/representation boundary**.

## Iteration 3 — preserve one negative route/profile response fact

### Hypothesis

When an unknown profile receives the public response `request_profile_not_applicable` on a route, that response establishes that the same profile should not be retried on the same route merely with a different object ID. Current ASEQ guards exact `(S,A)` self-loops, but different object IDs create different concrete actions; exact-action memory therefore cannot generalize this negative route/profile evidence.

### Single changed variable

Frozen diagnostic only: after observing `request_profile_not_applicable` for one `request_object(route, profile, object)`, remember only the public negative pair `(route, profile)` for the remainder of that episode and suppress other `request_object` actions with that same `(route, profile)` pair while alternatives exist.

This is not an oracle: it never reads the hidden correct profile, target object, or scenario role. It uses only the response body the mock HTTP server actually returned for the executed action.

### Explicitly unchanged

- environment dynamics and difficulty
- checkpoint weights
- state/action neural encodings
- Policy values and ranking among remaining actions
- reward
- ASEQ implementation
- Prophecy
- Critic
- Imagination disabled
- scenario seeds and episode cap

### Interpretation

- Success increase and lower lockout/failure pressure: missing route/profile negative-response memory is a major L2 defect.
- Fewer repeated bad-profile probes but still 0/8: negative memory matters but another downstream issue remains; next inspect whether positive profile-role discovery is learned/retained correctly.
- Little/no filtering: the hypothesis is wrong or the relevant negative response is not occurring where expected.

Do not promote the diagnostic filter itself to canonical AASSR. A positive result would justify a principled response-semantic memory representation in the plugin/observation boundary.