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

### Changed variable

Iteration 2 probe priority is held fixed. After observing `request_profile_not_applicable` for one `request_object(route, profile, object)`, remember the public negative pair `(route, profile)` for the remainder of that episode and suppress other `request_object` actions with that same pair while alternatives exist.

### Result

| Metric | Iteration 2 | Iteration 3 | Delta |
| --- | ---: | ---: | ---: |
| Success | 0/8 | 1/8 | +1 |
| Success rate | 0.0 | 0.125 | +0.125 |
| Failures | inferred lockout-dominated | 0 | improved |
| Truncations | 0 | 7 | +7 |
| Mean transitions | 59.0 | 90.75 | +31.75 |
| Mean ASEQ guards | 37.125 | 64.375 | +27.25 |
| Mean unknown attempts | 12.375 | 7.875 | -4.5 |
| Alias states | 99 | 353 | +254 |

Mechanism counters:

- 42 negative-response events
- 42 new invalid route/profile pairs
- 352 negative-memory filter events
- 5142 filtered follow-up actions
- 62 forced probe choices under the fixed Iteration-2 scaffold

### Decision

**KEEP as a supported defect signal, but do not yet promote the diagnostic filter to canonical AASSR.**

The change produced the first L2 success in this sequence and eliminated observed failure episodes, while the remaining seven episodes survived longer and ended by truncation. With only eight development seeds, 1/8 is not enough to claim the problem is solved, but the mechanism fired strongly and the failure mode changed in the predicted direction.

This supports the claim that current observation/memory loses useful public negative route/profile semantics. Another downstream defect remains because seven episodes still exhaust their budget.

## Iteration 4 — do not mark an object as globally tried when the profile itself was invalid

### Hypothesis

Current `ResponseMemory.observe` adds every `object_id` from every `request_object` action to the global `tried_objects` set before it knows whether the selected profile was applicable. Therefore an attempt such as:

`wrong-profile + object-X -> request_profile_not_applicable`

can still make later actions treat `object-X` as already tried. At L1 this is mostly harmless because there is essentially one relevant profile candidate. At L2, profile candidates multiply, so trying the target object under a wrong profile can poison the later relational state for the correct profile.

This is a stronger L1->L2-specific explanation than raw candidate identity alone: the memory relation `tried(object)` is too coarse and should depend on whether the request actually tested the object semantics.

### Single changed variable

Hold Iteration 2 probe priority and Iteration 3 negative route/profile memory fixed. Change only this bookkeeping rule during frozen diagnosis:

- if a `request_object` response contains public fact `request_profile_not_applicable`, that attempt must **not newly add** its `object_id` to `tried_objects`;
- if the object had already been legitimately tried before that invalid-profile action, keep the existing tried state;
- every non-`request_profile_not_applicable` object request keeps the original bookkeeping.

No hidden profile identity, target identity, or oracle information is read.

### Interpretation

- A material success increase over Iteration 3: global object-tried pollution from invalid profile probes is a major L2 defect.
- Similar 1/8 but clear counter activity: the bookkeeping is semantically wrong but not the main remaining bottleneck; move downstream to positive read-profile/target discovery retention.
- Little/no prevented pollution: reject this as a major explanation.

Do not change canonical runtime until this single-variable diagnostic is observed.
