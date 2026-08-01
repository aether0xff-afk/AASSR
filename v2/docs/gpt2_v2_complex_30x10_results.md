# GPT-2 v2_complex 30x10 benchmark results

## Status

This is a preliminary branch result, not a replacement for C5.

Run configuration:

```text
world = v2_complex
evaluation = 30 episodes x 10 seeds
step_limit = 120
academy pretraining = 100 episodes per seed
academy prophecy = sequence
GitHub Actions run = 30325845508
```

All four matrix jobs completed successfully and uploaded raw CSV/JSON artifacts.

## Main comparison

| Condition | Success | Steps | Repeats | Errors | Unique actions | Prophecy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C5 | 0.7167 | 83.42 | 24.72 | 4.02 | 58.70 | 0.0624 |
| GPT2_REWARD | 0.5500 | 93.27 | 45.08 | 8.55 | 47.30 | 0.0065 |
| GPT2_ACADEMY_MODEL | 0.5433 | 91.74 | 42.73 | 7.48 | 48.49 | 0.7768 |
| GPT2_ACADEMY_FULL | 0.6433 | 83.34 | 30.94 | 3.51 | 52.40 | 0.8625 |

The academy conditions use the new consequence reward. Their matched no-academy baseline was:

```text
success = 0.5667
steps = 90.57
repeats = 41.77
errors = 7.34
prophecy F1 = 0.6395
```

## Result 1: the first actionable reward failed

`GPT2_REWARD` was materially worse than C5:

```text
success delta = -0.1667
repeat delta = +20.37
error delta = +4.53
```

A paired seed-level exploratory t-test gives `p = 0.00028` for the success difference. This is not a final inferential analysis, but the degradation is large and consistent enough that the current reward must not replace C5.

### Failure diagnosis

The reward diagnostics show:

```text
mean newly unlocked actions per step = 2.288
mean newly locked actions per step = 2.338
steps with a positive unlock = 74.1%
mean unlock reward per step = +0.0285
mean cycle penalty per step = -0.0030
```

The unlocked and locked counts are nearly symmetric. The agent is therefore rewarded for transient candidate churn caused by movement/frontier changes, not only for durable capability gain.

The positive unlock signal is about an order of magnitude larger than the average cycle penalty. This creates a reward loophole: moving around can repeatedly create and remove executable candidates while still producing positive reward.

The next reward revision should therefore use one or more of:

1. persistent unlock: an action/affordance must remain available for `k` steps;
2. net unlock: reward only `max(0, unlocked - locked)`;
3. capability-class unlock: count new action schemas or dependency affordances, not coordinate-specific frontier candidates;
4. state-conditioned repeat: penalize returning to the same semantic state even if candidate identities changed;
5. zero or negative reward for pure candidate-pool churn.

## Result 2: model-only academy improved prediction but not behavior

Compared with its matched no-academy baseline:

```text
success: 0.5667 -> 0.5433
prophecy F1: 0.6395 -> 0.7768
```

The transition/Prophecy model became much more accurate, but this did not improve action selection. Resetting PolicyABC removed the behavioral knowledge needed to exploit the academy experience.

This means that, in the current architecture, better prediction alone is not sufficient. Prophecy quality and policy use are still weakly coupled.

Creativity was preserved:

```text
trajectory diversity = 1.000
trajectory entropy = 1.000
novel strategy rate = 0.994
policy override rate = 0.885
```

The model-only guardrail failed only because success decreased, not because strategy diversity collapsed.

## Result 3: retaining the academy policy prior helped

`GPT2_ACADEMY_FULL` compared with its matched no-academy baseline:

```text
success: 0.5667 -> 0.6433  (+0.0767)
steps: 90.57 -> 83.34
repeats: 41.77 -> 30.94
errors: 7.34 -> 3.51
prophecy F1: 0.6395 -> 0.8625
```

The paired seed-level exploratory success test gives `p = 0.061`. The direction is promising but not yet strong enough for a final claim.

The creativity guardrail passed:

```text
trajectory diversity = 1.000
trajectory entropy = 1.000
novel strategy rate = 1.000
policy override rate = 0.716
guardrail passed = true
```

Thus the policy prior did not force imitation of academy trajectories. The agent retained substantial override behavior and produced structurally novel successful paths.

## Result 4: GPT2_ACADEMY_FULL still did not beat C5

```text
C5 success = 0.7167
GPT2_ACADEMY_FULL success = 0.6433
delta = -0.0733
```

The exploratory paired seed-level test gives `p = 0.191`. The result does not establish a statistically reliable difference at ten seeds, but the current point estimate is still lower than C5.

Therefore:

- do not replace C5;
- keep `GPT2_ACADEMY_FULL` as an experimental condition;
- revise the actionable reward before larger experiments;
- retain policy-prior and model-only conditions as separate ablations;
- do not begin KK-update embedding yet.

## Main research interpretation

The academy itself is not the main failure. It successfully transferred predictive and policy knowledge without collapsing strategy diversity.

The central failure is the reward definition:

> instantaneous executable-candidate growth is not equivalent to durable problem-solving capability.

The next GPT-2 revision should define progress through persistent dependency/affordance transitions rather than raw candidate-set changes.
