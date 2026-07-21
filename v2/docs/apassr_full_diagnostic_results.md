# APASSR_FULL Diagnostic Results

Commands were run with 30 episodes, 10 seeds, and step limit 120. No reward,
policy, prophecy, rollout depth, or world-layout tuning was applied.

This document records the original uncalibrated `APASSR_FULL` diagnostic run.
The calibrated condition `APASSR_FULL_CAL` was added afterward as a separate
comparison condition; calibrated result tables should be generated with
`--include-apassr-full --include-apassr-full-cal` and reported separately.

## v2_complex 30x10

| Condition | Success | Steps | Repeat | Error |
| --- | ---: | ---: | ---: | ---: |
| C3 | 0.717 | 82.32 | 25.02 | 4.52 |
| C5 | 0.703 | 85.05 | 24.82 | 4.90 |
| APASSR_FULL | 0.647 | 86.45 | 29.30 | 5.74 |
| QLEARN | 0.500 | 93.85 | 45.49 | 4.99 |
| DQN_PARTIAL | 0.643 | 89.69 | 35.92 | 5.16 |
| ORACLE_MDP | 0.987 | 10.69 | 0.00 | 0.00 |

APASSR_FULL structural diagnostics:

| Metric | Value |
| --- | ---: |
| imagined state transitions / episode | 7555.237 |
| mean selected trajectory depth | 2.996 |
| newly unlocked actions / episode | 25596.443 |
| future dependency selection rate | 0.747 |
| setup actions / episode | 21.327 |
| KK precision / recall / F1 | 0.173 / 0.167 / 0.169 |
| imagined next exact / WHAT match | 0.176 / 0.612 |
| placeholder generated candidates / episode | 0.010 |
| placeholder execution attempts / episode | 0.000 |

## locked_bottleneck 30x10

| Condition | Success | Steps | Repeat | Error |
| --- | ---: | ---: | ---: | ---: |
| C3 | 0.400 | 105.39 | 46.37 | 6.80 |
| C5 | 0.397 | 105.54 | 50.97 | 3.31 |
| APASSR_FULL | 0.197 | 112.57 | 54.78 | 4.45 |
| QLEARN | 0.237 | 113.67 | 53.05 | 3.74 |
| DQN_PARTIAL | 0.397 | 106.77 | 47.22 | 5.76 |
| ORACLE_MDP | 1.000 | 15.09 | 0.00 | 0.00 |

APASSR_FULL structural diagnostics:

| Metric | Value |
| --- | ---: |
| imagined state transitions / episode | 8337.917 |
| mean selected trajectory depth | 2.899 |
| newly unlocked actions / episode | 30028.073 |
| future dependency selection rate | 0.808 |
| setup actions / episode | 16.993 |
| KK precision / recall / F1 | 0.475 / 0.469 / 0.470 |
| imagined next exact / WHAT match | 0.193 / 0.658 |
| placeholder generated candidates / episode | 2.230 |
| placeholder execution attempts / episode | 0.000 |

## Interpretation

The full structure is active: rollout depth is near 3, virtual transitions are
large, and many future actions are unlocked. The low `locked_bottleneck`
performance is not because imagination is inactive. The sharper issue is action
quality and follow-through: many future-dependent choices are made, but exact
imagined-next-action match remains below 0.20 in both environments.

Prophecy KK quality is environment-dependent. In `v2_complex`, APASSR_FULL KK
F1 is low at 0.169, lower than C3's 0.207. In `locked_bottleneck`, KK F1 rises
to 0.470 and exceeds C3/C5, yet success is only 0.197. That points away from
Prophecy accuracy as the sole cause and toward rollout scoring/future-candidate
selection producing too many weak future dependencies.

Placeholder use is measured and does not leak into real execution:
`placeholder_execution_attempt_count` stayed zero in smoke and both 30x10 runs.
