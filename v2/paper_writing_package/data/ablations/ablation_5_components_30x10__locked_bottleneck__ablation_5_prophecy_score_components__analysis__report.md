# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_5_components_30x10\locked_bottleneck\ablation_5_prophecy_score_components`
- Output: `runs\ablation_5_components_30x10\locked_bottleneck\ablation_5_prophecy_score_components\analysis`
- Conditions: A5 Full C3, A5 no error-avoidance score, A5 no flag-probability score, A5 no knowledge-gain score
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 Full C3 | 0.423 [0.290, 0.563] | 74.397 [69.300, 79.177] | 139.207 | 0.352 | 0.051 | 0.059 | 13.030 |
| A5 no error-avoidance score | 0.147 [0.067, 0.240] | 88.026 [78.318, 99.438] | 132.310 | 0.472 | 0.080 | 0.068 | 11.043 |
| A5 no flag-probability score | 0.443 [0.307, 0.573] | 75.864 [71.979, 79.920] | 140.910 | 0.353 | 0.051 | 0.059 | 13.219 |
| A5 no knowledge-gain score | 0.763 [0.747, 0.780] | 60.101 [59.311, 61.177] | 119.817 | 0.226 | 0.004 | 0.036 | 12.079 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
