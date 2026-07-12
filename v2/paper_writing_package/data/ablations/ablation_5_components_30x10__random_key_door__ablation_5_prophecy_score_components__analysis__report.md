# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_5_components_30x10\random_key_door\ablation_5_prophecy_score_components`
- Output: `runs\ablation_5_components_30x10\random_key_door\ablation_5_prophecy_score_components\analysis`
- Conditions: A5 Full C3, A5 no error-avoidance score, A5 no flag-probability score, A5 no knowledge-gain score
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 Full C3 | 0.970 [0.957, 0.983] | 23.144 [21.689, 24.619] | 51.907 | 0.087 | 0.042 | 0.092 | 6.170 |
| A5 no error-avoidance score | 0.973 [0.963, 0.987] | 22.964 [21.358, 24.734] | 51.683 | 0.088 | 0.044 | 0.094 | 6.143 |
| A5 no flag-probability score | 0.970 [0.947, 0.987] | 22.756 [21.499, 24.058] | 51.550 | 0.089 | 0.046 | 0.094 | 6.106 |
| A5 no knowledge-gain score | 0.923 [0.897, 0.950] | 23.173 [22.130, 24.150] | 52.487 | 0.083 | 0.011 | 0.066 | 6.101 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
