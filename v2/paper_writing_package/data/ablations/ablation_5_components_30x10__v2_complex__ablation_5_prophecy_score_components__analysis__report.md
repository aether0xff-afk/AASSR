# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_5_components_30x10\v2_complex\ablation_5_prophecy_score_components`
- Output: `runs\ablation_5_components_30x10\v2_complex\ablation_5_prophecy_score_components\analysis`
- Conditions: A5 Full C3, A5 no error-avoidance score, A5 no flag-probability score, A5 no knowledge-gain score
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 Full C3 | 0.717 [0.687, 0.753] | 72.093 [68.147, 75.812] | 137.940 | 0.254 | 0.060 | 0.069 | 13.732 |
| A5 no error-avoidance score | 0.683 [0.630, 0.720] | 68.875 [64.819, 72.833] | 136.987 | 0.253 | 0.073 | 0.075 | 13.471 |
| A5 no flag-probability score | 0.700 [0.667, 0.730] | 72.912 [69.874, 75.960] | 140.623 | 0.258 | 0.061 | 0.070 | 14.014 |
| A5 no knowledge-gain score | 0.750 [0.707, 0.797] | 64.991 [61.784, 67.830] | 130.400 | 0.193 | 0.034 | 0.053 | 13.242 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
