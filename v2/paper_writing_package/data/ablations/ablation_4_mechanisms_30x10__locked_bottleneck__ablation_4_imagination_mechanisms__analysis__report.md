# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_4_mechanisms_30x10\locked_bottleneck\ablation_4_imagination_mechanisms`
- Output: `runs\ablation_4_mechanisms_30x10\locked_bottleneck\ablation_4_imagination_mechanisms\analysis`
- Conditions: A4 Full C3, A4 no dependency bonus, A4 no policy prior, A4 no repeat penalty, A4 no rollout value, A4 one-step no dependency
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 Full C3 | 0.423 [0.290, 0.563] | 74.397 [69.300, 79.177] | 139.207 | 0.352 | 0.051 | 0.059 | 13.030 |
| A4 no dependency bonus | 0.470 [0.317, 0.633] | 74.109 [71.987, 76.446] | 143.317 | 0.336 | 0.053 | 0.062 | 13.621 |
| A4 no policy prior | 0.530 [0.447, 0.607] | 79.149 [74.026, 84.428] | 146.007 | 0.323 | 0.039 | 0.054 | 14.090 |
| A4 no repeat penalty | 0.077 [0.023, 0.140] | 61.625 [57.750, 66.425] | 105.720 | 0.621 | 0.048 | 0.046 | 7.383 |
| A4 no rollout value | 0.470 [0.317, 0.633] | 74.109 [71.987, 76.446] | 143.317 | 0.336 | 0.053 | 0.062 | 13.621 |
| A4 one-step no dependency | 0.470 [0.317, 0.633] | 74.109 [71.987, 76.446] | 143.317 | 0.336 | 0.053 | 0.062 | 13.621 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
