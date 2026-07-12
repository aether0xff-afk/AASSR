# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_4_mechanisms_30x10\random_key_door\ablation_4_imagination_mechanisms`
- Output: `runs\ablation_4_mechanisms_30x10\random_key_door\ablation_4_imagination_mechanisms\analysis`
- Conditions: A4 Full C3, A4 no dependency bonus, A4 no policy prior, A4 no repeat penalty, A4 no rollout value, A4 one-step no dependency
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 Full C3 | 0.970 [0.957, 0.983] | 23.144 [21.689, 24.619] | 51.907 | 0.087 | 0.042 | 0.092 | 6.170 |
| A4 no dependency bonus | 0.977 [0.960, 0.990] | 23.224 [21.771, 24.659] | 51.877 | 0.082 | 0.041 | 0.092 | 6.204 |
| A4 no policy prior | 0.957 [0.927, 0.983] | 23.404 [21.996, 24.526] | 51.643 | 0.096 | 0.036 | 0.087 | 6.057 |
| A4 no repeat penalty | 0.880 [0.853, 0.913] | 20.686 [19.071, 22.225] | 50.563 | 0.144 | 0.041 | 0.089 | 5.608 |
| A4 no rollout value | 0.977 [0.960, 0.990] | 23.224 [21.771, 24.659] | 51.877 | 0.082 | 0.041 | 0.092 | 6.204 |
| A4 one-step no dependency | 0.977 [0.960, 0.990] | 23.224 [21.771, 24.659] | 51.877 | 0.082 | 0.041 | 0.092 | 6.204 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
