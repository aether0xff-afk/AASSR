# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_4_mechanisms_30x10\v2_complex\ablation_4_imagination_mechanisms`
- Output: `runs\ablation_4_mechanisms_30x10\v2_complex\ablation_4_imagination_mechanisms\analysis`
- Conditions: A4 Full C3, A4 no dependency bonus, A4 no policy prior, A4 no repeat penalty, A4 no rollout value, A4 one-step no dependency
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 Full C3 | 0.717 [0.687, 0.753] | 72.093 [68.147, 75.812] | 137.940 | 0.254 | 0.060 | 0.069 | 13.732 |
| A4 no dependency bonus | 0.703 [0.677, 0.733] | 72.953 [70.509, 75.629] | 139.487 | 0.256 | 0.061 | 0.069 | 13.851 |
| A4 no policy prior | 0.730 [0.693, 0.773] | 72.257 [69.684, 75.013] | 137.890 | 0.244 | 0.054 | 0.066 | 13.826 |
| A4 no repeat penalty | 0.383 [0.327, 0.430] | 62.092 [57.137, 66.774] | 111.337 | 0.458 | 0.061 | 0.061 | 9.283 |
| A4 no rollout value | 0.703 [0.677, 0.733] | 72.953 [70.509, 75.629] | 139.487 | 0.256 | 0.060 | 0.069 | 13.878 |
| A4 one-step no dependency | 0.703 [0.677, 0.733] | 72.953 [70.509, 75.629] | 139.487 | 0.256 | 0.060 | 0.069 | 13.878 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
