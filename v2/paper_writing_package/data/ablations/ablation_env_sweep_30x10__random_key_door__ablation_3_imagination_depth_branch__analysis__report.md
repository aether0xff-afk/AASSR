# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_env_sweep_30x10\random_key_door\ablation_3_imagination_depth_branch`
- Output: `runs\ablation_env_sweep_30x10\random_key_door\ablation_3_imagination_depth_branch\analysis`
- Conditions: A3 depth 1, branch 1, A3 depth 1, branch 3, A3 depth 1, branch 5, A3 depth 2, branch 1, A3 depth 2, branch 3, A3 depth 2, branch 5, A3 depth 3, branch 1, A3 depth 3, branch 3, A3 depth 3, branch 5
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 depth 1, branch 1 | 0.977 [0.960, 0.990] | 23.224 [21.771, 24.659] | 51.877 | 0.082 | 0.041 | 0.092 | 6.204 |
| A3 depth 1, branch 3 | 0.977 [0.960, 0.990] | 23.224 [21.771, 24.659] | 51.877 | 0.082 | 0.041 | 0.092 | 6.204 |
| A3 depth 1, branch 5 | 0.977 [0.960, 0.990] | 23.224 [21.771, 24.659] | 51.877 | 0.082 | 0.041 | 0.092 | 6.204 |
| A3 depth 2, branch 1 | 0.970 [0.957, 0.983] | 23.122 [21.689, 24.574] | 51.907 | 0.087 | 0.041 | 0.092 | 6.172 |
| A3 depth 2, branch 3 | 0.970 [0.957, 0.983] | 23.144 [21.689, 24.619] | 51.907 | 0.087 | 0.042 | 0.092 | 6.170 |
| A3 depth 2, branch 5 | 0.970 [0.957, 0.983] | 23.136 [21.692, 24.610] | 51.907 | 0.087 | 0.041 | 0.092 | 6.171 |
| A3 depth 3, branch 1 | 0.973 [0.957, 0.987] | 23.235 [21.713, 24.756] | 51.920 | 0.086 | 0.042 | 0.092 | 6.187 |
| A3 depth 3, branch 3 | 0.967 [0.950, 0.983] | 23.068 [21.689, 24.529] | 51.917 | 0.088 | 0.041 | 0.092 | 6.158 |
| A3 depth 3, branch 5 | 0.967 [0.950, 0.983] | 23.093 [21.689, 24.585] | 51.903 | 0.089 | 0.042 | 0.092 | 6.154 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
