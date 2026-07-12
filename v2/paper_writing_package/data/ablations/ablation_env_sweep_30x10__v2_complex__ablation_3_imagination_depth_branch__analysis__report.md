# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_env_sweep_30x10\v2_complex\ablation_3_imagination_depth_branch`
- Output: `runs\ablation_env_sweep_30x10\v2_complex\ablation_3_imagination_depth_branch\analysis`
- Conditions: A3 depth 1, branch 1, A3 depth 1, branch 3, A3 depth 1, branch 5, A3 depth 2, branch 1, A3 depth 2, branch 3, A3 depth 2, branch 5, A3 depth 3, branch 1, A3 depth 3, branch 3, A3 depth 3, branch 5
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 depth 1, branch 1 | 0.703 [0.677, 0.733] | 72.953 [70.509, 75.629] | 139.487 | 0.256 | 0.060 | 0.069 | 13.878 |
| A3 depth 1, branch 3 | 0.703 [0.677, 0.733] | 72.953 [70.509, 75.629] | 139.487 | 0.256 | 0.060 | 0.069 | 13.878 |
| A3 depth 1, branch 5 | 0.703 [0.677, 0.733] | 72.953 [70.509, 75.629] | 139.487 | 0.256 | 0.060 | 0.069 | 13.878 |
| A3 depth 2, branch 1 | 0.737 [0.700, 0.770] | 73.864 [70.328, 77.369] | 139.393 | 0.250 | 0.057 | 0.069 | 13.988 |
| A3 depth 2, branch 3 | 0.717 [0.687, 0.753] | 72.093 [68.147, 75.812] | 137.940 | 0.254 | 0.060 | 0.069 | 13.732 |
| A3 depth 2, branch 5 | 0.713 [0.683, 0.747] | 72.721 [69.607, 75.919] | 138.397 | 0.256 | 0.060 | 0.069 | 13.756 |
| A3 depth 3, branch 1 | 0.723 [0.687, 0.763] | 72.991 [68.985, 76.685] | 139.093 | 0.251 | 0.059 | 0.068 | 13.875 |
| A3 depth 3, branch 3 | 0.703 [0.670, 0.743] | 71.766 [68.207, 75.231] | 137.393 | 0.256 | 0.061 | 0.069 | 13.611 |
| A3 depth 3, branch 5 | 0.717 [0.687, 0.757] | 72.265 [68.947, 75.653] | 138.143 | 0.253 | 0.061 | 0.069 | 13.742 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
