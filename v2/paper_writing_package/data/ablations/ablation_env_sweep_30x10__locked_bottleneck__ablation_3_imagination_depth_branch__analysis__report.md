# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_env_sweep_30x10\locked_bottleneck\ablation_3_imagination_depth_branch`
- Output: `runs\ablation_env_sweep_30x10\locked_bottleneck\ablation_3_imagination_depth_branch\analysis`
- Conditions: A3 depth 1, branch 1, A3 depth 1, branch 3, A3 depth 1, branch 5, A3 depth 2, branch 1, A3 depth 2, branch 3, A3 depth 2, branch 5, A3 depth 3, branch 1, A3 depth 3, branch 3, A3 depth 3, branch 5
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 depth 1, branch 1 | 0.470 [0.317, 0.633] | 74.109 [71.987, 76.446] | 143.317 | 0.336 | 0.053 | 0.062 | 13.621 |
| A3 depth 1, branch 3 | 0.470 [0.317, 0.633] | 74.109 [71.987, 76.446] | 143.317 | 0.336 | 0.053 | 0.062 | 13.621 |
| A3 depth 1, branch 5 | 0.470 [0.317, 0.633] | 74.109 [71.987, 76.446] | 143.317 | 0.336 | 0.053 | 0.062 | 13.621 |
| A3 depth 2, branch 1 | 0.453 [0.330, 0.580] | 77.496 [72.504, 82.430] | 141.670 | 0.346 | 0.051 | 0.059 | 13.378 |
| A3 depth 2, branch 3 | 0.423 [0.290, 0.563] | 74.397 [69.300, 79.177] | 139.207 | 0.352 | 0.051 | 0.059 | 13.030 |
| A3 depth 2, branch 5 | 0.393 [0.280, 0.510] | 75.020 [69.992, 79.961] | 139.283 | 0.365 | 0.051 | 0.058 | 12.928 |
| A3 depth 3, branch 1 | 0.493 [0.357, 0.633] | 74.888 [71.350, 78.052] | 142.167 | 0.326 | 0.050 | 0.059 | 13.594 |
| A3 depth 3, branch 3 | 0.427 [0.297, 0.563] | 76.659 [71.389, 82.287] | 139.903 | 0.358 | 0.052 | 0.059 | 13.068 |
| A3 depth 3, branch 5 | 0.427 [0.320, 0.533] | 74.376 [69.832, 78.863] | 140.157 | 0.354 | 0.051 | 0.059 | 13.130 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
