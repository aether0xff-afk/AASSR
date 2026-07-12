# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_env_sweep_30x10\locked_bottleneck\ablation_1_prophecy_model`
- Output: `runs\ablation_env_sweep_30x10\locked_bottleneck\ablation_1_prophecy_model\analysis`
- Conditions: A1 Table Prophecy C3, A1 Transformer Prophecy C3
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 Table Prophecy C3 | 0.423 [0.290, 0.563] | 74.397 [69.300, 79.177] | 139.207 | 0.352 | 0.051 | 0.059 | 13.030 |
| A1 Transformer Prophecy C3 | 0.397 [0.303, 0.487] | 76.775 [71.394, 81.870] | 110.047 | 0.464 | 0.028 | 0.106 | 11.230 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
