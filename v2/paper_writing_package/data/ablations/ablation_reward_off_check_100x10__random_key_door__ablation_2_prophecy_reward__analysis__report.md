# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\ablation_reward_off_check_100x10\random_key_door\ablation_2_prophecy_reward`
- Output: `runs\ablation_reward_off_check_100x10\random_key_door\ablation_2_prophecy_reward\analysis`
- Conditions: A2 Prophecy reward off, A2 Prophecy reward on
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 Prophecy reward off | 0.970 [0.962, 0.978] | 23.694 [22.672, 24.581] | 52.501 | 0.086 | 0.039 | 0.084 | 5.559 |
| A2 Prophecy reward on | 0.964 [0.955, 0.974] | 23.534 [22.519, 24.427] | 52.416 | 0.089 | 0.039 | 0.084 | 6.137 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
