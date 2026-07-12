# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\paper_rollout_v2_complex_30x10`
- Output: `runs\paper_rollout_v2_complex_30x10\analysis`
- Conditions: C0 Random, C1 PolicyABC, C2 PolicyABC + Prophecy, C3 PolicyABC + Prophecy + Imagination, C4 PolicyABC + Sequence Prophecy variant + Imagination, DQN partial-observation baseline, Oracle MDP, full-map upper bound, Q-learning baseline
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 Random | 0.427 [0.347, 0.500] | 74.705 [71.524, 77.531] | 119.417 | 0.504 | 0.059 | 0.000 | 8.440 |
| C1 PolicyABC | 0.540 [0.457, 0.620] | 76.169 [73.891, 78.521] | 127.897 | 0.363 | 0.047 | 0.000 | 10.341 |
| C2 PolicyABC + Prophecy | 0.530 [0.467, 0.593] | 75.124 [73.169, 77.050] | 128.243 | 0.371 | 0.061 | 0.059 | 11.691 |
| C3 PolicyABC + Prophecy + Imagination | 0.717 [0.687, 0.753] | 72.093 [68.147, 75.812] | 137.940 | 0.254 | 0.060 | 0.069 | 13.732 |
| C4 PolicyABC + Sequence Prophecy variant + Imagination | 0.703 [0.663, 0.750] | 62.863 [60.418, 65.362] | 131.140 | 0.208 | 0.034 | 0.075 | 13.730 |
| DQN partial-observation baseline | 0.593 [0.490, 0.697] | 73.312 [68.104, 78.863] | 127.477 | 0.354 | 0.054 | 0.000 | 10.338 |
| Oracle MDP, full-map upper bound | 0.987 [0.973, 1.000] | 9.207 [8.931, 9.508] | 0.000 | 0.000 | 0.000 | 0.000 | 0.987 |
| Q-learning baseline | 0.573 [0.520, 0.633] | 70.022 [67.896, 71.998] | 119.983 | 0.372 | 0.034 | 0.000 | 9.896 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
