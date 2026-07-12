# AASSR GridWorld Analysis

## Experiment Setting

- Input: `runs\paper_rollout_locked_30x10`
- Output: `runs\paper_rollout_locked_30x10\analysis`
- Conditions: C0 Random, C1 PolicyABC, C2 PolicyABC + Prophecy, C3 PolicyABC + Prophecy + Imagination, C4 PolicyABC + Sequence Prophecy variant + Imagination, DQN partial-observation baseline, Oracle MDP, full-map upper bound, Q-learning baseline
- Confidence interval: seed-level bootstrap 95% CI
- Steps to FLAG are averaged over successful episodes only.

## Summary

| condition | success_rate | steps_to_flag | semantic_gain | repeat_rate | error_rate | prophecy_error | total_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 Random | 0.337 [0.303, 0.373] | 109.050 [107.391, 110.624] | 138.317 | 0.558 | 0.053 | 0.000 | 9.142 |
| C1 PolicyABC | 0.397 [0.277, 0.487] | 109.665 [104.642, 113.792] | 146.283 | 0.462 | 0.045 | 0.000 | 10.771 |
| C2 PolicyABC + Prophecy | 0.190 [0.103, 0.277] | 111.066 [106.169, 116.620] | 140.757 | 0.513 | 0.071 | 0.066 | 11.482 |
| C3 PolicyABC + Prophecy + Imagination | 0.423 [0.303, 0.543] | 81.475 [76.058, 86.564] | 138.943 | 0.418 | 0.051 | 0.057 | 12.261 |
| C4 PolicyABC + Sequence Prophecy variant + Imagination | 0.527 [0.433, 0.607] | 73.984 [70.120, 78.092] | 114.937 | 0.416 | 0.020 | 0.056 | 10.395 |
| DQN partial-observation baseline | 0.553 [0.490, 0.617] | 98.071 [92.968, 103.220] | 146.277 | 0.410 | 0.044 | 0.000 | 11.468 |
| Oracle MDP, full-map upper bound | 1.000 [1.000, 1.000] | 15.087 [15.007, 15.167] | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| Q-learning baseline | 0.693 [0.617, 0.757] | 89.674 [86.824, 92.150] | 149.883 | 0.340 | 0.023 | 0.000 | 13.165 |

## Interpretation Notes

C3 is the main paper-aligned framework condition: PolicyABC + Prophecy Module + Imagination Cycle. The current C3 run uses TableProphecyModel as a lightweight Prophecy implementation, not as the framework contribution itself. C4 is an optional sequence-based Prophecy implementation variant and should not be interpreted as replacing the original framework. The Imagination Cycle performs depth-limited rollout over candidate branches using Prophecy predictions; it does not execute future actions or read the hidden map.

DQN_PARTIAL is a strong partial-observation baseline and may outperform C3 in some settings. ORACLE_MDP is a full-map upper bound and is not a same-information-condition baseline.

## Figures

- `figure_success_rate.png`
- `figure_steps_to_flag.png`
- `figure_semantic_gain.png`
- `figure_repeat_error_rate.png`
- `figure_learning_curve.png`
