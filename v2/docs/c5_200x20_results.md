# C5 200x20 Result Note

This note summarizes the C5 result needed for the paper draft.

## Condition Meaning

`C5` is an ablation-guided improved APASSR variant.

- It preserves the C3 closed-loop structure:
  `PolicyABC + Prophecy Module + Imagination Cycle`.
- It does not replace the vanilla paper-aligned condition `C3`.
- It removes the unconditional predicted knowledge-gain term from the Imagination score, following the A5 ablation result.
- It should be described as an improved variant, not as the core framework itself.

## v2_complex 200x20 Result

Experiment:

```text
world = v2_complex
episodes = 200
seeds = 20
step_limit = 120
total episodes = 4000
```

Command:

```powershell
$env:PYTHONPATH='src'
python -m aassr.experiment --condition C5 --world v2_complex --episodes 200 --seeds 20 --step-limit 120 --workers 6 --output-dir runs\c5_paper_v2_complex_200x20\C5
python -m aassr.analysis --input runs\c5_paper_v2_complex_200x20 --output runs\c5_paper_v2_complex_200x20\analysis
```

Summary:

| condition | success_rate | steps_to_flag | repeat_rate | error_rate | semantic_gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| C3 | 0.720 | 69.647 | 0.231 | 0.061 | 135.459 |
| C5 | 0.736 | 64.719 | 0.198 | 0.030 | 130.761 |
| DQN_PARTIAL | 0.753 | 66.931 | 0.221 | 0.059 | 133.437 |
| QLEARN | 0.617 | 70.089 | 0.353 | 0.036 | 122.134 |

Detailed C5 values:

```text
success_rate_mean       = 0.73575
success_rate_ci95_low   = 0.72225
success_rate_ci95_high  = 0.74900
steps_to_flag_mean      = 64.71901
steps_to_flag_ci95_low  = 64.02842
steps_to_flag_ci95_high = 65.30936
semantic_gain_mean      = 130.76125
repeat_rate_mean        = 0.19844
error_rate_mean         = 0.02969
prophecy_error_mean     = 0.04263
total_reward_mean       = 13.10195
```

## Safe Interpretation

C5 improves over vanilla C3 in the 200x20 `v2_complex` experiment:

- Success rate increases from `0.720` to `0.736`.
- Mean steps to flag decreases from `69.647` to `64.719`.
- Repeat rate decreases from `0.231` to `0.198`.
- Error rate decreases from `0.061` to `0.030`.

Compared with `DQN_PARTIAL`, C5 does not have the highest success rate:

- `DQN_PARTIAL` success rate is `0.753`.
- `C5` success rate is `0.736`.

However, C5 is more efficient on several behavior-quality metrics:

- C5 has lower steps to flag than DQN_PARTIAL: `64.719` vs `66.931`.
- C5 has lower repeat rate than DQN_PARTIAL: `0.198` vs `0.221`.
- C5 has lower error rate than DQN_PARTIAL: `0.030` vs `0.059`.

Therefore, the safe paper claim is:

```text
In the v2_complex 200x20 experiment, C5 improved over vanilla C3 and Q-learning.
DQN_PARTIAL achieved a slightly higher success rate, but C5 showed lower steps-to-flag,
repeat rate, and error rate. This suggests that the improved APASSR variant did not
universally dominate DQN, but produced more efficient and less repetitive behavior in
the tested knowledge-action dependency environment.
```

## Draft Replacement

Replace the older statement that only says DQN outperformed C3 with:

```text
In the larger v2_complex 200x20 experiment, vanilla C3 reached a success rate of
0.720, while DQN_PARTIAL reached 0.753. After applying the ablation-guided C5
variant, success increased to 0.736. Although DQN_PARTIAL still had a slightly
higher success rate, C5 reduced steps-to-flag, repeat rate, and error rate compared
with both C3 and DQN_PARTIAL. Thus, C5 should not be presented as universally
superior to DQN, but as an improved APASSR variant that provides more efficient
and less repetitive behavior in this controlled GridWorld setting.
```
