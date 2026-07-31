# transfer Final review

- Config: `D:\AASSR\configs\paper_transfer_final_v1.json`
- Started: 2026-07-31T11:05:03.085+00:00
- Pipeline completed: 2026-07-31T11:45:04.225+00:00
- Planned/actual rows: 762,000 / 762,000
- Completed research seeds: 30 / 30
- Missing seeds: []
- Failed/retried runs: []
- Artifact validator: PASS
- Config/resolved hash matches manifest: True / True
- Final acceptance gate hash match: True
- Frozen creativity rule: not applicable
- Pilot/Final research seed overlap: []
- Pilot/Final world seed overlap: []
- Final train/unseen world overlap: []
- Exact row duplicates: 0
- Grain duplicates: 0
- NaN/Inf: 0
- Invalid numeric: 0
- Abnormal domain values: 0
- Agent-visible private/oracle label leaks: 0
- Evaluation transitions with learning enabled: 0

## Largest episode CSV payload columns

| Column | Payload MiB | Share |
|---|---|---|
| branch_start_fingerprint | 42.85 | 16.4% |
| checkpoint_fingerprint_before | 42.85 | 16.4% |
| checkpoint_fingerprint_after | 36.62 | 14.0% |
| condition | 17.75 | 6.8% |
| phase | 17.45 | 6.7% |
| experiment | 16.71 | 6.4% |
| runtime_seconds | 15.05 | 5.8% |
| environment | 14.53 | 5.6% |
| model | 10.17 | 3.9% |
| suite | 5.81 | 2.2% |

Full per-condition/environment/phase counts and all integrity counters are in `integrity_report.json`.

## Primary seed-level results

| Condition | Environment | Phase | Metric | Seeds | Mean | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| from_scratch_contextual_policy | opaque_dependency_l6 | adaptation_curve | adaptation_auc | 30 | 0.1168 | [0.0901, 0.1446] |
| from_scratch_contextual_policy | opaque_dependency_l6 | adaptation | success | 30 | 0.0907 | [0.0683, 0.1134] |
| from_scratch_contextual_policy | opaque_dependency_l6 | adaptation | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| from_scratch_contextual_policy | opaque_dependency_l6 | adaptation | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| from_scratch_contextual_policy | opaque_dependency_l6 | evaluation_unseen_adaptation | success | 30 | 0.0633 | [0.0483, 0.0783] |
| from_scratch_contextual_policy | opaque_dependency_l6 | evaluation_unseen_adaptation | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| from_scratch_full_aassr | opaque_dependency_l6 | adaptation_curve | adaptation_auc | 30 | 0.1204 | [0.0905, 0.1526] |
| from_scratch_full_aassr | opaque_dependency_l6 | adaptation | success | 30 | 0.0880 | [0.0639, 0.1139] |
| from_scratch_full_aassr | opaque_dependency_l6 | adaptation | prediction_score | 30 | 0.9021 | [0.9007, 0.9034] |
| from_scratch_full_aassr | opaque_dependency_l6 | adaptation | holdout_gain | 30 | 0.0007 | [0.0006, 0.0008] |
| from_scratch_full_aassr | opaque_dependency_l6 | evaluation_unseen_adaptation | success | 30 | 0.0650 | [0.0483, 0.0808] |
| from_scratch_full_aassr | opaque_dependency_l6 | evaluation_unseen_adaptation | prediction_score | 30 | 0.7507 | [0.7440, 0.7571] |
| full_transfer | opaque_dependency_l6 | adaptation_curve | adaptation_auc | 30 | 0.1354 | [0.0981, 0.1802] |
| full_transfer | opaque_dependency_l6 | adaptation | success | 30 | 0.1044 | [0.0769, 0.1366] |
| full_transfer | opaque_dependency_l6 | adaptation | prediction_score | 30 | 0.9053 | [0.9020, 0.9083] |
| full_transfer | opaque_dependency_l6 | adaptation | holdout_gain | 30 | 0.0001 | [0.0001, 0.0001] |
| full_transfer | opaque_dependency_l6 | evaluation_unseen_adaptation | success | 30 | 0.0708 | [0.0517, 0.0942] |
| full_transfer | opaque_dependency_l6 | evaluation_unseen_adaptation | prediction_score | 30 | 0.7649 | [0.7526, 0.7767] |
| policy_reset_effect_retained | opaque_dependency_l6 | adaptation_curve | adaptation_auc | 30 | 0.1313 | [0.0960, 0.1720] |
| policy_reset_effect_retained | opaque_dependency_l6 | adaptation | success | 30 | 0.1014 | [0.0749, 0.1316] |
| policy_reset_effect_retained | opaque_dependency_l6 | adaptation | prediction_score | 30 | 0.9028 | [0.9014, 0.9041] |
| policy_reset_effect_retained | opaque_dependency_l6 | adaptation | holdout_gain | 30 | 0.0006 | [0.0005, 0.0007] |
| policy_reset_effect_retained | opaque_dependency_l6 | evaluation_unseen_adaptation | success | 30 | 0.0708 | [0.0517, 0.0933] |
| policy_reset_effect_retained | opaque_dependency_l6 | evaluation_unseen_adaptation | prediction_score | 30 | 0.7586 | [0.7529, 0.7643] |
| policy_reset_prophecy_retained | opaque_dependency_l6 | adaptation_curve | adaptation_auc | 30 | 0.1392 | [0.1118, 0.1697] |
| policy_reset_prophecy_retained | opaque_dependency_l6 | adaptation | success | 30 | 0.1074 | [0.0882, 0.1277] |
| policy_reset_prophecy_retained | opaque_dependency_l6 | adaptation | prediction_score | 30 | 0.9018 | [0.9004, 0.9032] |
| policy_reset_prophecy_retained | opaque_dependency_l6 | adaptation | holdout_gain | 30 | 0.0007 | [0.0006, 0.0008] |
| policy_reset_prophecy_retained | opaque_dependency_l6 | evaluation_unseen_adaptation | success | 30 | 0.0742 | [0.0600, 0.0908] |
| policy_reset_prophecy_retained | opaque_dependency_l6 | evaluation_unseen_adaptation | prediction_score | 30 | 0.7501 | [0.7436, 0.7568] |
| transfer_pretraining | opaque_dependency_l6 | training | success | 30 | 0.3548 | [0.3285, 0.3798] |
| transfer_pretraining | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| transfer_pretraining | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |

## Holm-significant paired comparisons

| Phase | Metric | Comparison | Paired seeds | Difference | 95% CI | Holm p |
|---|---|---|---|---|---|---|
| adaptation | prediction_score | full_transfer vs from_scratch_contextual_policy | 30 | 0.9053 | [0.9021, 0.9083] | 0.0002 |
| adaptation | holdout_score | full_transfer vs from_scratch_contextual_policy | 30 | 0.2470 | [0.2464, 0.2475] | 0.0002 |
| adaptation | holdout_gain | full_transfer vs from_scratch_contextual_policy | 30 | 0.0001 | [0.0001, 0.0001] | 0.0002 |
| adaptation | imagined_nodes | full_transfer vs from_scratch_contextual_policy | 30 | 98.2843 | [94.7616, 102.0556] | 0.0002 |
| adaptation | imagination_depth | full_transfer vs from_scratch_contextual_policy | 30 | 4.8595 | [4.7945, 4.9211] | 0.0002 |
| adaptation | runtime_seconds | full_transfer vs from_scratch_contextual_policy | 30 | 0.0033 | [0.0032, 0.0034] | 0.0002 |
| adaptation | imagined_transitions | full_transfer vs from_scratch_contextual_policy | 30 | 98.2843 | [94.7192, 102.0563] | 0.0002 |
| adaptation | holdout_score | full_transfer vs from_scratch_full_aassr | 30 | 0.0262 | [0.0253, 0.0270] | 0.0002 |
| adaptation | holdout_gain | full_transfer vs from_scratch_full_aassr | 30 | -0.0006 | [-0.0008, -0.0005] | 0.0002 |
| adaptation | runtime_seconds | full_transfer vs from_scratch_full_aassr | 30 | 0.0004 | [0.0002, 0.0005] | 0.0002 |
| adaptation | holdout_score | full_transfer vs policy_reset_effect_retained | 30 | 0.0287 | [0.0277, 0.0299] | 0.0002 |
| adaptation | holdout_gain | full_transfer vs policy_reset_effect_retained | 30 | -0.0005 | [-0.0006, -0.0004] | 0.0002 |
| adaptation | runtime_seconds | full_transfer vs policy_reset_effect_retained | 30 | 0.0004 | [0.0004, 0.0005] | 0.0002 |
| adaptation | holdout_score | full_transfer vs policy_reset_prophecy_retained | 30 | 0.0262 | [0.0254, 0.0269] | 0.0002 |
| adaptation | holdout_gain | full_transfer vs policy_reset_prophecy_retained | 30 | -0.0006 | [-0.0007, -0.0005] | 0.0002 |
| adaptation | imagined_nodes | full_transfer vs policy_reset_prophecy_retained | 30 | -3.2538 | [-5.4752, -1.1047] | 0.0187 |
| adaptation | runtime_seconds | full_transfer vs policy_reset_prophecy_retained | 30 | 0.0004 | [0.0003, 0.0004] | 0.0002 |
| adaptation | imagined_transitions | full_transfer vs policy_reset_prophecy_retained | 30 | -3.2538 | [-5.4358, -1.2135] | 0.0184 |
| adaptation_curve | unseen_prediction_calibration_error | full_transfer vs from_scratch_contextual_policy | 30 | 0.6310 | [0.5975, 0.6615] | 0.0002 |
| evaluation_unseen_adaptation | prediction_score | full_transfer vs from_scratch_contextual_policy | 30 | 0.7649 | [0.7526, 0.7767] | 0.0002 |
| evaluation_unseen_adaptation | holdout_score | full_transfer vs from_scratch_contextual_policy | 30 | 0.9874 | [0.9837, 0.9910] | 0.0002 |
| evaluation_unseen_adaptation | imagined_nodes | full_transfer vs from_scratch_contextual_policy | 30 | 68.6742 | [53.4867, 83.8942] | 0.0002 |
| evaluation_unseen_adaptation | imagination_depth | full_transfer vs from_scratch_contextual_policy | 30 | 1.8592 | [1.4733, 2.2267] | 0.0002 |
| evaluation_unseen_adaptation | runtime_seconds | full_transfer vs from_scratch_contextual_policy | 30 | 0.0030 | [0.0027, 0.0034] | 0.0002 |
| evaluation_unseen_adaptation | imagined_transitions | full_transfer vs from_scratch_contextual_policy | 30 | 68.6742 | [53.5025, 83.8275] | 0.0002 |
| evaluation_unseen_adaptation | holdout_score | full_transfer vs from_scratch_full_aassr | 30 | 0.4258 | [0.4189, 0.4328] | 0.0002 |
| evaluation_unseen_adaptation | runtime_seconds | full_transfer vs from_scratch_full_aassr | 30 | 0.0015 | [0.0012, 0.0019] | 0.0002 |
| evaluation_unseen_adaptation | holdout_score | full_transfer vs policy_reset_effect_retained | 30 | 0.4361 | [0.4304, 0.4422] | 0.0002 |
| evaluation_unseen_adaptation | runtime_seconds | full_transfer vs policy_reset_effect_retained | 30 | 0.0014 | [0.0014, 0.0015] | 0.0002 |
| evaluation_unseen_adaptation | holdout_score | full_transfer vs policy_reset_prophecy_retained | 30 | 0.4252 | [0.4185, 0.4320] | 0.0002 |
| evaluation_unseen_adaptation | imagined_nodes | full_transfer vs policy_reset_prophecy_retained | 30 | -4.4850 | [-7.7342, -1.5692] | 0.0187 |
| evaluation_unseen_adaptation | runtime_seconds | full_transfer vs policy_reset_prophecy_retained | 30 | 0.0014 | [0.0012, 0.0015] | 0.0002 |
| evaluation_unseen_adaptation | imagined_transitions | full_transfer vs policy_reset_prophecy_retained | 30 | -4.4850 | [-7.7025, -1.6067] | 0.0228 |
