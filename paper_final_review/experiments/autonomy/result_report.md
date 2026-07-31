# autonomy Final review

- Config: `D:\AASSR\configs\paper_autonomy_final_v1.json`
- Started: 2026-07-31T09:46:27.409+00:00
- Pipeline completed: 2026-07-31T10:28:18.029+00:00
- Planned/actual rows: 1,512,000 / 1,512,000
- Completed research seeds: 30 / 30
- Missing seeds: []
- Failed/retried runs: [{'error_type': 'PermissionError', 'failed_at_utc': '2026-07-31T09:48:55.319+00:00', 'reason': "[WinError 5] 액세스가 거부되었습니다: 'paper_results\\\\paper-autonomy-final-v1\\\\raw\\\\suite_runs\\\\autonomy\\\\progress.json.tmp' -> 'paper_results\\\\paper-autonomy-final-v1\\\\raw\\\\suite_runs\\\\autonomy\\\\progress.json'", 'suite': 'autonomy'}]
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
| experiment | 33.16 | 12.9% |
| runtime_seconds | 30.15 | 11.7% |
| environment | 28.84 | 11.2% |
| condition | 17.92 | 7.0% |
| model | 16.27 | 6.3% |
| phase | 14.66 | 5.7% |
| checkpoint_fingerprint_after | 13.63 | 5.3% |
| checkpoint_fingerprint_before | 13.63 | 5.3% |
| suite | 11.54 | 4.5% |
| action_family | 8.65 | 3.4% |

Full per-condition/environment/phase counts and all integrity counters are in `integrity_report.json`.

## Primary seed-level results

| Condition | Environment | Phase | Metric | Seeds | Mean | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| contextual_policy | opaque_dependency_l4 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l4 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l4 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l4 | training | success | 30 | 0.6543 | [0.6501, 0.6585] |
| contextual_policy | opaque_dependency_l4 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l4 | training | learning_auc | 30 | 0.6544 | [0.6502, 0.6584] |
| contextual_policy | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | training | success | 30 | 0.4351 | [0.4140, 0.4556] |
| contextual_policy | opaque_dependency_l6 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | training | learning_auc | 30 | 0.4352 | [0.4146, 0.4556] |
| contextual_policy | opaque_dependency_l8 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l8 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l8 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l8 | training | success | 30 | 0.1673 | [0.1412, 0.1952] |
| contextual_policy | opaque_dependency_l8 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l8 | training | learning_auc | 30 | 0.1674 | [0.1417, 0.1949] |
| dqn | opaque_dependency_l4 | evaluation_seen | success | 30 | 0.1000 | [0.0500, 0.1500] |
| dqn | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 0.0333 | [0.0125, 0.0583] |
| dqn | opaque_dependency_l4 | training | success | 30 | 0.1603 | [0.1370, 0.1860] |
| dqn | opaque_dependency_l4 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l4 | training | learning_auc | 30 | 0.1603 | [0.1370, 0.1851] |
| dqn | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l6 | training | success | 30 | 0.0361 | [0.0253, 0.0478] |
| dqn | opaque_dependency_l6 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l6 | training | learning_auc | 30 | 0.0361 | [0.0257, 0.0480] |
| dqn | opaque_dependency_l8 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l8 | training | success | 30 | 0.0028 | [0.0023, 0.0034] |
| dqn | opaque_dependency_l8 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| dqn | opaque_dependency_l8 | training | learning_auc | 30 | 0.0028 | [0.0023, 0.0034] |
| full_aassr | opaque_dependency_l4 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l4 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l4 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l4 | training | success | 30 | 0.6149 | [0.5958, 0.6328] |
| full_aassr | opaque_dependency_l4 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9901] |
| full_aassr | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l4 | training | learning_auc | 30 | 0.6150 | [0.5962, 0.6330] |
| full_aassr | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | training | success | 30 | 0.2737 | [0.2494, 0.2975] |
| full_aassr | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_aassr | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | training | learning_auc | 30 | 0.2737 | [0.2497, 0.2983] |
| full_aassr | opaque_dependency_l8 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l8 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l8 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l8 | training | success | 30 | 0.0343 | [0.0230, 0.0465] |
| full_aassr | opaque_dependency_l8 | training | prediction_score | 30 | 0.9900 | [0.9899, 0.9901] |
| full_aassr | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l8 | training | learning_auc | 30 | 0.0343 | [0.0228, 0.0463] |
| oracle_upper_bound | opaque_dependency_l4 | evaluation_seen | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l4 | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l4 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| oracle_upper_bound | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| oracle_upper_bound | opaque_dependency_l4 | training | learning_auc | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l6 | evaluation_seen | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l6 | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l6 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| oracle_upper_bound | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| oracle_upper_bound | opaque_dependency_l6 | training | learning_auc | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l8 | evaluation_seen | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l8 | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| oracle_upper_bound | opaque_dependency_l8 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| oracle_upper_bound | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| oracle_upper_bound | opaque_dependency_l8 | training | learning_auc | 30 | 1.0000 | [1.0000, 1.0000] |
| prophecy_no_imagination | opaque_dependency_l4 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l4 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l4 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l4 | training | success | 30 | 0.6541 | [0.6502, 0.6581] |
| prophecy_no_imagination | opaque_dependency_l4 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9901] |
| prophecy_no_imagination | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l4 | training | learning_auc | 30 | 0.6542 | [0.6502, 0.6582] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | training | success | 30 | 0.4395 | [0.4194, 0.4588] |
| prophecy_no_imagination | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| prophecy_no_imagination | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | training | learning_auc | 30 | 0.4396 | [0.4191, 0.4593] |
| prophecy_no_imagination | opaque_dependency_l8 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l8 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l8 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l8 | training | success | 30 | 0.1594 | [0.1338, 0.1855] |
| prophecy_no_imagination | opaque_dependency_l8 | training | prediction_score | 30 | 0.9900 | [0.9899, 0.9901] |
| prophecy_no_imagination | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l8 | training | learning_auc | 30 | 0.1595 | [0.1338, 0.1873] |
| q_learning | opaque_dependency_l4 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l4 | training | success | 30 | 0.1267 | [0.1055, 0.1488] |
| q_learning | opaque_dependency_l4 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l4 | training | learning_auc | 30 | 0.1267 | [0.1060, 0.1485] |
| q_learning | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l6 | training | success | 30 | 0.0114 | [0.0072, 0.0172] |
| q_learning | opaque_dependency_l6 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l6 | training | learning_auc | 30 | 0.0114 | [0.0071, 0.0174] |
| q_learning | opaque_dependency_l8 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l8 | training | success | 30 | 0.0030 | [0.0009, 0.0071] |
| q_learning | opaque_dependency_l8 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| q_learning | opaque_dependency_l8 | training | learning_auc | 30 | 0.0030 | [0.0009, 0.0071] |
| random | opaque_dependency_l4 | evaluation_seen | success | 30 | 0.0672 | [0.0602, 0.0747] |
| random | opaque_dependency_l4 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l4 | evaluation_unseen_zero_shot | success | 30 | 0.0602 | [0.0545, 0.0660] |
| random | opaque_dependency_l4 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l4 | training | success | 30 | 0.0626 | [0.0605, 0.0647] |
| random | opaque_dependency_l4 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l4 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l4 | training | learning_auc | 30 | 0.0626 | [0.0605, 0.0646] |
| random | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0152 | [0.0123, 0.0180] |
| random | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0145 | [0.0113, 0.0177] |
| random | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l6 | training | success | 30 | 0.0155 | [0.0148, 0.0163] |
| random | opaque_dependency_l6 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l6 | training | learning_auc | 30 | 0.0155 | [0.0148, 0.0162] |
| random | opaque_dependency_l8 | evaluation_seen | success | 30 | 0.0038 | [0.0025, 0.0052] |
| random | opaque_dependency_l8 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l8 | evaluation_unseen_zero_shot | success | 30 | 0.0043 | [0.0028, 0.0060] |
| random | opaque_dependency_l8 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l8 | training | success | 30 | 0.0036 | [0.0032, 0.0041] |
| random | opaque_dependency_l8 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l8 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| random | opaque_dependency_l8 | training | learning_auc | 30 | 0.0036 | [0.0031, 0.0041] |

## Holm-significant paired comparisons

| Phase | Metric | Comparison | Paired seeds | Difference | 95% CI | Holm p |
|---|---|---|---|---|---|---|
| evaluation_seen | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_seen | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0004 | [0.0003, 0.0005] | 0.0002 |
| evaluation_seen | success | full_aassr vs dqn | 30 | -0.1000 | [-0.1500, -0.0500] | 0.0034 |
| evaluation_seen | reward | full_aassr vs dqn | 30 | -0.1000 | [-0.1500, -0.0500] | 0.0038 |
| evaluation_seen | actual_return | full_aassr vs dqn | 30 | -0.1000 | [-0.1500, -0.0500] | 0.0038 |
| evaluation_seen | runtime_seconds | full_aassr vs dqn | 30 | 0.0009 | [0.0008, 0.0010] | 0.0002 |
| evaluation_seen | runtime_seconds | full_aassr vs q_learning | 30 | 0.0023 | [0.0022, 0.0023] | 0.0002 |
| evaluation_seen | success | full_aassr vs random | 30 | -0.0672 | [-0.0748, -0.0602] | 0.0002 |
| evaluation_seen | reward | full_aassr vs random | 30 | -0.0672 | [-0.0747, -0.0603] | 0.0002 |
| evaluation_seen | holdout_score | full_aassr vs random | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_seen | actual_return | full_aassr vs random | 30 | -0.0672 | [-0.0747, -0.0600] | 0.0002 |
| evaluation_seen | runtime_seconds | full_aassr vs random | 30 | 0.0004 | [0.0003, 0.0005] | 0.0002 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0003 | [0.0002, 0.0004] | 0.0002 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs dqn | 30 | 0.0008 | [0.0007, 0.0009] | 0.0002 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs q_learning | 30 | 0.0022 | [0.0021, 0.0023] | 0.0002 |
| evaluation_unseen_zero_shot | success | full_aassr vs random | 30 | -0.0602 | [-0.0657, -0.0545] | 0.0002 |
| evaluation_unseen_zero_shot | reward | full_aassr vs random | 30 | -0.0602 | [-0.0657, -0.0545] | 0.0002 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs random | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_unseen_zero_shot | actual_return | full_aassr vs random | 30 | -0.0602 | [-0.0658, -0.0547] | 0.0002 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs random | 30 | 0.0002 | [0.0001, 0.0003] | 0.0004 |
| training | success | full_aassr vs contextual_policy | 30 | -0.0393 | [-0.0571, -0.0231] | 0.0002 |
| training | reward | full_aassr vs contextual_policy | 30 | -0.0393 | [-0.0564, -0.0228] | 0.0002 |
| training | prediction_score | full_aassr vs contextual_policy | 30 | 0.9900 | [0.9898, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs contextual_policy | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs contextual_policy | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs contextual_policy | 30 | 42.5521 | [42.4524, 42.6535] | 0.0002 |
| training | imagination_depth | full_aassr vs contextual_policy | 30 | 3.5152 | [3.5091, 3.5212] | 0.0002 |
| training | actual_return | full_aassr vs contextual_policy | 30 | -0.0393 | [-0.0565, -0.0227] | 0.0002 |
| training | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0018 | [0.0018, 0.0019] | 0.0002 |
| training | imagined_transitions | full_aassr vs contextual_policy | 30 | 42.5521 | [42.4533, 42.6514] | 0.0002 |
| training | learning_auc | full_aassr vs contextual_policy | 30 | -0.0394 | [-0.0564, -0.0232] | 0.0002 |
| training | final_tail_success | full_aassr vs contextual_policy | 30 | -0.0603 | [-0.0852, -0.0368] | 0.0003 |
| training | success | full_aassr vs dqn | 30 | 0.4546 | [0.4253, 0.4838] | 0.0002 |
| training | reward | full_aassr vs dqn | 30 | 0.4546 | [0.4239, 0.4851] | 0.0002 |
| training | prediction_score | full_aassr vs dqn | 30 | 0.9900 | [0.9898, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs dqn | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs dqn | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs dqn | 30 | 42.5521 | [42.4545, 42.6475] | 0.0002 |
| training | imagination_depth | full_aassr vs dqn | 30 | 3.5152 | [3.5092, 3.5210] | 0.0002 |
| training | actual_return | full_aassr vs dqn | 30 | 0.4546 | [0.4248, 0.4834] | 0.0002 |
| training | runtime_seconds | full_aassr vs dqn | 30 | -0.0030 | [-0.0032, -0.0029] | 0.0002 |
| training | imagined_transitions | full_aassr vs dqn | 30 | 42.5521 | [42.4579, 42.6523] | 0.0002 |
| training | learning_auc | full_aassr vs dqn | 30 | 0.4547 | [0.4238, 0.4852] | 0.0002 |
| training | final_tail_success | full_aassr vs dqn | 30 | 0.5785 | [0.5193, 0.6347] | 0.0002 |
| training | success | full_aassr vs prophecy_no_imagination | 30 | -0.0392 | [-0.0566, -0.0231] | 0.0002 |
| training | reward | full_aassr vs prophecy_no_imagination | 30 | -0.0392 | [-0.0558, -0.0224] | 0.0002 |
| training | imagined_nodes | full_aassr vs prophecy_no_imagination | 30 | 42.5521 | [42.4540, 42.6529] | 0.0002 |
| training | imagination_depth | full_aassr vs prophecy_no_imagination | 30 | 3.5152 | [3.5092, 3.5211] | 0.0002 |
| training | actual_return | full_aassr vs prophecy_no_imagination | 30 | -0.0392 | [-0.0562, -0.0235] | 0.0002 |
| training | runtime_seconds | full_aassr vs prophecy_no_imagination | 30 | 0.0013 | [0.0013, 0.0013] | 0.0002 |
| training | imagined_transitions | full_aassr vs prophecy_no_imagination | 30 | 42.5521 | [42.4504, 42.6511] | 0.0002 |
| training | learning_auc | full_aassr vs prophecy_no_imagination | 30 | -0.0392 | [-0.0566, -0.0227] | 0.0002 |
| training | final_tail_success | full_aassr vs prophecy_no_imagination | 30 | -0.0603 | [-0.0848, -0.0367] | 0.0003 |
| training | success | full_aassr vs q_learning | 30 | 0.4882 | [0.4641, 0.5145] | 0.0002 |
| training | reward | full_aassr vs q_learning | 30 | 0.4882 | [0.4638, 0.5142] | 0.0002 |
| training | prediction_score | full_aassr vs q_learning | 30 | 0.9900 | [0.9898, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs q_learning | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs q_learning | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs q_learning | 30 | 42.5521 | [42.4543, 42.6495] | 0.0002 |
| training | imagination_depth | full_aassr vs q_learning | 30 | 3.5152 | [3.5091, 3.5211] | 0.0002 |
| training | actual_return | full_aassr vs q_learning | 30 | 0.4882 | [0.4646, 0.5132] | 0.0002 |
| training | runtime_seconds | full_aassr vs q_learning | 30 | 0.0020 | [0.0019, 0.0020] | 0.0002 |
| training | imagined_transitions | full_aassr vs q_learning | 30 | 42.5521 | [42.4537, 42.6526] | 0.0002 |
| training | learning_auc | full_aassr vs q_learning | 30 | 0.4883 | [0.4640, 0.5134] | 0.0002 |
| training | final_tail_success | full_aassr vs q_learning | 30 | 0.6713 | [0.6288, 0.7138] | 0.0002 |
| training | success | full_aassr vs random | 30 | 0.5523 | [0.5333, 0.5708] | 0.0002 |
| training | reward | full_aassr vs random | 30 | 0.5523 | [0.5331, 0.5698] | 0.0002 |
| training | prediction_score | full_aassr vs random | 30 | 0.9900 | [0.9898, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs random | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs random | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs random | 30 | 42.5521 | [42.4538, 42.6475] | 0.0002 |
| training | imagination_depth | full_aassr vs random | 30 | 3.5152 | [3.5091, 3.5212] | 0.0002 |
| training | actual_return | full_aassr vs random | 30 | 0.5523 | [0.5333, 0.5705] | 0.0002 |
| training | runtime_seconds | full_aassr vs random | 30 | 0.0019 | [0.0019, 0.0019] | 0.0002 |
| training | imagined_transitions | full_aassr vs random | 30 | 42.5521 | [42.4524, 42.6521] | 0.0002 |
| training | learning_auc | full_aassr vs random | 30 | 0.5524 | [0.5335, 0.5705] | 0.0002 |
| training | final_tail_success | full_aassr vs random | 30 | 0.7868 | [0.7592, 0.8127] | 0.0002 |
| evaluation_seen | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_seen | runtime_seconds | full_aassr vs dqn | 30 | 0.0013 | [0.0011, 0.0014] | 0.0002 |
| evaluation_seen | runtime_seconds | full_aassr vs q_learning | 30 | 0.0033 | [0.0032, 0.0035] | 0.0002 |
| evaluation_seen | success | full_aassr vs random | 30 | -0.0152 | [-0.0182, -0.0123] | 0.0002 |
| evaluation_seen | reward | full_aassr vs random | 30 | -0.0152 | [-0.0182, -0.0125] | 0.0002 |
| evaluation_seen | holdout_score | full_aassr vs random | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_seen | actual_return | full_aassr vs random | 30 | -0.0152 | [-0.0182, -0.0123] | 0.0002 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0003 | [0.0001, 0.0005] | 0.0038 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs dqn | 30 | 0.0014 | [0.0012, 0.0015] | 0.0002 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs q_learning | 30 | 0.0034 | [0.0033, 0.0036] | 0.0002 |
| evaluation_unseen_zero_shot | success | full_aassr vs random | 30 | -0.0145 | [-0.0177, -0.0113] | 0.0002 |
| evaluation_unseen_zero_shot | reward | full_aassr vs random | 30 | -0.0145 | [-0.0175, -0.0113] | 0.0002 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs random | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_unseen_zero_shot | actual_return | full_aassr vs random | 30 | -0.0145 | [-0.0177, -0.0113] | 0.0002 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs random | 30 | 0.0003 | [0.0002, 0.0005] | 0.0024 |
| training | success | full_aassr vs contextual_policy | 30 | -0.1615 | [-0.1906, -0.1326] | 0.0002 |
| training | reward | full_aassr vs contextual_policy | 30 | -0.1615 | [-0.1893, -0.1330] | 0.0002 |
| training | prediction_score | full_aassr vs contextual_policy | 30 | 0.9900 | [0.9898, 0.9902] | 0.0002 |
| training | holdout_score | full_aassr vs contextual_policy | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs contextual_policy | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs contextual_policy | 30 | 186.4483 | [185.9614, 186.9512] | 0.0002 |
| training | imagination_depth | full_aassr vs contextual_policy | 30 | 5.4436 | [5.4349, 5.4525] | 0.0002 |
| training | actual_return | full_aassr vs contextual_policy | 30 | -0.1615 | [-0.1913, -0.1319] | 0.0002 |
| training | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0066 | [0.0065, 0.0067] | 0.0002 |
| training | imagined_transitions | full_aassr vs contextual_policy | 30 | 186.4483 | [185.9771, 186.9379] | 0.0002 |
| training | learning_auc | full_aassr vs contextual_policy | 30 | -0.1615 | [-0.1914, -0.1314] | 0.0002 |
| training | final_tail_success | full_aassr vs contextual_policy | 30 | -0.2975 | [-0.3397, -0.2568] | 0.0002 |
| training | success | full_aassr vs dqn | 30 | 0.2376 | [0.2114, 0.2633] | 0.0002 |
| training | reward | full_aassr vs dqn | 30 | 0.2376 | [0.2112, 0.2634] | 0.0002 |
| training | prediction_score | full_aassr vs dqn | 30 | 0.9900 | [0.9898, 0.9902] | 0.0002 |
| training | holdout_score | full_aassr vs dqn | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs dqn | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs dqn | 30 | 186.4483 | [185.9718, 186.9253] | 0.0002 |
| training | imagination_depth | full_aassr vs dqn | 30 | 5.4436 | [5.4352, 5.4522] | 0.0002 |
| training | actual_return | full_aassr vs dqn | 30 | 0.2376 | [0.2111, 0.2644] | 0.0002 |
| training | runtime_seconds | full_aassr vs dqn | 30 | -0.0008 | [-0.0011, -0.0006] | 0.0002 |
| training | imagined_transitions | full_aassr vs dqn | 30 | 186.4483 | [185.9906, 186.9348] | 0.0002 |
| training | learning_auc | full_aassr vs dqn | 30 | 0.2376 | [0.2109, 0.2639] | 0.0002 |
| training | final_tail_success | full_aassr vs dqn | 30 | 0.3560 | [0.3085, 0.4008] | 0.0002 |
| training | success | full_aassr vs prophecy_no_imagination | 30 | -0.1659 | [-0.1965, -0.1355] | 0.0002 |
| training | reward | full_aassr vs prophecy_no_imagination | 30 | -0.1659 | [-0.1968, -0.1351] | 0.0002 |
| training | imagined_nodes | full_aassr vs prophecy_no_imagination | 30 | 186.4483 | [185.9810, 186.9214] | 0.0002 |
| training | imagination_depth | full_aassr vs prophecy_no_imagination | 30 | 5.4436 | [5.4347, 5.4520] | 0.0002 |
| training | actual_return | full_aassr vs prophecy_no_imagination | 30 | -0.1659 | [-0.1953, -0.1350] | 0.0002 |
| training | runtime_seconds | full_aassr vs prophecy_no_imagination | 30 | 0.0057 | [0.0056, 0.0058] | 0.0002 |
| training | imagined_transitions | full_aassr vs prophecy_no_imagination | 30 | 186.4483 | [185.9676, 186.9380] | 0.0002 |
| training | learning_auc | full_aassr vs prophecy_no_imagination | 30 | -0.1659 | [-0.1957, -0.1355] | 0.0002 |
| training | final_tail_success | full_aassr vs prophecy_no_imagination | 30 | -0.3038 | [-0.3470, -0.2605] | 0.0002 |
| training | success | full_aassr vs q_learning | 30 | 0.2622 | [0.2391, 0.2851] | 0.0002 |
| training | reward | full_aassr vs q_learning | 30 | 0.2622 | [0.2388, 0.2856] | 0.0002 |
| training | prediction_score | full_aassr vs q_learning | 30 | 0.9900 | [0.9898, 0.9902] | 0.0002 |
| training | holdout_score | full_aassr vs q_learning | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs q_learning | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs q_learning | 30 | 186.4483 | [185.9880, 186.9351] | 0.0002 |
| training | imagination_depth | full_aassr vs q_learning | 30 | 5.4436 | [5.4346, 5.4522] | 0.0002 |
| training | actual_return | full_aassr vs q_learning | 30 | 0.2622 | [0.2382, 0.2859] | 0.0002 |
| training | runtime_seconds | full_aassr vs q_learning | 30 | 0.0068 | [0.0067, 0.0069] | 0.0002 |
| training | imagined_transitions | full_aassr vs q_learning | 30 | 186.4483 | [185.9772, 186.9226] | 0.0002 |
| training | learning_auc | full_aassr vs q_learning | 30 | 0.2622 | [0.2387, 0.2856] | 0.0002 |
| training | final_tail_success | full_aassr vs q_learning | 30 | 0.4045 | [0.3660, 0.4428] | 0.0002 |
| training | success | full_aassr vs random | 30 | 0.2581 | [0.2338, 0.2821] | 0.0002 |
| training | reward | full_aassr vs random | 30 | 0.2581 | [0.2344, 0.2827] | 0.0002 |
| training | prediction_score | full_aassr vs random | 30 | 0.9900 | [0.9898, 0.9902] | 0.0002 |
| training | holdout_score | full_aassr vs random | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs random | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs random | 30 | 186.4483 | [185.9812, 186.9304] | 0.0002 |
| training | imagination_depth | full_aassr vs random | 30 | 5.4436 | [5.4348, 5.4526] | 0.0002 |
| training | actual_return | full_aassr vs random | 30 | 0.2581 | [0.2344, 0.2828] | 0.0002 |
| training | runtime_seconds | full_aassr vs random | 30 | 0.0067 | [0.0066, 0.0069] | 0.0002 |
| training | imagined_transitions | full_aassr vs random | 30 | 186.4483 | [185.9500, 186.9382] | 0.0002 |
| training | learning_auc | full_aassr vs random | 30 | 0.2582 | [0.2337, 0.2821] | 0.0002 |
| training | final_tail_success | full_aassr vs random | 30 | 0.3968 | [0.3593, 0.4347] | 0.0002 |
| evaluation_seen | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_seen | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0003 | [0.0001, 0.0005] | 0.0196 |
| evaluation_seen | runtime_seconds | full_aassr vs dqn | 30 | 0.0022 | [0.0020, 0.0024] | 0.0002 |
| evaluation_seen | runtime_seconds | full_aassr vs q_learning | 30 | 0.0050 | [0.0047, 0.0052] | 0.0002 |
| evaluation_seen | success | full_aassr vs random | 30 | -0.0038 | [-0.0052, -0.0025] | 0.0002 |
| evaluation_seen | reward | full_aassr vs random | 30 | -0.0038 | [-0.0053, -0.0025] | 0.0005 |
| evaluation_seen | holdout_score | full_aassr vs random | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_seen | actual_return | full_aassr vs random | 30 | -0.0038 | [-0.0052, -0.0025] | 0.0002 |
| evaluation_seen | runtime_seconds | full_aassr vs random | 30 | 0.0004 | [0.0001, 0.0006] | 0.0136 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs dqn | 30 | 0.0022 | [0.0019, 0.0024] | 0.0002 |
| evaluation_unseen_zero_shot | runtime_seconds | full_aassr vs q_learning | 30 | 0.0049 | [0.0046, 0.0051] | 0.0002 |
| evaluation_unseen_zero_shot | success | full_aassr vs random | 30 | -0.0043 | [-0.0060, -0.0028] | 0.0002 |
| evaluation_unseen_zero_shot | reward | full_aassr vs random | 30 | -0.0043 | [-0.0060, -0.0028] | 0.0002 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs random | 30 | 1.0000 | [1.0000, 1.0000] | 0.0001 |
| evaluation_unseen_zero_shot | actual_return | full_aassr vs random | 30 | -0.0043 | [-0.0060, -0.0028] | 0.0002 |
| training | success | full_aassr vs contextual_policy | 30 | -0.1330 | [-0.1605, -0.1066] | 0.0002 |
| training | reward | full_aassr vs contextual_policy | 30 | -0.1330 | [-0.1595, -0.1064] | 0.0002 |
| training | prediction_score | full_aassr vs contextual_policy | 30 | 0.9900 | [0.9899, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs contextual_policy | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs contextual_policy | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs contextual_policy | 30 | 525.2874 | [524.3373, 526.2215] | 0.0002 |
| training | imagination_depth | full_aassr vs contextual_policy | 30 | 7.4038 | [7.3959, 7.4124] | 0.0002 |
| training | actual_return | full_aassr vs contextual_policy | 30 | -0.1330 | [-0.1603, -0.1066] | 0.0002 |
| training | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0177 | [0.0173, 0.0181] | 0.0002 |
| training | imagined_transitions | full_aassr vs contextual_policy | 30 | 525.2874 | [524.3126, 526.2808] | 0.0002 |
| training | learning_auc | full_aassr vs contextual_policy | 30 | -0.1330 | [-0.1599, -0.1065] | 0.0002 |
| training | final_tail_success | full_aassr vs contextual_policy | 30 | -0.2442 | [-0.2943, -0.1940] | 0.0002 |
| training | success | full_aassr vs dqn | 30 | 0.0315 | [0.0191, 0.0437] | 0.0003 |
| training | reward | full_aassr vs dqn | 30 | 0.0315 | [0.0195, 0.0441] | 0.0002 |
| training | prediction_score | full_aassr vs dqn | 30 | 0.9900 | [0.9899, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs dqn | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs dqn | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs dqn | 30 | 525.2874 | [524.3534, 526.2473] | 0.0002 |
| training | imagination_depth | full_aassr vs dqn | 30 | 7.4038 | [7.3959, 7.4122] | 0.0002 |
| training | actual_return | full_aassr vs dqn | 30 | 0.0315 | [0.0194, 0.0441] | 0.0002 |
| training | runtime_seconds | full_aassr vs dqn | 30 | 0.0078 | [0.0074, 0.0082] | 0.0002 |
| training | imagined_transitions | full_aassr vs dqn | 30 | 525.2874 | [524.3315, 526.2339] | 0.0002 |
| training | learning_auc | full_aassr vs dqn | 30 | 0.0315 | [0.0195, 0.0436] | 0.0002 |
| training | final_tail_success | full_aassr vs dqn | 30 | 0.0483 | [0.0300, 0.0670] | 0.0003 |
| training | success | full_aassr vs prophecy_no_imagination | 30 | -0.1251 | [-0.1505, -0.0992] | 0.0002 |
| training | reward | full_aassr vs prophecy_no_imagination | 30 | -0.1251 | [-0.1506, -0.0987] | 0.0002 |
| training | imagined_nodes | full_aassr vs prophecy_no_imagination | 30 | 525.2874 | [524.3384, 526.2566] | 0.0002 |
| training | imagination_depth | full_aassr vs prophecy_no_imagination | 30 | 7.4038 | [7.3957, 7.4121] | 0.0002 |
| training | actual_return | full_aassr vs prophecy_no_imagination | 30 | -0.1251 | [-0.1508, -0.0995] | 0.0002 |
| training | runtime_seconds | full_aassr vs prophecy_no_imagination | 30 | 0.0163 | [0.0159, 0.0167] | 0.0002 |
| training | imagined_transitions | full_aassr vs prophecy_no_imagination | 30 | 525.2874 | [524.3093, 526.2343] | 0.0002 |
| training | learning_auc | full_aassr vs prophecy_no_imagination | 30 | -0.1251 | [-0.1509, -0.1000] | 0.0002 |
| training | final_tail_success | full_aassr vs prophecy_no_imagination | 30 | -0.2275 | [-0.2747, -0.1828] | 0.0002 |
| training | success | full_aassr vs q_learning | 30 | 0.0313 | [0.0199, 0.0432] | 0.0002 |
| training | reward | full_aassr vs q_learning | 30 | 0.0313 | [0.0202, 0.0427] | 0.0002 |
| training | prediction_score | full_aassr vs q_learning | 30 | 0.9900 | [0.9899, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs q_learning | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs q_learning | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs q_learning | 30 | 525.2874 | [524.3317, 526.2494] | 0.0002 |
| training | imagination_depth | full_aassr vs q_learning | 30 | 7.4038 | [7.3958, 7.4123] | 0.0002 |
| training | actual_return | full_aassr vs q_learning | 30 | 0.0313 | [0.0197, 0.0433] | 0.0002 |
| training | runtime_seconds | full_aassr vs q_learning | 30 | 0.0180 | [0.0176, 0.0184] | 0.0002 |
| training | imagined_transitions | full_aassr vs q_learning | 30 | 525.2874 | [524.3336, 526.2451] | 0.0002 |
| training | learning_auc | full_aassr vs q_learning | 30 | 0.0313 | [0.0198, 0.0432] | 0.0002 |
| training | final_tail_success | full_aassr vs q_learning | 30 | 0.0458 | [0.0277, 0.0648] | 0.0003 |
| training | first_success_real_transitions | full_aassr vs q_learning | 23 | 712.0000 | [160.6957, 1401.7391] | 0.0302 |
| training | success | full_aassr vs random | 30 | 0.0307 | [0.0191, 0.0426] | 0.0003 |
| training | reward | full_aassr vs random | 30 | 0.0307 | [0.0187, 0.0427] | 0.0002 |
| training | prediction_score | full_aassr vs random | 30 | 0.9900 | [0.9899, 0.9901] | 0.0002 |
| training | holdout_score | full_aassr vs random | 30 | 0.1235 | [0.1235, 0.1236] | 0.0002 |
| training | holdout_gain | full_aassr vs random | 30 | 0.0000 | [0.0000, 0.0000] | 0.0002 |
| training | imagined_nodes | full_aassr vs random | 30 | 525.2874 | [524.3407, 526.2408] | 0.0002 |
| training | imagination_depth | full_aassr vs random | 30 | 7.4038 | [7.3958, 7.4121] | 0.0002 |
| training | actual_return | full_aassr vs random | 30 | 0.0307 | [0.0192, 0.0426] | 0.0002 |
| training | runtime_seconds | full_aassr vs random | 30 | 0.0179 | [0.0175, 0.0183] | 0.0002 |
| training | imagined_transitions | full_aassr vs random | 30 | 525.2874 | [524.3774, 526.2520] | 0.0002 |
| training | learning_auc | full_aassr vs random | 30 | 0.0307 | [0.0190, 0.0427] | 0.0002 |
| training | final_tail_success | full_aassr vs random | 30 | 0.0473 | [0.0290, 0.0662] | 0.0003 |
