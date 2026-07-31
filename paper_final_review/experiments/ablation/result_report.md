# ablation Final review

- Config: `D:\AASSR\configs\paper_ablation_final_v1.json`
- Started: 2026-07-31T10:35:19.989+00:00
- Pipeline completed: 2026-07-31T11:03:02.798+00:00
- Planned/actual rows: 3,096,000 / 3,096,000
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
| condition | 76.35 | 12.0% |
| experiment | 67.91 | 10.7% |
| runtime_seconds | 60.40 | 9.5% |
| environment | 59.05 | 9.3% |
| model | 40.65 | 6.4% |
| holdout_score | 36.10 | 5.7% |
| root_imagined_value | 33.81 | 5.3% |
| checkpoint_fingerprint_after | 31.49 | 5.0% |
| checkpoint_fingerprint_before | 31.49 | 5.0% |
| phase | 30.02 | 4.7% |

Full per-condition/environment/phase counts and all integrity counters are in `integrity_report.json`.

## Primary seed-level results

| Condition | Environment | Phase | Metric | Seeds | Mean | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| contextual_policy | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | training | success | 30 | 0.4769 | [0.4532, 0.5004] |
| contextual_policy | opaque_dependency_l6 | training | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| contextual_policy | opaque_dependency_l6 | training | learning_auc | 30 | 0.4770 | [0.4528, 0.5003] |
| full_aassr | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | training | success | 30 | 0.3556 | [0.3205, 0.3910] |
| full_aassr | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_aassr | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_aassr | opaque_dependency_l6 | training | learning_auc | 30 | 0.3557 | [0.3201, 0.3900] |
| full_depth1_branch1_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3104, 0.3605] |
| full_depth1_branch1_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch1_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3100, 0.3607] |
| full_depth1_branch1_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3107, 0.3600] |
| full_depth1_branch1_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch1_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3109, 0.3602] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3104, 0.3611] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3105, 0.3609] |
| full_depth1_branch2_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3112, 0.3597] |
| full_depth1_branch2_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch2_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3108, 0.3598] |
| full_depth1_branch2_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3105, 0.3606] |
| full_depth1_branch2_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch2_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3102, 0.3609] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3095, 0.3609] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3110, 0.3605] |
| full_depth1_branch4_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3107, 0.3607] |
| full_depth1_branch4_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch4_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3099, 0.3609] |
| full_depth1_branch4_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3100, 0.3604] |
| full_depth1_branch4_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch4_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3111, 0.3610] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3106, 0.3607] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3095, 0.3609] |
| full_depth2_branch1_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3097, 0.3608] |
| full_depth2_branch1_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch1_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3111, 0.3612] |
| full_depth2_branch1_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3103, 0.3606] |
| full_depth2_branch1_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch1_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3097, 0.3605] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3105, 0.3602] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3106, 0.3619] |
| full_depth2_branch2_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3106, 0.3604] |
| full_depth2_branch2_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch2_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3107, 0.3613] |
| full_depth2_branch2_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3104, 0.3607] |
| full_depth2_branch2_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch2_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3104, 0.3605] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3098, 0.3604] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3112, 0.3609] |
| full_depth2_branch4_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3108, 0.3606] |
| full_depth2_branch4_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch4_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3097, 0.3616] |
| full_depth2_branch4_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3107, 0.3605] |
| full_depth2_branch4_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch4_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3104, 0.3607] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3107, 0.3611] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3108, 0.3600] |
| full_depth4_branch1_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3105, 0.3601] |
| full_depth4_branch1_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch1_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3103, 0.3604] |
| full_depth4_branch1_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3100, 0.3593] |
| full_depth4_branch1_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch1_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3105, 0.3611] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3108, 0.3605] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3110, 0.3600] |
| full_depth4_branch2_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3113, 0.3602] |
| full_depth4_branch2_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch2_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3102, 0.3597] |
| full_depth4_branch2_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3104, 0.3599] |
| full_depth4_branch2_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch2_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3114, 0.3604] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3096, 0.3603] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3114, 0.3605] |
| full_depth4_branch4_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3104, 0.3604] |
| full_depth4_branch4_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch4_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3103, 0.3603] |
| full_depth4_branch4_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3106, 0.3606] |
| full_depth4_branch4_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch4_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3098, 0.3599] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3101, 0.3608] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3107, 0.3603] |
| full_depth6_branch1_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3109, 0.3604] |
| full_depth6_branch1_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch1_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3101, 0.3607] |
| full_depth6_branch1_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3097, 0.3604] |
| full_depth6_branch1_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch1_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3105, 0.3608] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3109, 0.3592] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3103, 0.3609] |
| full_depth6_branch2_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3105, 0.3601] |
| full_depth6_branch2_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch2_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3101, 0.3601] |
| full_depth6_branch2_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3110, 0.3607] |
| full_depth6_branch2_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch2_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3105, 0.3608] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3111, 0.3600] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3110, 0.3597] |
| full_depth6_branch4_max | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_max | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_max | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3110, 0.3606] |
| full_depth6_branch4_max | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch4_max | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_max | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3101, 0.3609] |
| full_depth6_branch4_mean | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_mean | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_mean | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3104, 0.3608] |
| full_depth6_branch4_mean | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch4_mean | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_mean | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3103, 0.3603] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3106, 0.3604] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3111, 0.3601] |
| full_no_error_penalty | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_error_penalty | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_error_penalty | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_error_penalty | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_error_penalty | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3103, 0.3598] |
| full_no_error_penalty | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_no_error_penalty | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_error_penalty | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3096, 0.3609] |
| full_no_repeat_penalty | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_repeat_penalty | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_repeat_penalty | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_repeat_penalty | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_repeat_penalty | opaque_dependency_l6 | training | success | 30 | 0.3352 | [0.3107, 0.3608] |
| full_no_repeat_penalty | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_no_repeat_penalty | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_repeat_penalty | opaque_dependency_l6 | training | learning_auc | 30 | 0.3353 | [0.3102, 0.3622] |
| full_no_validated_information | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_validated_information | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_validated_information | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_validated_information | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_validated_information | opaque_dependency_l6 | training | success | 30 | 0.3355 | [0.3116, 0.3596] |
| full_no_validated_information | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| full_no_validated_information | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| full_no_validated_information | opaque_dependency_l6 | training | learning_auc | 30 | 0.3356 | [0.3113, 0.3597] |
| imagination_no_validated_value | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| imagination_no_validated_value | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| imagination_no_validated_value | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| imagination_no_validated_value | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| imagination_no_validated_value | opaque_dependency_l6 | training | success | 30 | 0.3355 | [0.3115, 0.3606] |
| imagination_no_validated_value | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| imagination_no_validated_value | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| imagination_no_validated_value | opaque_dependency_l6 | training | learning_auc | 30 | 0.3356 | [0.3113, 0.3599] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_seen | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_seen | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_unseen_zero_shot | success | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_unseen_zero_shot | prediction_score | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | training | success | 30 | 0.4725 | [0.4465, 0.4984] |
| prophecy_no_imagination | opaque_dependency_l6 | training | prediction_score | 30 | 0.9900 | [0.9898, 0.9902] |
| prophecy_no_imagination | opaque_dependency_l6 | training | holdout_gain | 30 | 0.0000 | [0.0000, 0.0000] |
| prophecy_no_imagination | opaque_dependency_l6 | training | learning_auc | 30 | 0.4726 | [0.4478, 0.4977] |

## Holm-significant paired comparisons

| Phase | Metric | Comparison | Paired seeds | Difference | 95% CI | Holm p |
|---|---|---|---|---|---|---|
| evaluation_seen | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0021 |
| evaluation_unseen_zero_shot | holdout_score | full_aassr vs contextual_policy | 30 | 1.0000 | [1.0000, 1.0000] | 0.0021 |
| training | success | full_aassr vs contextual_policy | 30 | -0.1213 | [-0.1617, -0.0799] | 0.0029 |
| training | reward | full_aassr vs contextual_policy | 30 | -0.1213 | [-0.1618, -0.0795] | 0.0021 |
| training | prediction_score | full_aassr vs contextual_policy | 30 | 0.9900 | [0.9898, 0.9902] | 0.0029 |
| training | holdout_score | full_aassr vs contextual_policy | 30 | 0.1235 | [0.1235, 0.1236] | 0.0021 |
| training | holdout_gain | full_aassr vs contextual_policy | 30 | 0.0000 | [0.0000, 0.0000] | 0.0029 |
| training | imagined_nodes | full_aassr vs contextual_policy | 30 | 186.4534 | [185.9878, 186.9297] | 0.0021 |
| training | imagination_depth | full_aassr vs contextual_policy | 30 | 5.4441 | [5.4353, 5.4532] | 0.0021 |
| training | actual_return | full_aassr vs contextual_policy | 30 | -0.1213 | [-0.1616, -0.0793] | 0.0021 |
| training | runtime_seconds | full_aassr vs contextual_policy | 30 | 0.0062 | [0.0061, 0.0063] | 0.0021 |
| training | imagined_transitions | full_aassr vs contextual_policy | 30 | 186.4534 | [185.9776, 186.9498] | 0.0021 |
| training | learning_auc | full_aassr vs contextual_policy | 30 | -0.1214 | [-0.1630, -0.0805] | 0.0029 |
| training | final_tail_success | full_aassr vs contextual_policy | 30 | -0.2228 | [-0.2833, -0.1605] | 0.0029 |
| training | imagined_nodes | full_aassr vs full_depth1_branch1_max | 30 | 177.3378 | [176.8792, 177.8019] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch1_max | 30 | 4.4732 | [4.4651, 4.4818] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch1_max | 30 | 0.0051 | [0.0050, 0.0052] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch1_max | 30 | 177.3378 | [176.8686, 177.8041] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch1_mean | 30 | 177.3378 | [176.8824, 177.8065] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch1_mean | 30 | 4.4732 | [4.4651, 4.4817] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch1_mean | 30 | 0.0051 | [0.0050, 0.0052] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch1_mean | 30 | 177.3378 | [176.8788, 177.8002] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch1_risk-adjusted | 30 | 177.3378 | [176.8742, 177.8207] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch1_risk-adjusted | 30 | 4.4732 | [4.4651, 4.4816] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch1_risk-adjusted | 30 | 0.0051 | [0.0050, 0.0052] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch1_risk-adjusted | 30 | 177.3378 | [176.8721, 177.7921] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch2_max | 30 | 172.7800 | [172.3354, 173.2363] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch2_max | 30 | 4.4732 | [4.4653, 4.4814] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch2_max | 30 | 0.0049 | [0.0048, 0.0051] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch2_max | 30 | 172.7800 | [172.3221, 173.2599] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch2_mean | 30 | 172.7800 | [172.3262, 173.2254] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch2_mean | 30 | 4.4732 | [4.4648, 4.4816] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch2_mean | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch2_mean | 30 | 172.7800 | [172.3202, 173.2401] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch2_risk-adjusted | 30 | 172.7800 | [172.3197, 173.2428] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch2_risk-adjusted | 30 | 4.4732 | [4.4652, 4.4815] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch2_risk-adjusted | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch2_risk-adjusted | 30 | 172.7800 | [172.3226, 173.2537] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch4_max | 30 | 172.7800 | [172.3351, 173.2513] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch4_max | 30 | 4.4732 | [4.4653, 4.4815] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch4_max | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch4_max | 30 | 172.7800 | [172.3208, 173.2458] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch4_mean | 30 | 172.7800 | [172.3380, 173.2598] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch4_mean | 30 | 4.4732 | [4.4654, 4.4816] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch4_mean | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch4_mean | 30 | 172.7800 | [172.3179, 173.2650] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth1_branch4_risk-adjusted | 30 | 172.7800 | [172.3257, 173.2491] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth1_branch4_risk-adjusted | 30 | 4.4732 | [4.4652, 4.4820] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth1_branch4_risk-adjusted | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth1_branch4_risk-adjusted | 30 | 172.7800 | [172.3379, 173.2351] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch1_max | 30 | 173.5410 | [173.0664, 174.0127] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch1_max | 30 | 3.5107 | [3.5031, 3.5184] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch1_max | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch1_max | 30 | 173.5410 | [173.0829, 174.0172] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch1_mean | 30 | 173.5410 | [173.0723, 174.0105] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch1_mean | 30 | 3.5107 | [3.5028, 3.5182] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch1_mean | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch1_mean | 30 | 173.5410 | [173.0889, 174.0037] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch1_risk-adjusted | 30 | 173.5410 | [173.0779, 174.0133] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch1_risk-adjusted | 30 | 3.5107 | [3.5029, 3.5184] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch1_risk-adjusted | 30 | 0.0049 | [0.0048, 0.0050] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch1_risk-adjusted | 30 | 173.5410 | [173.0715, 173.9966] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch2_max | 30 | 157.5931 | [157.1640, 158.0447] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch2_max | 30 | 3.5107 | [3.5035, 3.5182] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch2_max | 30 | 0.0044 | [0.0043, 0.0045] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch2_max | 30 | 157.5931 | [157.1462, 158.0444] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch2_mean | 30 | 157.5931 | [157.1632, 158.0419] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch2_mean | 30 | 3.5107 | [3.5031, 3.5185] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch2_mean | 30 | 0.0045 | [0.0043, 0.0046] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch2_mean | 30 | 157.5931 | [157.1525, 158.0434] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch2_risk-adjusted | 30 | 157.5931 | [157.1444, 158.0454] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch2_risk-adjusted | 30 | 3.5107 | [3.5031, 3.5181] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch2_risk-adjusted | 30 | 0.0042 | [0.0041, 0.0043] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch2_risk-adjusted | 30 | 157.5931 | [157.1559, 158.0391] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch4_max | 30 | 157.5931 | [157.1597, 158.0350] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch4_max | 30 | 3.5107 | [3.5030, 3.5186] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch4_max | 30 | 0.0045 | [0.0044, 0.0045] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch4_max | 30 | 157.5931 | [157.1545, 158.0335] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch4_mean | 30 | 157.5931 | [157.1571, 158.0523] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch4_mean | 30 | 3.5107 | [3.5032, 3.5183] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch4_mean | 30 | 0.0044 | [0.0043, 0.0045] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch4_mean | 30 | 157.5931 | [157.1514, 158.0374] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth2_branch4_risk-adjusted | 30 | 157.5931 | [157.1527, 158.0480] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth2_branch4_risk-adjusted | 30 | 3.5107 | [3.5033, 3.5184] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth2_branch4_risk-adjusted | 30 | 0.0042 | [0.0041, 0.0043] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth2_branch4_risk-adjusted | 30 | 157.5931 | [157.1474, 158.0547] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch1_max | 30 | 168.2274 | [167.7793, 168.6935] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch1_max | 30 | 1.6388 | [1.6335, 1.6442] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch1_max | 30 | 0.0047 | [0.0046, 0.0048] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch1_max | 30 | 168.2274 | [167.7723, 168.6905] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch1_mean | 30 | 168.2274 | [167.7649, 168.6792] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch1_mean | 30 | 1.6388 | [1.6333, 1.6443] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch1_mean | 30 | 0.0047 | [0.0046, 0.0048] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch1_mean | 30 | 168.2274 | [167.7766, 168.6908] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch1_risk-adjusted | 30 | 168.2274 | [167.7745, 168.6769] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch1_risk-adjusted | 30 | 1.6388 | [1.6335, 1.6445] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch1_risk-adjusted | 30 | 0.0047 | [0.0046, 0.0048] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch1_risk-adjusted | 30 | 168.2274 | [167.7872, 168.6852] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch2_max | 30 | 96.8745 | [96.5534, 97.2024] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch2_max | 30 | 1.6365 | [1.6313, 1.6421] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch2_max | 30 | 0.0027 | [0.0025, 0.0028] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch2_max | 30 | 96.8745 | [96.5416, 97.1990] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch2_mean | 30 | 96.8745 | [96.5589, 97.2052] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch2_mean | 30 | 1.6365 | [1.6313, 1.6419] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch2_mean | 30 | 0.0026 | [0.0025, 0.0027] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch2_mean | 30 | 96.8745 | [96.5498, 97.2084] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch2_risk-adjusted | 30 | 96.8745 | [96.5483, 97.2063] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch2_risk-adjusted | 30 | 1.6365 | [1.6313, 1.6419] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch2_risk-adjusted | 30 | 0.0024 | [0.0023, 0.0025] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch2_risk-adjusted | 30 | 96.8745 | [96.5579, 97.1834] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch4_max | 30 | 96.8745 | [96.5479, 97.1989] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch4_max | 30 | 1.6365 | [1.6313, 1.6420] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch4_max | 30 | 0.0026 | [0.0025, 0.0027] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch4_max | 30 | 96.8745 | [96.5542, 97.2034] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch4_mean | 30 | 96.8745 | [96.5543, 97.2029] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch4_mean | 30 | 1.6365 | [1.6311, 1.6418] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch4_mean | 30 | 0.0027 | [0.0025, 0.0028] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch4_mean | 30 | 96.8745 | [96.5567, 97.1959] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth4_branch4_risk-adjusted | 30 | 96.8745 | [96.5530, 97.1996] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth4_branch4_risk-adjusted | 30 | 1.6365 | [1.6313, 1.6420] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth4_branch4_risk-adjusted | 30 | 0.0024 | [0.0023, 0.0026] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth4_branch4_risk-adjusted | 30 | 96.8745 | [96.5412, 97.2077] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth6_branch1_max | 30 | 165.9560 | [165.5233, 166.4007] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth6_branch1_max | 30 | 0.0046 | [0.0033, 0.0060] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth6_branch1_max | 30 | 0.0046 | [0.0045, 0.0047] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth6_branch1_max | 30 | 165.9560 | [165.5014, 166.4081] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth6_branch1_mean | 30 | 165.9560 | [165.5179, 166.4019] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth6_branch1_mean | 30 | 0.0046 | [0.0033, 0.0060] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth6_branch1_mean | 30 | 0.0046 | [0.0045, 0.0047] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth6_branch1_mean | 30 | 165.9560 | [165.5220, 166.3899] | 0.0021 |
| training | imagined_nodes | full_aassr vs full_depth6_branch1_risk-adjusted | 30 | 165.9560 | [165.5148, 166.3941] | 0.0021 |
| training | imagination_depth | full_aassr vs full_depth6_branch1_risk-adjusted | 30 | 0.0046 | [0.0033, 0.0060] | 0.0021 |
| training | runtime_seconds | full_aassr vs full_depth6_branch1_risk-adjusted | 30 | 0.0046 | [0.0045, 0.0047] | 0.0021 |
| training | imagined_transitions | full_aassr vs full_depth6_branch1_risk-adjusted | 30 | 165.9560 | [165.5289, 166.4256] | 0.0021 |
| training | holdout_score | full_aassr vs full_no_validated_information | 30 | 0.1235 | [0.1235, 0.1236] | 0.0021 |
| training | holdout_gain | full_aassr vs full_no_validated_information | 30 | 0.0000 | [0.0000, 0.0000] | 0.0029 |
| training | runtime_seconds | full_aassr vs full_no_validated_information | 30 | 0.0008 | [0.0007, 0.0009] | 0.0021 |
| training | holdout_score | full_aassr vs imagination_no_validated_value | 30 | 0.1235 | [0.1235, 0.1236] | 0.0021 |
| training | holdout_gain | full_aassr vs imagination_no_validated_value | 30 | 0.0000 | [0.0000, 0.0000] | 0.0029 |
| training | runtime_seconds | full_aassr vs imagination_no_validated_value | 30 | 0.0008 | [0.0007, 0.0009] | 0.0021 |
| training | success | full_aassr vs prophecy_no_imagination | 30 | -0.1169 | [-0.1571, -0.0739] | 0.0029 |
| training | reward | full_aassr vs prophecy_no_imagination | 30 | -0.1169 | [-0.1595, -0.0753] | 0.0041 |
| training | imagined_nodes | full_aassr vs prophecy_no_imagination | 30 | 186.4534 | [185.9712, 186.9278] | 0.0021 |
| training | imagination_depth | full_aassr vs prophecy_no_imagination | 30 | 5.4441 | [5.4355, 5.4527] | 0.0021 |
| training | actual_return | full_aassr vs prophecy_no_imagination | 30 | -0.1169 | [-0.1570, -0.0748] | 0.0021 |
| training | runtime_seconds | full_aassr vs prophecy_no_imagination | 30 | 0.0054 | [0.0053, 0.0055] | 0.0021 |
| training | imagined_transitions | full_aassr vs prophecy_no_imagination | 30 | 186.4534 | [185.9790, 186.9177] | 0.0021 |
| training | learning_auc | full_aassr vs prophecy_no_imagination | 30 | -0.1169 | [-0.1567, -0.0771] | 0.0029 |
| training | final_tail_success | full_aassr vs prophecy_no_imagination | 30 | -0.2190 | [-0.2793, -0.1565] | 0.0029 |
