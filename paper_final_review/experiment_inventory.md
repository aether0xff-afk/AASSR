# Final experiment inventory

원본은 읽기 전용으로 조사했으며 이 디렉터리에는 경량 검토 산출물만 생성했다.

| Experiment | Config | Started (UTC) | Pipeline completed (UTC) | Planned rows | Actual rows | Seeds | Validator |
|---|---|---|---|---|---|---|---|
| autonomy | `D:\AASSR\configs\paper_autonomy_final_v1.json` | 2026-07-31T09:46:27.409+00:00 | 2026-07-31T10:28:18.029+00:00 | 1,512,000 | 1,512,000 | 30 | PASS |
| ablation | `D:\AASSR\configs\paper_ablation_final_v1.json` | 2026-07-31T10:35:19.989+00:00 | 2026-07-31T11:03:02.798+00:00 | 3,096,000 | 3,096,000 | 30 | PASS |
| transfer | `D:\AASSR\configs\paper_transfer_final_v1.json` | 2026-07-31T11:05:03.085+00:00 | 2026-07-31T11:45:04.225+00:00 | 762,000 | 762,000 | 30 | PASS |
| creativity | `D:\AASSR\configs\paper_creativity_final_v1.json` | 2026-07-31T11:45:48.040+00:00 | 2026-07-31T11:49:14.397+00:00 | 105,630 | 105,630 | 30 | PASS |
| safe_application | `D:\AASSR\configs\paper_safe_application_final_v1.json` | 2026-07-31T09:19:34.040+00:00 | 2026-07-31T09:19:35.485+00:00 | 600 | 600 | 30 | PASS |

## Storage finding

43GB를 넘는 것은 단일 CSV가 아니라 `paper-ablation-final-v1` 전체 디렉터리다. 최종 `raw/episodes.csv`는 약 0.74GiB이며, 대부분의 용량은 transition trace가 suite run, resume cache, merged raw에 반복 보관된 데서 발생한다.

## autonomy

- Directory: `D:\AASSR\paper_results\paper-autonomy-final-v1`
- Episode CSV: 0.310 GiB, 1,512,000 rows, 41 columns, 220.3 bytes/row
- Exact simulated gzip size: 0.021 GiB (6.7% of CSV)
- Full directory: 20.980 GiB

### Rows by condition, environment, and phase

| Condition | Environment | Phase | Rows |
|---|---|---|---|
| contextual_policy | opaque_dependency_l4 | evaluation_seen | 6,000 |
| contextual_policy | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| contextual_policy | opaque_dependency_l4 | training | 60,000 |
| contextual_policy | opaque_dependency_l6 | evaluation_seen | 6,000 |
| contextual_policy | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| contextual_policy | opaque_dependency_l6 | training | 60,000 |
| contextual_policy | opaque_dependency_l8 | evaluation_seen | 6,000 |
| contextual_policy | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| contextual_policy | opaque_dependency_l8 | training | 60,000 |
| dqn | opaque_dependency_l4 | evaluation_seen | 6,000 |
| dqn | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| dqn | opaque_dependency_l4 | training | 60,000 |
| dqn | opaque_dependency_l6 | evaluation_seen | 6,000 |
| dqn | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| dqn | opaque_dependency_l6 | training | 60,000 |
| dqn | opaque_dependency_l8 | evaluation_seen | 6,000 |
| dqn | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| dqn | opaque_dependency_l8 | training | 60,000 |
| full_aassr | opaque_dependency_l4 | evaluation_seen | 6,000 |
| full_aassr | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| full_aassr | opaque_dependency_l4 | training | 60,000 |
| full_aassr | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_aassr | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_aassr | opaque_dependency_l6 | training | 60,000 |
| full_aassr | opaque_dependency_l8 | evaluation_seen | 6,000 |
| full_aassr | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| full_aassr | opaque_dependency_l8 | training | 60,000 |
| oracle_upper_bound | opaque_dependency_l4 | evaluation_seen | 6,000 |
| oracle_upper_bound | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| oracle_upper_bound | opaque_dependency_l4 | training | 60,000 |
| oracle_upper_bound | opaque_dependency_l6 | evaluation_seen | 6,000 |
| oracle_upper_bound | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| oracle_upper_bound | opaque_dependency_l6 | training | 60,000 |
| oracle_upper_bound | opaque_dependency_l8 | evaluation_seen | 6,000 |
| oracle_upper_bound | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| oracle_upper_bound | opaque_dependency_l8 | training | 60,000 |
| prophecy_no_imagination | opaque_dependency_l4 | evaluation_seen | 6,000 |
| prophecy_no_imagination | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| prophecy_no_imagination | opaque_dependency_l4 | training | 60,000 |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_seen | 6,000 |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| prophecy_no_imagination | opaque_dependency_l6 | training | 60,000 |
| prophecy_no_imagination | opaque_dependency_l8 | evaluation_seen | 6,000 |
| prophecy_no_imagination | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| prophecy_no_imagination | opaque_dependency_l8 | training | 60,000 |
| q_learning | opaque_dependency_l4 | evaluation_seen | 6,000 |
| q_learning | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| q_learning | opaque_dependency_l4 | training | 60,000 |
| q_learning | opaque_dependency_l6 | evaluation_seen | 6,000 |
| q_learning | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| q_learning | opaque_dependency_l6 | training | 60,000 |
| q_learning | opaque_dependency_l8 | evaluation_seen | 6,000 |
| q_learning | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| q_learning | opaque_dependency_l8 | training | 60,000 |
| random | opaque_dependency_l4 | evaluation_seen | 6,000 |
| random | opaque_dependency_l4 | evaluation_unseen_zero_shot | 6,000 |
| random | opaque_dependency_l4 | training | 60,000 |
| random | opaque_dependency_l6 | evaluation_seen | 6,000 |
| random | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| random | opaque_dependency_l6 | training | 60,000 |
| random | opaque_dependency_l8 | evaluation_seen | 6,000 |
| random | opaque_dependency_l8 | evaluation_unseen_zero_shot | 6,000 |
| random | opaque_dependency_l8 | training | 60,000 |

## ablation

- Directory: `D:\AASSR\paper_results\paper-ablation-final-v1`
- Episode CSV: 0.742 GiB, 3,096,000 rows, 41 columns, 257.4 bytes/row
- Exact simulated gzip size: 0.050 GiB (6.8% of CSV)
- Full directory: 43.754 GiB

### Rows by condition, environment, and phase

| Condition | Environment | Phase | Rows |
|---|---|---|---|
| contextual_policy | opaque_dependency_l6 | evaluation_seen | 6,000 |
| contextual_policy | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| contextual_policy | opaque_dependency_l6 | training | 60,000 |
| full_aassr | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_aassr | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_aassr | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch1_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch1_max | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch1_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch1_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch1_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch2_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch2_max | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch2_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch2_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch2_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch4_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch4_max | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch4_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch4_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth1_branch4_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch1_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch1_max | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch1_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch1_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch1_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch2_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch2_max | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch2_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch2_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch2_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch4_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch4_max | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch4_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch4_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth2_branch4_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch1_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch1_max | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch1_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch1_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch1_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch2_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch2_max | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch2_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch2_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch2_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch4_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch4_max | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch4_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch4_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth4_branch4_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch1_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch1_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch1_max | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch1_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch1_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch1_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch1_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch2_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch2_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch2_max | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch2_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch2_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch2_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch2_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch4_max | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch4_max | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch4_max | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch4_mean | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch4_mean | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch4_mean | opaque_dependency_l6 | training | 60,000 |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_depth6_branch4_risk-adjusted | opaque_dependency_l6 | training | 60,000 |
| full_no_error_penalty | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_no_error_penalty | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_no_error_penalty | opaque_dependency_l6 | training | 60,000 |
| full_no_repeat_penalty | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_no_repeat_penalty | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_no_repeat_penalty | opaque_dependency_l6 | training | 60,000 |
| full_no_validated_information | opaque_dependency_l6 | evaluation_seen | 6,000 |
| full_no_validated_information | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| full_no_validated_information | opaque_dependency_l6 | training | 60,000 |
| imagination_no_validated_value | opaque_dependency_l6 | evaluation_seen | 6,000 |
| imagination_no_validated_value | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| imagination_no_validated_value | opaque_dependency_l6 | training | 60,000 |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_seen | 6,000 |
| prophecy_no_imagination | opaque_dependency_l6 | evaluation_unseen_zero_shot | 6,000 |
| prophecy_no_imagination | opaque_dependency_l6 | training | 60,000 |

## transfer

- Directory: `D:\AASSR\paper_results\paper-transfer-final-v1`
- Episode CSV: 0.285 GiB, 762,000 rows, 41 columns, 401.9 bytes/row
- Exact simulated gzip size: 0.012 GiB (4.0% of CSV)
- Full directory: 7.097 GiB

### Rows by condition, environment, and phase

| Condition | Environment | Phase | Rows |
|---|---|---|---|
| from_scratch_contextual_policy | opaque_dependency_l6 | adaptation | 20,400 |
| from_scratch_contextual_policy | opaque_dependency_l6 | evaluation_unseen_adaptation | 120,000 |
| from_scratch_full_aassr | opaque_dependency_l6 | adaptation | 20,400 |
| from_scratch_full_aassr | opaque_dependency_l6 | evaluation_unseen_adaptation | 120,000 |
| full_transfer | opaque_dependency_l6 | adaptation | 20,400 |
| full_transfer | opaque_dependency_l6 | evaluation_unseen_adaptation | 120,000 |
| policy_reset_effect_retained | opaque_dependency_l6 | adaptation | 20,400 |
| policy_reset_effect_retained | opaque_dependency_l6 | evaluation_unseen_adaptation | 120,000 |
| policy_reset_prophecy_retained | opaque_dependency_l6 | adaptation | 20,400 |
| policy_reset_prophecy_retained | opaque_dependency_l6 | evaluation_unseen_adaptation | 120,000 |
| transfer_pretraining | opaque_dependency_l6 | training | 60,000 |

## creativity

- Directory: `D:\AASSR\paper_results\paper-creativity-final-v1`
- Episode CSV: 0.024 GiB, 105,630 rows, 41 columns, 241.1 bytes/row
- Exact simulated gzip size: 0.002 GiB (9.7% of CSV)
- Full directory: 1.455 GiB

### Rows by condition, environment, and phase

| Condition | Environment | Phase | Rows |
|---|---|---|---|
| aassr_no_imagination | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| aassr_no_imagination | multi_solution_dependency | training | 15,000 |
| aassr_no_novelty | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| aassr_no_novelty | multi_solution_dependency | training | 15,000 |
| dqn | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| dqn | multi_solution_dependency | training | 15,000 |
| full_aassr | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| full_aassr | multi_solution_dependency | training | 15,000 |
| novelty_search | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| novelty_search | multi_solution_dependency | training | 15,000 |
| q_learning | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| q_learning | multi_solution_dependency | training | 15,000 |
| random | multi_solution_dependency | evaluation_unseen_adaptation | 90 |
| random | multi_solution_dependency | training | 15,000 |

## safe_application

- Directory: `D:\AASSR\paper_results\paper-safe-application-final-v1`
- Episode CSV: 0.000 GiB, 600 rows, 41 columns, 224.9 bytes/row
- Exact simulated gzip size: 0.000 GiB (2.7% of CSV)
- Full directory: 0.003 GiB

### Rows by condition, environment, and phase

| Condition | Environment | Phase | Rows |
|---|---|---|---|
| safe_rule_agent | docker_local_assessment | evaluation_unseen_zero_shot | 600 |
