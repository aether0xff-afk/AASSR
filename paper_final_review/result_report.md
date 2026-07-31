# AASSR Final 결과 무결성 및 경량 분석 보고서

## 기술 요약

- **질문 1 — 희소 보상 환경에서 정답 시범 없이 스스로 목표에 도달했는가? `Partial`.** Full AASSR는 training AUC가 0보다 높아 학습 중 목표 도달은 관찰됐지만, frozen `evaluation_seen`과 `evaluation_unseen_zero_shot` 성공률은 모든 길이에서 0이었다.
- **질문 2 — 기준선과 다른 유효·유용·재현 가능한 전략을 만들었는가? `No`.** Full AASSR 전략은 성공하고 재현됐지만 frozen novelty threshold를 넘은 전략이 0건이어서 최종 creative candidate도 0건이다.
- 다섯 Final suite는 계획 행 수와 실제 행 수가 일치하고 30개 research seed가 모두 존재한다. Autonomy에서 진행상태 파일 잠금으로 suite-level 재시도 1건이 있었지만 최종 데이터에는 누락 seed가 없다.
- 43.75GiB Ablation 디렉터리의 주원인은 episode CSV가 아니라 transition trace의 실행본·cache·병합본 중복이다. 원본은 수정하거나 삭제하지 않았다.

## RQ1 — 학습 중 도달했지만 frozen 평가에는 일반화되지 않았다

Training success AUC는 각 research seed 내부 episode 곡선을 먼저 적분한 뒤 30 seed 사이에서 집계했다. 평가 성공률도 episode가 아니라 seed별 평균을 통계 단위로 사용했다.

### Training success AUC

| Length | Condition | Mean AUC | Bootstrap 95% CI | Seeds |
|---|---|---|---|---|
| 4 | full_aassr | 0.6150 | [0.5962, 0.6330] | 30 |
| 4 | random | 0.0626 | [0.0605, 0.0646] | 30 |
| 4 | contextual_policy | 0.6544 | [0.6502, 0.6584] | 30 |
| 4 | q_learning | 0.1267 | [0.1060, 0.1485] | 30 |
| 4 | dqn | 0.1603 | [0.1370, 0.1851] | 30 |
| 4 | prophecy_no_imagination | 0.6542 | [0.6502, 0.6582] | 30 |
| 6 | full_aassr | 0.2737 | [0.2497, 0.2983] | 30 |
| 6 | random | 0.0155 | [0.0148, 0.0162] | 30 |
| 6 | contextual_policy | 0.4352 | [0.4146, 0.4556] | 30 |
| 6 | q_learning | 0.0114 | [0.0071, 0.0174] | 30 |
| 6 | dqn | 0.0361 | [0.0257, 0.0480] | 30 |
| 6 | prophecy_no_imagination | 0.4396 | [0.4191, 0.4593] | 30 |
| 8 | full_aassr | 0.0343 | [0.0228, 0.0463] | 30 |
| 8 | random | 0.0036 | [0.0031, 0.0041] | 30 |
| 8 | contextual_policy | 0.1674 | [0.1417, 0.1949] | 30 |
| 8 | q_learning | 0.0030 | [0.0009, 0.0071] | 30 |
| 8 | dqn | 0.0028 | [0.0023, 0.0034] | 30 |
| 8 | prophecy_no_imagination | 0.1595 | [0.1338, 0.1873] | 30 |

### Frozen evaluation success

| Length | Condition | Seen | Unseen zero-shot | Seeds |
|---|---|---|---|---|
| 4 | full_aassr | 0.0000 | 0.0000 | 30 |
| 4 | random | 0.0672 | 0.0602 | 30 |
| 4 | contextual_policy | 0.0000 | 0.0000 | 30 |
| 4 | q_learning | 0.0000 | 0.0000 | 30 |
| 4 | dqn | 0.1000 | 0.0333 | 30 |
| 4 | prophecy_no_imagination | 0.0000 | 0.0000 | 30 |
| 6 | full_aassr | 0.0000 | 0.0000 | 30 |
| 6 | random | 0.0152 | 0.0145 | 30 |
| 6 | contextual_policy | 0.0000 | 0.0000 | 30 |
| 6 | q_learning | 0.0000 | 0.0000 | 30 |
| 6 | dqn | 0.0000 | 0.0000 | 30 |
| 6 | prophecy_no_imagination | 0.0000 | 0.0000 | 30 |
| 8 | full_aassr | 0.0000 | 0.0000 | 30 |
| 8 | random | 0.0038 | 0.0043 | 30 |
| 8 | contextual_policy | 0.0000 | 0.0000 | 30 |
| 8 | q_learning | 0.0000 | 0.0000 | 30 |
| 8 | dqn | 0.0000 | 0.0000 | 30 |
| 8 | prophecy_no_imagination | 0.0000 | 0.0000 | 30 |

### Full AASSR training diagnostics

| Length | Final-tail success | First success transitions | Repeat rate | Error rate | Mean steps | Runtime (s) | Imagined nodes | Prediction | Holdout gain |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.8510 | 68.2667 | 0.0000 | 0.0000 | 4.0000 | 0.0021 | 42.5521 | 0.9900 | 0.0000 |
| 6 | 0.4120 | 321.2000 | 0.0000 | 0.0000 | 6.0000 | 0.0070 | 186.4483 | 0.9900 | 0.0000 |
| 8 | 0.0493 | 1936.2963 | 0.0000 | 0.0000 | 8.0000 | 0.0184 | 525.2874 | 0.9900 | 0.0000 |

`evaluation_unseen_adaptation`은 Autonomy suite에서 생성되지 않았으므로 해당 평가지표는 N/A다. Baseline별 동일 지표와 paired 차이는 경량 CSV에 모두 포함했다.

Full AASSR는 Random, Q-learning, DQN보다 training AUC가 높지만 Contextual Policy와 Prophecy without Imagination보다 낮다. 평가에서는 Full AASSR가 전 길이에서 0이고 L4의 Random/DQN은 0보다 높아, 자율 목표 달성 우수성은 지지되지 않는다.

## Ablation — 제거 효과는 명확하지 않고 길이 변화는 측정되지 않았다

Ablation Final config는 dependency length 6만 실행했다. 따라서 depth·branching·aggregation 효과가 환경 길이에 따라 어떻게 변하는지는 이 데이터로 답할 수 없다. 제거 조건 및 36개 imagination matrix 설정의 seed-first 비교는 `condition_comparisons.csv`와 `cross_seed_summary.csv`에 포함했다.

### Component removal comparisons

| Removed/changed component | Full AUC | Comparator AUC | Full - comparator | 95% CI | Holm p |
|---|---|---|---|---|---|
| Prophecy/Imagination removed | 0.3557 | 0.4770 | -0.1214 | [-0.1630, -0.0805] | 0.0029 |
| Imagination removed | 0.3557 | 0.4726 | -0.1169 | [-0.1567, -0.0771] | 0.0029 |
| Validated information value removed | 0.3557 | 0.3356 | 0.0201 | [-0.0186, 0.0574] | 1.0000 |
| Repeat penalty removed | 0.3557 | 0.3353 | 0.0204 | [-0.0203, 0.0604] | 1.0000 |
| Error penalty removed | 0.3557 | 0.3353 | 0.0204 | [-0.0204, 0.0591] | 1.0000 |
| Imagination without validated value | 0.3557 | 0.3356 | 0.0201 | [-0.0175, 0.0572] | 1.0000 |

### Imagination matrix marginal results (L6 only)

| Axis level | Learning AUC | Final-tail success | Prediction | Holdout gain |
|---|---|---|---|---|
| axis_aggregation=max | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_aggregation=mean | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_aggregation=risk-adjusted | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_branch=1 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_branch=2 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_branch=4 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_depth=1 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_depth=2 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_depth=4 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |
| axis_depth=6 | 0.3353 | 0.4995 | 0.9900 | 0.0000 |

Prophecy/Imagination 제거군인 Contextual Policy 및 Prophecy without Imagination은 Full AASSR보다 training AUC가 높았다. validated information value, repeat penalty, error penalty 제거군과 Full 간 차이는 작고 bootstrap CI가 0을 포함한다. 이는 해당 구성요소의 긍정적 기여를 입증하지 못한다.

## 구조 전이 — 작은 평균 차이는 있으나 유의한 sample-efficiency 이득은 없다

| Condition | Budget 0 | Budget 1 | Budget 4 | Budget 16 | Budget 64 |
|---|---|---|---|---|---|
| from_scratch_contextual_policy | 0.0000 | 0.0042 | 0.0292 | 0.0750 | 0.2083 |
| from_scratch_full_aassr | 0.0000 | 0.0042 | 0.0333 | 0.0917 | 0.1958 |
| full_transfer | 0.0000 | 0.0000 | 0.0250 | 0.0958 | 0.2333 |
| policy_reset_effect_retained | 0.0000 | 0.0000 | 0.0375 | 0.0875 | 0.2292 |
| policy_reset_prophecy_retained | 0.0000 | 0.0000 | 0.0375 | 0.1042 | 0.2292 |

| Condition | Adaptation AUC | Episodes to 50% | Episodes to 80% | Saving to 50% | Saving to 80% | Transfer gain | Calibration error |
|---|---|---|---|---|---|---|---|
| from_scratch_contextual_policy | 0.1168 | 64.0000 (n=1) | N/A | N/A | N/A | -0.0036 | 0.0633 |
| from_scratch_full_aassr | 0.1204 | 64.0000 (n=2) | N/A | 0.0000 (n=2) | N/A | 0.0000 | 0.6868 |
| full_transfer | 0.1354 | 32.0000 (n=3) | N/A | N/A | N/A | 0.0149 | 0.6943 |
| policy_reset_effect_retained | 0.1313 | 48.0000 (n=3) | N/A | N/A | N/A | 0.0109 | 0.6878 |
| policy_reset_prophecy_retained | 0.1392 | 64.0000 (n=2) | N/A | N/A | N/A | 0.0187 | 0.6760 |

Full transfer의 adaptation AUC는 from-scratch보다 소폭 높지만 paired bootstrap CI가 0을 포함하고 Holm 보정 후 유의하지 않다. Effect representation retained 조건도 명시적인 ID-retained 대조군이 없어 representation 자체의 인과적 이점을 판정할 수 없다. Effect-retained가 Prophecy-retained보다 우월하다는 증거도 없다.

모든 adaptation branch는 동일한 시작 checkpoint fingerprint에서 분기했고 `[0,1,4,16,64]` budget을 모두 포함했다. Transfer runner는 `evaluation_seen`과 `evaluation_unseen_zero_shot` 행을 생성하지 않았으므로, transfer suite 단독으로 seen-vs-zero-shot 질문은 답할 수 없다.

## RQ2 — 성공 전략은 많지만 기준선과 구조적으로 구별되지 않았다

| Condition | Strategies | Successful | Unique graphs | Novel | Utility-qualified | Reproduced | Candidates |
|---|---|---|---|---|---|---|---|
| aassr_no_imagination | 15000 | 15000 | 234 | 0 | 772 | 15000 | 0 |
| aassr_no_novelty | 15000 | 15000 | 234 | 0 | 870 | 15000 | 0 |
| dqn | 15000 | 15000 | 234 | 0 | 965 | 15000 | 0 |
| full_aassr | 15000 | 15000 | 234 | 0 | 899 | 15000 | 0 |
| novelty_search | 15000 | 15000 | 234 | 0 | 939 | 15000 | 0 |
| q_learning | 15000 | 15000 | 234 | 0 | 1057 | 15000 | 0 |
| random | 15000 | 15000 | 234 | 0 | 982 | 15000 | 0 |

| Solution family | Strategies | Share |
|---|---|---|
| tool_route | 26,130 | 24.9% |
| bypass_route | 24,522 | 23.4% |
| resource_route | 24,456 | 23.3% |
| information_route | 22,522 | 21.4% |
| emergent_combination | 7,370 | 7.0% |

Frozen novelty threshold는 0.06000000000000001이다. 총 105,000개 전략은 성공했지만 최소 baseline distance가 모두 0이어서 threshold를 넘은 전략과 최종 candidate는 0건이다. 행동 문자열은 graph key에 포함하지 않았고 causal effect graph 기준으로 중복 제거했다.

## 인간 비교 데이터 — 포함되지 않았다

실제 인간 path: False, 인간 rating: False, 참가자 수: 0. Approval ID와 승인 dataset manifest가 없고 merge가 비활성화됐다. 따라서 모든 reference는 **baseline-generated reference**이며 인간 전략으로 해석하면 안 된다.

## 무결성 및 이상 징후

- 모든 suite에서 계획 행 수와 실제 행 수가 일치하고 Final research seed 30개가 모두 존재한다.
- episode grain 및 exact row 중복, NaN/Inf, transition budget 초과, evaluation checkpoint mutation, agent-visible private-label leak은 `integrity_report.json`의 count로 기록했다.
- 여러 평가 조건이 0% 또는 100%이고 일부 ablation 결과가 완전히 동일하다. 이는 ceiling/floor effect로 비교 검정의 식별력을 크게 낮춘다.
- Seed-level 성공 벡터가 완전히 동일한 조건 묶음은 14개, 성공/AUC 절대합의 50%를 단일 seed가 차지한 묶음은 2개다. 구체적인 조건과 seed는 `integrity_report.json`에 있다.
- 평균 prediction score가 0.8 이상인데 seed-level success가 0.1 이하인 조건/환경/phase 묶음은 2개다.
- Full AASSR가 Contextual/Prophecy no-imagination comparator보다 유의하게 낮은 비교는 16개다. 단, 조건 간 차이가 imagination 하나뿐은 아니므로 imagination의 단독 인과효과로 해석하지 않는다.
- Creativity는 5개 solution family를 생성했지만 baseline pool에 동일한 causal graphs가 존재하여 novelty가 모두 0이다. Threshold가 낮아 대부분 통과한 것이 아니라 아무 것도 통과하지 못한 상태다.
- 가장 큰 solution family는 `tool_route`이며 전체의 24.9%다. 단일 family가 과반을 차지하지는 않는다.
- 원본 transition trace는 분석에 필요한 episode summary와 다른 grain이지만, 동일 merged/cache 파일이 중복 저장되어 전달 용량을 키운다.
- prediction score >= 0.8인데 실패한 episode가 발견됐다: ablation 1,632,659건, autonomy 224,109건, transfer 419,970건. 식별자와 앞뒤 문맥 표본은 `anomaly_samples.csv`에 있다.

## 방법

원본 CSV는 `csv.DictReader`로 한 행씩 읽었고, 중복 검사는 BLAKE2b-128 row/grain fingerprint를 임시 SQLite에 누적했다. gzip 크기는 원본 바이트를 메모리에 올리지 않고 zlib level 6으로 끝까지 통과시켜 실제 압축 바이트 수를 계산했다. Parquet은 pyarrow/duckdb가 설치되지 않아 생성하지 않았고 typed dictionary encoding을 가정한 범위만 제시했다.

통계 단위는 research seed다. episode를 먼저 seed 내부에서 집계한 후 seed 평균·표준편차·5,000회 bootstrap 95% CI를 계산했다. 조건 비교는 paired seed difference, 20,000회 paired permutation test, Cohen's dz, 동일 experiment/environment/phase/metric family 내 Holm correction을 사용했다. Oracle은 추론 비교에서 제외했다.

## 한계와 다음 단계

1. Ablation을 L4/L8에서도 실행해야 imagination 설정과 환경 길이의 상호작용을 검정할 수 있다.
2. Transfer에 명시적인 ID-retained 대조군과 zero-shot phase를 추가해야 representation advantage를 판정할 수 있다.
3. Creativity reference pool을 독립 인간 데이터 또는 사전 동결된 외부 baseline으로 구성해야 자기 데이터와 동일 graph가 reference에 들어가는 문제를 피할 수 있다.
4. Full AASSR의 frozen 평가 실패 원인을 training/evaluation world reset, learned policy use, reward propagation 관점에서 진단한 뒤 새로운 protocol version으로 재실행해야 한다. 현재 Final 결과를 사후 수정하면 안 된다.

## 추가 질문

- Training 성공이 frozen 평가에서 사라지는 원인이 checkpoint 복원, policy action selection, world-seed shift 중 어디에 있는가?
- Baseline과 동일한 creative graph가 대량 생성되는 것이 환경의 해 공간 제한인지 agent canonicalization의 과도한 압축인지?