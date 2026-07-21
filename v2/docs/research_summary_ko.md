# AASSR/APASSR v2 연구 정리

이 문서는 현재 v2 구현과 실험 결과를 논문/발표자료에 사용할 수 있도록
요약한 것이다. 핵심 결론은 다음과 같다.

```text
APASSR의 성능 향상은 특정 신경망 구조 하나가 아니라,
KK/KV Knowledge Storage, 행동 파라미터화, Prophecy Module,
Imagination Cycle, DMP 폐루프가 결합된 구조에서 나온다.
```

## 1. 핵심 프레임워크

본 프로젝트는 원래 APASSR 논문의 아이디어를 GridWorld에서 검증하기 위해
단순화한 구현이다. GridWorld는 원래 nmap 기반 펜테스팅 실험을 대체하는
환경이 아니라, 지식-행동 의존성 구조를 통제된 조건에서 검증하기 위한
추상 환경이다.

핵심 루프는 다음과 같다.

```text
Observation
-> Knowledge Storage update
-> KV candidates stored under KK slots
-> Action template parameter binding
-> Candidate action generation
-> PolicyABC / Prophecy / Imagination evaluation
-> Execution
-> New observation
```

연구 철학은 다음 문장으로 요약할 수 있다.

```text
행동이 지식을 만들고, 지식이 다음 행동을 만든다.
```

### Knowledge Storage

Knowledge Storage는 단순한 관측 기록소가 아니다.

```text
Knowledge Storage = memory + action parameter supplier
```

즉, 관측으로 얻은 KV는 나중에 행동 템플릿의 KK 슬롯에 대입된다.

예:

```text
MOVE_TOWARD {KK_FRONTIER_CELL}
INSPECT_CELL {KK_UNKNOWN_NEIGHBOR}
USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}
FOLLOW_HINT {KK_HINT_VALUE}
```

## 2. 조건 정의

현재 실험 조건은 다음과 같이 정리된다.

| Condition | 의미 | 해석 |
| --- | --- | --- |
| C0 | RandomScorer | 실행 가능한 후보 중 무작위 선택 |
| C1 | PolicyABC | WHAT/HOW/WHERE 확률표 기반 정책 학습 |
| C2 | PolicyABC + Prophecy reward | Prophecy 예측 오차를 보상에 반영 |
| C3 | PolicyABC + Prophecy Module + Imagination Cycle | legacy lightweight prototype |
| C4 | PolicyABC + sequence-based Prophecy + Imagination | 선택적 구현 변형/ablation |
| C5 | Improved APASSR | ablation 결과를 반영한 개선형 |
| APASSR_FULL | full predicted-state imagination | 논문 정렬형 구현 |
| APASSR_FULL_CAL | calibrated APASSR_FULL | FULL의 미래가치 과신을 보정한 비교 조건 |
| QLEARN | Tabular Q-learning baseline | 동일 정보 조건 baseline |
| DQN_PARTIAL | Partial-observation DQN baseline | Knowledge/candidate feature 기반 DQN |
| ORACLE_MDP | Full-map shortest-path oracle | 전체 지도를 아는 상한선, 공정 baseline 아님 |

C3/C5는 기존 실험 재현성을 위해 유지되는 legacy 조건이다.

```text
C3 = PolicyABC + Prophecy Module + Imagination Cycle
```

C3는 TableProphecyModel을 사용하지만, 프레임워크 자체가 table trick이라는
뜻은 아니다. TableProphecyModel은 현재 Prophecy Module의 경량 구현이다.
TransformerProphecyModel과 SequenceProphecyModel은 구현 변형이다.

C5는 C3를 대체하는 vanilla 조건이 아니라, A4/A5 ablation 결과를 반영한
개선형 조건이다.

```text
C5 = C3 loop
   + knowledge_weight = 0.0
   + repeat penalty 유지
   + error avoidance 유지
   + prediction-error reward 유지
```

새 논문 정렬형 구현은 `APASSR_FULL`로 분리한다.

```text
APASSR_FULL
= independent Policy A/B/C
+ richer Prophecy state/action/history
+ virtual Knowledge Store transition
+ future candidate regeneration
+ predicted-state multi-step imagination
```

정확한 구분은 다음과 같다.

```text
C3/C5:
Prophecy-guided candidate scoring with lightweight dependency lookahead

APASSR_FULL:
Predicted-state multi-step imagination with virtual Knowledge Store transitions
and future action regeneration

APASSR_FULL_CAL:
APASSR_FULL + candidate signature deduplication + confidence-discounted future
rollout value + placeholder grounding discount
```

`APASSR_FULL`의 rollout도 실제 hidden map을 시뮬레이션하는 것이 아니라
Knowledge Store와 Prophecy 예측에 기반한 belief/knowledge-state rollout이다.
`APASSR_FULL_CAL` 역시 같은 경계를 유지하며, 보상/환경/rollout 깊이/branching을
튜닝하지 않는다. 이 조건은 기존 FULL을 대체하지 않고, 과도하게 많은 imagined
future candidate가 미래 가치를 부풀리는지 확인하기 위한 보정 비교 조건이다.

진단 지표와 30x10 결과는 다음 문서에 정리했다.

```text
docs/apassr_full_diagnostic_metrics.md
docs/apassr_full_diagnostic_results.md
```

핵심 결과는 다음과 같다.

```text
v2_complex:
APASSR_FULL success = 0.647
C3 success = 0.717
DQN_PARTIAL success = 0.643

locked_bottleneck:
APASSR_FULL success = 0.197
C3 success = 0.400
DQN_PARTIAL success = 0.397
```

APASSR_FULL은 episode당 수천 개의 imagined state transition과 수만 개의
newly unlocked action을 생성하므로 구조가 비활성인 것은 아니다. 그러나
imagined next action의 exact match가 약 0.18-0.19 수준이어서, 상상한
미래 행동이 실제 다음 행동으로 이어지는 정도는 아직 낮다.

따라서 `APASSR_FULL_CAL`은 다음 네 가지 진단적 보정을 추가했다.

```text
1. placeholder candidate signature deduplication
2. raw/unique future candidate expansion metrics
3. confidence-discounted future rollout value
4. placeholder grounding discount
```

## 3. 구현 현황

주요 구현 파일은 다음과 같다.

| 파일 | 역할 |
| --- | --- |
| `src/aassr/knowledge.py` | KK/KV Knowledge Storage |
| `src/aassr/gridworld.py` | DMP, 행동 생성, 실행, 관측, 지식 갱신 |
| `src/aassr/policy.py` | C0 RandomScorer, C1 PolicyABC |
| `src/aassr/prophecy.py` | ProphecyModule, Table/Sequence/Transformer 구현 |
| `src/aassr/imagination.py` | Prophecy 기반 depth-limited Imagination Cycle |
| `src/aassr/experiment.py` | C0-C5 실험 러너 |
| `src/aassr/v2_compare.py` | APASSR 조건과 baseline 비교 |
| `src/aassr/ablation.py` | 논문용 ablation suite |
| `src/aassr/analysis.py` | summary table, CI, report, plot 생성 |
| `src/aassr/render_world.py` | 발표자료용 환경 PNG 렌더링 |

현재 CLI는 긴 실험에서 seed 단위 진행률과 ETA를 표시한다.

## 4. 실험 환경

대표 환경은 세 가지이다.

| Environment | 목적 |
| --- | --- |
| `random_key_door` | 기본 key-door-hint-flag 구조 |
| `v2_complex` | 더 큰 랜덤 맵, 다중 key/door/hint |
| `locked_bottleneck` | 문 bottleneck을 통과해야 하는 의존성 stress 환경 |

발표자료용 렌더링 이미지:

| Environment | Image |
| --- | --- |
| `random_key_door` | `artifacts/world_renders/random_key_door_seed_0.png` |
| `v2_complex` | `artifacts/world_renders/v2_complex_seed_0.png` |
| `locked_bottleneck` | `artifacts/world_renders/locked_bottleneck_seed_0.png` |

## 5. 주요 성능 결과

### v2_complex 30x10 기준

이전 paper-candidate run에서 C3는 동일 정보 조건 baseline보다 높은 성공률과
낮은 반복률을 보였다.

| Condition | Success rate | Repeat rate |
| --- | ---: | ---: |
| C3 | 0.717 | 0.254 |
| DQN_PARTIAL | 0.593 | 0.354 |
| QLEARN | 0.573 | 0.372 |
| C1 | 0.540 | 약 0.35 |
| C2 | 0.530 | 약 0.35 |
| C0 | 0.427 | 0.504 |

해석:

```text
C3는 단순히 더 많이 시도해서 성공한 것이 아니라,
반복 행동을 줄이면서 성공률을 높였다.
```

주의:

`ORACLE_MDP`는 전체 지도를 아는 상한선이므로 C3와 동일 정보 조건으로
비교하면 안 된다.

## 6. Ablation 결과

실험 경로:

```text
runs/ablation_env_sweep_30x10
```

설정:

```text
episodes = 30
seeds = 10
step_limit = 120
worlds = random_key_door, v2_complex, locked_bottleneck
```

### ablation_1: Table Prophecy vs Transformer Prophecy

| Environment | Table C3 | Transformer C3 | 해석 |
| --- | ---: | ---: | --- |
| random_key_door | 0.970 | 0.880 | Table이 더 안정적 |
| v2_complex | 0.717 | 0.680 | Table이 성공률 우세 |
| locked_bottleneck | 0.423 | 0.397 | Table이 약간 우세 |

결론:

```text
현재 GridWorld 규모에서는 Transformer가 더 복잡하지만 더 좋은 것은 아니다.
TableProphecyModel이 더 sample-efficient하고 안정적이었다.
```

논문 해석:

```text
성능 향상은 특정 neural prophecy 구조 때문이 아니라,
지식-행동 폐루프 구조 자체에서 나온다.
```

### ablation_2: Prophecy prediction-error reward ON/OFF

| Environment | Reward ON | Reward OFF | 해석 |
| --- | ---: | ---: | --- |
| random_key_door | 0.970 | 0.973 | 단순 환경은 포화 상태 |
| v2_complex | 0.717 | 0.700 | ON이 약간 우세 |
| locked_bottleneck | 0.423 | 0.370 | ON이 뚜렷하게 우세 |

결론:

```text
Prediction-error reward는 모든 환경에서 균일하게 큰 효과를 보이지는 않지만,
의존성이 강한 환경에서는 성공률 향상에 기여했다.
```

### ablation_3: Imagination rollout depth/branch

`v2_complex`:

| Setting | Success rate |
| --- | ---: |
| D2_B1 | 0.737 |
| D3_B1 | 0.723 |
| D2_B3 | 0.717 |
| D1_B1 | 0.703 |

`locked_bottleneck`:

| Setting | Success rate |
| --- | ---: |
| D3_B1 | 0.493 |
| D1_B1 | 0.470 |
| D2_B1 | 0.453 |
| D2_B3 | 0.423 |

`random_key_door`:

대부분 0.97 근처로 포화되어 차이가 작다.

결론:

```text
Rollout은 역할을 했다. 다만 성능 향상은 많은 후보를 넓게 펼치는
brute-force rollout이 아니라, 현재 행동이 다음 KK 슬롯을 열어주는지를
얕게 평가하는 dependency-aware rollout에서 나왔다.
```

즉:

```text
Rollout 있음 = 의미 있음
깊이 조금 있음 = 도움 됨
branch 많이 늘림 = 일관된 도움 없음
```

### 추가 ablation: mechanism/component 분석

더 풍부한 분석을 위해 다음 suite가 추가되었다.

`ablation_4_imagination_mechanisms`:

| Condition | 제거/변경 요소 | 질문 |
| --- | --- | --- |
| `A4_FULL_C3` | 없음 | 기준 C3 |
| `A4_NO_DEPENDENCY` | dependency bonus 제거 | 미래 KK 슬롯 enablement가 중요한가? |
| `A4_NO_REPEAT_PENALTY` | repeat penalty 제거 | 반복 감소가 penalty 때문인가? |
| `A4_NO_POLICY_PRIOR` | policy prior 제거 | Imagination 효과가 policy prior 때문인가? |
| `A4_NO_ROLLOUT_VALUE` | rollout value 제거 | depth-limited rollout 자체가 중요한가? |
| `A4_ONE_STEP_NO_DEP` | one-step, no dependency | 단순 one-step scoring과 비교 |

`ablation_5_prophecy_score_components`:

| Condition | 제거 요소 | 질문 |
| --- | --- | --- |
| `A5_FULL_C3` | 없음 | 기준 C3 |
| `A5_NO_KNOWLEDGE_GAIN` | predicted ΔK gain score 제거 | 지식 획득 예측이 중요한가? |
| `A5_NO_FLAG_PROB` | flag probability score 제거 | 목표 관련성 예측이 중요한가? |
| `A5_NO_ERROR_AVOIDANCE` | error avoidance score 제거 | 오류 회피 예측이 중요한가? |

이 추가 ablation들은 필수 최소 실험은 아니지만, 다음 질문에 방어적으로 답하기
위해 유용하다.

```text
성능 향상이 단순 점수 함수 때문인가,
아니면 knowledge-action dependency를 보는 구조 때문인가?
```

## 7. 논문에 넣을 수 있는 해석

### 영어 버전

```text
The ablation results show that APASSR's performance gain is not primarily caused
by a more complex neural Prophecy implementation. The lightweight table-based
Prophecy model was more stable than the Transformer variant in the tested
GridWorld settings. Prediction-error reward was most useful in dependency-heavy
environments, and shallow depth-limited Imagination was sufficient; increasing
the branching factor did not consistently improve performance.
```

```text
The rollout ablation suggests that shallow dependency-aware imagination improves
exploration efficiency, while increasing the branching factor does not
consistently improve performance. This indicates that the benefit comes from
evaluating whether a candidate action enables future knowledge-bound actions,
rather than from broad brute-force lookahead.
```

### 한국어 버전

```text
절제된 Table 기반 Prophecy와 얕은 dependency-aware Imagination이 현재
GridWorld 실험에서는 가장 안정적이었다. 이는 제안 방법의 핵심이 신경망
복잡도가 아니라, 지식 저장소와 행동 파라미터화, 예측, 상상 평가를
연결한 폐루프 구조에 있음을 보여준다.
```

```text
Rollout은 성능 향상에 기여했지만, 많은 미래 후보를 넓게 탐색하는 방식이
핵심은 아니었다. 오히려 현재 행동이 이후 행동 템플릿의 KK 슬롯을 채울
지식을 만들어내는지를 얕게 평가하는 구조가 중요했다.
```

## 8. 주의해야 할 주장

피해야 할 주장:

```text
이 방법은 항상 DQN보다 좋다.
GridWorld 결과가 곧바로 펜테스팅 성능을 증명한다.
현재 구현은 Transformer 기반 framework이다.
Imagination이 실제 환경을 여러 step 시뮬레이션한다.
```

안전한 주장:

```text
테스트한 v2_complex GridWorld에서는 C3가 QLEARN 및 DQN_PARTIAL보다 높은
성공률과 낮은 반복률을 보였다.
```

```text
GridWorld는 nmap 기반 펜테스팅 실험을 대체하는 것이 아니라, APASSR의
지식-행동 의존성 구조를 통제된 환경에서 검증하기 위한 추상 실험이다.
```

```text
현재 Imagination Cycle은 Prophecy 예측을 이용한 depth-limited candidate
evaluation이며, hidden map을 읽거나 실제 미래 행동을 실행하지 않는다.
```

## 9. 현재 판정

현재 상태는 다음과 같이 볼 수 있다.

```text
프레임워크 구현: 완료
C0-C5 조건 분리: 완료
APASSR_FULL/APASSR_FULL_CAL 조건 분리: 완료
QLEARN/DQN/ORACLE baseline: 완료
분석/그래프/보고서 자동화: 완료
ablation_1/2/3: 완료
환경 sweep: 완료
발표자료용 환경 렌더링: 완료
```

논문/발표에서 강조할 중심 주장은 다음이다.

```text
APASSR는 단순 policy 학습이 아니라,
지식을 행동 파라미터로 재사용하고,
Prophecy로 후보 행동의 지식 변화 가능성을 예측하며,
Imagination Cycle로 실행 전 후보를 평가하는
knowledge-parameterized closed-loop decision-making framework이다.
```
