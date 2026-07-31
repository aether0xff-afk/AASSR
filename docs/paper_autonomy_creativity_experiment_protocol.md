# AASSR v2 논문용 자율성·창의성 실험 프로토콜

## 0. 문서 목적

이 문서는 AASSR v2가 다음 두 연구 질문에 과학적으로 답할 수 있도록 실험을 분리하고, 구현·실행·통계·논문 작성 기준을 고정한다.

> **RQ1. 희소 보상 환경에서 에이전트가 정답 시범 없이 스스로 목표에 도달할 수 있는가?**

> **RQ2. 에이전트가 인간 및 기존 강화학습 기준선과 다른, 유효하고 유용한 해결 전략을 만들어낼 수 있는가?**

두 질문은 서로 다른 능력을 요구한다.

- RQ1은 **자율 발견과 학습 효율**에 관한 질문이다.
- RQ2는 **다중해 환경에서의 전략적 창의성**에 관한 질문이다.

따라서 하나의 환경과 하나의 지표로 두 질문을 동시에 검증하지 않는다. 자율 목표 달성 실험과 창의적 전략 실험을 별도 protocol로 수행한 뒤, 결과를 합쳐 해석한다.

이 문서는 구현 완료 보고서가 아니다. 현재 구현된 기능과 앞으로 구현해야 할 기능을 구분하며, 구현되지 않은 항목은 논문에서 완료된 것처럼 서술하지 않는다.

---

## 1. 핵심 용어와 주장 범위

### 1.1 자율 학습의 조작적 정의

본 연구에서 에이전트가 **자율적으로 학습했다**는 말은 다음 조건을 모두 만족한다는 뜻이다.

1. 정답 행동이나 정답 경로의 demonstration을 제공하지 않는다.
2. oracle transition이나 성공 경로로 Prophecy를 사전학습하지 않는다.
3. 중간 단계의 정답 여부를 외부 보상으로 알려주지 않는다.
4. 행동명과 상태 fact에 `safe`, `key`, `trap`, `finish` 같은 의미 단서를 넣지 않는다.
5. 최종 목표 달성 시에만 외부 성공 보상을 제공한다.
6. 에이전트는 자신이 직접 수집한 전이 경험으로 Policy와 Prophecy를 갱신한다.

단, 환경의 물리 규칙, 사용 가능한 행동 문법, 관측 형식, 최종 목표는 연구자가 제공한다. 따라서 본 연구는 “환경까지 스스로 창조하는 인공지능”을 주장하지 않는다.

### 1.2 창의적 전략의 조작적 정의

본 연구에서 **창의적 전략 후보**는 다음 네 조건을 모두 만족하는 성공 경로다.

1. **유효성(Validity)**: 실제 환경에서 최종 목표를 달성한다.
2. **새로움(Novelty)**: 인간 풀이와 기준선 에이전트 풀이의 알려진 전략 집합과 구조적으로 다르다.
3. **유용성(Utility)**: 단순히 다른 것에 그치지 않고 효율성, 안정성, 자원 사용, 일반성 중 하나 이상에서 의미 있는 이점을 가진다.
4. **재현성(Reproducibility)**: 우연히 한 번 나온 경로가 아니라 여러 seed 또는 동형 환경에서 반복된다.

다음은 창의성으로 인정하지 않는다.

- 실패한 특이 행동
- 행동 문자열만 다르고 인과 구조가 같은 경로
- 무작위 탐색 중 우연히 한 번 성공한 경로
- 인간 전략보다 다르지만 모든 효율·안정성 지표가 명백히 나쁜 경로
- 환경 제작자가 숨겨 둔 정답 경로를 단순히 암기한 결과

### 1.3 허용 가능한 최종 주장

RQ1의 기준을 만족하면 다음을 주장할 수 있다.

> AASSR v2는 불투명한 희소 보상 의존 환경에서 정답 시범 없이 반복 경험을 통해 성공 전략을 발견하였다.

RQ2의 기준을 만족하면 다음을 제한적으로 주장할 수 있다.

> AASSR v2는 다중해 환경에서 인간 및 기준선 전략과 구조적으로 다른 유효한 해결 경로를 생성했으며, 해당 경로는 하나 이상의 유용성 지표와 반복 재현성을 보였다.

다음 표현은 본 protocol만으로 주장하지 않는다.

- 인간보다 전반적으로 더 창의적이다.
- 모든 현실 환경에 일반화한다.
- 새로운 world의 정답을 아무 경험 없이 알아낸다.
- 인간의 환경 설계가 전혀 필요 없다.
- 실제 외부 시스템에서 자율 공격 능력을 입증했다.

---

## 2. 전체 실험 구조

논문용 실험은 다음 다섯 묶음으로 구성한다.

| 실험 | 목적 | 주 연구 질문 | 필수 여부 |
|---|---|---|---:|
| A. 자율 목표 달성 | 희소 보상에서 정답 시범 없이 성공 경로 발견 | RQ1 | 필수 |
| B. 모듈 기여도 | Prophecy, Imagination, 정보 가치의 실제 기여 분리 | RQ1 보강 | 필수 |
| C. 새로운 world 적응 | 암기와 구조 전이를 구분하고 few-shot 적응 측정 | RQ1 일반화 | 필수 |
| D. 창의적 전략 생성 | 인간·기준선과 다른 유효하고 유용한 다중해 전략 검증 | RQ2 | 필수 |
| E. 안전한 로컬 응용 | 지식-행동 의존 구조가 도구 사용 환경에서도 작동하는지 확인 | 응용 타당성 | 선택 |

실험 A의 높은 성공률만으로 실험 D의 창의성을 주장하지 않는다. Imagination을 사용했다는 사실만으로도 창의성을 주장하지 않는다.

---

## 3. 연구 가설

### H1. 자율 발견 가설

Full AASSR는 최종 희소 보상만 주어진 불투명 의존 환경에서 Random보다 높은 평가 성공률과 학습 곡선 AUC를 보인다.

### H2. 계획 효과 가설

Full AASSR는 동일한 실제 환경 상호작용 예산에서 `prophecy_no_imagination`보다 긴 의존 길이에서 높은 성공률 또는 빠른 최초 성공을 보인다.

### H3. 정보 가치 가설

Holdout으로 검증된 prediction improvement를 사용하는 조건은 이를 제거한 조건보다 학습 안정성, AUC 또는 오류율 중 하나 이상에서 개선된다.

### H4. 적응 전이 가설

여러 학습 world에서 형성된 효과 기반 표현과 Prophecy를 유지한 AASSR는 새로운 world에서 완전 초기화된 AASSR보다 적은 adaptation episode로 높은 성공률에 도달한다.

### H5. 전략 새로움 가설

다중해 환경에서 Full AASSR가 생성한 성공 전략 중 일부는 인간 풀이 집합 및 강화학습 기준선의 가장 가까운 전략보다 사전 정의된 구조 거리 임계값 이상 떨어져 있다.

### H6. 창의적 유용성 가설

새로운 AASSR 전략 후보는 인간 또는 기준선의 대표 전략과 비교했을 때 스텝 수, 오류율, 자원 소모, 위험도, 동형 환경 재사용 성능 중 하나 이상에서 통계적으로 또는 실질적으로 유의한 장점을 보인다.

가설이 지지되지 않으면 해당 주장을 포기하거나 제한한다. 실패 결과를 숨기지 않는다.

---

## 4. 공통 실험 원칙

### 4.1 Pilot과 Final 분리

- Pilot seed는 구현 오류 확인과 하이퍼파라미터 조정에만 사용한다.
- Final seed는 protocol 동결 이후 처음 공개하고, 결과를 본 뒤 변경하지 않는다.
- Pilot과 Final world seed 집합은 겹치지 않는다.
- Final 결과가 마음에 들지 않는다는 이유로 seed를 교체하지 않는다.

권장 규모:

- Pilot: 5 research seeds
- Final: 최소 20 research seeds, 가능하면 30 seeds

### 4.2 동일 예산 비교

모든 학습 알고리즘은 다음 예산을 동일하게 맞춘다.

- 실제 환경 transition 수
- train episode 수
- 평가 episode 수
- 행동 후보 수
- wall-clock 제한을 사용하는 경우 동일한 하드웨어 범주

Imagination 내부의 imagined transition은 실제 환경 transition과 분리해 보고한다. AASSR가 추가 계산을 사용했다면 계산 비용 표에 정직하게 표시한다.

### 4.3 Seed-first 통계

Episode를 독립 표본으로 세지 않는다.

1. 각 research seed 내부에서 episode 결과를 먼저 요약한다.
2. seed별 요약값을 통계 표본으로 사용한다.
3. 조건 비교는 같은 seed와 같은 world를 사용한 paired comparison으로 수행한다.

권장 보고:

- seed 평균
- 표준편차
- 95% bootstrap confidence interval
- paired mean difference
- paired permutation test 또는 Wilcoxon signed-rank test
- 여러 비교가 있을 때 Holm correction

p-value만 보고하지 않고 효과 크기와 신뢰구간을 중심으로 해석한다.

### 4.4 평가 중 학습 금지

`evaluation_seen`, `evaluation_unseen_zero_shot`에서는 다음을 모두 끈다.

- Policy update
- Prophecy update
- replay insertion
- epsilon exploration
- holdout set 변경
- Skill 생성 및 수정

`evaluation_unseen_adaptation`은 적응 구간과 평가 구간을 명시적으로 분리한다.

### 4.5 누수 방지

- 에이전트가 viable action, hidden solution family, 정답 branch ID에 접근하지 못하게 한다.
- 상태·행동 문자열에 의미 단서를 넣지 않는다.
- 인간 풀이 데이터는 평가용 비교에만 사용하고, 학습 입력으로 넣지 않는다.
- 환경 생성 코드와 분석 코드는 solution label을 별도 비공개 field로 유지한다.
- Final test template은 model selection에 사용하지 않는다.

---

## 5. 실험 A — 희소 보상 자율 목표 달성

### 5.1 목적

정답 시범과 중간 보상 없이 에이전트가 반복 경험만으로 성공 경로를 발견하는지 검증한다.

### 5.2 환경

기본 환경은 `OpaqueDependencyWorld(length=L)`을 사용한다.

- 각 단계에 의미 없는 행동 후보를 제공한다.
- 일부 행동은 성공 가능성을 보존하고 일부는 비가역 손상을 만든다.
- 중간 단계에서 외부 보상을 주지 않는다.
- 마지막 단계까지 손상 없이 도달한 경우에만 reward 1을 준다.
- 길이 `L = 4, 6, 8`을 사용해 의존 깊이를 변화시킨다.

현재 환경은 단일 성공 경로에 가까우므로 RQ1 검증에만 사용한다. 이 환경의 결과로 RQ2 창의성을 주장하지 않는다.

### 5.3 비교 조건

최소 조건:

1. Random
2. Contextual Policy
3. Q-learning
4. DQN 또는 부분관측 신경망 기준선
5. Prophecy without Imagination
6. Full AASSR

선택 조건:

- Oracle upper bound
- Tabular Prophecy Full AASSR
- GRU Prophecy Full AASSR

Oracle은 정보 접근 수준이 다른 상한선임을 명확히 표시하고 공정한 동급 baseline으로 해석하지 않는다.

### 5.4 주요 지표

- `evaluation_seen` 성공률
- 학습 곡선 AUC
- 최초 성공까지 실제 환경 transition 수
- 마지막 10% train 구간 성공률
- 목표 달성까지 평균 primitive step 수

보조 지표:

- 반복률
- 오류율
- prediction similarity
- holdout score와 holdout gain
- Imagination 사용 비율
- imagined node 수
- episode runtime

### 5.5 RQ1 성공 판정

다음 조건을 모두 만족해야 자율 목표 달성을 주장한다.

1. Full AASSR가 Random보다 평가 성공률 또는 AUC에서 명확히 높다.
2. 결과가 여러 seed에서 반복된다.
3. 정답 시범, oracle pretraining, 중간 외부 보상이 없음을 manifest로 검증한다.
4. 높은 성능이 특정 action string이나 단일 world seed 암기로만 설명되지 않는다.

Q-learning이나 DQN보다 높지 않아도 “자율 발견 자체”는 주장할 수 있지만, AASSR 구조의 우수성은 주장할 수 없다.

---

## 6. 실험 B — 모듈 기여도와 정밀 Ablation

### 6.1 목적

성능이 AASSR 전체 구조 때문인지, 단순 반복 억제나 오류 감점 때문인지 분리한다.

### 6.2 대표 환경

- 주 환경: `OpaqueDependencyWorld(length=6)`
- Final seed: 최소 20개
- 동일 train/eval 예산 사용

### 6.3 Ablation 조건

필수:

1. Contextual Policy only
2. + Prophecy
3. + Imagination
4. + Holdout-validated information value
5. Full AASSR
6. Full − repeat penalty
7. Full − error penalty
8. Full − validated information value

계획 민감도:

- imagination depth: 1, 2, 4, 6
- aggregation: max, mean, risk-adjusted
- branching factor: 1, 2, 4

### 6.4 해석 규칙

- Full과 `prophecy_no_imagination` 차이가 없으면 Imagination의 추가 이득을 주장하지 않는다.
- repeat/error penalty 제거가 가장 큰 하락을 만들면 이를 핵심 기여로 보고하고 숨기지 않는다.
- holdout gain이 0에 가깝고 제거 조건과 차이가 없으면 정보 가치 효과를 주장하지 않는다.
- 연산량이 크게 증가했지만 실제 interaction 효율이 개선되지 않으면 계산 비용 대비 이득이 없다고 기록한다.

---

## 7. 실험 C — 새로운 World의 Zero-shot 누수 검사와 Few-shot 적응

### 7.1 목적

고정 퍼즐 암기와 재사용 가능한 구조 학습을 구분한다.

### 7.2 세 평가 구간

#### A. `evaluation_seen`

학습에 사용한 world에서 기억과 숙련 성능을 평가한다.

#### B. `evaluation_unseen_zero_shot`

학습 중 보지 않은 행동 이름, 상태 표현, 정답 매핑의 world에 적응 없이 투입한다.

현재 opaque 환경에서 첫 행동을 구별할 정보가 없다면 zero-shot 성능은 random 수준일 수 있다. 따라서 이 구간은 일반화 우수성을 증명하기보다 **누수 검사용 음성 대조군**으로 사용한다.

#### C. `evaluation_unseen_adaptation`

새 world에서 제한된 학습 episode를 허용한 뒤, 학습을 다시 끄고 평가한다.

권장 adaptation budget:

```text
0, 1, 4, 16, 64 episodes
```

각 budget마다 별도 agent checkpoint에서 시작해 순서 효과를 방지한다.

### 7.3 비교 조건

1. From-scratch Contextual Policy
2. From-scratch Full AASSR
3. Policy reset + Prophecy retained
4. Policy reset + effect-based representation retained
5. Full transfer: 허용된 공유 표현과 Prophecy 유지

정확한 state/action ID table을 그대로 옮기는 것은 구조 전이가 아니라 암기 전이이므로 금지한다.

### 7.4 효과 기반 Relational Representation

새 action을 문자열 ID가 아니라 경험으로 얻은 효과 특징으로 표현한다.

후보 특징:

- 실행 횟수
- 오류율
- 평균 상태 변화량
- 새 fact 추가 및 제거 패턴
- 새 행동 unlock 여부
- corruption 또는 위험 지표 변화
- 목표 진행 변화
- prediction uncertainty
- 정보 획득량
- 다른 행동의 실행 가능성 변화

이 표현은 action의 진짜 역할을 직접 알려주는 oracle label이 아니라, 실제 실행에서 관측한 결과로만 계산해야 한다.

### 7.5 주요 지표

- adaptation success curve
- adaptation AUC
- 성공률 50% 또는 80% 도달까지 필요한 episode 수
- from-scratch 대비 sample saving
- 새 world prediction calibration
- retained representation의 transfer gain

### 7.6 구조 전이 성공 판정

다음이 만족될 때만 구조 전이를 주장한다.

1. zero-shot 결과가 누수 없이 예상 범위에 있다.
2. transfer 조건이 from-scratch보다 adaptation AUC에서 높다.
3. 효과가 여러 unseen world와 seed에서 반복된다.
4. 정확한 ID 암기나 seed 충돌로 설명되지 않는다.

seen만 높고 adaptation 차이가 없으면 “학습한 퍼즐 내 자율 발견”까지만 주장한다.

---

## 8. 실험 D — 창의적 전략 생성

### 8.1 목적

여러 성공 방법이 존재하는 환경에서 AASSR가 인간과 기준선이 주로 사용하는 경로와 다른, 유효하고 유용한 전략을 생성하는지 검증한다.

### 8.2 새 환경 요구사항

새 환경은 가칭 `MultiSolutionDependencyWorld` 또는 `CreativeEscapeWorld`로 구현한다.

필수 속성:

1. 하나의 최종 목표가 존재한다.
2. 서로 다른 인과 구조를 가진 성공 solution family가 최소 4개 존재한다.
3. 각 solution family 안에서도 행동 순서나 도구 조합의 변형이 가능하다.
4. 어떤 경로도 action 이름만으로 드러나지 않는다.
5. 한 경로가 모든 효율 지표에서 압도적인 유일 정답이 되지 않도록 trade-off를 둔다.
6. 환경 생성기는 solution family label을 분석용으로만 보관하고 agent에는 노출하지 않는다.
7. 인간이 미리 열거하지 않은 emergent combination이 성공할 수 있도록 규칙을 조합 가능하게 만든다.

### 8.3 권장 solution family 예시

GridWorld형 환경에서 다음처럼 서로 다른 원리를 사용할 수 있다.

- 열쇠를 획득해 문을 여는 경로
- 장애물을 이동시켜 우회 통로를 만드는 경로
- 긴 안전 경로로 잠긴 구역을 피하는 경로
- 두 도구를 조합해 장애물을 제거하는 경로
- 정보를 먼저 수집해 위험한 분기를 피하는 경로
- 기존 기술을 예상하지 않은 순서로 결합하는 경로

중요한 것은 정답 경로 목록을 agent에 제공하지 않는 것이다. 환경은 primitive rule만 제공하고, 전략은 agent가 행동 조합을 통해 형성한다.

### 8.4 인간 비교 데이터

권장 참가자:

- Pilot: 5명 내외
- Final: 10~20명

참가자에게 agent와 동일한 목표와 가능한 primitive action 설명을 제공한다. 숨겨진 정답 경로나 solution family 수는 알려주지 않는다.

수집 항목:

- 전체 행동 로그
- 성공 여부
- 스텝 수
- 오류 및 재시도
- 사용 자원
- 참가자가 설명한 전략 의도

개인 식별 정보는 저장하지 않고 익명 participant ID만 사용한다. 학교 연구 규정, 지도교사 승인, 참가자 동의가 필요한 경우 이를 먼저 충족한다.

### 8.5 기준선

- Random
- Q-learning
- DQN 또는 PPO 계열 기준선
- novelty bonus가 없는 AASSR
- Imagination이 없는 AASSR
- Full AASSR

가능하면 단순 novelty-search 기준선도 추가해 “새롭기만 한 행동”과 “새롭고 유용한 전략”을 구분한다.

### 8.6 전략의 정규화 표현

명령 문자열이 아니라 **인과 효과 그래프(causal effect graph)**로 전략을 표현한다.

노드 예시:

- 정보 획득
- 자원 획득
- 행동 가능성 unlock
- 장애물 제거
- 위험 감소
- 위치 또는 상태 전환
- 목표 달성

간선 예시:

- 어떤 결과가 다음 행동의 전제 조건이 됨
- 한 행동이 다른 행동을 가능하게 함
- 특정 정보가 parameter 선택에 사용됨

동일한 인과 구조를 가진 경로는 action 이름이나 사소한 순서가 달라도 같은 전략 family로 묶는다.

### 8.7 새로움 측정

한 AASSR 전략 `s`의 새로움은 인간 및 기준선 reference 전략 집합에서 가장 가까운 전략과의 거리로 측정한다.

후보 거리:

- 정규화 graph edit distance
- effect motif 집합의 Jaccard distance
- prerequisite edge 차이
- solution family 차이
- 행동 효과 sequence의 edit distance

하나의 거리만으로 결론내지 않고 최소 두 종류의 구조 거리로 교차 확인한다.

새로움 임계값은 Final 결과를 보기 전에 Pilot에서 고정한다.

### 8.8 유용성 측정

다음 지표를 개별적으로 보고한다.

- 성공률
- primitive step 수
- 오류 수
- 자원 소모
- 위험 상태 진입 횟수
- wall-clock 및 실제 환경 interaction 수
- 동형 환경에서의 재사용 성공률

창의성의 주 결론은 단일 가중합 점수보다 다차원 결과표를 기준으로 한다. 종합 creativity score를 만들더라도 보조 지표로만 사용하며, 가중치는 Final 전에 고정한다.

### 8.9 재현성 측정

전략 후보가 다음 중 하나를 만족해야 재현 가능한 것으로 본다.

- 서로 다른 3개 이상의 research seed에서 같은 effect-graph family가 발견됨
- 서로 다른 3개 이상의 동형 environment instance에서 성공함
- 저장된 strategy/skill을 새 instance에 적용했을 때 성공률이 사전 임계값 이상임

한 번만 나온 경로는 “흥미로운 사례”로 보고할 수 있지만 창의성의 핵심 증거로 사용하지 않는다.

### 8.10 블라인드 인간 평가

자동 거리 지표와 별도로 블라인드 평가를 수행한다.

평가자는 전략이 인간인지 AASSR인지 알지 못한 상태에서 다음을 5점 척도로 평가한다.

- 새로움
- 유용성
- 논리성
- 예상 밖의 정도

최소 2명의 평가자를 사용하고 평가자 간 일치도를 보고한다. 자동 지표와 인간 평가가 충돌하면 두 결과를 모두 공개하고 원인을 분석한다.

### 8.11 RQ2 성공 판정

다음을 모두 만족해야 “창의적 전략 후보를 생성했다”고 주장한다.

1. 전략이 실제로 성공한다.
2. 인간·기준선 reference 집합과 구조적으로 다르다.
3. 하나 이상의 유용성 지표에서 열등하지 않거나 의미 있는 장점이 있다.
4. 여러 seed 또는 환경에서 재현된다.
5. 블라인드 평가에서 새로움과 논리성이 확인된다.

Full AASSR가 다양한 경로를 만들었지만 유용하지 않으면 “전략 다양성 증가”까지만 주장한다. 인간과 다른 경로가 없으면 창의성 가설은 기각한다.

---

## 9. 실험 E — 선택적 안전한 로컬 응용 검증

펜테스팅 동기를 유지하려면 외부 시스템이 아닌 명시적으로 허가된 로컬 Docker/CTF 환경을 사용한다.

환경은 다음 지식-행동 의존 구조를 포함할 수 있다.

```text
관측 또는 정찰
→ 서비스·경로·설정 정보 획득
→ 획득한 정보를 후속 행동의 parameter로 사용
→ 여러 가능한 안전한 solution family
→ 로컬 FLAG 획득
```

변형 요소:

- 포트 및 서비스 배치
- 가짜 서비스와 distractor
- 경로 또는 설정 위치
- 필요한 정보 조합
- 행동 순서 trade-off

실제 외부 네트워크, 허가되지 않은 시스템, 실제 자격 증명은 사용하지 않는다. 이 실험의 목적은 실제 공격 성능이 아니라, 통제된 도구 사용 환경에서 지식-행동 결합과 다중 전략 형성이 재현되는지 확인하는 것이다.

---

## 10. 논문용 결과물과 자동 분석

### 10.1 권장 설정 파일

```text
configs/
├─ paper_autonomy_pilot_v1.json
├─ paper_autonomy_final_v1.json
├─ paper_ablation_final_v1.json
├─ paper_transfer_pilot_v1.json
├─ paper_transfer_final_v1.json
├─ paper_creativity_pilot_v1.json
└─ paper_creativity_final_v1.json
```

### 10.2 권장 실행·분석 스크립트

```text
scripts/
├─ run_paper_suite.py
├─ analyze_paper_results.py
├─ make_paper_tables.py
├─ make_paper_figures.py
└─ validate_paper_artifacts.py
```

### 10.3 출력 구조

```text
paper_results/<protocol_version>/
├─ raw/
│  ├─ episodes.csv
│  ├─ transitions.jsonl
│  ├─ strategies.jsonl
│  └─ human_paths.jsonl
├─ seed_level/
├─ statistics/
├─ tables/
├─ figures/
├─ manifests/
└─ report.md
```

### 10.4 필수 manifest

각 실행마다 다음을 저장한다.

- Git commit SHA
- protocol version
- config 원본과 SHA256
- 전체 research seed와 world seed 목록
- Python, PyTorch, CUDA 버전
- CPU, GPU, RAM 정보
- worker 수와 device 설정
- 시작·종료 시각
- 중단·실패·제외 run 목록과 사유
- train/eval/adaptation phase 정의
- human study dataset version

---

## 11. 논문 표와 그림 계획

### Figure 1. AASSR 전체 폐루프 구조

Observation → Knowledge → Policy/Prophecy → Imagination → Action → Actual transition → Holdout-validated update

### Figure 2. 실험 protocol

Pilot/Final 분리, train world, seen evaluation, unseen zero-shot, unseen adaptation, creativity human comparison을 한 그림에 표시한다.

### Figure 3. 자율 학습 곡선

의존 길이별 성공률 또는 cumulative success와 seed 95% CI를 표시한다.

### Figure 4. Seen/Adaptation 성능

seen 성공률과 unseen adaptation AUC를 분리해 표시한다. 서로 섞어 평균내지 않는다.

### Figure 5. Ablation forest plot

Full 대비 각 모듈 제거 효과와 95% CI를 표시한다.

### Figure 6. 창의성 전략 지도

인간·기준선·AASSR 전략의 effect-graph cluster 또는 거리 분포를 시각화한다.

### Table 1. 실험 protocol

환경, seed, world 수, episode, 보상, 행동 수, 평가 구간을 정리한다.

### Table 2. 자율성 메인 결과

성공률, AUC, 최초 성공, 반복률, 오류율, runtime을 평균과 95% CI로 보고한다.

### Table 3. 구조 전이 결과

adaptation budget별 성공률과 from-scratch 대비 sample saving을 보고한다.

### Table 4. 창의성 결과

유효성, 인간과의 최소 구조 거리, 유용성 지표, 재현 seed 수, 블라인드 평가를 전략별로 보고한다.

---

## 12. 결과별 주장 결정표

| 관찰 결과 | 허용되는 결론 |
|---|---|
| Seen 성공률만 높음 | 학습한 희소 보상 world에서 자율 발견과 기억에 성공 |
| Seen 높음, unseen zero-shot random 수준, adaptation 이득 없음 | 구조 전이는 검증되지 않음 |
| Transfer 조건의 adaptation AUC가 from-scratch보다 높음 | 새로운 world에서 few-shot 구조 전이가 관찰됨 |
| 다양한 성공 경로가 있지만 인간 전략과 구조적으로 같음 | 표현 다양성 또는 순서 다양성만 증가 |
| 인간과 다른 성공 경로지만 유용성·재현성 없음 | 흥미로운 novel 사례, 창의성 핵심 증거로는 부족 |
| 인간과 다른 유효·유용·재현 가능한 전략 | 제한적 의미의 창의적 전략 생성 지지 |
| Full과 no-imagination 차이 없음 | 해당 환경에서 Imagination 추가 효과 미검증 |
| repeat/error penalty 제거가 가장 큰 성능 하락 | 성능 주도 요인은 반복 억제·오류 회피임을 명시 |

---

## 13. 구현 우선순위

### P0 — 논문 신뢰성 기반

- 기존 자율 실험 회귀 테스트 유지
- train/seen/unseen seed 누수 검사
- evaluation 중 model mutation 금지 테스트
- manifest와 config hash 저장

### P1 — RQ1 완성

- Q-learning, DQN을 동일 protocol에 연결
- 논문용 autonomy config와 분석 스크립트
- seed-first 통계와 자동 표·그림 생성

### P2 — 구조 전이

- `evaluation_unseen_adaptation` phase
- checkpoint별 adaptation budget 평가
- effect-based relational representation
- from-scratch와 transfer 조건 분리

### P3 — 창의성 환경

- `MultiSolutionDependencyWorld`
- 최소 4개의 solution family와 trade-off
- strategy log와 causal effect graph 추출
- strategy canonicalization 및 거리 분석

### P4 — 인간 비교

- 익명 human path 수집 형식
- 블라인드 평가 UI 또는 worksheet
- 평가자 일치도와 자동 지표 비교

### P5 — 선택적 응용

- 허가된 로컬 CTF/Docker 다중해 환경
- 안전성과 재현성 manifest

P0~P3가 완료되기 전에는 Final paper experiment를 시작하지 않는다. Pilot 결과를 보고 P4의 평가 양식을 조정할 수 있지만, Final creativity threshold와 분석 규칙은 Final 실행 전에 동결한다.

---

## 14. 실행 가속 원칙

- 환경 시뮬레이션과 Tabular 조건은 CPU process pool을 사용한다.
- PyTorch GRU 조건은 CUDA를 사용한다.
- 단일 RTX 4090에서는 `cuda_workers=1`을 기본값으로 한다.
- CPU와 CUDA job pool은 동시에 실행할 수 있다.
- CUDA 사용 여부가 실제 환경 interaction budget을 바꾸지 않게 한다.
- wall-clock time과 algorithmic sample efficiency를 분리해 보고한다.

CUDA는 연구 가설이 아니라 실험 가속 수단이다. GPU 사용으로 더 많은 환경 경험을 제공했다면 공정 비교가 아니므로 금지한다.

---

## 15. Protocol 동결 체크리스트

Final 실행 전에 아래를 모두 확인한다.

- [ ] 연구 질문과 가설이 문서에 고정됨
- [ ] Pilot seed와 Final seed가 분리됨
- [ ] 모든 baseline이 동일 interaction budget을 사용함
- [ ] train, seen, zero-shot, adaptation world가 분리됨
- [ ] evaluation 중 학습이 꺼짐
- [ ] 창의성 environment에 최소 4개 solution family가 존재함
- [ ] 인간 풀이가 agent 학습에 유입되지 않음
- [ ] strategy canonicalization 규칙이 고정됨
- [ ] novelty threshold와 utility 기준이 고정됨
- [ ] 통계 검정과 correction 방법이 고정됨
- [ ] config hash, commit SHA, 하드웨어 manifest가 저장됨
- [ ] 자동 표·그림 생성이 raw data에서 재현됨
- [ ] 실패 run 처리 규칙이 고정됨

---

## 16. 최종 연구 해석

이 protocol의 목표는 “AASSR가 무조건 창의적임을 증명”하는 것이 아니다. 다음 경계를 구분하는 것이 목표다.

```text
희소 보상에서 우연히 성공
        ↓
반복 경험으로 성공 경로 학습
        ↓
새 world에 더 빠르게 적응
        ↓
여러 성공 방법 중 새로운 전략 형성
        ↓
새롭고 유용하며 재현 가능한 전략 형성
```

RQ1은 두 번째 단계까지를 검증한다. RQ2는 마지막 두 단계를 검증한다.

결과가 어느 단계에서 멈추는지 정직하게 보고하면, 성공 결과뿐 아니라 제한과 실패 결과도 AASSR의 실제 능력을 규명하는 연구 성과가 된다.
