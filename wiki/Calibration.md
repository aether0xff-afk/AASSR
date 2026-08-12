# Calibration

Calibration은 AASSR에서 **Prophecy가 낸 미래 예측을 실제 planning에 얼마나 믿고 써도 되는지** 측정하는 계층이다.

핵심은 probability와 reliability를 분리하는 것이다.

```text
outcome probability
= 환경에서 그 결과가 나올 확률

prediction reliability
= world model의 그 예측을 믿을 수 있는 정도
```

> [!IMPORTANT]
> 현재 manifest 계약: `semantic-probability-holdout-calibration-v3-status-aware`  
> 관련 구현: `src/aassr_v2/current_semantic_calibration.py`, `current_confidence_gate.py`

---

# 1. 연구 질문

> **world model의 평균 성능이 높아 보여도 실제 행동 결정에 중요한 오류가 숨어 있을 수 있는데, 어떤 예측을 planner가 믿어도 되는지 real holdout 경험으로 판단할 수 있는가?**

AASSR의 Imagination은 model error에 직접 노출된다.

따라서 "예측을 할 수 있는가?"와 "그 예측을 지금 믿어도 되는가?"는 다른 질문이다.

---

# 2. 왜 단순 confidence 하나로 부족한가?

예를 들어 Prophecy가 다음 outcome을 낸다고 하자.

```text
200 : probability 0.7
403 : probability 0.2
429 : probability 0.1
```

이 숫자는 **환경 outcome mass**에 대한 예측이다.

하지만 모델 자체가 이 state/action region을 거의 학습하지 않았다면 세 확률 모두 신뢰하기 어려울 수 있다.

따라서 별도로

```text
reliability = 0.2
```

같은 신뢰성 판단이 필요하다.

---

# 3. Holdout calibration

현재 calibration은 real replay에서 분리된 holdout transition을 사용한다.

개념적으로:

```text
real holdout transitions
(S,A,S')
      |
      v
Prophecy predicts S'
      |
      v
semantic correctness 측정
      |
      v
state/action region reliability
```

같은 relational action key에 해당하는 충분한 holdout sample이 없으면 calibration은 보수적으로 낮게 유지된다.

즉 데이터 부족을 근거 없는 높은 confidence로 바꾸지 않는다.

---

# 4. Probability-weighted semantic score

Stochastic Prophecy는 여러 outcome을 낼 수 있다.

따라서 실제 다음 상태와의 semantic correctness를 계산할 때 각 branch를 동일 가중치로 보는 대신 model이 부여한 outcome probability mass를 반영한다.

개념적으로:

```math
C(S,A,S') = \sum_i p_i \; score(\hat S'_i, S')
```

여기서:

- `p_i`: predicted outcome probability
- `score`: predicted semantic state와 actual next state의 일치도

이 값은 "가장 가까운 branch 하나만 맞으면 성공"이라는 지나치게 낙관적인 평가를 줄인다.

---

# 5. Frozen holdout

Evaluation과 비교 과정에서 calibration 기준 데이터가 계속 바뀌면 결과 해석이 어려워질 수 있다.

현재 구현은 holdout을 freeze할 수 있다.

```text
training / validation data 준비
        ↓
calibration holdout freeze
        ↓
frozen reliability 기준으로 evaluation
```

이것은 같은 checkpoint 비교에서 reliability 기준까지 흔들리는 것을 줄이기 위한 장치다.

---

# 6. 왜 status-aware calibration이 필요한가?

2026-08-11 2k diagnostic에서 중요한 문제가 드러났다.

당시 semantic prediction quality와 terminal match는 겉으로 높아 보였지만 Imagination이 실제로 바꾼 행동 상당수가 `403/404/429` 오류로 이어졌다.

즉:

```text
전체 semantic similarity 높음
!=
decision-critical outcome을 충분히 잘 예측함
```

이다.

그래서 current-generation은 latest HTTP status를 public state와 Prophecy supervision에 보존하고 calibration에서도 status-aware semantic quality를 본다.

---

# 7. Reliability는 value가 아니다

current-generation에서 매우 중요한 원칙이다.

잘못된 해석:

```text
Critic value 0.4
confidence 0.9
-> 0.4 + 0.9 bonus
```

이렇게 하면 confidence가 큰 branch가 실제 task return과 무관하게 더 좋은 행동처럼 보일 수 있다.

현재 원칙:

```text
reliability 충분?
  yes -> Critic value 비교 허용
  no  -> branch / override 제한
```

즉 confidence는 **gate**이지 **reward/value bonus**가 아니다.

---

# 8. Critic에서도 confidence를 제거하는 이유

과거 구조에서는 Critic input feature에 Prophecy confidence가 들어가면 네트워크가 confidence를 value signal처럼 사용할 가능성이 있었다.

current confidence gate는 기존 input shape는 유지하면서 그 scalar slot을 상수로 중립화한다.

```text
network shape 유지
+
confidence -> constant
```

따라서 branch ranking은 Critic의 sparse-return estimate로 이루어지고, confidence는 reliability eligibility에만 사용된다.

---

# 9. Global coverage gate

현재 state에서 legal actions 전체에 대해 Prophecy가 충분히 신뢰 가능한 prediction coverage를 갖는지 먼저 확인한다.

```text
coverage < threshold
-> Imagination eligible = false
-> Policy action 유지
```

이는 planner가 거의 모르는 상태에서 무리하게 전체 행동을 재평가하는 것을 막는다.

---

# 10. Per-root reliability gate

Global coverage가 충분해도 특정 root action prediction은 낮은 confidence일 수 있다.

그래서 각 root에도 reliability를 확인한다.

```text
reliable roots
-> Critic-only ranking 후보

unreliable roots
-> final override 후보에서 제외
```

특히 **Policy가 원래 선택한 root 자체의 prediction도 신뢰 가능해야 한다.**

그렇지 않으면 imagined alternative와 Policy baseline을 apples-to-apples로 비교할 수 없다.

---

# 11. 왜 Policy branch reliability가 필요한가?

예를 들어:

```text
Policy action value prediction : unreliable
Alternative value prediction   : reliable
```

이라면 두 값을 빼서

```text
advantage = V_alt - V_policy
```

라고 부르기 어렵다.

그래서 current gate는 Policy branch가 평가되지 않았거나 prediction confidence가 낮으면 override를 fail-closed 한다.

---

# 12. Calibration과 local Critic support의 차이

둘 다 "믿을 수 있는가?"를 묻지만 대상이 다르다.

```text
Calibration
= Prophecy의 predicted transition을 믿을 수 있는가?

Local Critic support
= 그 predicted/current region에서 Critic value를 믿을 실제 training support가 있는가?
```

둘 중 하나만 통과해서는 충분하지 않다.

예:

```text
Prophecy 정확함
+
Critic OOD
-> 잘 예측한 미래를 잘못 평가할 수 있음
```

반대도 가능하다.

```text
Critic supported
+
Prophecy 오류
-> 잘못된 미래를 정확하게 평가하는 셈
```

---

# 13. Calibration sample 부족

현재 calibration은 동일 relational action region의 holdout sample이 최소 수에 못 미치면 reliability를 `0`으로 두는 보수적 경로를 갖는다.

이 설계의 의미는:

```text
데이터 없음
!=
문제 없음
```

이다.

작은 transition budget에서는 이 때문에 Imagination intervention이 거의 발생하지 않을 수 있다.

하지만 그것은 무근거 extrapolation보다 방법론적으로 더 명확한 실패 모드다.

---

# 14. Cache와 refresh

Calibration을 매 decision마다 holdout 전체에 대해 다시 계산하면 비용이 크다.

현재 구현은

- relational action key
- holdout sample count 구간
- Prophecy gradient revision 구간

을 포함한 cache key를 사용해 reliability 계산을 재사용한다.

일정 sample/gradient stride를 넘으면 갱신한다.

이 최적화는 calibration 의미를 바꾸기보다 같은 계산의 반복 비용을 줄이기 위한 것이다.

---

# 15. Calibration이 직접 해결하지 않는 것

Calibration은 만능 안전장치가 아니다.

다음을 직접 해결하지 않는다.

- Policy 자체의 OOD
- Critic 자체의 OOD
- 아주 희귀한 status의 데이터 부족
- structural representation에서 이미 버린 중요한 정보
- planning depth에서 누적되는 model error 전체

그래서 AASSR은 calibration을 다른 gate와 조합한다.

---

# 16. 실패 모드

## 16.1 평균 metric blind spot

전체 semantic score는 높지만 중요한 failure/status channel이 틀릴 수 있다.

대응: status-aware metric, downstream intervention audit.

## 16.2 Sparse holdout

특정 action region의 holdout이 너무 적어 신뢰도 판단 불가능.

대응: fail-closed + real transition coverage 확대.

## 16.3 Confidence as value leakage

신뢰도가 value에 직접 섞이면 high-confidence branch를 좋은 branch로 오해할 수 있다.

대응: confidence-independent Critic encoding + reliability-only gate.

## 16.4 Stale reliability

모델이 크게 업데이트됐는데 calibration cache가 너무 오래 유지되면 현재 모델과 mismatch가 생긴다.

대응: model revision 기반 refresh.

---

# 17. 연구 가설

```text
H1. holdout semantic score가 실제 prediction correctness와 상관있는가?
H2. status-aware calibration이 decision-critical error를 더 잘 잡는가?
H3. low-reliability roots를 제거하면 intervention error가 감소하는가?
H4. reliability-only 설계가 confidence-value leakage보다 안정적인가?
H5. 너무 보수적이어서 useful intervention까지 전부 막지는 않는가?
```

마지막 질문이 중요하다.

Calibration은 intervention을 줄이는 것 자체가 목적이 아니라 **틀린 override를 줄이면서 유효한 override는 남기는 것**이 목적이다.

---

# 18. 관련 코드

```text
src/aassr_v2/current_semantic_calibration.py
  - probability_weighted_semantic_score
  - SemanticCalibratedProphecy

src/aassr_v2/current_confidence_gate.py
  - reliability-only decision gate
  - confidence-independent Critic encoding
```

---

다음으로 읽기:

- **[Prophecy](Prophecy)**
- **[Critic](Critic)**
- **[Imagination](Imagination)**
- **[Experiments](Experiments)**
