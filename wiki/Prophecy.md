# Prophecy

Prophecy는 AASSR의 **확률적 world model**이다.

목표는 단순하다.

> **현재 공개 상태에서 어떤 행동을 했을 때 다음에 어떤 공개 결과들이 얼마나 가능한지 예측한다.**

하지만 current-generation의 Prophecy는 단순한 `(S,A) -> S'` 회귀 모델이 아니다. 부분 관측, stochastic outcome, HTTP status, legal action surface, terminal class까지 함께 다루는 **relational conditional-mixture ensemble**이다.

> [!IMPORTANT]
> 현재 source of truth: `src/aassr_v2/current_manifest.py`  
> 핵심 구현: `src/aassr_v2/current_relational_mixture_model.py`

---

# 1. 연구 질문

> **미래의 public outcome 분포를 학습하면 희소 보상 환경에서 실제 행동 전에 더 나은 의사결정을 할 수 있는가?**

Prophecy 자체의 목표는 행동을 직접 선택하는 것이 아니다.

```text
Policy       = 지금 무엇을 할까?
Prophecy     = 그 행동을 하면 어떤 일이 일어날까?
Imagination  = 여러 미래를 이어 보면 어떤 행동이 더 나을까?
```

---

# 2. 왜 world model이 필요한가?

Model-free Policy는 현재 상태에서 행동 가치를 직접 학습한다.

```text
Q(S,A)
```

하지만 sparse reward에서는 행동 직후 reward가 대부분 `0`이기 때문에 장기 차이가 매우 늦게 드러날 수 있다.

Prophecy를 사용하면 실제 행동 전에 다음 질문을 계산할 수 있다.

```text
A를 하면 다음 상태는?
그 상태에서 legal action은?
실패 확률은?
다음 단계에서는 무엇을 할 수 있는가?
```

이 예측을 여러 단계 연결하는 것이 Imagination이다.

---

# 3. 입력

개념적으로 Prophecy의 입력은 다음과 같다.

```text
X_t = [R_t, A_t, K_t]
```

여기서:

- `R_t`: relational public state
- `A_t`: relational action representation
- `K_t`: 행동 전에 이미 획득한 episode-local Knowledge context

중요한 점은 concrete identifier 자체를 주요 lookup key로 사용하지 않는다는 것이다.

예:

```text
route-12
route-31
```

이름 자체보다

```text
catalog-like role
login-like role
object-like role
```

같은 관계적 역할이 transfer representation에 들어간다.

---

# 4. 출력

현재 Prophecy는 단순히 다음 벡터 하나만 예측하지 않는다.

각 mixture component는 개념적으로 다음 정보를 가진다.

```text
next relational descriptor
latest public HTTP status
legal action mask
terminal class
mixture probability
```

terminal class는 다음과 같이 분리한다.

```text
active
success
true failure
truncation
```

이 분리는 중요하다.

`true failure`와 `transition-cap` 또는 `rate-limit truncation`을 같은 종료로 취급하면 sparse return 의미가 달라진다.

---

# 5. 왜 deterministic prediction이 아닌가?

초기 형태의 world model을 다음처럼 생각할 수 있다.

```text
f(S,A) = S'
```

하지만 partial observability가 있는 환경에서는 같은 public `(S,A)`에서도 hidden condition에 따라 여러 outcome이 가능하다.

```text
(S,A)
  |-- 0.6 --> S1'
  |-- 0.3 --> S2'
  `-- 0.1 --> S3'
```

만약 이들을 하나의 평균으로 회귀하면 실제로 존재하지 않는 상태를 만들 수 있다.

```text
possible state A
possible state B
       |
       v
mean state C   <- 실제로는 존재하지 않을 수 있음
```

그래서 current-generation은 conditional mixture를 사용한다.

---

# 6. Mixture formulation

개념적으로 다음 분포를 근사한다.

```math
p(S_{t+1} \mid S_t, A_t, K_t)
= \sum_{m=1}^{M} \pi_m(X_t)\,p_m(S_{t+1}\mid X_t)
```

여기서:

- `M`: mixture component 수
- `pi_m`: 각 outcome mode의 probability mass
- `p_m`: 해당 mode가 예측하는 next-state distribution

Planner에서 중요한 것은 **각 branch가 단순 후보가 아니라 probability mass를 가진다**는 점이다.

---

# 7. Outcome probability와 reliability는 다르다

AASSR에서 자주 혼동하면 안 되는 두 값이다.

## Outcome probability

```text
이 결과가 환경에서 실제로 발생할 확률은 얼마인가?
```

예:

```text
200 outcome : 0.70
403 outcome : 0.20
429 outcome : 0.10
```

Chance node expectation에 사용된다.

## Prediction reliability

```text
이 world-model prediction 자체를 얼마나 믿을 수 있는가?
```

Calibration이 추정한다.

따라서

```text
high probability != high reliability
high reliability != good outcome
```

이다.

---

# 8. HTTP status를 왜 명시적으로 예측하는가?

2026-08-11의 repaired Imagination 2k 진단에서 중요한 실패가 발견됐다.

당시 semantic prediction metric은 꽤 높았지만, 실제 Imagination intervention은 많은 경우 `403/404/429`로 이어졌다.

원인 중 하나는 이전 relational representation이 **latest public HTTP status를 충분히 보존하지 않았다는 것**이었다.

이후 current-generation에서는 status를 public categorical channel로 명시적으로 보존하고 예측한다.

현재 대표 status vocabulary:

```text
200
302
400
401
403
404
409
429
```

---

# 9. 왜 categorical status objective인가?

HTTP status는 순서형 연속값이 아니다.

예를 들어 숫자 거리만 보면

```text
403 <-> 404
```

가 가깝지만 의미상 반드시 비슷한 outcome이라고 볼 수 없다.

따라서 status는 서로 배타적인 categorical public outcome으로 학습한다.

개념적 loss:

```math
L_{status} = -\sum_c w_c y_c \log \hat{p}_c
```

여기서 `w_c`는 class imbalance를 보정한다.

중요한 점은 사람이

```text
403 = 위험
429 = 나쁨
```

같은 의미 규칙을 직접 넣는 것이 아니라, **희귀 class가 학습에서 묻히지 않도록 frequency balance만 적용한다**는 것이다.

---

# 10. Legal action mask prediction

다음 상태 자체만 맞고 그 상태에서 가능한 행동 집합을 틀리면 planner가 존재하지 않는 행동을 상상할 수 있다.

그래서 Prophecy는 next-state의 legal action mask도 예측한다.

```text
predicted next state
+
predicted legal action surface
```

Imagination은 이 예측된 action mask를 사용해 다음 decision node의 후보 행동을 구성한다.

---

# 11. Terminal class prediction

다음 상태가 다음 중 무엇인지 구분해야 한다.

```text
active
success
true failure
truncation
```

왜냐하면 return 의미가 다르기 때문이다.

```text
success       -> +1
true failure  -> -1
truncation    ->  0
```

`failure`와 `truncation`을 합치면 Critic target과 planner value가 왜곡될 수 있다.

---

# 12. Calibration

좋은 world model은 단순히 평균 accuracy가 높은 모델이 아니다.

Planner가 필요한 것은 **decision-critical prediction이 언제 틀릴 가능성이 높은지 아는 것**이다.

현재 calibration은 holdout real transitions를 사용한다.

평가 대상에는 다음이 포함된다.

- relational semantic next-state quality
- legal-action-mask accuracy
- terminal-class accuracy
- HTTP-status accuracy
- probability-weighted semantic quality

Calibration 결과는 행동 value bonus로 더하지 않는다.

```text
reliable prediction
-> planner 사용 허용

unreliable prediction
-> Policy 유지 쪽으로 fail closed
```

---

# 13. Ensemble을 쓰는 이유

현재 manifest의 Prophecy는 ensemble 형태다.

Ensemble은 서로 다른 모델 prediction의 일치/불일치 패턴을 통해 uncertainty를 더 안정적으로 추정할 수 있게 한다.

단, ensemble disagreement 자체가 곧 environment stochasticity라는 뜻은 아니다.

두 종류를 구분해야 한다.

```text
Aleatoric uncertainty
= 환경 자체의 여러 가능한 outcome

Epistemic uncertainty
= 모델이 충분히 학습하지 못해 생기는 불확실성
```

Mixture는 주로 전자를 표현하고, ensemble/calibration은 후자의 신뢰도 판단에 도움을 준다.

---

# 14. 학습 데이터

Prophecy의 사실 근거는 **real transition**이다.

```text
real S_t
real A_t
real S_{t+1}
```

Imagined transition을 정답 데이터처럼 다시 Prophecy training target으로 사용하는 구조가 아니다.

이 원칙은 model hallucination이 자기 자신을 학습시키며 증폭되는 것을 피하기 위해 중요하다.

---

# 15. Knowledge leakage 방지

Prophecy input의 Knowledge는 반드시 행동 전에 이미 알고 있던 정보여야 한다.

잘못된 예:

```text
A 실행
-> response에서 token 획득
-> 그 token을 A 실행 전 prediction input에 사용
```

이것은 hindsight leak이다.

올바른 시간 순서:

```text
K_t
 ↓
predict(S_t, A_t, K_t)
 ↓
execute A_t
 ↓
observe response
 ↓
K_{t+1}
```

---

# 16. Planner와의 연결

Prophecy가 만든 branch는 Imagination에서 chance outcome으로 처리된다.

```text
root action A
   |
   |-- p1 --> predicted state 1
   |-- p2 --> predicted state 2
   `-- p3 --> predicted state 3
```

각 predicted state에서는 새로운 agent decision이 가능하다.

따라서 planner는

```text
chance expectation
+
decision max
```

를 번갈아 적용한다.

자세한 내용: **[Imagination](Imagination)**

---

# 17. Prophecy를 어떻게 평가하는가?

단일 MSE만으로는 충분하지 않다.

현재 중요한 지표:

- semantic top-k quality
- probability-weighted semantic quality
- legal-mask accuracy
- terminal accuracy
- HTTP status accuracy
- calibration reliability
- rare-status coverage
- mixture multimodality preservation

그리고 가장 중요한 downstream 질문:

> **Prophecy가 정확해졌을 때 실제 Imagination intervention 품질도 좋아지는가?**

world-model metric과 agent success를 따로 봐야 한다.

---

# 18. 알려진 실패 패턴

## 18.1 평균 semantic score는 높은데 중요한 status를 틀림

과거 2k 진단에서 실제로 관측됐다.

대응:

- latest HTTP status 보존
- categorical status supervision
- status-aware calibration

## 18.2 Training frontier 밖의 OOD state

Prophecy가 충분히 경험하지 않은 higher-level state에서 prediction reliability가 떨어질 수 있다.

대응:

- holdout calibration
- uncertainty-aware fail-closed gating
- larger real transition budget 검증

## 18.3 Multimodal collapse

여러 가능한 미래를 하나의 평균으로 합치는 문제.

대응:

- conditional mixture prediction
- mixture multimodality regression tests

---

# 19. 현재 연구 가설

현재 Prophecy에 대한 핵심 가설은 다음처럼 단계적으로 본다.

```text
H1. relational next-state structure를 학습할 수 있는가?
H2. stochastic multimodal outcome을 보존할 수 있는가?
H3. public status / legal mask / terminal을 함께 예측할 수 있는가?
H4. calibration이 잘못된 prediction을 걸러낼 수 있는가?
H5. 그 결과 Imagination intervention quality가 개선되는가?
H6. 최종적으로 agent success가 증가하는가?
```

H1~H4가 좋아도 H5/H6이 자동으로 성립하는 것은 아니다.

그래서 AASSR은 **world-model quality와 policy-level benefit을 분리해서 보고한다.**

---

다음으로 읽기:

- **[Imagination](Imagination)**
- **[Research Architecture](Research-Architecture)**
- **[Experiments](Experiments)**
