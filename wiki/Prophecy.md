# Prophecy — 미래 예측 모델

[Prophecy(미래 예측 모델)](Prophecy)는 AASSR의 **확률적 [world model](Model-Based-RL-and-World-Models)** 이다.

목표는 다음 질문에 답하는 것이다.

> **현재 공개 상태에서 어떤 행동을 했을 때 다음에 어떤 공개 결과들이 얼마나 가능한가?**

[현재 세대(current-generation)](Current-Status)의 [Prophecy](Prophecy)는 단순한 `(S,A) → S'` 회귀 모델이 아니다. [부분 관측](MDP-and-POMDP), [stochastic outcome](Stochasticity-Uncertainty-and-Probability), [공개된(public)](State-Representation) HTTP [상태 코드(status)](Terminology-Guide), legal [행동(action)](Reinforcement-Learning) surface, [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) class를 함께 다루는 **[관계 기반(relational)](Relational-Representation-and-Generalization) [조건부 혼합(conditional-mixture)](Prophecy) ensemble**이다.

> [!IMPORTANT]
> 현재 [최종 기준(source of truth)](Current-Status): `src/aassr_v2/current_manifest.py`  
> 핵심 구현 계열: `src/aassr_v2/current_relational_mixture_model.py`, [상태 코드까지 고려하는(status-aware)](Calibration) [현재(current)](Current-Status) [학습 모델(model)](Terminology-Guide)/repair modules

---

# 0. 먼저 알아두면 좋은 개념

[Prophecy](Prophecy) 문서를 제대로 이해하려면 다음 배경이 직접 연결된다.

- [MDP and POMDP](MDP-and-POMDP) — [상태(state)](State-Representation), [관측(observation)](MDP-and-POMDP), [숨은 환경 상태(hidden state)](MDP-and-POMDP), [부분 관측(partial observability)](MDP-and-POMDP)
- [Model-Based RL & World Models](Model-Based-RL-and-World-Models) — learned [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)와 [계획(planning)](Counterfactual-Planning-and-Search)
- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — [확률(probability)](Stochasticity-Uncertainty-and-Probability), aleatoric/[지식 부족에서 오는 불확실성(epistemic uncertainty)](Stochasticity-Uncertainty-and-Probability)
- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration) — [여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration) [예측(prediction)](Terminology-Guide), mixture weight, ensemble
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — [실제 개체를 구분하는(concrete)](State-Representation) ID 대신 구조를 학습하는 이유
- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance) — [범주형(categorical)](Loss-Functions-and-Class-Imbalance) 상태 코드, BCE/CE, [드문(rare)](Loss-Functions-and-Class-Imbalance) class
- [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation) — [Knowledge(에피소드 지식)](Knowledge) anti-hindsight boundary

---

# 1. 연구 질문

> **미래의 공개된 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability) 분포를 학습하면 [희소 보상](Sparse-Reward-and-Credit-Assignment) 환경에서 실제 행동 전에 더 나은 의사결정을 할 수 있는가?**

[Prophecy](Prophecy) 자체의 목표는 행동을 직접 선택하는 것이 아니다.

```text
Policy       = 지금 무엇을 할까?
Prophecy     = 그 행동을 하면 어떤 일이 일어날까?
Imagination  = 여러 미래를 이어 보면 어떤 행동이 더 나을까?
Critic       = 그 미래의 장기 sparse return은 얼마일까?
```

즉 [Prophecy](Prophecy)는 **[환경(environment)](Reinforcement-Learning) 환경의 상태 변화 규칙를 근사하는 예측 [처리 계층(layer)](Research-Architecture)**이고, 최종 행동 선택은 [Imagination](Imagination)과 [Critic](Critic), 그리고 여러 [신뢰도(reliability)](Calibration) [판정 관문(gate)](Terminology-Guide)를 거친다.

---

# 2. 왜 world model이 필요한가?

[Model-free](Reinforcement-Learning) [Policy(정책 모델)](Policy)는 현재 상태/행동의 장기 [가치(value)](Value-Functions-and-Bellman-Equation)를 직접 학습한다.

```text
Q(S,A)
```

하지만 [sparse reward](Sparse-Reward-and-Credit-Assignment)에서는 대부분의 즉시 [보상(reward)](Sparse-Reward-and-Credit-Assignment)가 `0`이다.

```text
A0 → 0
A1 → 0
A2 → 0
A3 → 0
A4 → +1
```

[TD learning](Q-Learning-DQN-and-TD)을 통해 최종 신호가 뒤로 전파될 수 있지만 성공 경험 자체가 적으면 매우 느릴 수 있다.

World 학습 모델이 있으면 실제 행동 전에 다음을 질문할 수 있다.

```text
A를 하면 어떤 public state가 가능한가?
각 결과는 얼마나 자주 일어날까?
그 결과에서 어떤 action이 legal한가?
success/failure/truncation인가?
몇 단계 더 전개하면 어떤 장기 outcome이 가능한가?
```

이 예측을 여러 단계 이어붙이는 것이 [counterfactual planning](Counterfactual-Planning-and-Search), 즉 AASSR의 [Imagination(가상 미래 탐색)](Imagination)이다.

---

# 3. True state를 예측하는가, public state를 예측하는가?

AASSR의 [학습 주체(learner)](Terminology-Guide)는 simulator의 모든 [숨겨진(hidden)](MDP-and-POMDP) truth를 볼 수 없다.

```text
Hidden simulator state
        ↓ public response
Observed state
        ↓ relational encoding
Relational State v3
```

따라서 [Prophecy](Prophecy)가 근사하는 것은 **숨겨진 simulator truth 자체가 아니라 학습 주체가 인과적으로 접근 가능한 공개된 future [표현(representation)](Relational-Representation-and-Generalization)**이다.

이 차이는 [POMDP](MDP-and-POMDP) 관점에서 중요하다.

```text
True dynamics:
P(S_hidden' | S_hidden, A)

Learner-side prediction:
P(R_public' | R_public, A, K)
```

여기서 `R_public`은 [Relational State](State-Representation), `K`는 현재까지 실제로 획득한 [Knowledge](Knowledge) context다.

---

# 4. 입력

개념적으로 [Prophecy](Prophecy) [입력(input)](Terminology-Guide)은 다음처럼 생각할 수 있다.

```text
X_t = [R_t, A_t, K_t]
```

- `R_t`: [relational public state](State-Representation)
- `A_t`: [relational action representation](Relational-Representation-and-Generalization)
- `K_t`: 행동 전에 이미 획득한 [현재 에피소드 안에서만 유지되는(episode-local)](Knowledge) [Knowledge](Knowledge)

중요한 점은 실제 개체를 구분하는 identifier 자체를 [전이(transfer)](Relational-Representation-and-Generalization) 학습 주체의 주요 [식별 방식(identity)](State-Representation)로 쓰지 않는다는 것이다.

```text
route-12
route-31
```

이름 자체보다:

```text
catalog-like role
login-like role
object-like role
```

같은 **관계적 구조**를 사용한다.

---

# 5. 왜 Knowledge가 input에 들어갈 수 있는가?

[POMDP](MDP-and-POMDP)에서는 현재 관측 하나만으로 과거에 알아낸 정보를 잃을 수 있다.

예:

```text
시점 t-2: response에서 token 획득
시점 t: 현재 raw response에는 token이 직접 다시 나오지 않음
```

현재 decision에 그 fact가 필요하다면 explicit memory가 필요하다.

AASSR의 [Knowledge](Knowledge)는 이런 **과거 [실제 환경에서 관측된(real)](Research-Jargon-Guide) [응답(response)](State-Representation)에서 이미 알게 된 사실**을 보존한다.

단, 시간 순서는 엄격하다.

```text
K_t
 ↓
Prophecy(S_t,A_t,K_t)
 ↓
A_t 실행
 ↓
새 response
 ↓
K_{t+1}
```

`K_{t+1}`를 `A_t` 실행 전 예측에 넣으면 [hindsight leakage](Causality-Leakage-and-Evaluation)다.

---

# 6. 출력

현재 [Prophecy](Prophecy)는 다음 상태 vector 하나만 내지 않는다.

각 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) 환경 결과 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)는 개념적으로 다음을 포함한다.

```text
next relational descriptor
latest public HTTP status
legal action mask
terminal class
outcome probability mass
prediction reliability는 별도 calibration 계층
```

이 각각은 [계획기(planner)](Counterfactual-Planning-and-Search)에서 다른 역할을 한다.

---

# 7. Next relational descriptor

다음 [공개 관측 상태(public state)](State-Representation)의 관계 구조를 예측한다.

Concrete ID를 그대로 생성하는 대신:

- known route/profile/object structure
- [역할(role)](Relational-Representation-and-Generalization) [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)
- 공개된 workflow-related relations
- available 행동 structure
- latest observed 상태 코드

같은 전이 가능한 [상태를 요약한 표현(descriptor)](State-Representation)를 중심으로 한다.

왜 이런 abstr행동을 쓰는지는 [Relational Representation & Generalization](Relational-Representation-and-Generalization)에서 다룬다.

---

# 8. 왜 deterministic prediction이 아닌가?

가장 단순한 [세계 모델(world model)](Model-Based-RL-and-World-Models):

```math
\hat S' = f_\theta(S,A)
```

은 하나의 future만 출력한다.

하지만 [부분 관측](MDP-and-POMDP)이나 실제 [stochasticity](Stochasticity-Uncertainty-and-Probability)가 있으면 같은 공개된 `(S,A)`에서도 여러 환경 결과이 가능하다.

```text
(S,A)
  |-- 0.60 → S1'
  |-- 0.30 → S2'
  `-- 0.10 → S3'
```

이들을 하나의 평균으로 회귀하면 실제로 존재하지 않는 **mean 상태**가 생길 수 있다.

```text
실제 outcome A
실제 outcome B
      ↓ 평균
가상의 C
```

특히 범주형 상태/행동 structure에서는 이 문제가 심각하다.

그래서 현재 세대은 [conditional mixture model](Mixture-Ensemble-and-Calibration)을 사용한다.

---

# 9. Mixture formulation

개념적으로 다음과 같은 분포을 근사한다.

```math
p(S_{t+1}|S_t,A_t,K_t)
=
\sum_{m=1}^{M}\pi_m(X_t)p_m(S_{t+1}|X_t)
```

여기서:

- `M`: mixture [구성요소(component)](Research-Architecture) 수
- `π_m(X_t)`: 구성요소 `m`의 [outcome probability mass](Stochasticity-Uncertainty-and-Probability)
- `p_m`: 해당 [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)의 next-state 분포

중요한 점:

```text
Mixture component
= 여러 모델의 의견이 아니라
  하나의 환경에서 가능한 여러 outcome mode
```

라는 것이다.

[Ensemble](Mixture-Ensemble-and-Calibration)과는 의미가 다르다.

---

# 10. Mixture와 Ensemble의 차이

AASSR에서 특히 헷갈리기 쉬운 구분이다.

```text
Mixture
→ world가 여러 결과를 낼 수 있다는 구조

Ensemble
→ 여러 learned models의 prediction/evidence
```

개념적으로:

```text
Mixture multimodality
→ aleatoric / hidden-state-induced outcome variability

Ensemble disagreement
→ epistemic uncertainty의 proxy가 될 수 있음
```

정확한 1:1 대응이라고 단정할 수는 없지만 연구 의미를 분리하는 데 유용하다.

자세히: [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 11. Outcome probability와 reliability는 다르다

가장 중요한 구분 중 하나다.

## Outcome probability

> **이 결과가 환경에서 실제로 발생할 [확률 질량(probability mass)](Stochasticity-Uncertainty-and-Probability)는 얼마인가?**

예:

```text
200 outcome : 0.70
403 outcome : 0.20
429 outcome : 0.10
```

이 값은 [chance-node expectation](Chance-and-Decision-Nodes)에 사용된다.

## Prediction reliability

> **이 world-model 예측 자체를 얼마나 믿을 수 있는가?**

이 값은 [Calibration](Calibration)이 실제 [검증용 분리 데이터(holdout)](Calibration) [증거(evidence)](Evidence-Matrix)로 판단한다.

따라서:

```text
high probability != high reliability
high reliability != high value
high value != high support
```

다.

전체 구분은 [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability)에서 본다.

---

# 12. HTTP status를 왜 명시적으로 예측하는가?

과거 repaired [Imagination](Imagination) 2k [진단 실험(diagnostic)](Evidence-Matrix)에서는 전체 [의미 기준(semantic)](State-Representation) 예측이 그럴듯해도 실제 계획기 [실제 행동 개입(intervention)](Imagination)이 `403/404/429` 같은 공개된 환경 결과으로 이어지는 문제가 드러났다.

그 결과 **[의사결정에 중요한(decision-critical)](Calibration) 공개된 variable을 abstr행동 과정에서 잃으면 전체 의미 기준 similarity만으로는 부족하다**는 점이 중요해졌다.

그래서 Relational [상태(State)](State-Representation) v3는 latest 공개된 HTTP 상태 코드를 명시적으로 보존하고 [Prophecy](Prophecy)도 이를 예측한다.

대표 상태 코드 vocabulary:

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

자세한 상태 [명세(contract)](Current-Status)는 [State Representation](State-Representation)에서 본다.

---

# 13. 왜 status를 continuous scalar로 보지 않는가?

HTTP 상태 코드 code의 숫자 차이는 task semantics의 거리라고 볼 수 없다.

```text
403과 404의 숫자 차이 = 1
```

이라고 해서 두 상태가 `200`과 `201`보다 의미상 반드시 비슷하다는 근거는 없다.

따라서 mutually exclusive [categorical target](Loss-Functions-and-Class-Imbalance)으로 다룬다.

개념적 cross-entropy:

```math
L_{status}=-\sum_c w_cy_c\log \hat p_c
```

`w_c`는 [class imbalance](Loss-Functions-and-Class-Imbalance)를 보정할 수 있다.

---

# 14. Class balancing은 reward shaping인가?

아니다.

```text
Rare 429 sample에 training weight를 더 줌
```

은 **예측 학습 모델이 드문 class를 무시하지 않도록 [학습 손실(loss)](Loss-Functions-and-Class-Imbalance)/sample 분포을 조정하는 것**이다.

반면:

```text
429가 나오면 reward -0.5
```

는 [에이전트(agent)](Reinforcement-Learning)의 task [학습 목표(objective)](Terminology-Guide)를 바꾸는 [reward shaping](Sparse-Reward-and-Credit-Assignment)이다.

둘은 완전히 다르다.

AASSR 현재 design은 전자를 사용할 수 있지만 후자로 [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment) 명세를 바꾸지 않는다.

---

# 15. Legal action mask prediction

다음 상태 표현을 대충 맞혀도 **그 상태에서 가능한 행동 집합**을 틀리면 계획기는 존재하지 않는 행동을 상상할 수 있다.

```text
Predicted state
+
Predicted legal action surface
```

가 함께 필요하다.

Legal 행동 mask는 여러 행동이 동시에 가능할 수 있으므로 [multi-label prediction](Loss-Functions-and-Class-Imbalance)과 연결된다.

평가에는 [Jaccard similarity](Loss-Functions-and-Class-Imbalance) 같은 set [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)을 사용할 수 있다.

---

# 16. Terminal class prediction

다음 공개된 환경 결과이:

```text
active
success
true failure
truncation
```

중 무엇인지 구분해야 한다.

왜냐하면 [return](Value-Functions-and-Bellman-Equation)과 [episode boundary](Replay-Buffer-and-Episode-Boundaries)의 의미가 다르기 때문이다.

```text
success       → +1
true failure  → -1
truncation    →  0
```

`true failure`와 administrative [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)을 같은 class로 합치면 계획기/[Critic(미래 가치 평가기)](Critic) semantics가 왜곡될 수 있다.

---

# 17. Prophecy training loss와 agent reward는 다르다

[Prophecy](Prophecy)는 여러 supervised 학습 목표를 사용할 수 있다.

개념적으로:

```math
L
=
\lambda_sL_{state}
+
\lambda_hL_{status}
+
\lambda_mL_{mask}
+
\lambda_tL_{terminal}
+
\lambda_{mix}L_{mixture}
```

이 학습 손실는 neural [신경망(network)](Neural-Networks-and-Optimization) [학습(training)](Terminology-Guide)을 위한 학습 목표다.

```text
Prophecy loss
!=
Environment reward
```

따라서 상태 코드 학습 손실 weight를 키운다고 에이전트에게 중간 보상를 주는 것이 아니다.

자세히: [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)

---

# 18. Calibration

좋은 세계 모델은 단순히 평균 학습 손실/accuracy가 높은 모델이 아니다.

Planner가 필요한 것은:

> **현재 query에서 예측을 실제 행동 판단에 사용해도 되는가?**

다.

현재 [Calibration](Calibration)은 실제 검증용 분리 데이터 [상태 전이(transition)](MDP-and-POMDP)을 기준으로 의미 기준 신뢰도를 평가한다.

평가 요소에는 다음이 포함될 수 있다.

- 관계 기반 의미 기준 next-state quality
- legal-행동-mask correctness
- 에피소드 종료-class correctness
- HTTP-status correctness
- probability-weighted 의미 기준 quality

[Calibration(예측 신뢰도 보정)](Calibration)은 가치 bonus가 아니다.

```text
reliability 충분
→ prediction 사용 가능

reliability 부족
→ override를 막고 Policy 쪽으로 fail closed
```

---

# 19. 왜 probability-weighted calibration이 필요한가?

Stochastic 학습 모델이 여러 결과 경로를 냈다고 하자.

```text
1% branch  → actual과 정확히 일치
99% branch → 크게 틀림
```

"하나라도 맞는 결과 경로가 있다"만 보면 모델이 좋아 보인다.

하지만 분포 전체는 사실상 틀렸다.

그래서:

```math
C=\sum_i p_i\,score(\hat s_i',s')
```

같은 probability-weighted 의미 기준 [평가 점수(score)](Terminology-Guide)가 더 적절할 수 있다.

관련 페이지: [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 20. Ensemble을 쓰는 이유

여러 학습 모델을 독립적 [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries)/initialization으로 학습하면 예측 disagreement를 관찰할 수 있다.

```text
Model 1 → A
Model 2 → A
Model 3 → A
```

과:

```text
Model 1 → A
Model 2 → B
Model 3 → C
```

는 epistemic 증거가 다를 수 있다.

하지만:

> ensemble이 동의한다고 정답이라는 보장은 없다.

모두 같은 biased data를 학습했다면 같이 틀릴 수 있다.

그래서 [real holdout calibration](Mixture-Ensemble-and-Calibration)이 필요하다.

---

# 21. Real transition만 factual target인가?

현재 research 원칙에서는 [Prophecy](Prophecy) 학습의 사실 근거는 **실제 환경 상태 전이**이다.

```text
real S_t
real A_t
real S_{t+1}
```

Imagined 상태 전이을 정답 data처럼 다시 [Prophecy](Prophecy)에 넣으면:

```text
model error
→ imagined sample
→ model이 자기 error를 truth로 학습
→ error amplification
```

이 생길 수 있다.

이 경계는 [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation)에서 더 일반적으로 다룬다.

---

# 22. One-step prediction과 multi-step planning

[Prophecy](Prophecy)는 기본적으로 다음 상태 전이 분포을 학습한다.

```text
(S_t,A_t) → distribution over S_{t+1}
```

[Imagination](Imagination)은 예측 상태를 다시 [Prophecy](Prophecy) 입력으로 사용한다.

```text
S0
 ↓ A0
Ŝ1
 ↓ A1
Ŝ2
 ↓ A2
Ŝ3
```

깊어질수록 예측 위에서 다시 예측하므로 [compounding model error](Model-Based-RL-and-World-Models)가 커질 수 있다.

그래서 계획 depth를 무조건 크게 하는 것이 좋은 전략은 아니다.

---

# 23. Planner와의 연결

한 [탐색의 첫 행동(root)](Imagination) 행동에 대해 [Prophecy](Prophecy)가 다음 분포를 만든다고 하자.

```text
root action A
   |
   |-- p1 → predicted state 1
   |-- p2 → predicted state 2
   `-- p3 → predicted state 3
```

이것은 [chance node](Chance-and-Decision-Nodes)다.

각 predicted 상태에서는 에이전트가 다음 행동을 선택할 수 있으므로 [decision node](Chance-and-Decision-Nodes)가 된다.

```text
Decision
  ↓ action
Chance
  ↓ stochastic outcome
Decision
  ↓ action
Chance
```

이 구조가 AASSR [Imagination](Imagination)의 핵심이다.

---

# 24. Prophecy와 Critic은 다르다

두 학습 모델이 모두 future와 관련 있어 보이지만 역할이 다르다.

```text
Prophecy
→ 어떤 state/outcome이 올까?

Critic
→ 그 trajectory/state의 future sparse return은 얼마일까?
```

예를 들어 [Prophecy](Prophecy)는 403 환경 결과을 정확하게 예측할 수 있다.

그 403이 task 학습 목표에서 얼마나 나쁜지는 [Critic](Critic)/[누적 보상(return)](Value-Functions-and-Bellman-Equation) semantics가 평가한다.

```text
prediction correctness
!=
task value
```

관련 페이지: [Critic](Critic)

---

# 25. Prophecy reliability와 Critic support도 다르다

```text
Prophecy reliability
→ transition prediction을 믿을 수 있나?

Critic local support
→ value estimate를 믿을 real training evidence가 있나?
```

둘 중 하나만 좋아도 충분하지 않다.

```text
미래는 정확히 예측
하지만 Critic이 그 region을 본 적 없음
→ value comparison은 위험
```

관련 페이지: [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 26. Prophecy를 어떻게 평가하는가?

단일 MSE만으로는 충분하지 않다.

## Model-level metric

- 관계 기반 의미 기준 quality
- probability-weighted 의미 기준 quality
- HTTP 상태 코드 accuracy / per-class recall
- legal-mask quality
- 에피소드 종료 accuracy
- NLL / likelihood류
- mixture 구성요소 usage
- ensemble disagreement
- calibration 신뢰도

## Planner-level metric

- reliable 탐색의 첫 행동 [데이터가 어느 영역까지 포함하는지(coverage)](Critic-Support-and-OOD)
- wrong-status 결과 경로 rate
- 결과 경로 pruning rate
- predicted vs actual 환경 결과 agreement

## Agent-level metric

- 실제 행동 개입 error rate
- direct success-producing 실제 행동 개입
- no-[Imagination](Imagination) 대비 [성공(success)](Terminology-Guide) difference

[Proxy metric](Ablation-Benchmarking-and-Reproducibility)과 final task 평가지표을 구분해야 한다.

---

# 27. 알려진 실패 패턴

## 27.1 평균 semantic score는 높은데 중요한 status를 틀림

**문제:** 표현/평가지표이 의사결정에 중요한 공개된 channel을 충분히 반영하지 않음.

**대응:**

- latest HTTP 상태 코드를 Relational 상태 v3에 보존
- 범주형 상태 코드 supervision
- 상태 코드까지 고려하는 의미 기준 calibration

## 27.2 Higher-level OOD

쉬운 [난이도 조절 학습(curriculum)](Curriculum-Learning) level에서만 충분한 data가 있고 higher level에서 학습 모델이 extrapolate할 수 있다.

**대응:**

- 실제 검증용 분리 데이터 신뢰도
- [Curriculum transfer](Curriculum-Learning) 분석
- larger 실제 상태 전이 budget
- [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) [조건부 통과 판단(gating)](Terminology-Guide)

## 27.3 Multimodal collapse

여러 가능한 future를 하나의 평균 또는 한 구성요소로 collapse.

**대응:**

- conditional mixture
- 구성요소/mass 진단 실험
- multimodality [회귀 테스트(regression test)](Ablation-Benchmarking-and-Reproducibility)s

## 27.4 Rare critical status 무시

Imbalanced data 때문에 majority 상태 코드만 잘 맞힘.

**대응:**

- [class-balanced training](Loss-Functions-and-Class-Imbalance)
- per-status 평가지표
- 의미 기준 calibration

## 27.5 Long rollout compounding error

한 단계 예측은 괜찮지만 깊은 [Imagination](Imagination)에서 drift.

**대응:**

- shallow/root-preserving 계획
- calibration 판정 관문
- re-plan after every 실제 행동

---

# 28. 연구 가설

[Prophecy](Prophecy)에 대한 질문은 단계적으로 분리하는 것이 좋다.

```text
H1. relational next-state structure를 학습할 수 있는가?
H2. stochastic multimodal outcome을 보존할 수 있는가?
H3. public status / legal mask / terminal을 함께 예측할 수 있는가?
H4. probability mass가 actual outcome frequency를 유용하게 나타내는가?
H5. calibration이 잘못된 prediction을 걸러낼 수 있는가?
H6. calibrated Prophecy가 Imagination intervention quality를 높이는가?
H7. 최종적으로 same-checkpoint Full success가 no-Imagination보다 높아지는가?
```

H1~H5가 좋아도 H6/H7은 자동으로 성립하지 않는다.

이것이 [world-model metric과 downstream task metric을 분리](Ablation-Benchmarking-and-Reproducibility)해야 하는 이유다.

---

# 29. 관련 코드 읽는 순서

```text
src/aassr_v2/current_manifest.py
        ↓ current component contract
current_relational_state_v3.py
        ↓ state representation
current_relational_mixture_model.py / current status model
        ↓ stochastic prediction
current_semantic_calibration.py
        ↓ reliability
current_planner.py / imagination tree
        ↓ planning consumption
```

신경망/optimizer/학습 손실 자체가 낯설다면:

- [Neural Networks & Optimization](Neural-Networks-and-Optimization)
- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)

부터 읽는 것이 좋다.

---

# 30. 한 문장 요약

> **[Prophecy](Prophecy)는 정답 미래 하나를 맞히는 모델이 아니라, 공개 관측으로부터 가능한 미래의 구조·확률을 예측하고, 별도의 신뢰도 증거와 함께 [Imagination](Imagination)에 공급하는 확률적 관계 기반 세계 모델이다.**

---

다음으로 읽기:

- **[Calibration](Calibration)**
- **[Imagination](Imagination)**
- **[Critic](Critic)**
- **[Research Architecture](Research-Architecture)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
