# Mixture Models, Ensembles and Calibration

이 페이지는 AASSR Prophecy의 세 가지 핵심 배경을 연결한다.

```text
Mixture Model
Ensemble
Calibration
```

세 개는 모두 "여러 예측"과 관련 있어 보이지만 역할이 다르다.

---

# 1. 왜 하나의 예측만으로 부족한가?

같은 public state/action에서 여러 다음 outcome이 실제로 가능할 수 있다.

```text
(S,A)
 |-- 0.6 → S1'
 |-- 0.3 → S2'
 `-- 0.1 → S3'
```

이런 distribution을 한 벡터로 평균내면 실제로 존재하지 않는 state가 생길 수 있다.

따라서 여러 **mode**를 표현할 필요가 있다.

---

# 2. Multimodal distribution

Distribution이 여러 개의 뚜렷한 mode를 가지면 multimodal이라고 한다.

예:

```text
HTTP outcome
200 근처 mode
403 근처 mode
429 근처 mode
```

이들을 단일 Gaussian/평균으로 표현하는 것은 부적절할 수 있다.

AASSR의 conditional mixture는 이런 여러 outcome mode를 명시적으로 보존하려는 설계다.

---

# 3. Mixture model

Mixture model은 여러 component distribution의 가중합이다.

```math
p(y\mid x)=\sum_{m=1}^{M}\pi_m(x)p_m(y\mid x)
```

여기서:

- `M`: component 수
- `π_m(x)`: component weight
- `p_m`: 각 component distribution

조건 `x`에 따라 mixture weight가 달라지면 **conditional mixture**다.

AASSR에서는 `x`가 relational state/action context에 해당한다.

---

# 4. Mixture weight

`π_m`은 각 mode의 probability mass다.

```math
\pi_m\ge0,
\qquad
\sum_m\pi_m=1
```

AASSR planner에서 이 mass는 chance outcome probability와 연결된다.

중요:

```text
mixture weight
!=
model reliability
```

어떤 mode에 90% mass를 줬더라도 모델이 해당 region을 거의 학습하지 않았다면 그 prediction 전체는 신뢰하기 어려울 수 있다.

---

# 5. Mixture collapse

학습이 잘못되면 여러 component가 사실상 같은 output을 내며 하나의 mode로 붕괴할 수 있다.

```text
component 1 → A
component 2 → A
component 3 → A
```

실제 environment가 multimodal인데 model이 한 mode만 남기면 위험 outcome이나 rare outcome을 잃을 수 있다.

AASSR에서는 multimodality preservation 자체를 regression test/diagnostic 대상으로 둘 가치가 있다.

---

# 6. Mode averaging

반대 failure는 여러 outcome을 하나의 평균으로 합치는 것이다.

```text
실제 outcomes:
A = success-like
B = failure-like

평균:
C = neither
```

Planner는 `C`에서 존재하지 않는 legal action을 상상하거나 잘못된 Critic value를 받을 수 있다.

Conditional mixture가 필요한 핵심 이유다.

---

# 7. Mixture와 categorical prediction

모든 feature를 continuous mixture로 예측해야 하는 것은 아니다.

HTTP status처럼 서로 배타적인 category는 categorical distribution이 자연스럽다.

```math
p(c\mid x)=softmax(z)_c
```

AASSR current status target은 대표 public status classes를 categorical하게 다룬다.

관련 페이지:

- [Prophecy](Prophecy)
- [State Representation](State-Representation)

---

# 8. Class imbalance

Training data에 특정 class가 압도적으로 많으면 model이 rare class를 무시할 수 있다.

```text
200: 90%
403: 4%
404: 3%
429: 1%
...
```

전체 accuracy만 보면 200 예측이 매우 좋아 보일 수 있다.

하지만 planning에서는 rare failure status가 decision-critical할 수 있다.

그래서 class weighting/balancing이 필요할 수 있다.

중요한 방법론 경계:

> class frequency를 보정하는 것과 사람이 `403=-10점` 같은 task value를 주입하는 것은 다르다.

---

# 9. Ensemble

Ensemble은 여러 독립/준독립 model을 함께 사용하는 방법이다.

```text
Model 1
Model 2
Model 3
   ↓
combined prediction / disagreement
```

각 model은 다른 initialization, bootstrap sample, training noise 등을 가질 수 있다.

---

# 10. Ensemble을 왜 쓰는가?

여러 model이 같은 input에 대해 비슷한 prediction을 내면 안정적인 evidence가 될 수 있다.

반대로 크게 disagree하면 model uncertainty가 높을 수 있다.

```text
M1 → A
M2 → A
M3 → A
→ agreement 높음

M1 → A
M2 → B
M3 → C
→ disagreement 큼
```

하지만 ensemble agreement만으로 완벽한 reliability를 보장하지는 않는다.

모든 model이 같은 biased data를 학습하면 함께 틀릴 수 있다.

---

# 11. Ensemble과 Mixture는 다르다

이 구분이 중요하다.

## Mixture

하나의 input에서 **환경에 여러 실제 outcome mode가 존재함**을 표현한다.

## Ensemble

여러 learned model을 사용해 **model uncertainty / robustness**에 대한 정보를 얻는다.

단순하게:

```text
Mixture → world의 여러 가능성
Ensemble → learner들의 여러 견해
```

라고 기억할 수 있다.

물론 실제 구현에서는 둘의 정보가 얽힐 수 있지만 개념적 역할은 분리하는 것이 좋다.

---

# 12. Aleatoric vs Epistemic 다시 연결

```text
Mixture multimodality
→ aleatoric/hidden-state induced outcome variability 표현

Ensemble disagreement
→ epistemic uncertainty에 대한 proxy
```

정확히 1:1 대응한다고 단정할 수는 없지만 AASSR 설계 의도를 이해하는 데 유용한 구분이다.

더 자세히:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 13. Calibration이란?

Calibration은 model이 내는 confidence/probability와 실제 correctness의 관계를 맞추는 개념이다.

Binary classification에서 이상적으로:

> confidence 0.8이라고 예측한 sample들의 약 80%가 실제로 맞는다.

와 같은 성질을 생각할 수 있다.

하지만 AASSR의 world model은 단순 binary classifier가 아니므로 calibration도 더 복잡하다.

---

# 14. Holdout calibration

Training에 직접 사용하지 않은 real transition을 이용해 model prediction quality를 평가한다.

```text
Training set
→ Prophecy update

Holdout set
→ reliability evaluation
```

AASSR에서는 relational action region별로 충분한 holdout sample이 있는지 확인하고 semantic prediction correctness를 계산한다.

관련 페이지:

- [Calibration](Calibration)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 15. Semantic calibration

World model의 correctness를 단순 vector MSE 하나로만 측정하면 decision-critical error를 놓칠 수 있다.

AASSR semantic evaluation은 개념적으로 다음을 함께 본다.

- relational state semantics
- legal action mask
- public HTTP status
- terminal class

즉:

```text
"숫자가 대충 비슷한가?"
```

보다:

```text
"planner 관점에서 중요한 미래 구조를 맞혔는가?"
```

를 보려는 것이다.

---

# 16. Probability-weighted semantic score

Stochastic model이 여러 outcome을 냈다면 단순히 "하나라도 actual과 비슷함"으로 평가하면 지나치게 낙관적일 수 있다.

예:

```text
prediction 1: actual과 정확히 같음, probability 0.01
prediction 2: 크게 틀림, probability 0.99
```

Top-1 matching branch만 보면 좋아 보이지만 model distribution은 사실상 틀렸다.

그래서 probability-weighted score:

```math
C=\sum_i p_i\,score(\hat s_i',s')
```

같은 평가가 유용하다.

---

# 17. Frozen holdout

Evaluation 중 calibration reference가 계속 변하면 same-checkpoint 비교가 불안정해질 수 있다.

AASSR current calibration은 holdout을 freeze하는 경로를 가진다.

```text
model/checkpoint 고정
holdout 고정
→ OFF / ON 평가
```

이렇게 하면 Imagination ON/OFF 비교에서 calibration 기준 자체가 움직이는 confound를 줄일 수 있다.

---

# 18. Data shortage와 fail-closed

특정 relational action region에 holdout sample이 거의 없다고 하자.

두 해석이 가능하다.

```text
A. 데이터가 없으니 문제 없다고 가정
B. 데이터가 없으니 reliability를 모른다고 가정
```

AASSR current gate는 B에 가깝다.

```text
insufficient evidence
→ low reliability
→ aggressive override 금지
```

이것이 fail-closed 원칙이다.

---

# 19. Reliability calibration과 task value

Calibration이 높다는 것은 prediction이 믿을 만하다는 뜻이다.

```text
Reliability = 0.95
```

그 prediction이 failure outcome일 수 있다.

따라서:

```text
high reliability
!=
high task value
```

AASSR current design은 confidence를 Critic value bonus로 넣지 않는다.

---

# 20. Calibration과 Critic support

둘 다 evidence gate처럼 보이지만 대상이 다르다.

```text
Calibration
→ Prophecy transition prediction을 믿을 수 있는가?

Critic support
→ 그 state/action에서 Critic value를 믿을 real training evidence가 있는가?
```

하나만 통과해도 충분하지 않다.

예:

```text
Prophecy 정확
Critic OOD
→ 미래는 잘 예측했지만 가치 평가가 틀릴 수 있음
```

반대도 가능하다.

---

# 21. Calibration metric 자체의 overfitting

Calibration metric을 반복해서 보며 model architecture/hyperparameter를 맞추면 사실상 validation set에 overfit할 수 있다.

그래서 최종 연구에서는:

- development validation
- frozen evaluation
- blind/unseen benchmark

같은 단계 분리가 중요하다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 22. Reliability diagram 개념

분류 probability calibration을 볼 때 prediction confidence bin과 empirical accuracy를 비교하는 reliability diagram을 사용할 수 있다.

```text
0.1 confidence bin → 실제 10% 맞음?
0.5 confidence bin → 실제 50% 맞음?
0.9 confidence bin → 실제 90% 맞음?
```

AASSR의 semantic world model에는 그대로 적용하기 어렵지만, **confidence가 실제 correctness를 반영해야 한다**는 기본 철학은 같다.

---

# 23. Expected Calibration Error 개념

일반 classification에서 ECE는 confidence bin별 confidence와 accuracy 차이를 가중 평균한다.

개념적으로:

```math
ECE=\sum_b\frac{|B_b|}{N}|acc(B_b)-conf(B_b)|
```

AASSR current semantic calibration은 ECE 하나로 정의되는 구조는 아니지만, 관련 연구 배경으로 알아두면 좋다.

---

# 24. AASSR의 흐름

```text
Real transition data
       ↓
Conditional Mixture Prophecy
       ↓
여러 stochastic outcomes + probability mass
       ↓
Ensemble / holdout evidence
       ↓
Semantic Calibration
       ↓
reliable branches만 planner에 허용
```

---

# 25. 핵심 오해

## "Mixture component 수가 많을수록 무조건 좋다"

아니다. 너무 많으면 학습/식별이 어렵고 component collapse/redundancy가 생길 수 있다.

## "Ensemble이 동의하면 정답이다"

아니다. shared bias가 있으면 모두 같이 틀릴 수 있다.

## "Softmax 0.99면 reliability 0.99다"

아니다. neural classifier는 OOD에서 과도하게 confident할 수 있다.

## "Calibration이 높으면 좋은 행동이다"

아니다. 정확하게 예측된 실패도 calibration은 높을 수 있다.

---

# 26. 다음으로 읽기

- [Prophecy](Prophecy)
- [Calibration](Calibration)
- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)
- [Critic, Support and OOD](Critic-Support-and-OOD)
- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)

관련 색인: **[Concept Index](Concept-Index)**