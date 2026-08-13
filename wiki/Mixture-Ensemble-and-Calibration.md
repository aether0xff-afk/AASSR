# Mixture Models, Ensembles and Calibration

이 페이지는 AASSR [Prophecy(미래 예측 모델)](Prophecy)의 세 가지 핵심 배경을 연결한다.

```text
Mixture Model
Ensemble
Calibration
```

세 개는 모두 "여러 예측"과 관련 있어 보이지만 역할이 다르다.

---

# 1. 왜 하나의 예측만으로 부족한가?

같은 [공개 관측 상태(public state)](State-Representation)/[행동(action)](Reinforcement-Learning)에서 여러 다음 outcome이 실제로 가능할 수 있다.

```text
(S,A)
 |-- 0.6 → S1'
 |-- 0.3 → S2'
 `-- 0.1 → S3'
```

이런 distribution을 한 벡터로 평균내면 실제로 존재하지 않는 [상태(state)](State-Representation)가 생길 수 있다.

따라서 여러 **mode**를 표현할 필요가 있다.

---

# 2. Multimodal distribution

Distribution이 여러 개의 뚜렷한 mode를 가지면 [여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration)이라고 한다.

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

Mixture [학습 모델(model)](Terminology-Guide)은 여러 [구성요소(component)](Research-Architecture) distribution의 가중합이다.

```math
p(y\mid x)=\sum_{m=1}^{M}\pi_m(x)p_m(y\mid x)
```

여기서:

- `M`: 구성요소 수
- `π_m(x)`: 구성요소 weight
- `p_m`: 각 구성요소 distribution

조건 `x`에 따라 mixture weight가 달라지면 **conditional mixture**다.

AASSR에서는 `x`가 [관계 기반(relational)](Relational-Representation-and-Generalization) 상태/행동 context에 해당한다.

---

# 4. Mixture weight

`π_m`은 각 mode의 [확률 질량(probability mass)](Stochasticity-Uncertainty-and-Probability)다.

```math
\pi_m\ge0,
\qquad
\sum_m\pi_m=1
```

AASSR [계획기(planner)](Counterfactual-Planning-and-Search)에서 이 mass는 chance [결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability)와 연결된다.

중요:

```text
mixture weight
!=
model reliability
```

어떤 mode에 90% mass를 줬더라도 모델이 해당 region을 거의 학습하지 않았다면 그 [예측(prediction)](Terminology-Guide) 전체는 신뢰하기 어려울 수 있다.

---

# 5. Mixture collapse

학습이 잘못되면 여러 구성요소가 사실상 같은 [출력(output)](Terminology-Guide)을 내며 하나의 mode로 붕괴할 수 있다.

```text
component 1 → A
component 2 → A
component 3 → A
```

실제 [환경(environment)](Reinforcement-Learning)가 여러 결과 형태를 가진인데 학습 모델이 한 mode만 남기면 위험 outcome이나 [드문(rare)](Loss-Functions-and-Class-Imbalance) outcome을 잃을 수 있다.

AASSR에서는 multimodality preservation 자체를 [회귀 테스트(regression test)](Ablation-Benchmarking-and-Reproducibility)/[진단 실험(diagnostic)](Evidence-Matrix) 대상으로 둘 가치가 있다.

---

# 6. Mode averaging

반대 [실패(failure)](Replay-Buffer-and-Episode-Boundaries)는 여러 outcome을 하나의 평균으로 합치는 것이다.

```text
실제 outcomes:
A = success-like
B = failure-like

평균:
C = neither
```

Planner는 `C`에서 존재하지 않는 legal 행동을 상상하거나 잘못된 [Critic(미래 가치 평가기)](Critic) [가치(value)](Value-Functions-and-Bellman-Equation)를 받을 수 있다.

Conditional mixture가 필요한 핵심 이유다.

---

# 7. Mixture와 categorical prediction

모든 feature를 continuous mixture로 예측해야 하는 것은 아니다.

HTTP [상태 코드(status)](Terminology-Guide)처럼 서로 배타적인 category는 [범주형(categorical)](Loss-Functions-and-Class-Imbalance) distribution이 자연스럽다.

```math
p(c\mid x)=softmax(z)_c
```

AASSR [현재(current)](Current-Status) 상태 코드 target은 대표 [공개된(public)](State-Representation) 상태 코드 classes를 범주형하게 다룬다.

관련 페이지:

- [Prophecy](Prophecy)
- [State Representation](State-Representation)

---

# 8. Class imbalance

Training data에 특정 class가 압도적으로 많으면 학습 모델이 드문 class를 무시할 수 있다.

```text
200: 90%
403: 4%
404: 3%
429: 1%
...
```

전체 accuracy만 보면 200 예측이 매우 좋아 보일 수 있다.

하지만 [계획(planning)](Counterfactual-Planning-and-Search)에서는 드문 실패 상태 코드가 [의사결정에 중요한(decision-critical)](Calibration)할 수 있다.

그래서 class weighting/balancing이 필요할 수 있다.

중요한 방법론 경계:

> class frequency를 보정하는 것과 사람이 `403=-10점` 같은 task 가치를 주입하는 것은 다르다.

---

# 9. Ensemble

Ensemble은 여러 독립/준독립 학습 모델을 함께 사용하는 방법이다.

```text
Model 1
Model 2
Model 3
   ↓
combined prediction / disagreement
```

각 학습 모델은 다른 initialization, [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries) sample, [학습(training)](Terminology-Guide) noise 등을 가질 수 있다.

---

# 10. Ensemble을 왜 쓰는가?

여러 학습 모델이 같은 [입력(input)](Terminology-Guide)에 대해 비슷한 예측을 내면 안정적인 [증거(evidence)](Evidence-Matrix)가 될 수 있다.

반대로 크게 disagree하면 학습 모델 uncertainty가 높을 수 있다.

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

하지만 ensemble agreement만으로 완벽한 [신뢰도(reliability)](Calibration)를 보장하지는 않는다.

모든 학습 모델이 같은 biased data를 학습하면 함께 틀릴 수 있다.

---

# 11. Ensemble과 Mixture는 다르다

이 구분이 중요하다.

## Mixture

하나의 입력에서 **환경에 여러 실제 outcome mode가 존재함**을 표현한다.

## Ensemble

여러 learned 학습 모델을 사용해 **학습 모델 uncertainty / robustness**에 대한 정보를 얻는다.

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

[Calibration(예측 신뢰도 보정)](Calibration)은 학습 모델이 내는 confidence/probability와 실제 correctness의 관계를 맞추는 개념이다.

Binary classification에서 이상적으로:

> confidence 0.8이라고 예측한 sample들의 약 80%가 실제로 맞는다.

와 같은 성질을 생각할 수 있다.

하지만 AASSR의 [세계 모델(world model)](Model-Based-RL-and-World-Models)은 단순 binary classifier가 아니므로 calibration도 더 복잡하다.

---

# 14. Holdout calibration

Training에 직접 사용하지 않은 real [상태 전이(transition)](MDP-and-POMDP)을 이용해 학습 모델 예측 quality를 평가한다.

```text
Training set
→ Prophecy update

Holdout set
→ reliability evaluation
```

AASSR에서는 관계 기반 행동 region별로 충분한 [검증용 분리 데이터(holdout)](Calibration) sample이 있는지 확인하고 semantic 예측 correctness를 계산한다.

관련 페이지:

- [Calibration](Calibration)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 15. Semantic calibration

World 학습 모델의 correctness를 단순 vector MSE 하나로만 측정하면 의사결정에 중요한 error를 놓칠 수 있다.

AASSR semantic [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)은 개념적으로 다음을 함께 본다.

- 관계 기반 상태 semantics
- [가능 행동 마스크(legal action mask)](Prophecy)
- 공개된 HTTP 상태 코드
- [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) class

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

Stochastic 학습 모델이 여러 outcome을 냈다면 단순히 "하나라도 actual과 비슷함"으로 평가하면 지나치게 낙관적일 수 있다.

예:

```text
prediction 1: actual과 정확히 같음, probability 0.01
prediction 2: 크게 틀림, probability 0.99
```

Top-1 matching branch만 보면 좋아 보이지만 학습 모델 distribution은 사실상 틀렸다.

그래서 probability-weighted score:

```math
C=\sum_i p_i\,score(\hat s_i',s')
```

같은 평가가 유용하다.

---

# 17. Frozen holdout

Evaluation 중 calibration reference가 계속 변하면 [같은 체크포인트(same-checkpoint)](Experiments) 비교가 불안정해질 수 있다.

AASSR 현재 calibration은 검증용 분리 데이터을 freeze하는 경로를 가진다.

```text
model/checkpoint 고정
holdout 고정
→ OFF / ON 평가
```

이렇게 하면 [Imagination(가상 미래 탐색)](Imagination) ON/OFF 비교에서 calibration 기준 자체가 움직이는 confound를 줄일 수 있다.

---

# 18. Data shortage와 fail-closed

특정 관계 기반 행동 region에 검증용 분리 데이터 sample이 거의 없다고 하자.

두 해석이 가능하다.

```text
A. 데이터가 없으니 문제 없다고 가정
B. 데이터가 없으니 reliability를 모른다고 가정
```

AASSR 현재 [판정 관문(gate)](Terminology-Guide)는 B에 가깝다.

```text
insufficient evidence
→ low reliability
→ aggressive override 금지
```

이것이 [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) 원칙이다.

---

# 19. Reliability calibration과 task value

[Calibration](Calibration)이 높다는 것은 예측이 믿을 만하다는 뜻이다.

```text
Reliability = 0.95
```

그 예측이 실패 outcome일 수 있다.

따라서:

```text
high reliability
!=
high task value
```

AASSR 현재 design은 confidence를 [Critic](Critic) 가치 bonus로 넣지 않는다.

---

# 20. Calibration과 Critic support

둘 다 증거 판정 관문처럼 보이지만 대상이 다르다.

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

[Calibration](Calibration) [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)을 반복해서 보며 학습 모델 [구조(architecture)](Research-Architecture)/hyperparameter를 맞추면 사실상 [검증(validation)](Ablation-Benchmarking-and-Reproducibility) set에 overfit할 수 있다.

그래서 최종 연구에서는:

- development 검증
- [학습을 멈춘(frozen)](Ablation-Benchmarking-and-Reproducibility) 평가
- blind/[학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)

같은 단계 분리가 중요하다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 22. Reliability diagram 개념

분류 probability calibration을 볼 때 예측 confidence bin과 empirical accuracy를 비교하는 신뢰도 diagram을 사용할 수 있다.

```text
0.1 confidence bin → 실제 10% 맞음?
0.5 confidence bin → 실제 50% 맞음?
0.9 confidence bin → 실제 90% 맞음?
```

AASSR의 semantic 세계 모델에는 그대로 적용하기 어렵지만, **confidence가 실제 correctness를 반영해야 한다**는 기본 철학은 같다.

---

# 23. Expected Calibration Error 개념

일반 classification에서 ECE는 confidence bin별 confidence와 accuracy 차이를 가중 평균한다.

개념적으로:

```math
ECE=\sum_b\frac{|B_b|}{N}|acc(B_b)-conf(B_b)|
```

AASSR 현재 semantic calibration은 ECE 하나로 정의되는 구조는 아니지만, 관련 연구 배경으로 알아두면 좋다.

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

아니다. 너무 많으면 학습/식별이 어렵고 구성요소 collapse/redundancy가 생길 수 있다.

## "Ensemble이 동의하면 정답이다"

아니다. shared bias가 있으면 모두 같이 틀릴 수 있다.

## "Softmax 0.99면 reliability 0.99다"

아니다. neural classifier는 [학습 분포 밖(OOD)](Critic-Support-and-OOD)에서 과도하게 confident할 수 있다.

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