# Loss Functions and Class Imbalance

이 페이지는 neural model이 **무엇을 틀렸다고 판단하고 어떻게 학습 신호를 만드는지** 설명한다.

AASSR에서는 DQN, Prophecy, Critic이 서로 다른 prediction target을 가지므로 loss의 의미도 다르다.

---

# 1. Loss function이란?

Model prediction과 target의 차이를 scalar로 측정한다.

```math
L(\theta)
```

Training은 보통:

```math
\min_\theta L(\theta)
```

를 수행하는 과정이다.

Loss는 **환경 reward와 다르다.**

```text
Reward
→ RL task objective의 외부 신호

Loss
→ model parameter를 학습시키는 optimization objective
```

AASSR에서 status classification loss를 추가한다고 외부 reward shaping이 생기는 것은 아니다.

---

# 2. Mean Squared Error

Regression에서 대표적인 loss:

```math
MSE=\frac1N\sum_i(\hat y_i-y_i)^2
```

큰 error에 제곱으로 강한 penalty를 준다.

Value regression이나 continuous prediction에 사용할 수 있다.

---

# 3. Mean Absolute Error

```math
MAE=\frac1N\sum_i|\hat y_i-y_i|
```

MSE보다 outlier에 덜 민감하다.

---

# 4. Huber / Smooth L1 Loss

작은 error에서는 제곱, 큰 error에서는 절대값에 가까운 형태를 쓴다.

대표 형태:

```math
L_\delta(a)=
\begin{cases}
\frac12a^2,& |a|\le\delta\\
\delta(|a|-\frac12\delta),& \text{otherwise}
\end{cases}
```

MSE의 smoothness와 MAE의 outlier robustness를 절충한다.

DQN/return regression에서 자주 사용된다.

AASSR current Critic도 Smooth L1 계열과 연결된다.

---

# 5. Classification

Target이 discrete category인 문제다.

예:

```text
HTTP status class
200 / 302 / 400 / 401 / 403 / 404 / 409 / 429
```

이를 continuous scalar regression으로 보면 status 숫자 사이의 거리 자체에 의미가 있다고 잘못 가정할 수 있다.

AASSR은 status를 categorical prediction으로 다룬다.

---

# 6. Softmax

Logit `z_c`를 category probability로 바꾼다.

```math
p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}
```

각 `p_c`는 0~1이고 전체 합은 1이다.

---

# 7. Cross-Entropy Loss

정답 category의 predicted probability가 높아지도록 학습한다.

One-hot target `y`에 대해:

```math
L=-\sum_c y_c\log p_c
```

정답 class 하나라면 사실상:

```math
L=-\log p_{correct}
```

이다.

---

# 8. Binary Cross Entropy

여러 label이 독립적으로 참/거짓일 수 있는 multi-label 문제에서는 sigmoid + BCE를 사용할 수 있다.

```math
L=-[y\log p+(1-y)\log(1-p)]
```

Legal action mask처럼 여러 action availability bit를 각각 예측하는 문제와 개념적으로 연결될 수 있다.

---

# 9. Categorical과 Multi-label 차이

HTTP status:

```text
한 transition에서 하나의 latest status
→ mutually exclusive categorical
```

Legal action mask:

```text
여러 action이 동시에 legal일 수 있음
→ multi-label binary vector
```

Prediction target의 구조에 맞는 loss를 써야 한다.

---

# 10. Class imbalance

어떤 category가 압도적으로 많을 수 있다.

```text
200 → 90%
403 → 4%
404 → 3%
429 → 1%
기타 → 2%
```

항상 200이라고 예측해도 accuracy가 90%가 된다.

하지만 rare `429` 같은 outcome이 planner에 매우 중요할 수 있다.

---

# 11. Accuracy paradox

Imbalanced dataset에서 단순 accuracy가 model quality를 과장할 수 있다.

예:

```text
1000 samples
990 normal
10 critical failure
```

모두 normal이라고 예측:

```text
accuracy = 99%
critical failure recall = 0%
```

따라서 decision-critical rare class는 별도 metric이 필요하다.

---

# 12. Class weighting

Rare class의 loss contribution을 크게 줄 수 있다.

```math
L=-\sum_c w_cy_c\log p_c
```

`w_c`는 frequency 기반으로 정할 수 있다.

중요:

```text
rare class weight ↑
```

는 "그 status가 task에서 더 나쁘다"는 reward 의미가 아니다.

단지 **training dataset imbalance를 보정**하는 optimization 선택이다.

---

# 13. Oversampling

Rare class sample을 minibatch에 더 자주 뽑는다.

```text
raw frequency
→ balanced sampler
→ training batch
```

Class weighting과 비슷한 목적이지만 구현 방식이 다르다.

---

# 14. Undersampling

Majority class sample 일부를 줄인다.

장점:

- balance 개선

단점:

- 실제 data를 버릴 수 있음

---

# 15. Focal Loss

쉬운 majority sample의 loss weight를 줄이고 어려운 sample에 더 집중하도록 설계된 classification loss다.

대표 형태:

```math
FL(p_t)=-(1-p_t)^\gamma\log p_t
```

AASSR current status model이 반드시 focal loss를 쓴다는 뜻은 아니다. Class imbalance 문제를 이해하기 위한 관련 일반 개념이다.

---

# 16. Multi-task loss

World model은 여러 target을 동시에 예측할 수 있다.

예:

```text
next relational state
status
legal action mask
terminal class
mixture probability
```

전체 loss를:

```math
L=\lambda_1L_{state}+\lambda_2L_{status}+\lambda_3L_{mask}+\lambda_4L_{terminal}+\cdots
```

처럼 조합할 수 있다.

여기서 `λ`는 각 training objective의 상대적인 최적화 scale을 정한다.

이 weight 역시 environment reward와는 별개다.

---

# 17. Loss weight와 Reward weight는 다르다

매우 중요한 구분이다.

```text
Status loss weight ↑
→ world model이 status prediction을 더 강하게 학습

Status reward +0.2/-0.2
→ agent의 task objective 자체를 바꿈
```

AASSR current repair는 decision-critical status를 **prediction target/validation metric에서 강화**하지만 sparse external reward를 status-based shaping으로 바꾸지 않는다.

---

# 18. Terminal classification

다음 state가:

```text
active
success
true failure
truncation
```

중 무엇인지 예측해야 할 수 있다.

Failure와 truncation이 같은 class로 뭉치면 planner/value semantics가 왜곡될 수 있다.

관련 페이지:

- [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)
- [Prophecy](Prophecy)

---

# 19. Mask prediction loss

Predicted legal action mask와 actual mask를 비교한다.

단순 exact-match만 보면 action 하나 차이도 전체 실패가 된다.

다른 metric으로:

- per-bit BCE
- precision / recall
- Jaccard similarity

같은 방법을 사용할 수 있다.

AASSR semantic evaluation에서는 legal action structure가 decision-critical하므로 mask quality를 별도 요소로 본다.

---

# 20. Jaccard similarity

두 set `A`, `B`의 overlap:

```math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
```

Legal action set 비교에 자연스럽다.

둘 다 빈 set인 경우 convention을 별도로 정해야 한다.

---

# 21. Calibration loss와 accuracy

Classification accuracy가 높아도 probability가 잘 calibrated되어 있지 않을 수 있다.

```text
항상 99.9% confidence
실제 accuracy 80%
```

이면 overconfident하다.

AASSR에서는 raw softmax confidence 대신 real holdout semantic reliability를 별도로 본다.

관련 페이지:

- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)
- [Calibration](Calibration)

---

# 22. NLL

Negative Log-Likelihood는 probabilistic model이 실제 data에 얼마나 높은 likelihood를 주는지 평가하는 대표 objective다.

```math
NLL=-\sum_i\log p_\theta(y_i|x_i)
```

Mixture model training에서도 likelihood-based objective가 자연스럽다.

---

# 23. Mixture likelihood

Mixture distribution:

```math
p(y|x)=\sum_m\pi_m(x)p_m(y|x)
```

의 likelihood를 최대화한다.

여러 component가 같은 mode로 collapse하거나 한 component만 모든 mass를 가져가는 failure가 생길 수 있어 diagnostic이 필요하다.

관련 페이지:

- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 24. Validation metric과 Training loss

같은 것이 아닐 수 있다.

Training:

```text
continuous descriptor loss
status CE
mask BCE
terminal CE
```

Validation:

```text
semantic score
status accuracy
mask Jaccard
probability-weighted quality
```

처럼 **optimizer가 직접 최소화하는 값과 연구자가 실제 usefulness를 측정하는 metric**이 다를 수 있다.

---

# 25. Proxy objective 문제

Training loss를 낮추는 것이 최종 agent success를 높인다는 보장은 없다.

```text
World-model MSE ↓
```

이어도:

```text
Imagination success contribution = 0
```

일 수 있다.

그래서 AASSR은 model metric과 downstream intervention/task metric을 분리한다.

관련 페이지:

- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 26. Reward loss / TD loss

DQN은 supervised label dataset 대신 Bellman TD target을 만든다.

```math
y=r+\gamma\max_{a'}Q_{target}(s',a')
```

그리고:

```math
L=(Q_\theta(s,a)-y)^2
```

같은 regression loss로 학습한다.

즉 RL에서도 결국 neural network optimization 수준에서는 loss function이 존재한다.

관련 페이지:

- [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD)

---

# 27. Critic regression

AASSR Critic은 실제 sparse-return target을 sequence input에서 회귀한다.

```text
sequence → predicted return
```

Sparse target이 대부분 0이면 class imbalance와 비슷한 **target starvation** 문제가 생길 수 있다.

성공/failure trajectory가 너무 적으면 Critic이 branch를 구분하기 어렵다.

---

# 28. Loss scale

Multi-task loss에서 한 항의 숫자 scale이 지나치게 크면 gradient를 지배할 수 있다.

```text
L_state ≈ 0.01
L_status ≈ 4.0
```

같다면 단순 합산 시 status가 최적화를 지배할 수 있다.

Weighting/normalization을 설계할 때 각 objective의 의미와 scale을 함께 봐야 한다.

---

# 29. AASSR에서 어디에 연결되는가?

```text
DQN
→ TD regression loss

Prophecy
→ state/status/mask/terminal/mixture objectives

Critic
→ sparse return regression

Calibration
→ training loss가 아니라 holdout correctness/reliability 평가
```

---

# 30. 다음으로 읽기

- [Neural Networks & Optimization](Neural-Networks-and-Optimization)
- [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD)
- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)
- [Prophecy](Prophecy)
- [Critic](Critic)

관련 색인: **[Concept Index](Concept-Index)**