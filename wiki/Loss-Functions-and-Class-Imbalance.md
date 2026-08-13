# Loss Functions and Class Imbalance

이 페이지는 [신경망 기반(neural)](Neural-Networks-and-Optimization) [학습 모델(model)](Terminology-Guide)이 **무엇을 틀렸다고 판단하고 어떻게 학습 신호를 만드는지** 설명한다.

AASSR에서는 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD), [Prophecy(미래 예측 모델)](Prophecy), [Critic(미래 가치 평가기)](Critic)이 서로 다른 [예측(prediction)](Terminology-Guide) [대상 또는 학습 목표값(target)](Terminology-Guide)을 가지므로 [학습 손실(loss)](Loss-Functions-and-Class-Imbalance)의 의미도 다르다.

---

# 1. Loss function이란?

Model 예측과 대상/목표값의 차이를 [숫자 하나인 스칼라(scalar)](Neural-Networks-and-Optimization)로 측정한다.

```math
L(\theta)
```

[학습(Training)](Reinforcement-Learning)은 보통:

```math
\min_\theta L(\theta)
```

를 수행하는 과정이다.

[학습 손실(Loss)](Loss-Functions-and-Class-Imbalance)는 **환경 [보상(reward)](Sparse-Reward-and-Credit-Assignment)와 다르다.**

```text
Reward
→ RL task objective의 외부 신호

Loss
→ model parameter를 학습시키는 optimization objective
```

AASSR에서 [상태 코드(status)](Terminology-Guide) classification 학습 손실를 추가한다고 외부 보상 [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)이 생기는 것은 아니다.

---

# 2. Mean Squared Error

Regression에서 대표적인 학습 손실:

```math
MSE=\frac1N\sum_i(\hat y_i-y_i)^2
```

큰 [오차(error)](Loss-Functions-and-Class-Imbalance)에 제곱으로 강한 penalty를 준다.

Value [회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)이나 continuous 예측에 사용할 수 있다.

---

# 3. Mean Absolute Error

```math
MAE=\frac1N\sum_i|\hat y_i-y_i|
```

MSE보다 outlier에 덜 민감하다.

---

# 4. Huber / Smooth L1 Loss

작은 오차에서는 제곱, 큰 오차에서는 절대값에 가까운 형태를 쓴다.

대표 형태:

```math
L_\delta(a)=
\begin{cases}
\frac12a^2,& |a|\le\delta\\
\delta(|a|-\frac12\delta),& \text{otherwise}
\end{cases}
```

MSE의 smoothness와 MAE의 outlier robustness를 절충한다.

[DQN](Q-Learning-DQN-and-TD)/[누적 보상(return)](Value-Functions-and-Bellman-Equation) 회귀 검증에서 자주 사용된다.

AASSR [현재(current)](Current-Status) [Critic](Critic)도 Smooth L1 계열과 연결된다.

---

# 5. Classification

Target이 discrete [범주(category)](Loss-Functions-and-Class-Imbalance)인 문제다.

예:

```text
HTTP status class
200 / 302 / 400 / 401 / 403 / 404 / 409 / 429
```

이를 continuous 스칼라 회귀 검증으로 보면 상태 코드 숫자 사이의 거리 자체에 의미가 있다고 잘못 가정할 수 있다.

AASSR은 상태 코드를 [범주형(categorical)](Loss-Functions-and-Class-Imbalance) 예측으로 다룬다.

---

# 6. Softmax

Logit `z_c`를 범주 [확률(probability)](Stochasticity-Uncertainty-and-Probability)로 바꾼다.

```math
p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}
```

각 `p_c`는 0~1이고 전체 합은 1이다.

---

# 7. Cross-Entropy Loss

정답 범주의 [예측된(predicted)](Terminology-Guide) 확률가 높아지도록 학습한다.

One-hot 대상/목표값 `y`에 대해:

```math
L=-\sum_c y_c\log p_c
```

정답 [범주(class)](Loss-Functions-and-Class-Imbalance) 하나라면 사실상:

```math
L=-\log p_{correct}
```

이다.

---

# 8. Binary Cross Entropy

여러 [정답 범주 표시(label)](Loss-Functions-and-Class-Imbalance)이 독립적으로 참/거짓일 수 있는 multi-label 문제에서는 sigmoid + BCE를 사용할 수 있다.

```math
L=-[y\log p+(1-y)\log(1-p)]
```

Legal [행동(action)](Reinforcement-Learning) [가능/불가능을 표시하는 마스크(mask)](Terminology-Guide)처럼 여러 행동 availability bit를 각각 예측하는 문제와 개념적으로 연결될 수 있다.

---

# 9. Categorical과 Multi-label 차이

HTTP 상태 코드:

```text
한 transition에서 하나의 latest status
→ mutually exclusive categorical
```

Legal 행동 마스크:

```text
여러 action이 동시에 legal일 수 있음
→ multi-label binary vector
```

Prediction 대상/목표값의 구조에 맞는 학습 손실를 써야 한다.

---

# 10. Class imbalance

어떤 범주가 압도적으로 많을 수 있다.

```text
200 → 90%
403 → 4%
404 → 3%
429 → 1%
기타 → 2%
```

항상 200이라고 예측해도 [정확도(accuracy)](Ablation-Benchmarking-and-Reproducibility)가 90%가 된다.

하지만 [드문(rare)](Loss-Functions-and-Class-Imbalance) `429` 같은 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)이 [계획기(planner)](Counterfactual-Planning-and-Search)에 매우 중요할 수 있다.

---

# 11. Accuracy paradox

Imbalanced [데이터 묶음(dataset)](Ablation-Benchmarking-and-Reproducibility)에서 단순 정확도가 학습 모델 [품질(quality)](Ablation-Benchmarking-and-Reproducibility)를 과장할 수 있다.

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

따라서 [의사결정에 중요한(decision-critical)](Calibration) 드문 범주는 별도 [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)이 필요하다.

---

# 12. Class weighting

Rare 범주의 학습 손실 contribution을 크게 줄 수 있다.

```math
L=-\sum_c w_cy_c\log p_c
```

`w_c`는 frequency 기반으로 정할 수 있다.

중요:

```text
rare class weight ↑
```

는 "그 상태 코드가 [연구 과제(task)](Sparse-Reward-Problem)에서 더 나쁘다"는 보상 의미가 아니다.

단지 **[학습(training)](Terminology-Guide) 데이터셋 [데이터 수의 불균형(imbalance)](Loss-Functions-and-Class-Imbalance)를 보정**하는 [최적화(optimization)](Neural-Networks-and-Optimization) 선택이다.

---

# 13. Oversampling

Rare 범주 [표본(sample)](Ablation-Benchmarking-and-Reproducibility)을 minibatch에 더 자주 뽑는다.

```text
raw frequency
→ balanced sampler
→ training batch
```

Class weighting과 비슷한 목적이지만 구현 방식이 다르다.

---

# 14. Undersampling

Majority 범주 표본 일부를 줄인다.

장점:

- balance 개선

단점:

- 실제 [데이터(data)](Terminology-Guide)를 버릴 수 있음

---

# 15. Focal Loss

쉬운 majority 표본의 학습 손실 [가중치(weight)](Neural-Networks-and-Optimization)를 줄이고 어려운 표본에 더 집중하도록 설계된 classification 학습 손실다.

대표 형태:

```math
FL(p_t)=-(1-p_t)^\gamma\log p_t
```

AASSR 현재 상태 코드 학습 모델이 반드시 focal 학습 손실를 쓴다는 뜻은 아니다. Class 불균형 문제를 이해하기 위한 관련 일반 개념이다.

---

# 16. Multi-task loss

[세계(World)](Model-Based-RL-and-World-Models) 학습 모델은 여러 대상/목표값을 동시에 예측할 수 있다.

예:

```text
next relational state
status
legal action mask
terminal class
mixture probability
```

전체 학습 손실를:

```math
L=\lambda_1L_{state}+\lambda_2L_{status}+\lambda_3L_{mask}+\lambda_4L_{terminal}+\cdots
```

처럼 조합할 수 있다.

여기서 `λ`는 각 학습 [학습 목표(objective)](Terminology-Guide)의 상대적인 최적화 scale을 정한다.

이 가중치 역시 [환경(environment)](Reinforcement-Learning) 보상와는 별개다.

---

# 17. Loss weight와 Reward weight는 다르다

매우 중요한 구분이다.

```text
Status loss weight ↑
→ world model이 status prediction을 더 강하게 학습

Status reward +0.2/-0.2
→ agent의 task objective 자체를 바꿈
```

AASSR 현재 [문제 수정(repair)](Development-History)는 의사결정에 중요한 상태 코드를 **예측 대상/목표값/[검증(validation)](Ablation-Benchmarking-and-Reproducibility) 평가지표에서 강화**하지만 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [환경이 주는 외부(external)](Terminology-Guide) 보상를 status-based 형태 조정으로 바꾸지 않는다.

---

# 18. Terminal classification

다음 [상태(state)](State-Representation)가:

```text
active
success
true failure
truncation
```

중 무엇인지 예측해야 할 수 있다.

Failure와 [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)이 같은 범주로 뭉치면 계획기/[가치(value)](Value-Functions-and-Bellman-Equation) [의미 규칙(semantics)](State-Representation)가 왜곡될 수 있다.

관련 페이지:

- [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)
- [Prophecy](Prophecy)

---

# 19. Mask prediction loss

Predicted [가능 행동 마스크(legal action mask)](Prophecy)와 actual 마스크를 비교한다.

단순 exact-match만 보면 행동 하나 차이도 전체 실패가 된다.

다른 평가지표으로:

- per-bit BCE
- precision / [놓치지 않고 찾아낸 비율인 재현율(recall)](Ablation-Benchmarking-and-Reproducibility)
- Jaccard [유사도(similarity)](Critic-Support-and-OOD)

같은 방법을 사용할 수 있다.

AASSR [의미 기준(semantic)](State-Representation) [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)에서는 [현재 허용된(legal)](Terminology-Guide) 행동 [구조(structure)](Research-Architecture)가 의사결정에 중요한하므로 마스크 품질를 별도 요소로 본다.

---

# 20. Jaccard similarity

두 [집합(set)](Terminology-Guide) `A`, `B`의 overlap:

```math
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
```

Legal 행동 집합 비교에 자연스럽다.

둘 다 빈 집합인 경우 convention을 별도로 정해야 한다.

---

# 21. Calibration loss와 accuracy

Classification 정확도가 높아도 확률가 잘 calibrated되어 있지 않을 수 있다.

```text
항상 99.9% confidence
실제 accuracy 80%
```

이면 overconfident하다.

AASSR에서는 [가공하지 않은 원본(raw)](State-Representation) softmax [예측 신뢰 정도(confidence)](Calibration) 대신 [실제 환경에서 관측된(real)](Research-Jargon-Guide) [검증용 분리 데이터(holdout)](Calibration) 의미 기준 [신뢰도(reliability)](Calibration)를 별도로 본다.

관련 페이지:

- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)
- [Calibration](Calibration)

---

# 22. NLL

Negative Log-Likelihood는 probabilistic 학습 모델이 실제 데이터에 얼마나 높은 likelihood를 주는지 평가하는 대표 학습 목표다.

```math
NLL=-\sum_i\log p_\theta(y_i|x_i)
```

Mixture 학습 모델 학습에서도 likelihood-based 학습 목표가 자연스럽다.

---

# 23. Mixture likelihood

Mixture [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability):

```math
p(y|x)=\sum_m\pi_m(x)p_m(y|x)
```

의 likelihood를 최대화한다.

여러 [구성요소(component)](Research-Architecture)가 같은 [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)로 [여러 결과가 하나로 뭉개지는 붕괴(collapse)](Mixture-Ensemble-and-Calibration)하거나 한 구성요소만 모든 [확률 질량(mass)](Stochasticity-Uncertainty-and-Probability)를 가져가는 [실패(failure)](Replay-Buffer-and-Episode-Boundaries)가 생길 수 있어 [진단 실험(diagnostic)](Evidence-Matrix)이 필요하다.

관련 페이지:

- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 24. Validation metric과 Training loss

같은 것이 아닐 수 있다.

학습:

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

처럼 **[신경망 파라미터를 갱신하는 최적화 알고리즘(optimizer)](Neural-Networks-and-Optimization)가 직접 최소화하는 값과 연구자가 실제 usefulness를 측정하는 평가지표**이 다를 수 있다.

---

# 25. Proxy objective 문제

학습 학습 손실를 낮추는 것이 최종 [에이전트(agent)](Reinforcement-Learning) [성공(success)](Terminology-Guide)를 높인다는 보장은 없다.

```text
World-model MSE ↓
```

이어도:

```text
Imagination success contribution = 0
```

일 수 있다.

그래서 AASSR은 학습 모델 평가지표과 downstream [실제 행동 개입(intervention)](Imagination)/연구 과제 평가지표을 분리한다.

관련 페이지:

- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 26. Reward loss / TD loss

[DQN](Q-Learning-DQN-and-TD)은 supervised 라벨 데이터셋 대신 Bellman TD 대상/목표값을 만든다.

```math
y=r+\gamma\max_{a'}Q_{target}(s',a')
```

그리고:

```math
L=(Q_\theta(s,a)-y)^2
```

같은 회귀 검증 학습 손실로 학습한다.

즉 RL에서도 결국 신경망 기반 [신경망(network)](Neural-Networks-and-Optimization) 최적화 수준에서는 학습 손실 [함수(function)](Terminology-Guide)이 존재한다.

관련 페이지:

- [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD)

---

# 27. Critic regression

AASSR [Critic](Critic)은 실제 sparse-누적 보상 대상/목표값을 [순서열(sequence)](GRU-and-Sequence-Models) [입력(input)](Terminology-Guide)에서 회귀한다.

```text
sequence → predicted return
```

Sparse 대상/목표값이 대부분 0이면 범주 불균형와 비슷한 **대상/목표값 starvation** 문제가 생길 수 있다.

성공/실패 [경험 경로(trajectory)](Reinforcement-Learning)가 너무 적으면 [Critic](Critic)이 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)를 구분하기 어렵다.

---

# 28. Loss scale

Multi-task 학습 손실에서 한 항의 숫자 scale이 지나치게 크면 [기울기(gradient)](Neural-Networks-and-Optimization)를 지배할 수 있다.

```text
L_state ≈ 0.01
L_status ≈ 4.0
```

같다면 단순 합산 시 상태 코드가 최적화를 지배할 수 있다.

Weighting/[수치 범위를 맞추는 정규화(normalization)](Neural-Networks-and-Optimization)을 설계할 때 각 학습 목표의 의미와 scale을 함께 봐야 한다.

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