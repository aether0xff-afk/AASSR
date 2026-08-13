# Neural Networks and Optimization

AASSR의 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD), [Prophecy(미래 예측 모델)](Prophecy), [Critic(미래 가치 평가기)](Critic)은 모두 어떤 형태로든 **함수 근사([함수(function)](Terminology-Guide) [근사(approximation)](Value-Functions-and-Bellman-Equation))** 를 사용한다. 이 페이지는 신경망을 처음 보는 독자가 AASSR 구현 문서를 따라갈 수 있을 정도의 기초를 정리한다.

---

# 1. 함수 근사

우리가 알고 싶은 함수가 있다고 하자.

```math
f(x)=y
```

하지만 정확한 식을 모른다.

Neural [신경망(network)](Neural-Networks-and-Optimization)는 [학습 파라미터(parameter)](Neural-Networks-and-Optimization) `θ`를 가진 함수:

```math
f_\theta(x)
```

를 학습해 원하는 mapping을 근사한다.

강화학습에서는 예를 들어:

```math
Q_\theta(s,a)
```

로 [행동(action)](Reinforcement-Learning) [가치(value)](Value-Functions-and-Bellman-Equation)를 근사할 수 있다.

[세계(World)](Model-Based-RL-and-World-Models) [학습 모델(model)](Terminology-Guide)에서는:

```math
\hat P_\theta(s'|s,a)
```

를 근사할 수 있다.

---

# 2. Neuron / Linear layer

가장 기본적인 [처리 계층(layer)](Research-Architecture)는:

```math
z=Wx+b
```

이다.

`W`는 [가중치(weight)](Neural-Networks-and-Optimization) matrix, `b`는 [편향(bias)](Ablation-Benchmarking-and-Reproducibility)다.

Linear transformation만 여러 번 쌓으면 전체가 다시 linear transformation이므로 nonlinear activation이 필요하다.

---

# 3. Activation function

대표적인 activation:

- ReLU
- GELU
- tanh
- sigmoid

예:

```math
ReLU(x)=\max(0,x)
```

Nonlinearity 덕분에 [신경망 기반(neural)](Neural-Networks-and-Optimization) 신경망가 복잡한 함수를 표현할 수 있다.

---

# 4. Forward pass

Input을 신경망에 넣어 [출력(output)](Terminology-Guide)을 계산하는 과정이다.

```text
input
 ↓
layer 1
 ↓
activation
 ↓
layer 2
 ↓
output
```

[DQN](Q-Learning-DQN-and-TD)에서는 [Q값(Q-value)](Value-Functions-and-Bellman-Equation)가 나오고, [Prophecy](Prophecy)에서는 [미래(future)](Counterfactual-Planning-and-Search) [예측(prediction)](Terminology-Guide) 파라미터가 나오며, [Critic](Critic)에서는 [누적 보상(return)](Value-Functions-and-Bellman-Equation) [추정값(estimate)](Value-Functions-and-Bellman-Equation)가 나올 수 있다.

---

# 5. Loss function

Prediction과 [대상 또는 학습 목표값(target)](Terminology-Guide)의 차이를 하나의 [숫자 하나인 스칼라(scalar)](Neural-Networks-and-Optimization)로 만든다.

```math
L(\theta)
```

[학습(Training)](Reinforcement-Learning)은 [학습 손실(loss)](Loss-Functions-and-Class-Imbalance)를 작게 만드는 `θ`를 찾는 과정이다.

더 자세한 학습 손실 종류는 **[Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)** 에서 본다.

---

# 6. Gradient

[학습 손실(Loss)](Loss-Functions-and-Class-Imbalance)가 파라미터마다 어느 방향으로 얼마나 변하는지를 나타낸다.

```math
\nabla_\theta L
```

Gradient descent의 기본 [학습 갱신(update)](Neural-Networks-and-Optimization):

```math
\theta \leftarrow \theta-\alpha\nabla_\theta L
```

`α`는 [학습(learning)](Reinforcement-Learning) [비율(rate)](Terminology-Guide)다.

---

# 7. Backpropagation

Network 출력에서 학습 손실를 계산한 뒤 chain [규칙(rule)](Terminology-Guide)로 각 파라미터 [기울기(gradient)](Neural-Networks-and-Optimization)를 뒤로 전파하는 알고리즘이다.

```text
forward
x → network → prediction → loss

backward
loss → gradients → parameters update
```

---

# 8. Learning rate

한 학습 갱신에서 파라미터를 얼마나 크게 움직일지 정한다.

```text
너무 큼
→ overshoot / instability

너무 작음
→ training 매우 느림
```

Hyperparameter tuning에서 가장 중요한 값 중 하나다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 9. Optimizer

Gradient를 이용해 파라미터를 업데이트하는 알고리즘이다.

대표적으로:

- SGD
- Adam
- AdamW

Adam은 각 파라미터의 기울기 1차/2차 moment를 추적해 adaptive [단계(step)](Terminology-Guide) size를 사용한다.

AASSR의 여러 신경망 기반 [구성요소(component)](Research-Architecture) 역시 [신경망 파라미터를 갱신하는 최적화 알고리즘(optimizer)](Neural-Networks-and-Optimization)를 통해 기울기 학습 갱신를 수행한다.

---

# 10. Batch와 Minibatch

전체 [데이터 묶음(dataset)](Ablation-Benchmarking-and-Reproducibility)을 한 번에 쓰지 않고 일부 [표본(sample)](Ablation-Benchmarking-and-Reproducibility) 묶음으로 기울기를 계산한다.

```text
Replay data
 ↓ sample batch
[transition 1, ..., transition B]
 ↓
network update
```

Batch size가 너무 작으면 기울기 [잡음(noise)](Stochasticity-Uncertainty-and-Probability)가 크고, 너무 크면 [기억(memory)](GRU-and-Sequence-Models)/[계산(compute)](Reproduction)가 증가한다.

---

# 11. Epoch와 Gradient update

Supervised 학습에서는 데이터셋 전체를 한 번 본 것을 epoch라고 자주 부른다.

RL [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries) [학습(training)](Terminology-Guide)에서는 고정 데이터셋 epoch보다 **[환경(environment)](Reinforcement-Learning) [상태 전이(transition)](MDP-and-POMDP)s 수와 기울기 학습 갱신 수**가 더 자연스러운 단위일 수 있다.

AASSR 문서에서 `2k transitions`와 `gradient_updates`를 구분해야 하는 이유다.

---

# 12. Overfitting

학습 [데이터(data)](Terminology-Guide)에는 매우 잘 맞지만 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 데이터에는 잘 맞지 않는 현상이다.

```text
train error ↓
test error ↑
```

AASSR에서는 특히 [실제 개체를 구분하는(concrete)](State-Representation) ID [이름이나 사례를 그대로 외우는 암기(memorization)](Relational-Representation-and-Generalization)과 [검증(validation)](Ablation-Benchmarking-and-Reproducibility)/[검사 또는 테스트(test)](Ablation-Benchmarking-and-Reproducibility) 반복 tuning이 overfitting을 만들 수 있다.

관련 페이지:

- [Relational Representation & Generalization](Relational-Representation-and-Generalization)
- [Causality, Leakage & Evaluation](Causality-Leakage-and-Evaluation)

---

# 13. Underfitting

Model capacity가 부족하거나 학습이 충분하지 않아 [학습 데이터(training data)](Terminology-Guide)조차 잘 설명하지 못하는 상태다.

[Prophecy](Prophecy)가 미래 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability) [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)를 제대로 분리하지 못하거나 [Critic](Critic)이 모든 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)에 비슷한 값을 내는 경우 capacity/학습 부족 가능성을 생각할 수 있다.

---

# 14. Regularization

Overfitting을 줄이기 위한 방법들이다.

예:

- 가중치 decay
- dropout
- 데이터 augmentation
- early stopping

하지만 RL에서는 [데이터 분포 변화(distribution shift)](Critic-Support-and-OOD)와 대상/목표값 drift가 있어 일반 supervised 학습보다 진단이 복잡할 수 있다.

---

# 15. Train / Validation / Test

```text
Train
→ parameter 학습

Validation
→ hyperparameter / model selection

Test
→ 최종 성능 평가
```

AASSR에서는 [실제 환경에서 관측된(real)](Research-Jargon-Guide) 경험 재사용 [검증용 분리 데이터(holdout)](Calibration), development [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility), [최종(final)](Ablation-Benchmarking-and-Reproducibility) 학습 중 보지 못한 표준 비교 실험를 목적에 따라 분리한다.

---

# 16. Normalization

Feature scale이 크게 다르면 [최적화(optimization)](Neural-Networks-and-Optimization)이 어려워질 수 있다.

예:

```text
feature A: 0~1
feature B: 0~100000
```

[관계 기반(Relational)](Relational-Representation-and-Generalization) [상태를 요약한 표현(descriptor)](State-Representation)에서 [횟수(count)](Terminology-Guide)를 일정 범위로 normalize하는 것도 같은 일반적 문제와 연결된다.

---

# 17. One-hot encoding

Categorical 값을 각 [범주(category)](Loss-Functions-and-Class-Imbalance)별 dimension으로 표현한다.

예:

```text
status 200 → [1,0,0,...]
status 403 → [0,0,0,1,...]
```

HTTP [상태 코드(status)](Terminology-Guide)를 numeric continuous 가치로 해석하지 않고 [범주형(categorical)](Loss-Functions-and-Class-Imbalance) [정보 채널(channel)](Causality-Leakage-and-Evaluation)로 처리하는 AASSR 설계와 연결된다.

관련 페이지:

- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)
- [State Representation](State-Representation)

---

# 18. Embedding

Discrete 범주를 [학습된(learned)](Neural-Networks-and-Optimization) continuous [수치 벡터(vector)](Neural-Networks-and-Optimization)로 mapping하는 방법이다.

```text
ID/category
 ↓ embedding table
vector
```

AASSR [현재(current)](Current-Status) [관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)은 실제 개체를 구분하는 ID embedding 암기보다 [역할(role)](Relational-Representation-and-Generalization)/relationship [학습에 사용하는 특징(features)](Terminology-Guide)를 강조한다.

---

# 19. Function approximation과 Generalization

Neural 신경망는 정확히 본 표본뿐 아니라 비슷한 [입력(input)](Terminology-Guide)에도 출력을 만든다.

이것이 [일반화(generalization)](Relational-Representation-and-Generalization)의 장점이다.

하지만 학습 [데이터 근거(support)](Critic-Support-and-OOD) 밖에서도 숫자를 출력하므로 [학습 분포 밖(OOD)](Critic-Support-and-OOD) [학습 범위 밖으로 값을 추정하는 외삽(extrapolation)](Critic-Support-and-OOD)의 원인이 되기도 한다.

관련 페이지:

- [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 20. Gradient clipping

Gradient norm이 너무 커지는 것을 제한한다.

```math
\|g\|>c
\Rightarrow
g\leftarrow c\frac{g}{\|g\|}
```

RNN/[GRU(게이트 순환 유닛)](GRU-and-Sequence-Models) 학습 안정화에 자주 사용된다.

관련 페이지:

- [GRU & Sequence Models](GRU-and-Sequence-Models)

---

# 21. Ensemble training

여러 신경망를 독립적으로 학습할 수 있다.

차이를 만들기 위해:

- initialization
- [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries) 표본
- [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) 최적화

등을 다르게 할 수 있다.

AASSR [Prophecy](Prophecy) [여러 모델을 함께 쓰는 앙상블(ensemble)](Mixture-Ensemble-and-Calibration)의 [불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability) [증거(evidence)](Evidence-Matrix)와 연결된다.

관련 페이지:

- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 22. GPU batching

GPU는 작은 학습 모델 call을 수천 번 순차 실행하는 것보다 큰 tensor [여러 입력 묶음(batch)](Reproduction)를 한 번에 처리할 때 효율적인 경우가 많다.

AASSR [현재 세대(current-generation)](Current-Status)에서 [Prophecy](Prophecy)/[Critic](Critic) [탐색 깊이(depth)](Counterfactual-Planning-and-Search) [묶음 처리(batching)](Reproduction)이 큰 성능 최적화였던 이유다.

```text
scalar calls × N
→ overhead 큼

batch of N
→ parallel tensor compute
```

---

# 23. CPU ↔ GPU synchronization

GPU 연산 중간에 `.item()`처럼 값을 CPU로 자주 가져오면 GPU가 완료될 때까지 기다리는 synchronization [지점(point)](Terminology-Guide)가 생길 수 있다.

많은 작은 sync는 성능을 크게 떨어뜨릴 수 있다.

AASSR 현재 hardware optimizations가 묶음 [전이(transfer)](Relational-Representation-and-Generalization)와 bulk 학습 손실 bookkeeping을 사용하는 이유와 연결된다.

---

# 24. TF32 / Mixed precision

현대 GPU는 일부 matrix multiplication에서 낮은 precision 형식을 사용해 속도를 높일 수 있다.

하지만 reproducibility와 numerical [행동 양상(behavior)](Experiments)가 달라질 수 있어 experimental config에 기록하는 것이 좋다.

---

# 25. Neural network가 '이해'한다는 말

Network가 높은 예측 [정확도(accuracy)](Ablation-Benchmarking-and-Reproducibility)를 보인다고 해서 내부적으로 인간과 같은 의미를 이해한다고 자동으로 말할 수는 없다.

연구 문서에서는 가능한 한 operational [연구 주장(claim)](Evidence-Matrix)을 사용한다.

```text
"이 관계를 이해한다"
보다
"unseen identifier permutation에서 relational feature를 이용해 prediction accuracy를 유지한다"
```

처럼 검증 가능한 표현이 더 안전하다.

---

# 26. AASSR에서 어디에 쓰이나?

```text
Relational DQN
→ Q-function approximation

Prophecy
→ stochastic future model

Critic
→ sparse-return sequence value approximation
```

각 신경망는 대상/목표값과 역할이 다르다.

---

# 27. 다음으로 읽기

- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)
- [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD)
- [GRU & Sequence Models](GRU-and-Sequence-Models)
- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)
- [Critic, Support & OOD](Critic-Support-and-OOD)

관련 색인: **[Concept Index](Concept-Index)**