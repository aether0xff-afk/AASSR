# Neural Networks and Optimization

AASSR의 DQN, Prophecy, Critic은 모두 어떤 형태로든 **함수 근사(function approximation)** 를 사용한다. 이 페이지는 신경망을 처음 보는 독자가 AASSR 구현 문서를 따라갈 수 있을 정도의 기초를 정리한다.

---

# 1. 함수 근사

우리가 알고 싶은 함수가 있다고 하자.

```math
f(x)=y
```

하지만 정확한 식을 모른다.

Neural network는 parameter `θ`를 가진 함수:

```math
f_\theta(x)
```

를 학습해 원하는 mapping을 근사한다.

강화학습에서는 예를 들어:

```math
Q_\theta(s,a)
```

로 action value를 근사할 수 있다.

World model에서는:

```math
\hat P_\theta(s'|s,a)
```

를 근사할 수 있다.

---

# 2. Neuron / Linear layer

가장 기본적인 layer는:

```math
z=Wx+b
```

이다.

`W`는 weight matrix, `b`는 bias다.

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

Nonlinearity 덕분에 neural network가 복잡한 함수를 표현할 수 있다.

---

# 4. Forward pass

Input을 network에 넣어 output을 계산하는 과정이다.

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

DQN에서는 Q-value가 나오고, Prophecy에서는 future prediction parameter가 나오며, Critic에서는 return estimate가 나올 수 있다.

---

# 5. Loss function

Prediction과 target의 차이를 하나의 scalar로 만든다.

```math
L(\theta)
```

Training은 loss를 작게 만드는 `θ`를 찾는 과정이다.

더 자세한 loss 종류는 **[Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)** 에서 본다.

---

# 6. Gradient

Loss가 parameter마다 어느 방향으로 얼마나 변하는지를 나타낸다.

```math
\nabla_\theta L
```

Gradient descent의 기본 update:

```math
\theta \leftarrow \theta-\alpha\nabla_\theta L
```

`α`는 learning rate다.

---

# 7. Backpropagation

Network output에서 loss를 계산한 뒤 chain rule로 각 parameter gradient를 뒤로 전파하는 알고리즘이다.

```text
forward
x → network → prediction → loss

backward
loss → gradients → parameters update
```

---

# 8. Learning rate

한 update에서 parameter를 얼마나 크게 움직일지 정한다.

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

Gradient를 이용해 parameter를 업데이트하는 알고리즘이다.

대표적으로:

- SGD
- Adam
- AdamW

Adam은 각 parameter의 gradient 1차/2차 moment를 추적해 adaptive step size를 사용한다.

AASSR의 여러 neural component 역시 optimizer를 통해 gradient update를 수행한다.

---

# 10. Batch와 Minibatch

전체 dataset을 한 번에 쓰지 않고 일부 sample 묶음으로 gradient를 계산한다.

```text
Replay data
 ↓ sample batch
[transition 1, ..., transition B]
 ↓
network update
```

Batch size가 너무 작으면 gradient noise가 크고, 너무 크면 memory/compute가 증가한다.

---

# 11. Epoch와 Gradient update

Supervised learning에서는 dataset 전체를 한 번 본 것을 epoch라고 자주 부른다.

RL replay training에서는 고정 dataset epoch보다 **environment transitions 수와 gradient update 수**가 더 자연스러운 단위일 수 있다.

AASSR 문서에서 `2k transitions`와 `gradient_updates`를 구분해야 하는 이유다.

---

# 12. Overfitting

Training data에는 매우 잘 맞지만 unseen data에는 잘 맞지 않는 현상이다.

```text
train error ↓
test error ↑
```

AASSR에서는 특히 concrete ID memorization과 validation/test 반복 tuning이 overfitting을 만들 수 있다.

관련 페이지:

- [Relational Representation & Generalization](Relational-Representation-and-Generalization)
- [Causality, Leakage & Evaluation](Causality-Leakage-and-Evaluation)

---

# 13. Underfitting

Model capacity가 부족하거나 training이 충분하지 않아 training data조차 잘 설명하지 못하는 상태다.

Prophecy가 미래 outcome mode를 제대로 분리하지 못하거나 Critic이 모든 branch에 비슷한 값을 내는 경우 capacity/training 부족 가능성을 생각할 수 있다.

---

# 14. Regularization

Overfitting을 줄이기 위한 방법들이다.

예:

- weight decay
- dropout
- data augmentation
- early stopping

하지만 RL에서는 distribution shift와 target drift가 있어 일반 supervised learning보다 진단이 복잡할 수 있다.

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

AASSR에서는 real replay holdout, development benchmark, final unseen benchmark를 목적에 따라 분리한다.

---

# 16. Normalization

Feature scale이 크게 다르면 optimization이 어려워질 수 있다.

예:

```text
feature A: 0~1
feature B: 0~100000
```

Relational descriptor에서 count를 일정 범위로 normalize하는 것도 같은 일반적 문제와 연결된다.

---

# 17. One-hot encoding

Categorical 값을 각 category별 dimension으로 표현한다.

예:

```text
status 200 → [1,0,0,...]
status 403 → [0,0,0,1,...]
```

HTTP status를 numeric continuous value로 해석하지 않고 categorical channel로 처리하는 AASSR 설계와 연결된다.

관련 페이지:

- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)
- [State Representation](State-Representation)

---

# 18. Embedding

Discrete category를 learned continuous vector로 mapping하는 방법이다.

```text
ID/category
 ↓ embedding table
vector
```

AASSR current relational representation은 concrete ID embedding 암기보다 role/relationship features를 강조한다.

---

# 19. Function approximation과 Generalization

Neural network는 정확히 본 sample뿐 아니라 비슷한 input에도 output을 만든다.

이것이 generalization의 장점이다.

하지만 training support 밖에서도 숫자를 출력하므로 OOD extrapolation의 원인이 되기도 한다.

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

RNN/GRU training 안정화에 자주 사용된다.

관련 페이지:

- [GRU & Sequence Models](GRU-and-Sequence-Models)

---

# 21. Ensemble training

여러 network를 독립적으로 학습할 수 있다.

차이를 만들기 위해:

- initialization
- bootstrap sample
- stochastic optimization

등을 다르게 할 수 있다.

AASSR Prophecy ensemble의 uncertainty evidence와 연결된다.

관련 페이지:

- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 22. GPU batching

GPU는 작은 model call을 수천 번 순차 실행하는 것보다 큰 tensor batch를 한 번에 처리할 때 효율적인 경우가 많다.

AASSR current-generation에서 Prophecy/Critic depth batching이 큰 성능 최적화였던 이유다.

```text
scalar calls × N
→ overhead 큼

batch of N
→ parallel tensor compute
```

---

# 23. CPU ↔ GPU synchronization

GPU 연산 중간에 `.item()`처럼 값을 CPU로 자주 가져오면 GPU가 완료될 때까지 기다리는 synchronization point가 생길 수 있다.

많은 작은 sync는 성능을 크게 떨어뜨릴 수 있다.

AASSR current hardware optimizations가 batch transfer와 bulk loss bookkeeping을 사용하는 이유와 연결된다.

---

# 24. TF32 / Mixed precision

현대 GPU는 일부 matrix multiplication에서 낮은 precision 형식을 사용해 속도를 높일 수 있다.

하지만 reproducibility와 numerical behavior가 달라질 수 있어 experimental config에 기록하는 것이 좋다.

---

# 25. Neural network가 '이해'한다는 말

Network가 높은 prediction accuracy를 보인다고 해서 내부적으로 인간과 같은 의미를 이해한다고 자동으로 말할 수는 없다.

연구 문서에서는 가능한 한 operational claim을 사용한다.

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

각 network는 target과 역할이 다르다.

---

# 27. 다음으로 읽기

- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)
- [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD)
- [GRU & Sequence Models](GRU-and-Sequence-Models)
- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)
- [Critic, Support & OOD](Critic-Support-and-OOD)

관련 색인: **[Concept Index](Concept-Index)**