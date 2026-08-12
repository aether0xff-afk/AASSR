# Q-Learning, DQN and Temporal Difference

이 페이지는 AASSR의 model-free Policy를 이해하기 위한 핵심 배경인 **Q-learning**, **DQN(Deep Q-Network)**, **TD(Temporal-Difference) learning**을 설명한다.

---

# 1. Q-learning의 목표

Q-learning은 최적 action value:

```math
Q^*(s,a)
```

를 학습하려 한다.

이 값은:

> state `s`에서 action `a`를 먼저 한 뒤 최적으로 행동했을 때 기대되는 장기 return

이다.

행동 선택은 보통:

```math
a^*=\arg\max_a Q(s,a)
```

로 한다.

---

# 2. Q-learning update

Tabular Q-learning의 전형적인 update:

```math
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha\left[
 r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)
\right]
```

대괄호 안이 TD error다.

```math
\delta_t
=
r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)
```

---

# 3. Temporal-Difference learning

TD learning은 실제 최종 return 전체를 기다리지 않고 **현재 reward와 다음 state의 value estimate를 이용해 update**한다.

```math
target=r_t+\gamma V(s_{t+1})
```

Q-learning에서는:

```math
target=r_t+\gamma\max_{a'}Q(s_{t+1},a')
```

이다.

이것이 [bootstrapping](Value-Functions-and-Bellman-Equation#9-bootstrapping)이다.

---

# 4. TD error의 의미

```math
\delta=target-current\ estimate
```

이다.

예:

```text
현재 Q = 0.2
reward = 0
next max Q = 0.8
γ = 0.9
```

이면:

```math
target=0+0.9(0.8)=0.72
```

```math
\delta=0.72-0.2=0.52
```

현재 Q를 위쪽으로 수정한다.

희소 보상에서는 terminal `+1`에서 생긴 value가 이런 TD update를 통해 이전 transition들로 전파될 수 있다.

---

# 5. Q-learning이 off-policy인 이유

Q-learning target은 실제 behavior가 다음에 어떤 action을 했는지가 아니라:

```math
\max_{a'}Q(s',a')
```

를 사용한다.

즉 행동 데이터를 만든 behavior policy와 학습하려는 target policy가 다를 수 있다.

그래서 과거 experience를 [Replay Buffer](Replay-Buffer-and-Episode-Boundaries)에서 재사용하기 좋다.

---

# 6. Tabular Q-learning의 한계

State/action space가 작으면 table을 둘 수 있다.

```text
Q[state][action]
```

하지만 실제 환경의 state가 고차원이고 거의 연속적이면 table이 불가능하다.

그래서 neural network로 Q-function을 근사한다.

```math
Q_\theta(s,a)
```

이 방향이 DQN이다.

---

# 7. DQN

**Deep Q-Network**는 neural network를 Q-function approximator로 사용한다.

입력:

```text
state representation
```

출력 방식은 구현에 따라 다르지만 일반적으로 각 action Q-value 또는 state/action pair score를 만든다.

AASSR current DQN은 relational state/action structure를 이용하는 변형된 action scoring path를 사용한다.

관련 페이지:

- [Policy](Policy)
- [State Representation](State-Representation)

---

# 8. 왜 그냥 neural network 하나로 Q-learning을 하면 불안정할 수 있나?

Q-learning + function approximation에서는 target 자체가 같은 network의 output에 의존한다.

```text
network가 Q 예측
      ↓
그 Q로 target 계산
      ↓
같은 network 업데이트
      ↓
target도 다시 움직임
```

또 연속 trajectory sample은 강하게 상관되어 있다.

이 때문에 DQN에서는 대표적으로:

- experience replay
- target network

같은 장치를 사용한다.

---

# 9. Experience Replay

과거 transition을 buffer에 저장한다.

```text
(S,A,R,S',done)
```

그리고 현재 trajectory의 바로 다음 sample만 쓰지 않고 과거 buffer에서 minibatch를 뽑아 학습한다.

장점:

- sample 재사용
- temporal correlation 완화
- off-policy 학습과 잘 맞음

더 자세히:

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 10. Target Network

Online network를 `Q_θ`, target network를 `Q_{θ^-}`라 하자.

Target:

```math
y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')
```

Online network는 이 target에 맞추어 학습한다.

Target network는 더 느리게 갱신하여 target이 지나치게 빠르게 움직이는 것을 줄인다.

---

# 11. DQN loss

기본적으로:

```math
L(\theta)
=
\mathbb{E}\left[
\left(y-Q_\theta(s,a)\right)^2
\right]
```

같은 regression 형태를 사용한다.

실제 구현에서는 MSE 대신 Huber/Smooth L1 loss를 사용할 수도 있다.

---

# 12. Terminal transition

진짜 terminal이라면 future Q를 더하면 안 된다.

```math
y=r
```

Non-terminal:

```math
y=r+\gamma\max_{a'}Q(s',a')
```

따라서 `done`/terminal flag는 매우 중요하다.

AASSR에서 한때 reset이 일어났는데 replay에서 non-terminal로 취급되어 **새 episode state를 이전 episode의 미래처럼 bootstrap**하는 mismatch가 문제가 된 이유가 이것이다.

관련 페이지:

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 13. Reward 0과 terminal false는 같은 말이 아니다

다음 두 transition은 reward가 모두 `0`일 수 있다.

```text
A. 정상 진행
reward = 0
next state는 같은 episode

B. administrative reset
reward = 0
다음 observation은 새 episode
```

A에서는 bootstrap이 자연스럽다.

```math
y=0+\gamma\max Q(s',a')
```

B에서 같은 식을 쓰면 **새 episode의 value가 이전 episode 행동에 연결**된다.

그래서 reward 의미와 episode boundary를 분리해야 한다.

---

# 14. Epsilon-greedy

DQN training에서 exploration을 위해:

```text
확률 ε → random action
그 외 → argmax Q
```

를 사용할 수 있다.

AASSR current Policy도 이 기본 exploration mechanism을 가진다.

자세한 내용:

- [Exploration and Exploitation](Exploration-and-Exploitation)

---

# 15. Overestimation bias

Q-learning의 `max`는 noisy estimate 중 큰 값을 선택하기 때문에 value를 과대평가할 수 있다.

```text
실제 값은 비슷한데
noise 때문에 어떤 action Q가 우연히 높음
→ max가 그 값을 선택
→ optimistic bias
```

Double DQN은 action selection과 evaluation을 분리해 이 문제를 줄이는 대표 방법이다.

AASSR의 Imagination에서도 비슷하게 **max를 어디에 사용해도 되는지**가 중요하다.

환경 stochastic outcome에 max를 쓰면 더 심각한 optimistic planning 오류가 생긴다.

관련 페이지:

- [Chance and Decision Nodes](Chance-and-Decision-Nodes)

---

# 16. Distribution shift

DQN은 training 중 경험한 state/action distribution에서 학습한다.

새로운 unseen region에서는 function approximator가 근거 없는 Q-value를 낼 수 있다.

```text
training support 안
→ interpolation

training support 밖
→ extrapolation
```

AASSR에서는 Policy뿐 아니라 Critic에서도 이 문제가 중요하다.

특히 Imagination이 model-generated state를 평가하면 OOD risk가 더 커질 수 있다.

관련 페이지:

- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 17. Raw DQN과 Relational DQN

AASSR benchmark에서 representation 효과를 분리하기 위해:

```text
dqn_raw
vs
dqn_relational
```

을 비교한다.

핵심 차이는 "DQN이냐 아니냐"가 아니라 **state/action representation**이다.

Relational DQN은 concrete ID 자체보다 역할/관계 구조를 사용해 seed-renaming transfer를 노린다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [State Representation](State-Representation)

---

# 18. AASSR Policy의 Q와 information residual

Current Policy의 기본 개념:

```math
score(S,A)=Q_{task}(S,A)+I(S,A)
```

여기서 `Q_task`는 DQN이 external sparse reward를 학습한 값이다.

`I`는 별도의 information residual이다.

중요:

```text
I를 DQN reward target에 합쳐서 학습하는 것이 아니다.
```

관련 페이지:

- [Policy](Policy)

---

# 19. Imagination과 DQN의 관계

DQN은 model-free하게 기본 action을 제안한다.

Imagination은 learned world model을 사용해 여러 root action을 미래 관점에서 재평가한다.

```text
DQN Policy action
       ↓
Prophecy / Planner / Critic
       ↓
충분한 evidence가 있으면 override
```

따라서 AASSR Full은 DQN을 제거한 시스템이 아니다.

**DQN Policy가 fallback이자 baseline decision**이다.

---

# 20. TD와 Critic training의 차이

AASSR의 Policy DQN과 GRU Critic은 둘 다 미래 return을 다루지만 학습 계약이 다르다.

```text
Policy DQN
→ primitive action external Q
→ TD learning

GRU Critic
→ trajectory/suffix sparse-return evaluation
→ Imagination branch 평가
```

둘을 같은 value network라고 생각하면 안 된다.

관련 페이지:

- [Critic](Critic)
- [GRU and Sequence Models](GRU-and-Sequence-Models)

---

# 21. 핵심 오해

## "Q가 높으면 실제 성공 확률인가?"

아니다. Q는 reward 정의와 discounting 아래의 **expected return**이다. 성공 확률과 일치할 수도 있지만 일반적으로 같은 값은 아니다.

## "Reward가 0이면 TD target도 0인가?"

Non-terminal이면 다음 state Q가 들어가므로 아니다.

## "DQN은 world model을 학습하나?"

기본 DQN은 명시적인 transition world model 없이 Q-value를 직접 학습한다.

## "Relational DQN이면 model-based인가?"

아니다. representation이 relational일 뿐 DQN 자체는 model-free다.

---

# 22. 다음으로 읽기

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)
- [Policy](Policy)
- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Critic, Support and OOD](Critic-Support-and-OOD)

관련 색인: **[Concept Index](Concept-Index)**