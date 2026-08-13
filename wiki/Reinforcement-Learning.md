# 강화학습 (Reinforcement Learning)

[강화학습(Reinforcement Learning, RL)](Reinforcement-Learning)은 **에이전트([에이전트(agent)](Reinforcement-Learning))가 환경([환경(environment)](Reinforcement-Learning))과 상호작용하면서 장기적인 누적 보상을 최대화하는 행동 규칙을 학습하는 문제**다.

AASSR을 이해하려면 먼저 강화학습의 기본 언어를 정확히 구분해야 한다. 특히 `state`, `observation`, `reward`, `return`, `policy`, `value`, `world model`은 서로 다른 개념이다.

---

# 1. 가장 기본적인 상호작용

강화학습은 보통 다음 반복으로 표현한다.

```text
현재 정보
  ↓
Agent가 action 선택
  ↓
Environment가 변함
  ↓
새 observation + reward
  ↓
다시 action 선택
```

수식으로는 시점 `t`에서:

```math
A_t \sim \pi(\cdot\mid O_t)
```

환경이 행동을 받은 뒤:

```math
S_{t+1} \sim P(\cdot\mid S_t,A_t)
```

그리고 [보상(reward)](Sparse-Reward-and-Credit-Assignment):

```math
R_{t+1} = R(S_t,A_t,S_{t+1})
```

을 내놓는 식으로 생각할 수 있다.

완전 관측 환경에서는 에이전트가 `S_t` 자체를 볼 수 있지만, [부분 관측(POMDP)](MDP-and-POMDP#5-pomdp-partially-observable-markov-decision-process)에서는 실제 내부 상태 `S_t` 대신 [관측(observation)](MDP-and-POMDP) `O_t`만 볼 수 있다.

AASSR은 후자에 더 가까운 문제를 다룬다.

---

# 2. Agent

**[에이전트(Agent)](Reinforcement-Learning)**는 행동을 선택하는 주체다.

AASSR에서는 하나의 신경망만 에이전트인 것이 아니다. 실제 decision은 여러 구성요소의 조합으로 만들어진다.

```text
Relational State
    ↓
Policy
    ↓
기본 행동 후보
    ↓
Prophecy + Imagination + Critic
    ↓
필요하면 행동 override
    ↓
실제 concrete action 실행
```

즉 AASSR 전체가 에이전트이며, `Policy`는 그 안의 기본 행동 선택기다.

관련 페이지:

- [Policy](Policy)
- [Research Architecture](Research-Architecture)
- [Imagination](Imagination)

---

# 3. Environment

**[환경(Environment)](Reinforcement-Learning)**는 에이전트의 행동을 받아 상태를 변화시키고 관측과 보상를 돌려주는 외부 세계다.

AASSR 실험에서는 실제 외부 시스템 대신 안전한 in-process [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility) 환경을 사용한다.

중요한 구분:

```text
Environment가 내부적으로 알고 있는 것
!=
Agent가 observation으로 볼 수 있는 것
```

환경 simulator가 [숨겨진(hidden)](MDP-and-POMDP) session TTL, 숨겨진 workflow stage, 정답 target [식별 방식(identity)](State-Representation)를 알고 있다고 해서 [학습 주체(learner)](Terminology-Guide)에게 그 정보를 주면 안 된다.

이 경계는 [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)에서 더 깊게 다룬다.

---

# 4. State

**[상태(State)](State-Representation)**는 환경의 현재 상황을 기술하는 정보다.

이론적 [MDP](MDP-and-POMDP#2-mdp-markov-decision-process)에서는 `S_t`가 미래를 예측하는 데 필요한 모든 정보를 포함한다고 가정한다.

즉 Markov property:

```math
P(S_{t+1}\mid S_0,A_0,\ldots,S_t,A_t)
=
P(S_{t+1}\mid S_t,A_t)
```

가 성립한다.

하지만 실제 에이전트가 이 완전한 상태를 항상 볼 수 있는 것은 아니다.

AASSR에서는 **환경의 숨겨진 [실제 환경 상태(true state)](MDP-and-POMDP)**와 **에이전트가 가진 [공개된(public)](State-Representation) [관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)**을 의도적으로 구분한다.

관련 페이지:

- [MDP and POMDP](MDP-and-POMDP)
- [State Representation](State-Representation)

---

# 5. Observation

**[관측(Observation)](MDP-and-POMDP)**은 에이전트가 실제로 관측할 수 있는 정보다.

완전 관측이면:

```text
Observation ≈ State
```

일 수 있지만 부분 관측이면:

```text
Hidden true state S
      ↓ observation function
Public observation O
      ↓
Agent
```

가 된다.

AASSR의 `response-causal public observation contract`는 학습 주체가 실제 [응답(response)](State-Representation)에서 인과적으로 알 수 있는 정보만 사용하도록 제한한다.

예:

- 실제로 발견한 route 관계
- 실제로 관측한 HTTP [상태 코드(status)](Terminology-Guide)
- 실제 legal [행동(action)](Reinforcement-Learning) surface

반면 다음은 직접 관측으로 주지 않는다.

- 정답 route
- 숨겨진 countdown
- 숨겨진 [난이도 조절 학습(curriculum)](Curriculum-Learning) level
- 미래 결과

---

# 6. Action

**[행동(Action)](Reinforcement-Learning)**은 에이전트가 환경에 가하는 선택이다.

AASSR에서는 두 수준의 행동 식별 방식를 구분한다.

```text
Concrete action
= 실제 환경에서 실행되는 정확한 대상/명령

Relational action
= transfer 학습을 위해 역할과 구조만 표현한 action
```

예:

```text
GET /route-12
```

와

```text
catalog-like route를 request
```

는 같은 행동을 서로 다른 abstr행동 level에서 본 것이다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [State Representation](State-Representation)

---

# 7. Reward

**[보상(Reward)](Sparse-Reward-and-Credit-Assignment)**는 환경이 한 [상태 전이(transition)](MDP-and-POMDP)에 대해 주는 즉각적인 scalar 학습 신호다.

```math
r_t \in \mathbb{R}
```

AASSR의 핵심은 보상를 자주 주지 않는다는 것이다.

대표 외부 보상 [명세(contract)](Current-Status):

```text
success       +1
true failure  -1
otherwise      0
```

따라서 route를 하나 찾았다고 자동으로 `+0.2` 같은 보상을 주지 않는다.

이것이 [Sparse Reward](Sparse-Reward-and-Credit-Assignment) 문제를 만든다.

---

# 8. Reward와 Return은 다르다

자주 혼동하는 부분이다.

## Reward

현재 상태 전이에서 바로 받은 값:

```math
R_{t+1}
```

## Return

현재 시점부터 미래에 받을 보상들의 누적값:

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+\cdots
```

즉 희소 보상에서는 현재 보상가 `0`이어도 현재 행동의 **장기 [누적 보상(return)](Value-Functions-and-Bellman-Equation)**은 매우 클 수 있다.

AASSR [Critic(미래 가치 평가기)](Critic)이 학습하려는 것은 바로 이 sparse-누적 보상 구조다.

관련 페이지:

- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
- [Critic](Critic)

---

# 9. Discount factor γ

`γ`는 먼 미래 보상를 얼마나 할인할지 정하는 값이다.

```math
0 \le \gamma \le 1
```

`γ`가 작으면 가까운 보상를 더 중요하게 보고, `γ`가 1에 가까우면 먼 미래도 강하게 고려한다.

예를 들어 성공 `+1`이 4단계 뒤에 있고 다른 보상가 없다면 누적 보상은 대략:

```math
\gamma^3
```

가 된다.

AASSR [Critic](Critic)의 discounted sparse-누적 보상 target을 이해하려면 이 개념이 필요하다.

---

# 10. Policy

**[Policy(정책 모델)](Policy) `π`**는 주어진 정보에서 어떤 행동을 선택할지 정의하는 규칙이다.

확률적 policy:

```math
\pi(a\mid s)=P(A_t=a\mid S_t=s)
```

결정론적 policy:

```math
a=\pi(s)
```

AASSR의 [Policy](Policy)는 [관계 기반(relational)](Relational-Representation-and-Generalization) [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) 기반 행동 점수와 별도 [정보 가치 잔차(information residual)](Policy)을 이용해 기본 행동을 정한다.

중요한 점:

> AASSR의 [Imagination(가상 미래 탐색)](Imagination)은 [Policy](Policy) 자체와 동일한 개념이 아니다.

[Policy](Policy)가 기본 행동을 제안하고, [Imagination](Imagination)은 충분한 근거가 있을 때 그 행동을 바꿀 수 있다.

---

# 11. Value function

Value function은 **미래의 누적 보상 기대값**을 나타낸다.

## State value

```math
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
```

## Action value

```math
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
```

[DQN](Q-Learning-DQN-and-TD)은 `Q(s,a)`를 근사한다.

AASSR [Critic](Critic)도 넓은 의미에서는 미래 누적 보상을 근사하는 [가치(value)](Value-Functions-and-Bellman-Equation) estimator지만, 입력과 학습 계약이 [Policy](Policy) [DQN](Q-Learning-DQN-and-TD)과 다르다.

관련 페이지:

- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Critic](Critic)

---

# 12. Trajectory

**Trajectory**는 시간에 따른 경험의 연속이다.

```text
S0, A0, R1, S1, A1, R2, S2, ...
```

또는 상태 전이 중심으로:

```text
(S0,A0,S1)
(S1,A1,S2)
(S2,A2,S3)
```

AASSR의 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)는 실제 `(S,A,S')` 상태 전이을 핵심 경험 단위로 본다.

관련 페이지:

- [ASEQ](ASEQ)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 13. Episode

**Episode**는 하나의 시작부터 종료까지 이어지는 trajectory다.

종료 이유는 모두 같은 의미가 아니다.

```text
success
true failure
truncation
stalled reset
transition cap
```

특히 RL 구현에서는 **환경 의미의 실패**와 **학습상 [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries)을 끊어야 하는 boundary**를 구분할 필요가 있다.

AASSR의 stalled/rate-limit/reset 관련 설계는 [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)에서 자세히 설명한다.

---

# 14. Model-Free RL

**Model-free RL**은 환경 [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models) `P(S'|S,A)`를 명시적으로 학습해 계획하지 않고, policy나 가치를 직접 학습하는 계열이다.

대표 예:

- Q-learning
- [DQN](Q-Learning-DQN-and-TD)
- 많은 actor-critic 방법

AASSR의 기본 [Policy](Policy) [DQN](Q-Learning-DQN-and-TD)은 model-free [구성요소(component)](Research-Architecture)다.

```text
State → Q-values → Action
```

장점:

- [세계 모델(world model)](Model-Based-RL-and-World-Models) 오류를 직접 겪지 않음
- 구조가 상대적으로 단순함

단점:

- 실제 경험 없이 미래를 명시적으로 전개하기 어려움
- [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)에서 긴 [보상 책임 배분(credit assignment)](Sparse-Reward-and-Credit-Assignment)가 힘들 수 있음

---

# 15. Model-Based RL

**[모델 기반 강화학습(Model-based RL)](Model-Based-RL-and-World-Models)**은 환경의 상태 전이이나 보상를 모델링하고 그 모델을 이용해 [계획(planning)](Counterfactual-Planning-and-Search)을 수행한다.

```text
현재 state
  ↓
행동 후보
  ↓ learned model
예측 미래
  ↓ planning
행동 결정
```

AASSR에서는:

```text
Prophecy = learned world model
Imagination = planning
Critic = long-horizon value estimator
```

로 볼 수 있다.

관련 페이지:

- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Prophecy](Prophecy)
- [Imagination](Imagination)

---

# 16. On-policy와 Off-policy

## On-policy

현재 행동을 만드는 policy에서 나온 경험으로 그 policy를 학습한다.

## Off-policy

다른 behavior policy에서 나온 과거 경험도 현재 학습 주체가 재사용할 수 있다.

Q-learning/[DQN](Q-Learning-DQN-and-TD)은 대표적인 off-policy 계열이다.

그래서 [Replay Buffer](Replay-Buffer-and-Episode-Boundaries)가 자연스럽게 사용된다.

AASSR에서도 실제 상태 전이을 replay해 [Policy](Policy), [Prophecy(미래 예측 모델)](Prophecy), [Critic](Critic)의 학습 근거로 사용한다.

---

# 17. Exploration과 Exploitation

에이전트가 이미 좋다고 아는 행동만 고르면 **[활용(exploitation)](Exploration-and-Exploitation)**이다.

새 행동을 시도해 정보를 얻는 것은 **[탐색(exploration)](Exploration-and-Exploitation)**이다.

희소 보상에서는 탐색이 특히 어렵다.

```text
성공이 매우 드묾
→ 어떤 행동이 좋은지 초기에는 거의 모름
→ random exploration만으로 성공까지 가기 어려움
```

관련 페이지:

- [Exploration and Exploitation](Exploration-and-Exploitation)
- [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 18. 학습과 Planning은 다르다

**Learning**은 경험을 이용해 [학습 모델(model)](Terminology-Guide)/[Policy](Policy)/[Critic](Critic)의 파라미터나 통계를 바꾸는 과정이다.

**Planning**은 현재 가진 모델을 이용해 행동 전에 미래를 계산하는 과정이다.

AASSR [현재(current)](Current-Status) [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)에서는 이 구분이 매우 중요하다.

```text
Real transitions
  → Policy / Prophecy / Critic 학습

Imagined transitions
  → 실행 전 planning
  → real truth처럼 persistent learner를 직접 학습시키지 않음
```

이 경계를 통해 학습 모델 hallucination이 자기 자신을 학습시키는 문제를 줄인다.

---

# 19. AASSR을 RL 용어로 다시 쓰면

AASSR은 대략 다음 조합으로 볼 수 있다.

```text
Partially observed sparse-reward environment
+
relational representation
+
model-free Q Policy
+
stochastic learned world model
+
real-return Critic
+
counterfactual tree planning
+
reliability / OOD gates
+
experience-derived skills
```

즉 완전히 새로운 강화학습의 기본 정의를 만드는 것이 아니라, **기존 RL의 여러 문제를 명시적으로 분리하고 결합한 연구 시스템**이다.

---

# 20. 다음 개념

강화학습 기초를 이해했다면 다음 순서를 추천한다.

1. [MDP and POMDP](MDP-and-POMDP)
2. [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)
3. [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
4. [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
5. [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
6. [Research Architecture](Research-Architecture)

관련 색인: **[Concept Index](Concept-Index)**