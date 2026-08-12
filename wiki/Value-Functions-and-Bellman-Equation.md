# Value Functions and Bellman Equation

강화학습에서 **value function**은 현재 상태 또는 행동이 장기적으로 얼마나 좋은지를 **미래 누적 reward의 기대값**으로 나타낸다.

AASSR의 DQN Policy와 sparse-return Critic을 이해하는 핵심 기초다.

---

# 1. Reward와 Return

즉시 reward:

```math
R_{t+1}
```

미래 누적 return:

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+\cdots
```

즉 현재 reward가 `0`이라고 해서 현재 행동의 가치가 `0`인 것은 아니다.

몇 단계 뒤 `+1` 성공으로 이어진다면 현재 state/action은 양의 장기 value를 가질 수 있다.

---

# 2. Discount factor

`γ`는 미래 reward의 시간적 가중치다.

```math
0\le\gamma\le1
```

예:

```text
γ = 0.9
1단계 뒤 +1 → 1
2단계 뒤 +1 → 0.9
3단계 뒤 +1 → 0.81
```

더 늦은 reward가 더 많이 할인된다.

`γ=1`에 가까우면 장기 horizon을 더 강하게 고려하지만, 문제 성질과 안정성을 함께 고려해야 한다.

---

# 3. State value V

Policy `π`를 따를 때 state `s`에서 기대되는 return:

```math
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
```

뜻:

> 현재 state에 있고 앞으로 policy `π`를 따르면 평균적으로 얼마나 많은 discounted return을 받을까?

---

# 4. Action value Q

State `s`에서 action `a`를 먼저 한 뒤 policy `π`를 따를 때의 기대 return:

```math
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
```

DQN이 근사하는 것은 보통 이 action-value 계열이다.

AASSR Policy에서:

```text
Q_task(relational_state, action)
```

는 외부 sparse task return을 학습하는 부분이다.

---

# 5. Advantage

Action `a`가 state 평균보다 얼마나 좋은지 나타내는 값:

```math
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
```

AASSR의 Imagination intervention에서도 비슷한 비교가 등장한다.

```text
imagined candidate value
-
Policy action value
=
intervention advantage
```

다만 AASSR의 구현상 advantage는 planner root evaluation 차이이며, 일반 actor-critic의 학습 advantage와 완전히 같은 객체라고 볼 필요는 없다.

---

# 6. Bellman expectation equation

Value는 재귀적으로 표현할 수 있다.

```math
V^\pi(s)
=
\mathbb{E}_{a\sim\pi,s'\sim P}
\left[R_{t+1}+\gamma V^\pi(s')\right]
```

Action value는:

```math
Q^\pi(s,a)
=
\mathbb{E}_{s'\sim P}
\left[R_{t+1}+\gamma\mathbb{E}_{a'\sim\pi}Q^\pi(s',a')\right]
```

핵심은 현재 value를 **즉시 reward + 다음 state의 value**로 분해할 수 있다는 것이다.

---

# 7. Bellman optimality equation

최적 value는 다음 행동에서 최선의 선택을 한다고 본다.

```math
V^*(s)=\max_a\mathbb{E}\left[R_{t+1}+\gamma V^*(S_{t+1})\right]
```

Q-value는:

```math
Q^*(s,a)=
\mathbb{E}\left[R_{t+1}+\gamma\max_{a'}Q^*(S_{t+1},a')\right]
```

이 식이 [Q-learning](Q-Learning-DQN-and-TD)의 핵심 target으로 이어진다.

---

# 8. Backup이란?

강화학습에서 **backup**은 미래/다음 state의 정보를 현재 value로 가져오는 연산을 말한다.

```text
S_t
 ↓ action
S_{t+1}
 ↓ value estimate
현재 Q target 계산
```

Bellman backup:

```math
y=r+\gamma\max_{a'}Q(s',a')
```

AASSR Imagination에서도 tree의 자식 node value를 부모로 올리는 **planning backup**이 존재한다.

하지만 chance node와 decision node의 backup rule은 다르다.

관련 페이지:

- [Chance and Decision Nodes](Chance-and-Decision-Nodes)
- [Imagination](Imagination)

---

# 9. Bootstrapping

현재 target을 계산할 때 **이미 학습 중인 value estimate를 다시 사용하는 것**을 bootstrapping이라고 한다.

```math
target=r+\gamma V_{estimate}(s')
```

장점:

- episode가 끝날 때까지 기다리지 않고 업데이트 가능
- reward 신호를 단계적으로 전파 가능

단점:

- 잘못된 value가 target으로 다시 들어가 bias/error를 퍼뜨릴 수 있음
- terminal/boundary 처리 실수가 큰 오류를 만듦

더 자세히:

- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 10. Monte Carlo와 TD 비교

## Monte Carlo

실제 episode return을 끝까지 관측하고 사용한다.

```math
G_t=R_{t+1}+\gamma R_{t+2}+\cdots
```

장점:

- bootstrap하지 않음

단점:

- 종료까지 기다려야 함
- variance가 클 수 있음

## TD

한 단계 뒤 estimate를 사용한다.

```math
r+\gamma V(s')
```

장점:

- 빠른 online update

단점:

- estimate error가 target에 들어감

---

# 11. Sparse reward에서 Bellman propagation

다음 episode가 있다고 하자.

```text
S0 → S1 → S2 → S3 → success(+1)
```

처음에는 `S3` 근처에서만 성공 신호가 직접 보인다.

TD update를 반복하면:

```text
S3 value ↑
 ↓
S2 target에 반영
 ↓
S2 value ↑
 ↓
S1 target에 반영
 ↓
...
```

처럼 성공 신호가 뒤로 전파된다.

문제는 **성공 episode 자체가 너무 드물면 이 propagation이 시작될 sample이 부족하다**는 것이다.

관련 페이지:

- [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 12. Terminal state의 Bellman target

Episode가 진짜 terminal이면 그 뒤의 미래 value를 이어붙이면 안 된다.

보통:

```math
y=r
```

이다.

Non-terminal이면:

```math
y=r+\gamma\max_{a'}Q(s',a')
```

이다.

즉 terminal flag는 **reward 값과 별개의 학습 의미**를 가진다.

AASSR에서 stalled/reset이 reward `0`이더라도 TD bootstrap boundary를 끊을 수 있는 이유가 여기에 있다.

관련 페이지:

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 13. Truncation의 복잡성

Time limit 때문에 episode를 강제로 끊었다고 하자.

환경의 실제 underlying task는 terminal이 아닐 수도 있다.

그래서 일반 RL에서는 truncation을 무조건 true terminal과 동일하게 처리할지 주의해야 한다.

AASSR의 current training에서는 **world reset으로 이어져 다음 observation이 같은 episode의 연속이 아니면 TD bootstrap을 끊어야 하는 경우**가 있다.

이것은 "truncation reward를 failure로 바꾼다"와는 다르다.

```text
reward semantics
!=
bootstrap boundary semantics
```

---

# 14. Expected value와 max의 차이

Bellman optimality에서는 **agent가 선택할 수 있는 다음 action**에 대해 `max`를 쓴다.

하지만 stochastic environment outcome은 agent가 선택할 수 없다.

따라서 환경 outcome을 backup할 때는 확률적 expectation이 필요하다.

```math
\mathbb{E}[V]=\sum_i p_iV_i
```

AASSR Imagination이:

```text
chance node → expectation
decision node → max
```

를 구분하는 이유가 이 Bellman 의미와 연결된다.

---

# 15. Function approximation

작은 tabular problem에서는 각 state/action마다 Q-table을 둘 수 있다.

```text
Q[s,a]
```

큰 state space에서는 neural network 같은 function approximator를 쓴다.

```math
Q_\theta(s,a)
```

DQN이 대표적이다.

Function approximation은 generalization을 가능하게 하지만 동시에 OOD extrapolation 문제를 만든다.

관련 페이지:

- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 16. Critic이란?

넓은 의미에서 Critic은 state/action/trajectory의 미래 return을 평가하는 value estimator다.

Actor-critic에서는 actor가 policy를 만들고 critic이 그 policy의 value/advantage를 평가한다.

AASSR의 Critic은 조금 다르게 사용된다.

```text
Prophecy가 만든 imagined branch
      ↓
GRU sparse-return Critic
      ↓
branch long-term value
```

즉 AASSR Critic은 planner의 leaf/branch 평가기로 사용된다.

관련 페이지:

- [Critic](Critic)
- [GRU and Sequence Models](GRU-and-Sequence-Models)

---

# 17. Value와 probability는 다르다

어떤 outcome이 자주 일어난다고 좋은 것은 아니다.

```text
Outcome A: probability 0.9, value -1
Outcome B: probability 0.1, value +1
```

Chance expectation:

```math
0.9(-1)+0.1(+1)=-0.8
```

이 된다.

AASSR에서:

- Prophecy outcome probability
- Calibration reliability
- Critic value
- local support

는 전부 다른 의미다.

관련 페이지:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 18. Value와 information value도 다르다

AASSR Policy는 외부 task Q와 information residual을 분리한다.

```text
Q_task
= external sparse return

I
= future decision에 도움이 될 정보의 내부 가치
```

둘을 같은 reward target으로 학습시키지 않는다.

관련 페이지:

- [Policy](Policy)
- [Exploration and Exploitation](Exploration-and-Exploitation)

---

# 19. AASSR 연결 요약

```text
Bellman / return
   ├→ DQN Policy의 external Q
   ├→ TD bootstrap
   ├→ episode boundary 처리
   ├→ sparse reward propagation
   └→ Imagination의 decision backup 개념

Critic
   └→ real sparse return으로 imagined future 평가
```

---

# 20. 다음으로 읽기

- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)
- [Policy](Policy)
- [Critic](Critic)
- [Chance and Decision Nodes](Chance-and-Decision-Nodes)

관련 색인: **[Concept Index](Concept-Index)**