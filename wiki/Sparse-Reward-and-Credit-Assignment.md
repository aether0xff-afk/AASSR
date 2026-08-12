# Sparse Reward and Credit Assignment

이 페이지는 **희소 보상(sparse reward)** 과 **credit assignment**를 강화학습 일반 개념 수준에서 설명한다.

AASSR 자체의 구체적인 문제 정의는 **[Sparse Reward Problem](Sparse-Reward-Problem)** 에서 다룬다.

---

# 1. Dense reward와 Sparse reward

## Dense reward

에이전트가 목표로 가는 과정에서도 자주 reward를 받는다.

```text
좋은 방향으로 이동   +0.1
장애물 회피          +0.05
목표에 가까워짐      +0.2
목표 도달            +1.0
```

이런 환경에서는 현재 행동이 유용한지 비교적 빠르게 알 수 있다.

## Sparse reward

보상이 매우 드물다.

```text
A0 → A1 → A2 → A3 → A4 → success
 0    0    0    0    0      +1
```

대부분의 행동이 `0`을 받기 때문에 성공을 한 번도 발견하지 못하면 learner가 행동 간 차이를 배울 신호 자체가 거의 없다.

---

# 2. Delayed reward

**Delayed reward**는 행동의 결과가 한참 뒤에 보상으로 나타나는 상황이다.

```text
A0
 ↓
A1
 ↓
A2
 ↓
A3
 ↓
+1
```

성공 시점에서 바로 앞의 `A3`만 중요했던 것이 아닐 수 있다.

예를 들어 `A0`가 이후 모든 행동을 가능하게 하는 정보를 얻었을 수 있다.

이때 "최종 보상에 어떤 과거 행동들이 얼마나 기여했는가?"가 어려워진다.

---

# 3. Credit assignment란?

**Credit assignment problem**은 최종 결과의 책임 또는 공로를 과거의 행동/상태에 어떻게 배분할지에 대한 문제다.

강화학습에서는 특히 **temporal credit assignment**가 중요하다.

```text
오래 전 행동 A
     ↓
여러 중간 transition
     ↓
최종 성공
```

`A`가 성공에 기여했다면 그 신호를 시간적으로 거슬러 전달해야 한다.

[TD learning](Q-Learning-DQN-and-TD)과 [Bellman backup](Value-Functions-and-Bellman-Equation)이 이런 신호 전달의 핵심 수학적 메커니즘이다.

---

# 4. 희소 보상에서는 왜 credit assignment가 더 어려운가?

Dense reward에서는 중간 단계마다 힌트가 있다.

```text
A0 +0.1
A1 +0.2
A2 -0.1
A3 +0.4
```

Sparse reward에서는:

```text
A0 0
A1 0
A2 0
A3 0
A4 +1
```

이므로 처음 성공하기 전에는 거의 모든 경험이 동일한 `0`처럼 보일 수 있다.

즉 두 문제가 연결된다.

```text
Sparse reward
    ↓
성공 경험 부족
    ↓
value target 부족
    ↓
credit assignment 느림
    ↓
좋은 행동을 더 찾기 어려움
```

---

# 5. Exploration 문제와 연결

희소 보상에서는 credit assignment 이전에 **성공 trajectory 자체를 발견하지 못하는 문제**가 있다.

행동 수가 각 단계마다 `b`, 성공까지 필요한 깊이가 `d`라고 단순화하면 random exploration으로 특정 경로를 고를 확률은 대략:

```math
\left(\frac{1}{b}\right)^d
```

이다.

예를 들어 매 단계 10개 행동 중 하나가 맞고 6단계가 필요하면:

```math
10^{-6}
```

수준이 된다.

실제 환경은 이보다 복잡하지만 **행동 공간과 horizon이 함께 커지면 무작위 성공 발견이 급격히 어려워진다**는 직관은 같다.

관련 페이지:

- [Exploration and Exploitation](Exploration-and-Exploitation)

---

# 6. Long horizon

**Horizon**은 현재 행동이 영향을 미치는 미래의 길이를 말한다.

긴 horizon에서는:

- reward가 더 멀리 있음
- model-based planning에서는 prediction error가 더 많이 누적됨
- value learning에서는 bootstrap chain이 길어짐
- exploration에서 가능한 경로 수가 커짐

AASSR이 Imagination depth를 무조건 크게 하지 않는 이유도 여기에 있다.

관련 페이지:

- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)
- [Imagination](Imagination)

---

# 7. Reward shaping

희소 보상 문제를 쉽게 만드는 대표적인 방법은 **reward shaping**이다.

원래 reward:

```text
goal +1
otherwise 0
```

shaped reward:

```text
subgoal A +0.1
subgoal B +0.2
goal      +1.0
```

잘 설계하면 학습을 크게 빠르게 할 수 있다.

하지만 연구자가 subgoal을 미리 알고 있어야 하는 경우가 있다.

AASSR의 핵심 질문이 "중간 목표를 사람이 정해주지 않아도 되는가?"이므로 current benchmark는 외부 shaping reward를 의도적으로 제한한다.

---

# 8. Reward shaping이 항상 잘못인가?

아니다.

Reward shaping은 정당한 RL 기법이다. 문제는 **무엇을 연구하려는가**다.

예를 들어 실제 제품 최적화가 목적이라면 사람이 유용한 intermediate objective를 설계하는 것이 합리적일 수 있다.

반면 AASSR 연구 질문은:

> sparse external reward에서도 경험 구조와 planning이 도움이 되는가?

이므로 강한 hand-crafted shaping을 넣으면 원래 질문을 흐릴 수 있다.

---

# 9. Potential-based reward shaping

고전적으로 shaping이 최적 policy를 바꾸지 않도록 설계하는 방법 중 하나가 potential-based shaping이다.

형태:

```math
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
```

여기서 `Φ(s)`는 potential function이다.

이 방식은 특정 조건에서 원래 optimal policy를 보존하는 이론적 성질을 가진다.

하지만 AASSR current benchmark는 더 강한 질문을 유지하기 위해 **외부 중간 reward 자체를 사용하지 않는 방향**을 택한다.

---

# 10. Intrinsic reward

환경이 주는 외부 reward와 별도로 agent 내부에서 exploration을 위한 reward를 만들 수도 있다.

예:

- novelty
- curiosity
- prediction error
- information gain

이를 **intrinsic motivation**이라고 부른다.

AASSR의 information residual은 "정보의 가치"와 관련 있지만, current 설계에서는 이를 **외부 DQN task reward와 분리된 residual**로 유지한다.

즉:

```text
external sparse reward
!=
internal information value
```

관련 페이지:

- [Exploration and Exploitation](Exploration-and-Exploitation)
- [Policy](Policy)

---

# 11. Hindsight Experience Replay와의 차이

희소 보상에서 유명한 방법 중 하나는 실패 trajectory도 다른 goal 관점에서 성공처럼 재해석하는 **Hindsight Experience Replay(HER)** 류의 아이디어다.

핵심 직관:

```text
원래 목표에는 실패
하지만 실제 도달한 상태를 다른 목표로 보면 성공 경험으로 재사용
```

AASSR의 현재 benchmark와는 목적과 계약이 다르다.

AASSR은 행동 후 알게 된 미래 정보를 행동 전 prediction에 넣는 [hindsight leakage](Causality-Leakage-and-Evaluation)를 특히 경계한다.

HER 자체가 leakage라는 뜻은 아니다. **목표 재라벨링을 명시적으로 허용하는 알고리즘적 설정**과 **원래 시점에 알 수 없었던 정보를 몰래 입력으로 사용하는 것**은 완전히 다른 문제다.

---

# 12. Monte Carlo credit assignment

Episode가 끝난 뒤 실제 return을 계산해서 각 state/action에 학습 신호를 줄 수 있다.

```math
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}
```

장점:

- 실제 observed return 사용
- bootstrap bias가 없음

단점:

- episode 종료까지 기다려야 함
- sparse reward에서는 유용한 성공 episode가 드물 수 있음
- variance가 클 수 있음

AASSR Critic은 실제 sparse-return을 기반으로 하지만 recurrent suffix training이라는 별도 구조를 사용한다.

---

# 13. Temporal Difference credit assignment

TD는 실제 final return을 전부 기다리지 않고 다음 value estimate를 이용해 현재 value를 업데이트한다.

```math
y_t=r_t+\gamma V(s_{t+1})
```

또는 Q-learning:

```math
y_t=r_t+\gamma\max_{a'}Q(s_{t+1},a')
```

이것을 **bootstrapping**이라고 한다.

장점:

- online update 가능
- reward 신호를 단계적으로 뒤로 전파 가능

단점:

- 잘못된 value estimate가 target에 다시 들어감
- boundary 처리 오류가 큰 문제를 만들 수 있음

관련 페이지:

- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 14. Model-based planning과 credit assignment

World model을 이용하면 실제 reward가 오기 전에 미래를 전개해볼 수 있다.

```text
현재 state
  ↓
행동 A를 상상
  ↓
예측 미래
  ↓
그 뒤 성공 가능성/가치 계산
```

이것은 value learning의 temporal propagation과 다른 방식으로 장기 의사결정을 보완할 수 있다.

AASSR의 Prophecy + Imagination은 이 방향에 해당한다.

하지만 world model이 틀리면 planning도 틀릴 수 있다.

그래서 다음이 필요하다.

- [Calibration](Calibration)
- [Critic Support and OOD](Critic-Support-and-OOD)

---

# 15. Self-loop와 sparse reward

Reward가 계속 0이면 agent가 같은 행동을 반복해도 강한 부정 신호가 없을 수 있다.

```text
S → A → S → A → S → A → S
```

AASSR의 ASEQ는 **반복 행동 자체**가 아니라 실제로 관측된 진전 없는:

```text
S → A → S
```

만 좁게 억제한다.

이것은 reward를 추가하는 것이 아니라 action selection constraint에 가깝다.

관련 페이지:

- [ASEQ](ASEQ)

---

# 16. 정보 수집 행동의 credit

어떤 행동은 즉시 reward를 주지 않지만 정보를 얻어 미래의 action space를 바꿀 수 있다.

```text
정보 탐색 action
    ↓ reward 0
새 route 발견
    ↓
새 action 가능
    ↓
몇 단계 뒤 success +1
```

단순 TD만으로도 이 관계를 장기적으로 학습할 수 있지만 성공 sample이 매우 적으면 느릴 수 있다.

AASSR Policy는 external Q-value와 별도로 delayed information residual을 관리한다.

중요한 점은 **그 residual을 외부 reward로 위장하지 않는 것**이다.

---

# 17. Sparse reward benchmark에서 꼭 확인할 것

희소 보상 환경을 만들었다고 해서 좋은 benchmark가 되는 것은 아니다.

다음이 중요하다.

## Solvability

Oracle 또는 검증된 solver가 성공할 수 있어야 한다.

## Non-triviality

Random이 너무 쉽게 성공하면 sparse exploration 연구가 의미가 없다.

## No shortcut

hidden 정답 정보가 observation에 들어가면 안 된다.

## Stable failure semantics

failure와 truncation을 구분해야 한다.

## Comparable budgets

baseline과 proposed model이 같은 transition budget을 가져야 한다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 18. AASSR과 연결

AASSR은 sparse reward 문제를 한 가지 trick으로 해결하려 하지 않는다.

```text
Sparse reward
  ├→ exploration difficulty        → Policy / information residual
  ├→ self-loop                     → ASEQ
  ├→ delayed consequences          → Critic
  ├→ future uncertainty            → Prophecy
  ├→ long-horizon action choice    → Imagination
  ├→ model error                   → Calibration
  └→ OOD value error               → local Critic support
```

각 구성요소가 실제로 필요한지는 반드시 [ablation](Ablation-Benchmarking-and-Reproducibility)으로 분리해야 한다.

---

# 19. 핵심 오해

## "reward가 0이면 학습이 아예 안 된다"

항상 그렇지는 않다. bootstrap이나 이후 terminal reward가 이전 state/action으로 전파될 수 있다. 문제는 **성공 경험이 너무 적으면 그 전파를 시작할 sample 자체가 부족하다**는 것이다.

## "중간 reward를 주면 sparse reward가 해결된다"

실용적으로는 도움이 될 수 있지만, 사람이 좋은 subgoal을 알고 있어야 한다면 원래 문제를 일부 외부 지식으로 푼 셈일 수 있다.

## "정보를 얻으면 reward를 주면 되지 않나?"

가능하지만 그 순간 연구 objective가 바뀔 수 있다. AASSR은 information signal과 external task reward를 분리해 분석한다.

---

# 20. 다음으로 읽기

- [Exploration and Exploitation](Exploration-and-Exploitation)
- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Sparse Reward Problem](Sparse-Reward-Problem)
- [Research Questions](Research-Questions)

관련 색인: **[Concept Index](Concept-Index)**