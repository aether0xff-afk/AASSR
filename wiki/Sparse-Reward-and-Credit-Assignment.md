# Sparse Reward and Credit Assignment

이 페이지는 **[희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)** 과 **[보상 책임 배분(credit assignment)](Sparse-Reward-and-Credit-Assignment)**를 강화학습 일반 개념 수준에서 설명한다.

AASSR 자체의 구체적인 문제 정의는 **[Sparse Reward Problem](Sparse-Reward-Problem)** 에서 다룬다.

---

# 1. Dense reward와 Sparse reward

## Dense reward

에이전트가 목표로 가는 과정에서도 자주 [보상(reward)](Sparse-Reward-and-Credit-Assignment)를 받는다.

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

대부분의 행동이 `0`을 받기 때문에 성공을 한 번도 발견하지 못하면 [학습 주체(learner)](Terminology-Guide)가 행동 간 차이를 배울 신호 자체가 거의 없다.

---

# 2. Delayed reward

**Delayed 보상**는 행동의 결과가 한참 뒤에 보상으로 나타나는 상황이다.

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

강화학습에서는 특히 **[시간 순서를 고려하는(temporal)](GRU-and-Sequence-Models) 보상 책임 배분**가 중요하다.

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

Dense 보상에서는 중간 단계마다 힌트가 있다.

```text
A0 +0.1
A1 +0.2
A2 -0.1
A3 +0.4
```

Sparse 보상에서는:

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

희소 보상에서는 보상 책임 배분 이전에 **성공 [경험 경로(trajectory)](Reinforcement-Learning) 자체를 발견하지 못하는 문제**가 있다.

행동 수가 각 단계마다 `b`, 성공까지 필요한 깊이가 `d`라고 단순화하면 [무작위(random)](Ablation-Benchmarking-and-Reproducibility) [탐색(exploration)](Exploration-and-Exploitation)으로 특정 경로를 고를 확률은 대략:

```math
\left(\frac{1}{b}\right)^d
```

이다.

예를 들어 매 단계 10개 행동 중 하나가 맞고 6단계가 필요하면:

```math
10^{-6}
```

수준이 된다.

실제 환경은 이보다 복잡하지만 **행동 공간과 [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search)이 함께 커지면 무작위 성공 발견이 급격히 어려워진다**는 직관은 같다.

관련 페이지:

- [Exploration and Exploitation](Exploration-and-Exploitation)

---

# 6. Long horizon

**Horizon**은 현재 행동이 영향을 미치는 미래의 길이를 말한다.

긴 미래 탐색 범위에서는:

- 보상가 더 멀리 있음
- [환경 모델을 사용하는(model-based)](Model-Based-RL-and-World-Models) [계획(planning)](Counterfactual-Planning-and-Search)에서는 [예측(prediction)](Terminology-Guide) [오차(error)](Loss-Functions-and-Class-Imbalance)가 더 많이 누적됨
- [가치(value)](Value-Functions-and-Bellman-Equation) [학습(learning)](Reinforcement-Learning)에서는 [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries) chain이 길어짐
- 탐색에서 가능한 경로 수가 커짐

AASSR이 [Imagination(가상 미래 탐색)](Imagination) [탐색 깊이(depth)](Counterfactual-Planning-and-Search)를 무조건 크게 하지 않는 이유도 여기에 있다.

관련 페이지:

- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)
- [Imagination](Imagination)

---

# 7. Reward shaping

희소 보상 문제를 쉽게 만드는 대표적인 방법은 **보상 [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)**이다.

원래 보상:

```text
goal +1
otherwise 0
```

shaped 보상:

```text
subgoal A +0.1
subgoal B +0.2
goal      +1.0
```

잘 설계하면 학습을 크게 빠르게 할 수 있다.

하지만 연구자가 subgoal을 미리 알고 있어야 하는 경우가 있다.

AASSR의 핵심 질문이 "중간 목표를 사람이 정해주지 않아도 되는가?"이므로 [현재(current)](Current-Status) [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)는 외부 형태 조정 보상를 의도적으로 제한한다.

---

# 8. Reward shaping이 항상 잘못인가?

아니다.

[보상(Reward)](Sparse-Reward-and-Credit-Assignment) 형태 조정은 정당한 RL 기법이다. 문제는 **무엇을 연구하려는가**다.

예를 들어 실제 제품 최적화가 목적이라면 사람이 유용한 [중간(intermediate)](Sparse-Reward-and-Credit-Assignment) [학습 목표(objective)](Terminology-Guide)를 설계하는 것이 합리적일 수 있다.

반면 AASSR 연구 질문은:

> [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [환경이 주는 외부(external)](Terminology-Guide) 보상에서도 경험 구조와 계획이 도움이 되는가?

이므로 강한 hand-crafted 형태 조정을 넣으면 원래 질문을 흐릴 수 있다.

---

# 9. Potential-based reward shaping

고전적으로 형태 조정이 최적 [정책(policy)](Policy)를 바꾸지 않도록 설계하는 방법 중 하나가 potential-based 형태 조정이다.

형태:

```math
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
```

여기서 `Φ(s)`는 potential [함수(function)](Terminology-Guide)이다.

이 방식은 특정 조건에서 원래 optimal 정책를 보존하는 이론적 성질을 가진다.

하지만 AASSR 현재 표준 비교 실험는 더 강한 질문을 유지하기 위해 **외부 중간 보상 자체를 사용하지 않는 방향**을 택한다.

---

# 10. Intrinsic reward

환경이 주는 외부 보상와 별도로 [에이전트(agent)](Reinforcement-Learning) 내부에서 탐색을 위한 보상를 만들 수도 있다.

예:

- [새로움(novelty)](Information-Theory-and-Intrinsic-Motivation)
- [새 정보를 찾아보려는 호기심 기반 탐색(curiosity)](Information-Theory-and-Intrinsic-Motivation)
- 예측 오차
- [정보(information)](Information-Theory-and-Intrinsic-Motivation) [증가량(gain)](Ablation-Benchmarking-and-Reproducibility)

이를 **[내재 동기(intrinsic motivation)](Information-Theory-and-Intrinsic-Motivation)**이라고 부른다.

AASSR의 [정보 가치 잔차(information residual)](Policy)은 "정보의 가치"와 관련 있지만, 현재 설계에서는 이를 **외부 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) [연구 과제(task)](Sparse-Reward-Problem) 보상와 분리된 [기본 값에 더하는 잔차(residual)](Policy)**로 유지한다.

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

희소 보상에서 유명한 방법 중 하나는 실패 경험 경로도 다른 [최종 목표(goal)](Sparse-Reward-Problem) 관점에서 성공처럼 재해석하는 **Hindsight Experience Replay(HER)** 류의 아이디어다.

핵심 직관:

```text
원래 목표에는 실패
하지만 실제 도달한 상태를 다른 목표로 보면 성공 경험으로 재사용
```

AASSR의 현재 표준 비교 실험와는 목적과 계약이 다르다.

AASSR은 행동 후 알게 된 미래 정보를 행동 전 예측에 넣는 [hindsight leakage](Causality-Leakage-and-Evaluation)를 특히 경계한다.

HER 자체가 [정보 누출(leakage)](Causality-Leakage-and-Evaluation)라는 뜻은 아니다. **목표 재라벨링을 명시적으로 허용하는 알고리즘적 설정**과 **원래 시점에 알 수 없었던 정보를 몰래 입력으로 사용하는 것**은 완전히 다른 문제다.

---

# 12. Monte Carlo credit assignment

Episode가 끝난 뒤 실제 [누적 보상(return)](Value-Functions-and-Bellman-Equation)을 계산해서 각 [상태(state)](State-Representation)/[행동(action)](Reinforcement-Learning)에 학습 신호를 줄 수 있다.

```math
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}
```

장점:

- 실제 [실제로 관측된(observed)](Causality-Leakage-and-Evaluation) 누적 보상 사용
- 다음 상태 가치 이어받기 [편향(bias)](Ablation-Benchmarking-and-Reproducibility)가 없음

단점:

- [한 번의 문제 풀이 구간(episode)](Terminology-Guide) 종료까지 기다려야 함
- 희소 보상에서는 유용한 성공 한 번의 문제 풀이 구간가 드물 수 있음
- [분산(variance)](Stochasticity-Uncertainty-and-Probability)가 클 수 있음

AASSR [Critic(미래 가치 평가기)](Critic)은 실제 sparse-누적 보상을 기반으로 하지만 [과거 정보를 이어가는 순환형(recurrent)](GRU-and-Sequence-Models) [후속 구간(suffix)](GRU-and-Sequence-Models) [학습(training)](Terminology-Guide)이라는 별도 구조를 사용한다.

---

# 13. Temporal Difference credit assignment

TD는 실제 [최종(final)](Ablation-Benchmarking-and-Reproducibility) 누적 보상을 전부 기다리지 않고 다음 가치 [추정값(estimate)](Value-Functions-and-Bellman-Equation)를 이용해 현재 가치를 업데이트한다.

```math
y_t=r_t+\gamma V(s_{t+1})
```

또는 [Q-러닝(Q-learning)](Q-Learning-DQN-and-TD):

```math
y_t=r_t+\gamma\max_{a'}Q(s_{t+1},a')
```

이것을 **다음 상태 가치 이어받기ping**이라고 한다.

장점:

- [경험이 들어올 때마다 갱신하는 온라인 방식(online)](Neural-Networks-and-Optimization) [학습 갱신(update)](Neural-Networks-and-Optimization) 가능
- 보상 신호를 단계적으로 뒤로 전파 가능

단점:

- 잘못된 가치 추정값가 [대상 또는 학습 목표값(target)](Terminology-Guide)에 다시 들어감
- [경계(boundary)](Replay-Buffer-and-Episode-Boundaries) 처리 오류가 큰 문제를 만들 수 있음

관련 페이지:

- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 14. Model-based planning과 credit assignment

[세계(World)](Model-Based-RL-and-World-Models) [학습 모델(model)](Terminology-Guide)을 이용하면 실제 보상가 오기 전에 미래를 전개해볼 수 있다.

```text
현재 state
  ↓
행동 A를 상상
  ↓
예측 미래
  ↓
그 뒤 성공 가능성/가치 계산
```

이것은 가치 학습의 시간 순서 기반 propagation과 다른 방식으로 장기 의사결정을 보완할 수 있다.

AASSR의 [Prophecy(미래 예측 모델)](Prophecy) + [Imagination](Imagination)은 이 방향에 해당한다.

하지만 [세계 모델(world model)](Model-Based-RL-and-World-Models)이 틀리면 계획도 틀릴 수 있다.

그래서 다음이 필요하다.

- [Calibration](Calibration)
- [Critic Support and OOD](Critic-Support-and-OOD)

---

# 15. Self-loop와 sparse reward

보상가 계속 0이면 에이전트가 같은 행동을 반복해도 강한 부정 신호가 없을 수 있다.

```text
S → A → S → A → S → A → S
```

AASSR의 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)는 **반복 행동 자체**가 아니라 실제로 관측된 진전 없는:

```text
S → A → S
```

만 좁게 억제한다.

이것은 보상를 추가하는 것이 아니라 행동 selection constraint에 가깝다.

관련 페이지:

- [ASEQ](ASEQ)

---

# 16. 정보 수집 행동의 credit

어떤 행동은 즉시 보상를 주지 않지만 정보를 얻어 미래의 행동 [공간(space)](MDP-and-POMDP)를 바꿀 수 있다.

```text
정보 탐색 action
    ↓ reward 0
새 route 발견
    ↓
새 action 가능
    ↓
몇 단계 뒤 success +1
```

단순 TD만으로도 이 관계를 장기적으로 학습할 수 있지만 성공 [표본(sample)](Ablation-Benchmarking-and-Reproducibility)이 매우 적으면 느릴 수 있다.

AASSR [Policy(정책 모델)](Policy)는 환경이 주는 외부 [Q값(Q-value)](Value-Functions-and-Bellman-Equation)와 별도로 delayed 정보 가치 잔차을 관리한다.

중요한 점은 **그 잔차을 외부 보상로 위장하지 않는 것**이다.

---

# 17. Sparse reward benchmark에서 꼭 확인할 것

희소 보상 환경을 만들었다고 해서 좋은 표준 비교 실험가 되는 것은 아니다.

다음이 중요하다.

## Solvability

[정답을 알고 있는 기준(Oracle)](Ablation-Benchmarking-and-Reproducibility) 또는 검증된 solver가 성공할 수 있어야 한다.

## Non-triviality

[무작위(Random)](Ablation-Benchmarking-and-Reproducibility)이 너무 쉽게 성공하면 희소한 탐색 연구가 의미가 없다.

## No shortcut

[숨겨진(hidden)](MDP-and-POMDP) 정답 정보가 [관측(observation)](MDP-and-POMDP)에 들어가면 안 된다.

## Stable failure semantics

[실패(failure)](Replay-Buffer-and-Episode-Boundaries)와 [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)을 구분해야 한다.

## Comparable budgets

[비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)과 proposed 학습 모델이 같은 [상태 전이(transition)](MDP-and-POMDP) [실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility)을 가져야 한다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 18. AASSR과 연결

AASSR은 희소 보상 문제를 한 가지 trick으로 해결하려 하지 않는다.

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

항상 그렇지는 않다. 다음 상태 가치 이어받기이나 이후 [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) 보상가 이전 상태/행동으로 전파될 수 있다. 문제는 **성공 경험이 너무 적으면 그 전파를 시작할 표본 자체가 부족하다**는 것이다.

## "중간 reward를 주면 sparse reward가 해결된다"

실용적으로는 도움이 될 수 있지만, 사람이 좋은 subgoal을 알고 있어야 한다면 원래 문제를 일부 외부 지식으로 푼 셈일 수 있다.

## "정보를 얻으면 reward를 주면 되지 않나?"

가능하지만 그 순간 연구 학습 목표가 바뀔 수 있다. AASSR은 정보 [학습 신호(signal)](Information-Theory-and-Intrinsic-Motivation)과 환경이 주는 외부 연구 과제 보상를 분리해 분석한다.

---

# 20. 다음으로 읽기

- [Exploration and Exploitation](Exploration-and-Exploitation)
- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Sparse Reward Problem](Sparse-Reward-Problem)
- [Research Questions](Research-Questions)

관련 색인: **[Concept Index](Concept-Index)**