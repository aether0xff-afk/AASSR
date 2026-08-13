# Sparse Reward Problem

AASSR은 **보상이 드문 환경에서 장기 행동 구조를 스스로 만들어야 하는 문제**를 연구하기 위해 시작되었다.

이 페이지에서는 AASSR에서 말하는 희소 보상 문제가 정확히 무엇인지, 그리고 단순히 "[보상(reward)](Sparse-Reward-and-Credit-Assignment)가 적다"는 말보다 왜 더 어려운지를 정리한다.

---

## 1. 희소 보상이란 무엇인가?

강화학습에서 에이전트는 보통 누적 보상를 최대화하도록 학습한다.

```text
state -> action -> next state -> reward
```

Dense 보상 환경에서는 진행 중에도 자주 학습 신호가 나온다.

예:

```text
목표 방향 이동      +0.1
속도 증가           +0.05
충돌                -0.2
목표 도달           +1.0
```

반면 AASSR이 겨냥하는 환경에서는 대부분의 행동이 즉시 보상을 주지 않는다.

```text
S0 -> A0 -> S1 -> A1 -> S2 -> A2 -> ... -> success
          0           0                  +1
```

현재 pentest [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)의 외부 보상 [명세(contract)](Current-Status)는 다음처럼 매우 좁다.

```text
proof success       +1
true failure        -1
stall                0
rate-limit trunc.    0
transition-cap       0
ordinary transition  0
```

---

# 2. 단순히 reward가 적어서 어려운가?

아니다. AASSR에서 문제는 여러 어려움이 동시에 겹친다는 데 있다.

## 2.1 긴 credit assignment

성공까지 여러 단계가 필요하면 어떤 과거 행동이 성공에 기여했는지 판단하기 어렵다.

```text
A0 -> A1 -> A2 -> A3 -> A4 -> success
```

성공 시점에서는 `A4`뿐 아니라 이전 행동들이 만든 상태 구조도 중요할 수 있다.

---

## 2.2 큰 행동 공간

현재 가능한 행동 수가 많으면 무작위 탐색으로 성공 [경험 경로(trajectory)](Reinforcement-Learning)를 만날 확률이 급격히 낮아진다.

특히 이름만 다른 [실제 실행 행동(concrete action)](State-Representation)이 많이 존재할 수 있다.

그래서 AASSR은 [실제 개체 구분(concrete identity)](State-Representation)와 [관계 기반(relational)](Relational-Representation-and-Generalization) [역할(role)](Relational-Representation-and-Generalization)을 분리한다.

---

## 2.3 Partial observability

환경의 내부 상태를 전부 볼 수 없으면 동일해 보이는 [공개 관측 상태(public state)](State-Representation)에서 서로 다른 미래가 나올 수 있다.

즉 실제 [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)는 단순히

```text
(S,A) -> 하나의 S'
```

가 아닐 수 있다.

```text
(S,A)
  |-- p1 --> S1'
  |-- p2 --> S2'
  `-- p3 --> S3'
```

이 때문에 현재 [Prophecy(미래 예측 모델)](Prophecy)는 [같은 입력이면 항상 같은 결과인 결정론적(deterministic)](Stochasticity-Uncertainty-and-Probability) [회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)이 아니라 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [조건부(conditional)](Stochasticity-Uncertainty-and-Probability) [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration)를 사용한다.

---

## 2.4 비가역적 실패

어떤 행동은 단순히 한 번의 보상 손실로 끝나지 않고 [한 번의 문제 풀이 구간(episode)](Terminology-Guide) 전체를 망칠 수 있다.

예:

- true [복구할 수 없는 실패 잠금(lockout)](Replay-Buffer-and-Episode-Boundaries)
- irreversible workflow [실패(failure)](Replay-Buffer-and-Episode-Boundaries)
- [한 번의 접속 세션(session)](Terminology-Guide) [학습 손실(loss)](Loss-Functions-and-Class-Imbalance) 후 복구 불가능한 상태

따라서 실제로 행동하기 전에 미래 위험을 추정하는 능력이 중요해진다.

---

## 2.5 Self-loop

보상이 없으면 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)이 특정 행동을 계속 반복하면서도 자신이 진행하지 못하고 있다는 사실을 늦게 배울 수 있다.

```text
S -> A -> S -> A -> S -> A -> S
```

AASSR은 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)를 사용해 실제로 관측된 `S -> A -> S` 반복만 좁게 억제한다.

---

## 2.6 이름이 바뀐 unseen 환경

훈련 중 `route-12`가 중요했다고 해서 평가에서도 같은 이름이 등장한다는 보장은 없다.

따라서 [실제 개체를 구분하는(concrete)](State-Representation) ID 암기는 [전이(transfer)](Relational-Representation-and-Generalization)를 방해할 수 있다.

AASSR은 다음처럼 관계 구조를 사용한다.

```text
concrete ID
    ↓ 제거
role / relation / public status
    ↓
transfer representation
```

---

# 3. 왜 중간 보상을 넣지 않는가?

가장 쉬운 해결책 중 하나는 사람이 중간 [최종 목표(goal)](Sparse-Reward-Problem)을 만들어 보상 [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)을 하는 것이다.

예:

```text
route 발견      +0.1
login 성공      +0.2
object 발견     +0.2
csrf 획득       +0.2
proof           +1.0
```

하지만 이렇게 하면 에이전트가 **스스로 장기 문제 구조를 학습했는지**와 **사람이 만들어준 subgoal을 따라갔는지**를 분리하기 어렵다.

AASSR은 이 문제를 연구하기 때문에 현재 표준 비교 실험에서는 [중간(intermediate)](Sparse-Reward-and-Credit-Assignment) 형태 조정 보상를 사용하지 않는다.

이 선택은 성능을 쉽게 만드는 방법을 일부러 포기하는 대신 연구 질문을 더 명확하게 만든다.

---

# 4. AASSR이 추가하는 것은 reward가 아니다

AASSR의 여러 구성 요소는 학습을 돕지만 외부 [연구 과제(task)](Sparse-Reward-Problem) 보상 자체를 바꾸지 않는다.

예를 들어:

- [관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)은 **표현 방식**을 바꾼다.
- [ASEQ](ASEQ)는 **진전 없는 반복 행동 후보**를 억제한다.
- [Prophecy](Prophecy)는 **미래 상태 분포**를 예측한다.
- [Calibration(예측 신뢰도 보정)](Calibration)은 **예측 신뢰도**를 측정한다.
- [Critic(미래 가치 평가기)](Critic)은 **실제 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [누적 보상(return)](Value-Functions-and-Bellman-Equation)**을 학습한다.
- [Imagination(가상 미래 탐색)](Imagination)은 **실행 전 [반사실적 계획(counterfactual planning)](Counterfactual-Planning-and-Search)**을 한다.

즉 내부 계산은 복잡해져도 외부 보상 명세는 그대로 유지된다.

---

# 5. AASSR의 문제 정의

AASSR이 실제로 다루는 문제를 조금 더 엄밀하게 쓰면 다음 조건의 조합이다.

```text
Sparse reward
+ Long horizon
+ Large dynamic action surface
+ Partial observability
+ Hidden stochastic factors
+ Irreversible failure risk
+ Identifier permutation / transfer
```

그래서 단순한 `DQN vs AASSR` 비교만으로는 충분하지 않다.

각 어려움에 대해 어떤 설계가 영향을 주는지 분리해야 한다.

---

# 6. 현재 benchmark는 어떻게 이 문제를 만든가?

AASSR의 [현재(current)](Current-Status) HTTP pentest lab은 실제 공격 대신 safe in-process [환경 시뮬레이터(simulator)](MDP-and-POMDP)를 사용한다.

환경은 다음 요소를 포함한다.

- route [스스로 새로운 성공 경로를 발견하는 것(discovery)](Research-Questions)
- authentication / 접속 세션
- CSRF
- object authorization
- state-changing workflow
- decoy routes
- [공정성과 구현을 점검하는 감사(audit)](Causality-Leakage-and-Evaluation) / 실패 잠금
- [비율(rate)](Terminology-Guide) [제한(limit)](Terminology-Guide)
- 접속 세션 expiration
- opaque [식별자(identifier)](State-Representation) [이름 순서를 바꾸는 순열(permutation)](Relational-Representation-and-Generalization)
- HTTP-like [공개된(public)](State-Representation) [상태 코드(status)](Terminology-Guide)

대표적인 진행 구조는 다음과 같다.

```text
Entry
  ↓
Discovery
  ↓
Login / Session
  ↓
Object reasoning
  ↓
Authorization boundary
  ↓
State change prerequisite
  ↓
Proof
```

중간 단계 대부분은 보상 `0`이다.

---

# 7. 이 환경이 정말 풀 수 있는가?

희소 보상 실험에서는 환경 자체가 사실상 불가능하면 [에이전트(agent)](Reinforcement-Learning) 성능을 비교할 수 없다.

그래서 표준 비교 실험 [검증(validation)](Ablation-Benchmarking-and-Reproducibility) 단계에서는 다음을 따로 확인한다.

- [정답을 알고 있는 기준(Oracle)](Ablation-Benchmarking-and-Reproducibility)은 일관되게 성공하는가?
- [무작위(Random)](Ablation-Benchmarking-and-Reproducibility)은 거의 성공하지 못하는가?
- 단순 heuristic은 낮은 난도에서는 일부 성공하지만 높은 난도에서 무너지는가?

즉 표준 비교 실험의 목표는

```text
너무 쉬움 X
불가능 X
구조적인 reasoning 필요 O
```

인 구간을 만드는 것이다.

자세한 수치는 **[Experiments](Experiments)** 에서 관리한다.

---

# 8. AASSR의 연구 방향

희소 보상 문제를 해결하기 위해 AASSR은 인간의 사고를 그대로 복제한다고 주장하지 않는다.

대신 인간의 장기 의사결정에서 관찰할 수 있는 몇 가지 기능을 계산 가능한 형태로 분해한다.

```text
실제 경험을 기억한다        -> ASEQ / Knowledge
관계 구조를 알아본다        -> Relational representation
미래 결과를 예측한다        -> Prophecy
예측을 얼마나 믿을지 본다   -> Calibration
여러 미래를 비교한다        -> Imagination
장기 결과를 평가한다        -> Critic
```

핵심 질문은 이러한 기능들이 실제로 **희소 보상에서 추가적인 학습/계획 이득을 만드는가**이다.

---

다음으로 읽기:

- **[Research Questions](Research-Questions)**
- **[Research Architecture](Research-Architecture)**
- **[ASEQ](ASEQ)**
- **[Experiments](Experiments)**
