# MDP and POMDP

이 페이지는 강화학습의 수학적 환경 모델인 **MDP(Markov Decision Process)** 와 **POMDP(Partially Observable Markov Decision Process)** 를 설명한다.

AASSR에서 [부분 관측](MDP-and-POMDP#5-pomdp-partially-observable-markov-decision-process), [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [Prophecy(미래 예측 모델)](Prophecy), [Knowledge(에피소드 지식)](Knowledge), [과거 정보를 이어가는 순환형(recurrent)](GRU-and-Sequence-Models) [Critic(미래 가치 평가기)](Critic)이 왜 필요한지 이해하려면 이 개념이 중요하다.

---

# 1. Markov property

어떤 상태 표현 `S_t`가 **현재 이후의 미래를 예측하는 데 필요한 과거 정보를 충분히 담고 있다**면 Markov property를 가진다고 한다.

수식으로:

```math
P(S_{t+1}\mid S_0,A_0,S_1,A_1,\ldots,S_t,A_t)
=
P(S_{t+1}\mid S_t,A_t)
```

뜻은:

> 현재 상태 `S_t`를 알고 있다면 다음 상태를 예측하기 위해 과거 전체를 다시 볼 필요가 없다.

이것은 "과거가 중요하지 않다"는 뜻이 아니다.

과거의 중요한 정보가 **현재 [상태(state)](State-Representation)에 충분히 요약되어 있다**는 뜻이다.

---

# 2. MDP: Markov Decision Process

일반적인 MDP는 다음 tuple로 정의한다.

```math
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)
```

각 항은:

- `S`: 상태 [공간(space)](MDP-and-POMDP)
- `A`: [행동(action)](Reinforcement-Learning) 공간
- `P`: [상태 전이(transition)](MDP-and-POMDP) [확률(probability)](Stochasticity-Uncertainty-and-Probability)
- `R`: [보상(reward)](Sparse-Reward-and-Credit-Assignment) [함수(function)](Terminology-Guide)
- `γ`: [미래 보상의 할인율(discount)](Value-Functions-and-Bellman-Equation) [실험에서 바꾸어 보는 요인(factor)](Ablation-Benchmarking-and-Reproducibility)

이다.

---

# 3. Transition probability

환경은 현재 상태와 행동으로부터 다음 상태를 만든다.

```math
P(s'\mid s,a)
```

환경이 [같은 입력이면 항상 같은 결과인 결정론적(deterministic)](Stochasticity-Uncertainty-and-Probability)이면 어떤 `(s,a)`에 대해 사실상 하나의 `s'`만 나온다.

```text
(S,A) → S'
```

확률적하면 여러 결과가 확률적으로 가능하다.

```text
(S,A)
 |-- 0.7 → S1'
 |-- 0.2 → S2'
 `-- 0.1 → S3'
```

AASSR의 [Prophecy](Prophecy)는 이 상태 전이 [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)을 [공개된(public)](State-Representation) [관계 기반(relational)](Relational-Representation-and-Generalization) 공간에서 학습하려 한다.

---

# 4. 완전 관측 MDP의 이상적 상황

완전 관측이면 [에이전트(agent)](Reinforcement-Learning)가 현재 `S_t`를 직접 볼 수 있다.

```text
True environment state S_t
          ↓
        Agent
```

이때 상태 표현이 정말 Markov하다면 [세계 모델(world model)](Model-Based-RL-and-World-Models)은 원칙적으로:

```math
P(S_{t+1}\mid S_t,A_t)
```

만 알면 된다.

그러나 실제 문제에서는 [학습 주체(learner)](Terminology-Guide)가 [실제 환경 상태(true state)](MDP-and-POMDP) 전체를 보지 못하는 경우가 많다.

---

# 5. POMDP: Partially Observable Markov Decision Process

POMDP에서는 실제 [숨은 환경 상태(hidden state)](MDP-and-POMDP) `S_t`가 있지만 에이전트는 [관측(observation)](MDP-and-POMDP) `O_t`만 받는다.

보통 다음처럼 확장한다.

```math
\mathcal{P}=(\mathcal{S},\mathcal{A},P,R,\Omega,O,\gamma)
```

여기서:

- `Ω`: 관측 공간
- `O(o|s)`: 숨은 환경 상태에서 관측이 나오는 분포

를 추가한다.

구조:

```text
Hidden state S_t
      ↓
Observation process
      ↓
Public observation O_t
      ↓
Agent
```

---

# 6. 같은 observation인데 미래가 달라질 수 있다

부분 관측에서는 서로 다른 숨은 환경 상태가 같은 관측을 만들 수 있다.

```text
Hidden S_A ─┐
            ├→ same observation O
Hidden S_B ─┘
```

에이전트 입장에서는 둘을 구분할 수 없다.

그 상태에서 같은 행동 `A`를 하면:

```text
O + action A
  |-- hidden S_A였음 → outcome X
  `-- hidden S_B였음 → outcome Y
```

처럼 여러 미래가 가능하다.

이것이 AASSR의 [Prophecy](Prophecy)가 단일 결정론적 `S'`보다 **[여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration) 확률적 분포**을 표현해야 하는 이유 중 하나다.

관련 페이지:

- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)
- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)

---

# 7. Observation은 state와 같은가?

항상 아니다.

다음 네 개념을 구분하는 것이 좋다.

```text
True hidden state
    ↓
Raw observation
    ↓ feature/representation
Agent state representation
    ↓ memory/context
Effective decision information
```

AASSR에서는 이 구분이 특히 중요하다.

## True hidden state

[환경 시뮬레이터(simulator)](MDP-and-POMDP)가 내부적으로 가진 모든 정보.

예:

- [숨겨진(hidden)](MDP-and-POMDP) workflow stage
- 숨겨진 [한 번의 접속 세션(session)](Terminology-Guide) [남은 횟수 카운트다운(countdown)](Causality-Leakage-and-Evaluation)
- 정답 [식별 방식(identity)](State-Representation)

## Public observation

에이전트가 실제 [응답(response)](State-Representation)를 통해 볼 수 있는 정보.

## Relational State v3

공개된 관측을 [전이(transfer)](Relational-Representation-and-Generalization)하기 좋은 구조로 변환한 [표현(representation)](Relational-Representation-and-Generalization).

## Knowledge / sequence context

과거 공개된 관측s에서 얻은 정보를 현재 [의사결정(decision)](Chance-and-Decision-Nodes)에 보존하는 추가 [문맥 정보(context)](GRU-and-Sequence-Models).

관련 페이지:

- [State Representation](State-Representation)
- [Knowledge](Knowledge)
- [GRU and Sequence Models](GRU-and-Sequence-Models)

---

# 8. Belief state

POMDP의 고전적인 해법 중 하나는 **[관측을 바탕으로 추정한 상태 믿음(belief)](MDP-and-POMDP) 상태**를 유지하는 것이다.

상태 추정 상태는 숨은 환경 상태에 대한 확률분포다.

```math
b_t(s)=P(S_t=s\mid O_{0:t},A_{0:t-1})
```

즉:

```text
"현재 hidden state가 무엇인지 정확히 모른다"
        ↓
가능한 hidden states에 확률을 둔다
```

AASSR이 [명시적인(explicit)](Causality-Leakage-and-Evaluation) Bayesian belief-state solver라고 주장하는 것은 아니다.

하지만 개념적으로 다음 요소들이 부분 관측을 보완한다.

- 공개된 관계 기반 상태
- [현재 에피소드 안에서만 유지되는(episode-local)](Knowledge) [Knowledge](Knowledge)
- 확률적 [Prophecy](Prophecy)
- 순환형 [Critic](Critic)

이들은 숨은 환경 상태를 직접 알려주는 것이 아니라 **관측 가능한 [기록(history)](Development-History)와 [미래(future)](Counterfactual-Planning-and-Search) 분포을 이용해 의사결정에 필요한 구조를 복원하려는 장치**다.

---

# 9. Memory가 필요한 이유

현재 관측만으로 과거에 얻은 중요한 정보가 사라질 수 있다.

예:

```text
시점 1: token을 발견함
시점 2: 화면에는 token 정보가 직접 없음
시점 3: token이 필요한 행동 결정
```

현재 관측 하나만 보면 시점 1의 사실을 잊게 된다.

이 때문에 POMDP에서 다음이 사용될 수 있다.

- 순환형 [신경망 기반(neural)](Neural-Networks-and-Optimization) [신경망(network)](Neural-Networks-and-Optimization)
- 상태 추정 상태
- 명시적인 [기억(memory)](GRU-and-Sequence-Models)
- 기록 window

AASSR에서는 [Knowledge](Knowledge)와 [GRU Critic](GRU-and-Sequence-Models)이 서로 다른 목적으로 기록 정보를 다룬다.

---

# 10. Partial observability와 stochasticity는 같은가?

아니다.

## Partial observability

실제 상태의 일부만 본다.

## Stochasticity

같은 실제 상태/행동에서도 환경 자체가 확률적으로 다른 결과를 낼 수 있다.

두 현상 모두 에이전트의 공개된 관점에서는 여러 미래를 만들 수 있지만 원인은 다르다.

```text
여러 미래
  ├─ hidden state를 구분 못해서
  └─ 환경 자체가 랜덤해서
```

AASSR 세계 모델은 둘을 공개된 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability) 분포에서 함께 다뤄야 할 수 있다.

더 자세히:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 11. POMDP와 model error도 다르다

세 번째로 분리해야 하는 것이 **[학습 모델(model)](Terminology-Guide) [불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability)**다.

```text
A. 실제 환경이 stochastic함
B. hidden state를 못 봄
C. 모델이 충분히 학습하지 못함
```

A와 B 때문에 여러 환경 결과이 실제로 가능할 수 있다.

C는 세계 모델 자체의 무지다.

AASSR에서는 개념적으로:

```text
Mixture outcome probability
→ 가능한 environment outcomes의 mass

Calibration / ensemble evidence
→ model prediction reliability
```

로 분리한다.

---

# 12. Action space도 state에 따라 달라질 수 있다

일반적인 교과서 MDP에서는 `A`를 고정된 행동 [집합(set)](Terminology-Guide)처럼 쓰지만 실제 환경에서는 상태마다 [현재 허용된(legal)](Terminology-Guide) 행동이 달라질 수 있다.

```math
\mathcal{A}(s) \subseteq \mathcal{A}
```

AASSR에서는 현재 [공개 관측 상태(public state)](State-Representation)의 `available_actions`가 매우 중요하다.

그래서 [Prophecy](Prophecy)는 다음 상태의 표현뿐 아니라 **[가능 행동 마스크(legal action mask)](Prophecy)/[현재 선택 가능한 영역(surface)](Terminology-Guide)**도 예측한다.

그렇지 않으면 [계획기(planner)](Counterfactual-Planning-and-Search)가 존재하지 않는 행동을 상상할 수 있다.

---

# 13. Terminal state

[에피소드 종료(Terminal)](Replay-Buffer-and-Episode-Boundaries) 상태는 [한 번의 문제 풀이 구간(episode)](Terminology-Guide)가 끝나는 상태다.

그러나 구현에서는 여러 종류의 종료를 구분해야 한다.

```text
success
true failure
truncation
administrative reset
```

MDP 의미에서 [최종 목표(goal)](Sparse-Reward-Problem)/[실패(failure)](Replay-Buffer-and-Episode-Boundaries) termination과, 학습 데이터 수집 과정에서 발생하는 상태 전이 cap은 의미가 다르다.

관련 페이지:

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 14. POMDP에서 Q(s,a)를 그대로 쓰기 어려운 이유

실제 환경 상태 `s`를 볼 수 없다면 엄밀히는 에이전트가 `Q(s,a)`를 직접 계산할 수 없다.

대신 관측/기록/상태 추정 표현에 대해 근사한다.

```text
Q(observation representation, action)
```

AASSR [Policy(정책 모델)](Policy)는 [가공하지 않은 원본(raw)](State-Representation) 숨은 환경 상태가 아니라 공개된 [관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)을 입력으로 사용한다.

따라서 엄밀하게는 학습 주체가 만든 **에이전트-side 의사결정 상태**에 대한 [Q값(Q-value)](Value-Functions-and-Bellman-Equation)라고 보는 것이 맞다.

---

# 15. State representation이 충분하지 않으면 생기는 일

두 숨겨진 situations가 실제로 중요한 차이가 있는데 표현에서 같은 값으로 압축되면 **상태 aliasing**이 생긴다.

```text
Situation A ─┐
             ├→ same representation R
Situation B ─┘
```

그런데 최적 행동이 다르면 [Policy](Policy)가 모순된 [대상 또는 학습 목표값(target)](Terminology-Guide)을 받는다.

[세계(World)](Model-Based-RL-and-World-Models) 학습 모델도 여러 incompatible [다음(next)](Terminology-Guide) states를 보게 된다.

AASSR의 [조건부(conditional)](Stochasticity-Uncertainty-and-Probability) [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration)는 이런 multimodality를 일부 흡수할 수 있지만, [의사결정에 중요한(decision-critical)](Calibration) 공개된 [학습 신호(signal)](Information-Theory-and-Intrinsic-Motivation)을 표현 자체에서 버리면 한계가 있다.

그래서 [관계 기반(Relational)](Relational-Representation-and-Generalization) [상태(State)](State-Representation) v3에서 [가장 최근의(latest)](Current-Status) 공개된 HTTP [상태 코드(status)](Terminology-Guide)를 명시적으로 보존했다.

관련 페이지:

- [State Representation](State-Representation)
- [Prophecy](Prophecy)

---

# 16. Relational abstraction과 Markov property의 trade-off

더 강하게 abstr행동할수록 전이는 쉬워질 수 있다.

하지만 중요한 차이를 너무 많이 지우면 Markov성이 약해질 수 있다.

```text
Concrete representation
→ 정보 많음
→ memorization 위험

Relational abstraction
→ transfer 좋음
→ 과도하면 state aliasing 위험
```

AASSR에서 [실제 개체 구분(concrete identity)](State-Representation)와 관계 기반 전이 식별 방식를 동시에 유지하는 이유가 여기에 있다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [State Representation](State-Representation)

---

# 17. AASSR을 POMDP 관점에서 보기

AASSR의 환경을 개념적으로 다음처럼 볼 수 있다.

```text
Hidden simulator state
    ↓ public response
Observation
    ↓ relational encoder
Relational State v3
    ├→ Policy
    ├→ Prophecy
    └→ Critic

Past real responses
    ↓
Knowledge / learned memory
```

[Prophecy](Prophecy)는:

```math
P(R_{t+1}\mid R_t,A_t,K_t)
```

에 가까운 공개된 미래 분포을 근사한다.

여기서 `R_t`는 true 숨은 환경 상태가 아니라 관계 기반 공개된 표현이다.

---

# 18. 핵심 오해 정리

## "POMDP면 Markov property가 없는가?"

Hidden 실제 환경 상태 과정은 Markov일 수 있다. **에이전트가 그 상태를 직접 관측하지 못하는 것**이 문제다.

## "같은 observation에서 여러 미래면 환경이 랜덤인가?"

반드시 그렇지 않다. 서로 다른 숨겨진 states가 같은 관측으로 [같은 구조를 가리키는 다른 이름(alias)](State-Representation)된 것일 수 있다.

## "history를 쓰면 무조건 POMDP가 해결되는가?"

아니다. 기록가 숨은 환경 상태를 완전히 식별하지 못할 수 있고, 모델 capacity/학습 데이터도 필요하다.

## "relational abstraction은 항상 좋나?"

아니다. [식별자(identifier)](State-Representation) [이름이나 사례를 그대로 외우는 암기(memorization)](Relational-Representation-and-Generalization)을 줄이지만 중요한 상태 [변수(variable)](Terminology-Guide)까지 버리면 aliasing이 커질 수 있다.

---

# 19. 다음으로 읽기

1. [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)
2. [State Representation](State-Representation)
3. [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)
4. [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
5. [Prophecy](Prophecy)

관련 색인: **[Concept Index](Concept-Index)**