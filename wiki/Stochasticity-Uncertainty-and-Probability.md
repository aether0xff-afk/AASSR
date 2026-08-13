# Stochasticity, Uncertainty and Probability

AASSR에서 가장 자주 혼동하기 쉬운 네 단어는 다음이다.

```text
probability
uncertainty
reliability
value
```

여기에 `support`까지 더하면 다섯 가지가 된다.

이 페이지의 핵심 목표는 이들을 **절대로 같은 scalar 의미로 섞지 않는 것**이다.

---

# 1. Probability

Probability는 어떤 사건이 일어날 가능성을 `0~1` 사이의 값으로 표현한다.

```math
0\le P(E)\le1
```

예:

```text
P(200) = 0.7
P(403) = 0.2
P(429) = 0.1
```

이 값들의 합이 1이면 환경 outcome distribution으로 볼 수 있다.

AASSR [Prophecy(미래 예측 모델)](Prophecy)의 **[결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability)**가 이 의미다.

---

# 2. Random variable

확률적으로 여러 값을 가질 수 있는 변수를 random variable이라고 한다.

예:

```text
X = 다음 HTTP status
```

일 수 있다.

```text
X=200 with p=0.7
X=403 with p=0.2
X=429 with p=0.1
```

World [학습 모델(model)](Terminology-Guide)은 이런 next-outcome random variable의 distribution을 근사할 수 있다.

---

# 3. Expected value

여러 outcome의 [가치(value)](Value-Functions-and-Bellman-Equation)가 다를 때 확률 가중 평균을 구한다.

```math
\mathbb{E}[V]
=
\sum_i p_iV_i
```

예:

```text
70% → value +1
20% → value  0
10% → value -1
```

이면:

```math
0.7(1)+0.2(0)+0.1(-1)=0.6
```

이다.

AASSR의 [chance node](Chance-and-Decision-Nodes)는 환경 outcome을 이런 expectation으로 backup한다.

---

# 4. Variance

기대값이 같아도 distribution의 퍼짐은 다를 수 있다.

```math
Var(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]
```

예:

```text
A: 항상 0.5
B: 50%로 0, 50%로 1
```

둘의 expectation은 0.5지만 B가 훨씬 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability)하다.

Risk-sensitive [계획(planning)](Counterfactual-Planning-and-Search)에서는 variance 같은 정보도 중요할 수 있다.

AASSR [현재(current)](Current-Status) [계획기(planner)](Counterfactual-Planning-and-Search)의 기본 chance semantics는 결과 확률의 expectation을 중심으로 하며, variance 자체를 [보상(reward)](Sparse-Reward-and-Credit-Assignment)로 추가하는 구조는 아니다.

---

# 5. Stochasticity

**Stochasticity**는 환경 자체의 결과가 확률적으로 달라질 수 있다는 뜻이다.

```text
같은 true S, 같은 A
  |-- p1 → S1'
  |-- p2 → S2'
  `-- p3 → S3'
```

이것은 모델이 멍청해서 생기는 것이 아니다.

완벽한 학습 모델을 가지고 있어도 outcome을 하나로 확정할 수 없을 수 있다.

---

# 6. Partial observability로 인한 apparent stochasticity

환경 자체는 deterministic하더라도 [에이전트(agent)](Reinforcement-Learning)가 [숨은 환경 상태(hidden state)](MDP-and-POMDP)를 못 보면 [공개된(public)](State-Representation) 관점에서는 여러 outcome이 가능해 보일 수 있다.

```text
hidden H1 + public O + action A → outcome X
hidden H2 + public O + action A → outcome Y
```

에이전트는 `H1/H2`를 구분하지 못하므로:

```text
public (O,A)
→ X or Y
```

처럼 보인다.

관련 페이지:

- [MDP and POMDP](MDP-and-POMDP)

---

# 7. Uncertainty

Uncertainty는 "모른다" 또는 "결과가 정해져 있지 않다"는 더 넓은 개념이다.

Machine learning에서는 크게 두 종류를 구분하는 경우가 많다.

```text
Aleatoric uncertainty
Epistemic uncertainty
```

---

# 8. Aleatoric uncertainty

데이터/환경 자체의 randomness 또는 관측만으로 제거할 수 없는 uncertainty다.

예:

```text
동일한 public 정보에서도 실제 outcome이 확률적으로 달라짐
```

데이터를 무한히 더 모아도 환경 자체가 확률적하면 이 uncertainty는 사라지지 않는다.

AASSR에서는 **mixture outcome distribution**이 이런 [여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration) uncertainty를 표현하는 역할을 한다.

---

# 9. Epistemic uncertainty

모델이 충분히 배우지 못해서 생기는 uncertainty다.

예:

```text
이 state/action region을 거의 본 적 없음
→ model이 무엇이 일어날지 잘 모름
```

더 많은 적절한 [학습 데이터(training data)](Terminology-Guide)가 있으면 줄어들 수 있다.

AASSR에서는:

- ensemble
- [검증용 분리 데이터(holdout)](Calibration) calibration
- [국소 데이터 근거(local support)](Critic-Support-and-OOD)

등이 epistemic risk를 판단하는 데 관련된다.

---

# 10. Aleatoric과 Epistemic을 섞으면 생기는 문제

World 학습 모델의 여러 [예측(prediction)](Terminology-Guide)이 다르다고 하자.

그 이유가:

```text
A. 환경이 실제로 여러 outcome을 가짐
B. 모델들이 서로 동의하지 못함
```

중 무엇인지 구분해야 한다.

A는 확률적 [환경(environment)](Reinforcement-Learning) structure이고 B는 학습 모델 ignorance일 수 있다.

그래서 AASSR은 개념적으로:

```text
Mixture components
→ environment outcome modes

Ensemble/calibration
→ model reliability evidence
```

를 분리하려 한다.

---

# 11. Outcome probability

AASSR에서 결과 확률는:

> 이 예측 branch가 나타내는 공개된 outcome이 환경에서 발생할 [확률 질량(probability mass)](Stochasticity-Uncertainty-and-Probability)

다.

[환경 결과 노드(Chance node)](Chance-and-Decision-Nodes)에서:

```math
V=\sum_i p_iV_i
```

의 `p_i`로 사용된다.

이 값을 [Critic(미래 가치 평가기)](Critic) 가치에 bonus처럼 더하지 않는다.

---

# 12. Reliability

Reliability는:

> [세계 모델(world model)](Model-Based-RL-and-World-Models)이 이 [상태(state)](State-Representation)/[행동(action)](Reinforcement-Learning) region에서 내놓는 예측을 실제로 얼마나 믿을 수 있는가?

라는 질문이다.

예:

```text
outcome distribution:
200 0.9
403 0.1

reliability:
0.15
```

일 수 있다.

즉 학습 모델은 90%라고 말하지만 **그 90% 자체를 잘 믿을 수 없는 상황**이다.

AASSR [Calibration(예측 신뢰도 보정)](Calibration)이 이 의미를 담당한다.

관련 페이지:

- [Calibration](Calibration)
- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)

---

# 13. Confidence라는 단어의 위험

`confidence`라는 단어는 여러 라이브러리/논문에서 서로 다른 의미로 쓰인다.

예:

- softmax probability
- ensemble agreement
- calibration score
- posterior probability
- heuristic certainty

그래서 AASSR 위키에서는 가능하면 더 구체적으로:

```text
outcome probability
prediction reliability
local support
Critic value
```

라고 부르는 편이 안전하다.

---

# 14. Value

Value는 **미래 task [누적 보상(return)](Value-Functions-and-Bellman-Equation)의 기대값**이다.

```math
V(s)=\mathbb{E}[G_t\mid S_t=s]
```

또는 행동 가치:

```math
Q(s,a)=\mathbb{E}[G_t\mid S_t=s,A_t=a]
```

중요:

```text
reliability 높음
!=
value 높음
```

예측이 확실한 실패는 [신뢰도(reliability)](Calibration)가 높고 가치는 낮을 수 있다.

---

# 15. Support

AASSR의 local [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD)는:

> 현재 상태/행동의 가치 estimate가 실제 [Critic](Critic) 학습 데이터 근처에 있는가?

를 나타낸다.

```text
support 높음
!=
좋은 action
```

단지 가치 estimate를 신뢰할 empirical basis가 더 있다는 뜻이다.

관련 페이지:

- [Critic, Support and OOD](Critic-Support-and-OOD)
- [Critic](Critic)

---

# 16. 다섯 값을 한 예제로 비교

어떤 행동 `A`에 대해:

```text
Predicted outcomes:
70% → state X
30% → state Y

Prediction reliability = 0.8
Critic values:
V(X)=+0.5
V(Y)=-0.5
Local critic support = 0.9
```

라고 하자.

Chance 가치:

```math
0.7(0.5)+0.3(-0.5)=0.2
```

여기서:

```text
0.7 / 0.3 → outcome probabilities
0.8       → world-model reliability
+0.5/-0.5 → task-return values
0.9       → Critic support
0.2       → expected branch value
```

이다.

전부 다른 의미다.

---

# 17. Calibration과 probability

Probability 학습 모델이 calibration되어 있다는 것은 대략:

> 모델이 70%라고 말한 사건들이 장기적으로 실제로 약 70% 빈도로 일어나는가?

라는 의미와 연결된다.

하지만 AASSR 현재 semantic calibration은 단순 binary confidence calibration만이 아니라 **semantic next-state correctness, [상태 코드(status)](Terminology-Guide), legal mask, [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) 등 [의사결정에 중요한(decision-critical)](Calibration) correctness를 검증용 분리 데이터으로 평가**하는 구조다.

따라서 일반 확률 calibration 개념과 AASSR의 operational 신뢰도 [판정 관문(gate)](Terminology-Guide)를 구분해서 이해하는 것이 좋다.

---

# 18. Distribution shift

Training distribution과 [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility) distribution이 달라지면 학습 모델 uncertainty가 커질 수 있다.

```text
Training: Level 0/1 region
Evaluation: higher-level unseen region
```

모델은 높은 softmax probability를 내더라도 실제로 [학습 분포 밖(OOD)](Critic-Support-and-OOD)일 수 있다.

이런 이유로 raw 학습 모델 confidence만으로 충분하지 않다.

관련 페이지:

- [Critic, Support and OOD](Critic-Support-and-OOD)
- [Relational Representation and Generalization](Relational-Representation-and-Generalization)

---

# 19. Class imbalance

Rare outcome이 중요한데 학습 데이터에서 매우 적으면 학습 모델이 다수 class만 잘 맞혀도 높은 accuracy를 얻을 수 있다.

예:

```text
200: 95%
429:  1%
기타: 4%
```

항상 200만 예측해도 naive accuracy는 높다.

하지만 429를 놓치면 decision quality에 치명적일 수 있다.

AASSR status-supervised [Prophecy](Prophecy)가 class balance를 고려하는 이유다.

관련 페이지:

- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)
- [Prophecy](Prophecy)

---

# 20. Uncertainty penalty의 위험

Planner 가치에 uncertainty penalty를 직접 넣는 방식:

```math
V'=V-\lambda U
```

도 가능하다.

하지만 AASSR 현재 design은 신뢰도를 **가치 자체와 분리된 판정 관문**로 보는 방향을 택한다.

왜냐하면 uncertainty가 높은 행동이 반드시 나쁜 행동은 아니며, 신뢰도와 task [학습 목표(objective)](Terminology-Guide)를 섞으면 해석이 어려워질 수 있기 때문이다.

```text
reliability gate 통과
→ Critic value 비교

통과 실패
→ fail closed
```

---

# 21. Risk와 uncertainty도 다르다

Risk는 일반적으로 outcome distribution에서 나쁜 결과가 발생할 가능성과 그 크기를 말한다.

Uncertainty는 distribution 자체를 얼마나 알고 있는지까지 포함할 수 있다.

예:

```text
Action A
실패 확률 40%를 정확히 앎
→ high risk, low epistemic uncertainty

Action B
실패 확률이 1%인지 80%인지 잘 모름
→ uncertainty 큼
```

AASSR 현재 계획기의 기본 학습 목표는 expected sparse 누적 보상이지만, 위험과 uncertainty를 구분하는 것이 해석에 중요하다.

---

# 22. AASSR 연결 표

| 개념 | AASSR에서의 대표 역할 |
|---|---|
| [결과 확률(Outcome probability)](Stochasticity-Uncertainty-and-Probability) | [Prophecy](Prophecy) mixture branch mass |
| Aleatoric uncertainty | 여러 확률적 outcome mode |
| Epistemic uncertainty | 학습 모델 knowledge 부족 |
| Reliability | [Calibration](Calibration) 판정 관문 |
| Value | [Policy(정책 모델)](Policy) [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) / [Critic](Critic) 누적 보상 estimate |
| Support | [Critic](Critic) real-training neighborhood [증거(evidence)](Evidence-Matrix) |
| Expected 가치 | [Imagination(가상 미래 탐색)](Imagination) chance backup |

---

# 23. 다음으로 읽기

- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)
- [Chance and Decision Nodes](Chance-and-Decision-Nodes)
- [Calibration](Calibration)
- [Critic, Support and OOD](Critic-Support-and-OOD)
- [Prophecy](Prophecy)

관련 색인: **[Concept Index](Concept-Index)**