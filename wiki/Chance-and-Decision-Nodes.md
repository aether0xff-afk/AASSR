# Chance Nodes and Decision Nodes

AASSR [Imagination(가상 미래 탐색)](Imagination)에서 가장 중요한 수학적 구분 중 하나는 **[환경 결과 노드(Chance node)](Chance-and-Decision-Nodes)**와 **[행동 선택 노드(Decision node)](Chance-and-Decision-Nodes)**다.

한 줄 요약:

```text
환경이 정하는 것 → expectation
Agent가 정하는 것 → max
```

---

# 1. 왜 node 종류를 구분해야 하는가?

Planning [탐색 트리(tree)](Counterfactual-Planning-and-Search)에는 두 종류의 [여러 미래로 갈라지는 분기(branching)](Chance-and-Decision-Nodes)이 있다.

```text
A. Agent가 여러 action 중 하나를 고름
B. 하나의 action 뒤 환경에서 여러 outcome이 확률적으로 발생함
```

겉으로는 둘 다 "여러 자식 [탐색 트리의 한 지점(node)](Chance-and-Decision-Nodes)"지만 의사결정 의미는 완전히 다르다.

---

# 2. Decision node

행동 선택 노드에서는 [에이전트(agent)](Reinforcement-Learning)가 [행동(action)](Reinforcement-Learning)을 선택할 수 있다.

```text
현재 state S
 ├→ action A
 ├→ action B
 └→ action C
```

[에이전트(Agent)](Reinforcement-Learning)가 최적 행동을 고른다고 가정하면:

```math
V_{decision}(S)=\max_aV(S,a)
```

이다.

이 `max`는 [Bellman optimality equation](Value-Functions-and-Bellman-Equation#7-bellman-optimality-equation)과 연결된다.

---

# 3. Chance node

환경 결과 노드에서는 이미 행동을 선택했고, 그 뒤 환경 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)이 확률적으로 갈린다.

```text
Action A 실행
  ├→ 70% outcome X
  ├→ 20% outcome Y
  └→ 10% outcome Z
```

에이전트는 X/Y/Z 중 하나를 고를 수 없다.

따라서 [확률을 고려해 기대되는(expected)](Chance-and-Decision-Nodes) [가치(value)](Value-Functions-and-Bellman-Equation):

```math
V_{chance}(A)=\sum_i p_iV_i
```

를 사용한다.

---

# 4. 가장 단순한 예

[행동(Action)](Reinforcement-Learning) A:

```text
10% → +1
90% → -1
```

행동 B:

```text
100% → +0.2
```

A에서 좋은 환경 결과만 max하면:

```text
A value = +1
```

처럼 보인다.

하지만 [확률 기댓값(expectation)](Chance-and-Decision-Nodes)은:

```math
0.1(1)+0.9(-1)=-0.8
```

이다.

B의 가치는:

```math
0.2
```

이므로 expected-[누적 보상(return)](Value-Functions-and-Bellman-Equation) 관점에서는 B가 더 좋다.

---

# 5. Optimistic stochastic backup

환경 환경 결과에:

```math
\max_iV_i
```

를 쓰면 에이전트가 실제로 통제할 수 없는 [무작위성(randomness)](Stochasticity-Uncertainty-and-Probability)를 선택할 수 있는 것처럼 취급한다.

이를 여기서는 **optimistic [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)**이라고 부른다.

```text
위험한 action
→ 아주 낮은 확률의 jackpot outcome 존재
→ max가 jackpot만 봄
→ action을 과대평가
```

AASSR [현재(current)](Current-Status) [계획기(planner)](Counterfactual-Planning-and-Search)는 이 오류를 피하기 위해 [환경의 확률 분기(chance)](Chance-and-Decision-Nodes)/[의사결정(decision)](Chance-and-Decision-Nodes) [의미 규칙(semantics)](State-Representation)를 분리한다.

---

# 6. Expectation은 risk-neutral objective다

Expected 가치만 최적화하는 것은 기본적으로 risk-neutral한 관점이다.

두 행동이 같은 확률 기댓값을 가져도 [분산(variance)](Stochasticity-Uncertainty-and-Probability)는 다를 수 있다.

```text
A: 항상 0
B: 50% +1, 50% -1
```

둘의 확률 기댓값은 0이다.

Risk-sensitive [학습 목표(objective)](Terminology-Guide)라면 분산, CVaR 같은 다른 기준을 사용할 수도 있다.

AASSR 현재 main 계획기는 기본 sparse-누적 보상 확률 기댓값 의미 규칙를 유지한다.

관련 페이지:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 7. Probability mass가 왜 중요한가?

Chance 가치 되돌림 계산에 [확률(probability)](Stochasticity-Uncertainty-and-Probability)가 없다면 여러 환경 결과을 공정하게 합칠 수 없다.

예:

```text
X value +1, probability 0.01
Y value  0, probability 0.99
```

단순 평균:

```math
(1+0)/2=0.5
```

는 실제 [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)을 반영하지 못한다.

올바른 확률 기댓값:

```math
0.01(1)+0.99(0)=0.01
```

이다.

그래서 AASSR [Prophecy(미래 예측 모델)](Prophecy)의 [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration) [결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability)를 [Imagination](Imagination)이 보존해야 한다.

---

# 8. Reliability는 probability weight가 아니다

어떤 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)가 [신뢰도(reliability)](Calibration) `0.2`라고 해서:

```math
0.2\times V
```

로 환경 확률 분기 확률처럼 처리하면 의미가 바뀐다.

AASSR 현재 [설계(design)](Design-Rationale):

```text
Outcome probability
→ chance expectation weight

Prediction reliability
→ branch 사용 가능 여부 gate
```

이다.

관련 페이지:

- [Calibration](Calibration)
- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 9. Decision max와 Policy

행동 선택 노드에서 `max`를 쓴다고 해서 실제 [Policy(정책 모델)](Policy)가 반드시 계획기의 모든 [미래(future)](Counterfactual-Planning-and-Search) 의사결정을 완벽히 실행한다는 뜻은 아니다.

[계획기(Planner)](Counterfactual-Planning-and-Search)는 미래 [상태(state)](State-Representation)에서 available 행동s를 평가하고 최선의 [계속 진행되는 상태(continuation)](Chance-and-Decision-Nodes)을 가정한다.

이것은 [환경 모델을 사용하는(model-based)](Model-Based-RL-and-World-Models) lookahead의 기본 가정이다.

실제 [환경(environment)](Reinforcement-Learning)에서는 첫 행동만 실행하고 다시 관측해 [계획(planning)](Counterfactual-Planning-and-Search)하므로 미래 [모델이 상상한(imagined)](Research-Jargon-Guide) decisions는 **[실제로 하지 않은 경우를 가정하는 반사실적(counterfactual)](Counterfactual-Planning-and-Search) [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)용**이다.

---

# 10. Alternating tree

AASSR의 계획 탐색 트리는 개념적으로 다음처럼 번갈아 간다.

```text
Decision: root action 선택
   ↓
Chance: 그 action의 stochastic outcome
   ↓
Decision: predicted state에서 다음 action
   ↓
Chance: 그 action의 stochastic outcome
   ↓
...
```

수식 구조:

```math
V(s)
=
\max_a
\left[
\sum_{s'}\hat P(s'|s,a)
V(s')
\right]
```

즉 classic 확률적 [효과를 비교하기 위한 대조 조건(control)](Ablation-Benchmarking-and-Reproducibility)의 Bellman optimality 구조와 연결된다.

---

# 11. Terminal outcome

Chance 결과 경로가 [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) [성공(success)](Terminology-Guide)/[실패(failure)](Replay-Buffer-and-Episode-Boundaries)로 끝나면 계속 진행 의사결정이 없다.

```text
Action A
 ├→ success terminal → +1
 └→ active state     → future decision
```

[에피소드 종료(Terminal)](Replay-Buffer-and-Episode-Boundaries) [범주(class)](Loss-Functions-and-Class-Imbalance) [예측(prediction)](Terminology-Guide)을 틀리면 탐색 트리 의미 규칙 자체가 바뀔 수 있다.

그래서 AASSR [Prophecy](Prophecy)는 에피소드 종료 범주를 [의사결정에 중요한(decision-critical)](Calibration) [대상 또는 학습 목표값(target)](Terminology-Guide)으로 본다.

---

# 12. Truncation outcome

Truncation은 true 실패와 다를 수 있다.

계획기에서 [예측된(predicted)](Terminology-Guide) [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)을 어떻게 평가할지는 [실험(experiment)](Experiments) [명세(contract)](Current-Status)와 일관되어야 한다.

AASSR 외부 [보상(reward)](Sparse-Reward-and-Credit-Assignment)에서는 administrative 외부 제한 종료을 실패 `-1`로 자동 바꾸지 않는다.

관련 페이지:

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

---

# 13. Chance branch pruning

Probability가 매우 낮은 결과 경로를 모두 버리면 [드문(rare)](Loss-Functions-and-Class-Imbalance) catastrophic 환경 결과을 잃을 수 있다.

반대로 모든 tiny 결과 경로를 유지하면 계산량이 폭발한다.

따라서 확률적 계획기에는:

- 환경 결과 [표본(sample)](Ablation-Benchmarking-and-Reproducibility) [횟수(count)](Terminology-Guide)
- [유망 후보만 남기는 빔 탐색(beam)](Counterfactual-Planning-and-Search) width
- [확률 질량(probability mass)](Stochasticity-Uncertainty-and-Probability) [의미 보존(preservation)](Ablation-Benchmarking-and-Reproducibility)

같은 [한쪽을 얻으면 다른 쪽을 잃는 상충 관계(trade-off)](Terminology-Guide)가 있다.

AASSR [Skill(성공 절차 재사용)](Skills) [Prophecy](Prophecy)에서도 빔 탐색을 자를 때 retained [확률 질량(mass)](Stochasticity-Uncertainty-and-Probability)를 다시 정규화한다.

관련 페이지:

- [Skills](Skills)
- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)

---

# 14. Expected value와 sampled rollout

모든 환경 결과 분포을 완전히 열거하지 않고 표본을 뽑아 확률 기댓값을 근사할 수 있다.

```math
\mathbb{E}[V]\approx\frac1N\sum_{j=1}^{N}V^{(j)}
```

하지만 드문 환경 결과을 표본하지 못할 수 있다.

AASSR [Prophecy](Prophecy)는 혼합 분포 [갈라진 결과 경로(branches)](Chance-and-Decision-Nodes)와 환경 결과 probabilities를 명시적으로 다루는 방향을 사용한다.

---

# 15. Decision branch pruning

행동 선택 노드에서는 모든 행동을 확장하기 비싸므로 [선택 후보(candidate)](Terminology-Guide) [후보 순위(ranking)](Policy)/빔 탐색/[중복 계산 제거(dedup)](Reproduction)을 사용할 수 있다.

하지만 이는 환경 확률 분기 환경 결과을 버리는 것과 의미가 다르다.

```text
Decision pruning
→ agent가 고려할 action 후보를 줄임

Chance pruning
→ environment outcome distribution 근사를 줄임
```

연구에서 둘을 구분해 보고해야 한다.

---

# 16. Structural alias와 Decision node

[실제 개체를 구분하는(Concrete)](State-Representation) 행동이 많지만 relationally 같은 구조라면 expensive 가치 computation을 공유할 수 있다.

```text
concrete A1 ─┐
concrete A2 ─┼→ same structural decision
concrete A3 ─┘
```

하지만 최종 실제 행동은 [실제 개체 구분(concrete identity)](State-Representation)로 다시 [역할을 실제 객체에 연결(bind)](Skills)한다.

이것은 행동 확률를 합치는 환경 확률 분기 operation이 아니라 **의사결정 [계산(compute)](Reproduction) deduplication**이다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)

---

# 17. Critic leaf와 chance backup

Depth [제한(limit)](Terminology-Guide)에서 각 미래 상태의 [Critic(미래 가치 평가기)](Critic) 가치가 나온다고 하자.

```text
Outcome X → Critic 0.8
Outcome Y → Critic -0.2
```

Probability:

```text
X 0.25
Y 0.75
```

Root 행동 기대되는 가치:

```math
0.25(0.8)+0.75(-0.2)=0.05
```

[Critic](Critic) 가치가 큰 결과 경로 하나만 고르면 안 된다.

---

# 18. Model reliability와 chance tree

[예측 신뢰도(Prediction reliability)](Calibration)가 낮은 환경 결과 분포은 정확한 확률 기댓값 계산 자체가 의미 없을 수 있다.

그래서:

```text
먼저 reliability gate
그 뒤 probability expectation
```

라는 의미 순서가 중요하다.

AASSR 현재 [판정 관문(gate)](Terminology-Guide)는 low-confidence [탐색의 첫 행동(root)](Imagination) 예측을 [기본 행동 덮어쓰기(override)](Imagination) 후보에서 제외할 수 있다.

---

# 19. Policy baseline 비교

계획기가 preferred 탐색의 첫 행동 `B`를 찾았다고 하자.

실제 기본 행동 덮어쓰기는 단순:

```text
B is argmax
```

만으로 결정하지 않는다.

AASSR 현재 flow에는:

- [Policy](Policy) 탐색의 첫 행동 [예측 신뢰도(prediction reliability)](Calibration)
- 선택 후보 신뢰도
- [현재 주변에 한정된 국소적(local)](Critic-Support-and-OOD) [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD)
- [실제 행동 개입(intervention)](Imagination) [최소 차이 기준(margin)](Imagination)

등이 포함된다.

즉 환경 확률 분기/의사결정 탐색 트리는 **탐색의 첫 행동 가치를 만드는 내부 계획 의미 규칙**이고, 최종 [실제 실행(execution)](Research-Jargon-Guide) 판정 관문는 별도다.

---

# 20. AASSR의 가장 중요한 기억법

```text
Agent가 고를 수 있나?
    |
   yes → max
    |
    no → probability-weighted expectation
```

그리고:

```text
그 prediction 자체를 믿을 수 있나?
→ Calibration

그 value를 믿을 training evidence가 있나?
→ Critic support
```

이다.

---

# 21. 다음으로 읽기

- [Imagination](Imagination)
- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)
- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)
- [Critic](Critic)
- [Calibration](Calibration)

관련 색인: **[Concept Index](Concept-Index)**