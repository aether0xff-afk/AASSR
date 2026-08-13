# Counterfactual Planning and Search

**Counterfactual [계획(planning)](Counterfactual-Planning-and-Search)**은 실제 행동을 하기 전에 "다른 행동을 했다면 어떤 미래가 생길까?"를 계산하고 비교하는 과정이다.

AASSR의 [Imagination](Imagination)은 [학습된(learned)](Neural-Networks-and-Optimization) [세계 모델(world model)](Model-Based-RL-and-World-Models)인 [Prophecy](Prophecy)를 이용해 이런 [실제로 하지 않은 경우를 가정하는 반사실적(counterfactual)](Counterfactual-Planning-and-Search) [미래(future)](Counterfactual-Planning-and-Search)를 전개한다.

---

# 1. Planning이란?

Planning은 현재 가진 환경 [학습 모델(model)](Terminology-Guide)을 이용해 미래 consequences를 계산하고 행동을 선택하는 과정이다.

```text
현재 상태
  ↓
가능한 행동 후보
  ↓
미래 결과 예측
  ↓
미래 가치 비교
  ↓
현재 행동 하나 선택
```

Learning과 다르다.

```text
Learning
→ 경험으로 model/policy parameter를 바꿈

Planning
→ 현재 parameter를 이용해 행동 전에 계산함
```

AASSR [현재(current)](Current-Status) main [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)에서 [Imagination(가상 미래 탐색)](Imagination)은 주로 계획 역할이다.

---

# 2. Counterfactual

Counterfactual은 실제로 일어나지 않은 대안을 묻는다.

```text
실제로 A를 실행했음
```

과 별개로:

```text
만약 B를 했으면?
만약 C를 했으면?
```

를 모델을 통해 계산한다.

AASSR에서는 아직 실행하지 않은 여러 [탐색의 첫 행동(root)](Imagination) [행동(action)](Reinforcement-Learning)을 [Prophecy(미래 예측 모델)](Prophecy)로 전개한다.

---

# 3. Lookahead

현재에서 `k` [단계(step)](Terminology-Guide) 앞까지 미래를 보는 것을 lookahead라고 생각할 수 있다.

```text
Depth 0: current state
Depth 1: one-step futures
Depth 2: two-step futures
Depth 3: ...
```

Lookahead [탐색 깊이(depth)](Counterfactual-Planning-and-Search)가 길수록 장기 consequence를 더 볼 수 있지만 학습 모델 [오차(error)](Loss-Functions-and-Class-Imbalance)와 [계산(compute)](Reproduction)가 증가한다.

---

# 4. Planning horizon

[계획기(Planner)](Counterfactual-Planning-and-Search)가 명시적으로 [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search)하는 길이를 [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search)이라고 할 수 있다.

```text
short horizon
→ 빠르고 model error 적음
→ 먼 결과 놓칠 수 있음

long horizon
→ 먼 결과 고려
→ model error / compute 증가
```

AASSR의 imagination 탐색 깊이는 이 [한쪽을 얻으면 다른 쪽을 잃는 상충 관계(trade-off)](Terminology-Guide) 안에 있다.

---

# 5. Search tree

여러 행동과 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)을 전개하면 [탐색 트리(tree)](Counterfactual-Planning-and-Search)가 된다.

```text
S0
 ├→ Action A
 │    ├→ outcome A1
 │    └→ outcome A2
 │
 ├→ Action B
 │    ├→ outcome B1
 │    └→ outcome B2
 │
 └→ Action C
      └→ ...
```

하지만 이 탐색 트리에는 서로 다른 종류의 [여러 미래로 갈라지는 분기(branching)](Chance-and-Decision-Nodes)이 섞여 있다.

- [에이전트(agent)](Reinforcement-Learning)가 고르는 행동 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)
- [환경(environment)](Reinforcement-Learning)가 만드는 확률적 환경 결과 결과 경로

AASSR은 이를 [Decision node와 Chance node](Chance-and-Decision-Nodes)로 분리한다.

---

# 6. Branching factor

한 [탐색 트리의 한 지점(node)](Chance-and-Decision-Nodes)에서 확장하는 자식 수를 분기 [실험에서 바꾸어 보는 요인(factor)](Ablation-Benchmarking-and-Reproducibility)라고 한다.

[행동(Action)](Reinforcement-Learning) 후보가 `b`, 탐색 깊이가 `d`이면 단순 full 탐색 트리 노드 수는 대략 지수적으로 증가할 수 있다.

```math
O(b^d)
```

실제로 확률적 환경 결과 결과 경로까지 있으면 더 커질 수 있다.

따라서 [계획기(planner)](Counterfactual-Planning-and-Search)에는 [유망하지 않은 탐색 가지를 제거하는 가지치기(pruning)](Counterfactual-Planning-and-Search), [유망 후보만 남기는 빔 탐색(beam)](Counterfactual-Planning-and-Search) [탐색(search)](Counterfactual-Planning-and-Search), [중복 계산 제거(dedup)](Reproduction), [묶음 처리(batching)](Reproduction) 같은 계산 전략이 필요하다.

---

# 7. Beam search

각 탐색 깊이에서 모든 결과 경로를 유지하지 않고 일부 높은 priority 결과 경로만 유지한다.

```text
Depth 1: 100 branches
   ↓ top 20 유지
Depth 2: 확장
   ↓ top 20 유지
...
```

장점:

- 계산량 제한

단점:

- 초기에 낮게 평가된 중요한 결과 경로가 잘릴 수 있음

AASSR의 계획기/[Skill(성공 절차 재사용)](Skills) 가상 미래 전개에서도 제한된 분기을 관리하는 아이디어가 사용된다.

---

# 8. Pruning

명백히 쓸모없거나 [신뢰하기 어려운(unreliable)](Calibration)한 결과 경로를 더 이상 확장하지 않는 것이다.

예:

```text
terminal failure
→ 더 확장할 필요 없음

model reliability 너무 낮음
→ branch expansion 중단 가능
```

하지만 가지치기이 탐색의 첫 행동 행동 자체를 삭제하게 되면 실제 [현재 허용된(legal)](Terminology-Guide) 행동 비교가 왜곡될 수 있다.

그래서 AASSR에서는 **탐색의 첫 행동 [의미 보존(preservation)](Ablation-Benchmarking-and-Reproducibility)**이 중요하다.

---

# 9. Root preservation

어떤 탐색의 첫 행동 행동의 깊은 가상 미래 전개이 실패해도 이미 계산한 shallow [가치(value)](Value-Functions-and-Bellman-Equation)까지 잃을 필요는 없다.

```text
Action A
→ depth 1까지 usable
→ depth 2에서 uncertainty 너무 큼
```

잘못된 처리:

```text
A root 전체 삭제
```

보수적인 처리:

```text
depth 2 expansion 중단
하지만 root A는 shallower evidence로 유지
```

이것이 AASSR 현재 계획기의 중요한 설계 원칙이다.

---

# 10. Structural deduplication

실제 행동 [현재 선택 가능한 영역(surface)](Terminology-Guide)에는 [실제 개체를 구분하는(concrete)](State-Representation) name만 다른 행동이 많을 수 있다.

```text
route-12 request
route-31 request
route-44 request
```

[관계 기반(Relational)](Relational-Representation-and-Generalization) [역할(role)](Relational-Representation-and-Generalization)이 같다면 세계 모델/[Critic(미래 가치 평가기)](Critic) 계산도 같은 구조일 수 있다.

```text
많은 concrete aliases
      ↓ group
하나의 structural root
      ↓ expensive model evaluation 1회
결과를 concrete aliases에 fan-out
```

이를 통해 계획 complexity를 크게 줄일 수 있다.

하지만 실제 실행은 [실제 실행 행동(concrete action)](State-Representation)을 유지해야 한다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [Imagination](Imagination)

---

# 11. Compute identity와 Execution identity

AASSR에서 매우 중요한 분리다.

```text
Compute identity
= relational structure

Execution identity
= concrete action
```

두 행동이 계획 계산상 같은 구조라고 해서 실제 환경에서 같은 객체는 아니다.

따라서 중복 제거은 **계산 공유**이지 **실제 [식별 방식(identity)](State-Representation) 병합**이 아니다.

---

# 12. Leaf evaluation

Planning 탐색 트리를 무한히 펼칠 수 없으므로 어떤 탐색 깊이에서 멈추고 leaf의 미래 가치를 추정한다.

```text
S0 → Ŝ1 → Ŝ2 → Ŝ3
                 ↓
               Critic
```

AASSR의 [Critic](Critic)이 이 역할을 맡는다.

---

# 13. Search와 Value function

계획기는 두 정보원을 조합할 수 있다.

```text
Explicit rollout value
+
Leaf value estimate
```

짧은 미래 탐색 범위에서 학습 모델 [예측(prediction)](Terminology-Guide)을 쓰고, 그 이후는 [Critic](Critic)이 요약된 long-term 가치를 제공한다.

이런 구조는 다양한 계획 알고리즘에서 흔한 아이디어다.

---

# 14. Chance와 Decision backup

행동을 고르는 노드:

```math
V_{decision}=\max_aV(a)
```

환경 환경 결과 노드:

```math
V_{chance}=\sum_ip_iV_i
```

둘을 구분하지 않으면 환경 [무작위성(randomness)](Stochasticity-Uncertainty-and-Probability)를 에이전트 choice처럼 취급하는 오류가 생긴다.

더 자세히:

- [Chance and Decision Nodes](Chance-and-Decision-Nodes)

---

# 15. Expected planning vs Optimistic planning

환경 환경 결과에서 가장 좋은 결과만 고르면:

```math
V=\max_iV_i
```

가 된다.

하지만 에이전트가 환경 결과 `i`를 선택할 수 없다면 이것은 지나치게 optimistic하다.

AASSR은 확률적 환경 결과을 [확률로 가중한(probability-weighted)](Chance-and-Decision-Nodes) [확률 기댓값(expectation)](Chance-and-Decision-Nodes)으로 [미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)한다.

---

# 16. Model Predictive Control과의 개념적 유사점

Model Predictive Control(MPC)은 현재 시점에서 미래 미래 탐색 범위을 최적화한 뒤 **첫 [효과를 비교하기 위한 대조 조건(control)](Ablation-Benchmarking-and-Reproducibility) 행동만 실행하고 다시 관측해서 재계획**하는 방식이다.

AASSR [Imagination](Imagination)도 넓은 의미에서 비슷한 receding-horizon 구조를 가진다.

```text
현재에서 여러 미래 계산
→ 첫 action 하나 실행
→ 실제 response 관측
→ 다시 planning
```

하지만 AASSR은 학습된 확률적 [관계 기반(relational)](Relational-Representation-and-Generalization) 학습 모델과 RL [Policy(정책 모델)](Policy)/[Critic](Critic)을 사용하므로 전통적 continuous-control MPC와 동일한 알고리즘이라고 부르는 것은 부정확하다.

---

# 17. Receding horizon

한 번 긴 계획을 만든 뒤 그대로 끝까지 실행하는 것이 아니라 매 실제 단계마다 다시 계획한다.

```text
Plan A0,A1,A2,A3
→ A0만 실행
→ 실제 S1 관측
→ 새로 plan
```

장점:

- 학습 모델 예측과 실제 환경 결과 차이를 다음 [의사결정(decision)](Chance-and-Decision-Nodes)에서 즉시 반영
- open-loop 오차 누적 감소

AASSR은 실제 행동 하나 실행 후 다시 [공개된(public)](State-Representation) [응답(response)](State-Representation)를 읽는 구조다.

---

# 18. Open-loop와 Closed-loop

## Open-loop

미리 만든 행동 [순서열(sequence)](GRU-and-Sequence-Models)를 실제 환경 결과에 관계없이 계속 실행.

## Closed-loop

매 [관측(observation)](MDP-and-POMDP)을 보고 다음 행동을 다시 결정.

AASSR [Imagination](Imagination)은 내부적으로 미래 순서열를 상상하지만 실제 실행은 closed-loop에 가깝다.

```text
상상: multi-step
실행: first step only
재관측 후 다시 결정
```

---

# 19. Planning with uncertainty

[세계(World)](Model-Based-RL-and-World-Models) 학습 모델 예측이 신뢰하기 어려운한 결과 경로까지 강하게 최적화하면 학습 모델 오차 [활용(exploitation)](Exploration-and-Exploitation)이 생길 수 있다.

그래서 계획기는:

- 학습 모델 [신뢰도(reliability)](Calibration)
- critic [데이터 근거(support)](Critic-Support-and-OOD)

를 확인할 수 있다.

AASSR 현재 [설계(design)](Design-Rationale)은 [불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability)를 가치 [추가 점수(bonus)](Information-Theory-and-Intrinsic-Motivation)/penalty로 섞기보다 **eligibility [판정 관문(gate)](Terminology-Guide)**로 분리한다.

관련 페이지:

- [Calibration](Calibration)
- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 20. Intervention margin

계획기가 [Policy](Policy)보다 아주 미세하게 높은 탐색의 첫 행동를 찾았다고 바로 [행동 전환(switch)](Imagination)하면 [잡음(noise)](Stochasticity-Uncertainty-and-Probability)에 민감할 수 있다.

```math
V_{candidate}-V_{policy}\ge m
```

일 때만 [기본 행동 덮어쓰기(override)](Imagination)하도록 [최소 차이 기준(margin)](Imagination)을 둘 수 있다.

AASSR 현재 [Imagination](Imagination)은 [고정된(fixed)](Ablation-Benchmarking-and-Reproducibility) [실제 행동 개입(intervention)](Imagination) 최소 차이 기준을 사용한다.

중요한 점은 최소 차이 기준이 [보상(reward)](Sparse-Reward-and-Credit-Assignment)가 아니라 **행동 행동 전환 의사결정 [판정 기준값(threshold)](Terminology-Guide)**라는 것이다.

---

# 21. Same-checkpoint planning evaluation

계획기의 순수 [다른 조건이 같을 때의 추가 기여(marginal)](Ablation-Benchmarking-and-Reproducibility) [효과(effect)](Ablation-Benchmarking-and-Reproducibility)를 보려면:

```text
one trained checkpoint
       ↓ freeze
OFF evaluation
ON evaluation
```

을 비교해야 한다.

[학습(Training)](Reinforcement-Learning) 중 계획기 실제 행동 개입이 [경험 경로(trajectory)](Reinforcement-Learning)를 바꾸면 두 모델은 더 이상 같은 학습 조건이 아니다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)
- [Imagination](Imagination)

---

# 22. Planner가 action을 바꾼 것과 좋은 action을 고른 것은 다르다

세 단계가 다르다.

```text
1. plan을 만들 수 있음
2. Policy와 다른 candidate를 선택할 수 있음
3. 그 candidate가 실제로 더 좋은 결과를 만듦
```

AASSR의 과거 [진단 실험(diagnostic)](Evidence-Matrix)은 1, 2는 성립해도 3이 자동으로 성립하지 않음을 보여줬다.

그래서 실제 행동 개입 [횟수(count)](Terminology-Guide)와 [성공(success)](Terminology-Guide)를 따로 본다.

---

# 23. Planning failure modes

## Branch explosion

행동 × 환경 결과 × 탐색 깊이로 계산량 증가.

## Model exploitation

Model이 틀리는 방향을 계획기가 선택.

## OOD leaf value

[Critic](Critic)이 경험하지 않은 [상태(state)](State-Representation)에서 큰 값을 출력.

## Over-pruning

중요한 탐색의 첫 행동가 너무 일찍 삭제.

## Optimistic chance backup

환경 환경 결과에 max를 사용.

## Inert planner

모든 탐색의 첫 행동가 비슷하거나 판정 관문가 너무 강해 행동을 한 번도 바꾸지 못함.

---

# 24. AASSR Imagination과 연결

```text
Policy root candidates
       ↓
structural dedup
       ↓
Prophecy stochastic expansion
       ↓
chance/decision alternating tree
       ↓
Critic leaf/branch values
       ↓
Calibration + local support
       ↓
advantage margin
       ↓
Policy override or fallback
```

---

# 25. 다음으로 읽기

- [Chance and Decision Nodes](Chance-and-Decision-Nodes)
- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Critic, Support and OOD](Critic-Support-and-OOD)
- [Imagination](Imagination)
- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

관련 색인: **[Concept Index](Concept-Index)**