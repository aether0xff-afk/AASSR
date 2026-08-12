# Counterfactual Planning and Search

**Counterfactual planning**은 실제 행동을 하기 전에 "다른 행동을 했다면 어떤 미래가 생길까?"를 계산하고 비교하는 과정이다.

AASSR의 [Imagination](Imagination)은 learned world model인 [Prophecy](Prophecy)를 이용해 이런 counterfactual future를 전개한다.

---

# 1. Planning이란?

Planning은 현재 가진 환경 model을 이용해 미래 consequences를 계산하고 행동을 선택하는 과정이다.

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

AASSR current main protocol에서 Imagination은 주로 planning 역할이다.

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

AASSR에서는 아직 실행하지 않은 여러 root action을 Prophecy로 전개한다.

---

# 3. Lookahead

현재에서 `k` step 앞까지 미래를 보는 것을 lookahead라고 생각할 수 있다.

```text
Depth 0: current state
Depth 1: one-step futures
Depth 2: two-step futures
Depth 3: ...
```

Lookahead depth가 길수록 장기 consequence를 더 볼 수 있지만 model error와 compute가 증가한다.

---

# 4. Planning horizon

Planner가 명시적으로 rollout하는 길이를 horizon이라고 할 수 있다.

```text
short horizon
→ 빠르고 model error 적음
→ 먼 결과 놓칠 수 있음

long horizon
→ 먼 결과 고려
→ model error / compute 증가
```

AASSR의 imagination depth는 이 trade-off 안에 있다.

---

# 5. Search tree

여러 action과 stochastic outcome을 전개하면 tree가 된다.

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

하지만 이 tree에는 서로 다른 종류의 branching이 섞여 있다.

- agent가 고르는 action branch
- environment가 만드는 stochastic outcome branch

AASSR은 이를 [Decision node와 Chance node](Chance-and-Decision-Nodes)로 분리한다.

---

# 6. Branching factor

한 node에서 확장하는 자식 수를 branching factor라고 한다.

Action 후보가 `b`, depth가 `d`이면 단순 full tree node 수는 대략 지수적으로 증가할 수 있다.

```math
O(b^d)
```

실제로 stochastic outcome branch까지 있으면 더 커질 수 있다.

따라서 planner에는 pruning, beam search, dedup, batching 같은 계산 전략이 필요하다.

---

# 7. Beam search

각 depth에서 모든 branch를 유지하지 않고 일부 높은 priority branch만 유지한다.

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

- 초기에 낮게 평가된 중요한 branch가 잘릴 수 있음

AASSR의 planner/Skill rollout에서도 제한된 branching을 관리하는 아이디어가 사용된다.

---

# 8. Pruning

명백히 쓸모없거나 unreliable한 branch를 더 이상 확장하지 않는 것이다.

예:

```text
terminal failure
→ 더 확장할 필요 없음

model reliability 너무 낮음
→ branch expansion 중단 가능
```

하지만 pruning이 root action 자체를 삭제하게 되면 실제 legal action 비교가 왜곡될 수 있다.

그래서 AASSR에서는 **root preservation**이 중요하다.

---

# 9. Root preservation

어떤 root action의 깊은 rollout이 실패해도 이미 계산한 shallow value까지 잃을 필요는 없다.

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

이것이 AASSR current planner의 중요한 설계 원칙이다.

---

# 10. Structural deduplication

실제 action surface에는 concrete name만 다른 action이 많을 수 있다.

```text
route-12 request
route-31 request
route-44 request
```

Relational role이 같다면 world model/Critic 계산도 같은 구조일 수 있다.

```text
많은 concrete aliases
      ↓ group
하나의 structural root
      ↓ expensive model evaluation 1회
결과를 concrete aliases에 fan-out
```

이를 통해 planning complexity를 크게 줄일 수 있다.

하지만 실제 실행은 concrete action을 유지해야 한다.

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

두 action이 planning 계산상 같은 구조라고 해서 실제 환경에서 같은 객체는 아니다.

따라서 dedup은 **계산 공유**이지 **실제 identity 병합**이 아니다.

---

# 12. Leaf evaluation

Planning tree를 무한히 펼칠 수 없으므로 어떤 depth에서 멈추고 leaf의 미래 가치를 추정한다.

```text
S0 → Ŝ1 → Ŝ2 → Ŝ3
                 ↓
               Critic
```

AASSR의 [Critic](Critic)이 이 역할을 맡는다.

---

# 13. Search와 Value function

Planner는 두 정보원을 조합할 수 있다.

```text
Explicit rollout value
+
Leaf value estimate
```

짧은 horizon에서 model prediction을 쓰고, 그 이후는 Critic이 요약된 long-term value를 제공한다.

이런 구조는 다양한 planning 알고리즘에서 흔한 아이디어다.

---

# 14. Chance와 Decision backup

Action을 고르는 node:

```math
V_{decision}=\max_aV(a)
```

환경 outcome node:

```math
V_{chance}=\sum_ip_iV_i
```

둘을 구분하지 않으면 environment randomness를 agent choice처럼 취급하는 오류가 생긴다.

더 자세히:

- [Chance and Decision Nodes](Chance-and-Decision-Nodes)

---

# 15. Expected planning vs Optimistic planning

환경 outcome에서 가장 좋은 결과만 고르면:

```math
V=\max_iV_i
```

가 된다.

하지만 agent가 outcome `i`를 선택할 수 없다면 이것은 지나치게 optimistic하다.

AASSR은 stochastic outcome을 probability-weighted expectation으로 backup한다.

---

# 16. Model Predictive Control과의 개념적 유사점

Model Predictive Control(MPC)은 현재 시점에서 미래 horizon을 최적화한 뒤 **첫 control action만 실행하고 다시 관측해서 재계획**하는 방식이다.

AASSR Imagination도 넓은 의미에서 비슷한 receding-horizon 구조를 가진다.

```text
현재에서 여러 미래 계산
→ 첫 action 하나 실행
→ 실제 response 관측
→ 다시 planning
```

하지만 AASSR은 learned stochastic relational model과 RL Policy/Critic을 사용하므로 전통적 continuous-control MPC와 동일한 알고리즘이라고 부르는 것은 부정확하다.

---

# 17. Receding horizon

한 번 긴 계획을 만든 뒤 그대로 끝까지 실행하는 것이 아니라 매 실제 step마다 다시 계획한다.

```text
Plan A0,A1,A2,A3
→ A0만 실행
→ 실제 S1 관측
→ 새로 plan
```

장점:

- model prediction과 실제 outcome 차이를 다음 decision에서 즉시 반영
- open-loop error 누적 감소

AASSR은 실제 action 하나 실행 후 다시 public response를 읽는 구조다.

---

# 18. Open-loop와 Closed-loop

## Open-loop

미리 만든 action sequence를 실제 outcome에 관계없이 계속 실행.

## Closed-loop

매 observation을 보고 다음 action을 다시 결정.

AASSR Imagination은 내부적으로 미래 sequence를 상상하지만 실제 실행은 closed-loop에 가깝다.

```text
상상: multi-step
실행: first step only
재관측 후 다시 결정
```

---

# 19. Planning with uncertainty

World model prediction이 unreliable한 branch까지 강하게 최적화하면 model error exploitation이 생길 수 있다.

그래서 planner는:

- model reliability
- critic support

를 확인할 수 있다.

AASSR current design은 uncertainty를 value bonus/penalty로 섞기보다 **eligibility gate**로 분리한다.

관련 페이지:

- [Calibration](Calibration)
- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 20. Intervention margin

Planner가 Policy보다 아주 미세하게 높은 root를 찾았다고 바로 switch하면 noise에 민감할 수 있다.

```math
V_{candidate}-V_{policy}\ge m
```

일 때만 override하도록 margin을 둘 수 있다.

AASSR current Imagination은 fixed intervention margin을 사용한다.

중요한 점은 margin이 reward가 아니라 **action switch decision threshold**라는 것이다.

---

# 21. Same-checkpoint planning evaluation

Planner의 순수 marginal effect를 보려면:

```text
one trained checkpoint
       ↓ freeze
OFF evaluation
ON evaluation
```

을 비교해야 한다.

Training 중 planner intervention이 trajectory를 바꾸면 두 모델은 더 이상 같은 학습 조건이 아니다.

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

AASSR의 과거 diagnostic은 1, 2는 성립해도 3이 자동으로 성립하지 않음을 보여줬다.

그래서 intervention count와 success를 따로 본다.

---

# 23. Planning failure modes

## Branch explosion

Action × outcome × depth로 계산량 증가.

## Model exploitation

Model이 틀리는 방향을 planner가 선택.

## OOD leaf value

Critic이 경험하지 않은 state에서 큰 값을 출력.

## Over-pruning

중요한 root가 너무 일찍 삭제.

## Optimistic chance backup

환경 outcome에 max를 사용.

## Inert planner

모든 root가 비슷하거나 gate가 너무 강해 행동을 한 번도 바꾸지 못함.

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