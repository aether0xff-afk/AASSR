# Imagination

Imagination은 AASSR의 **[counterfactual planner](Counterfactual-Planning-and-Search)** 다.

핵심 질문은 다음과 같다.

> **실제 행동을 하기 전에 [Prophecy](Prophecy)가 예측한 여러 미래를 몇 단계 전개해 보면, 현재 [Policy](Policy)가 고른 행동보다 더 나은 첫 행동을 선택할 수 있는가?**

AASSR에서 Imagination은 current main protocol 기준으로 imagined data를 사실처럼 학습시키는 장치가 아니라 **실행 전 planning 장치**다.

> [!IMPORTANT]
> 현재 핵심 구현: `src/aassr_v2/current_planner.py`  
> 신뢰도 gate: `src/aassr_v2/current_confidence_gate.py`  
> Critic support: `src/aassr_v2/current_critic_support.py`

---

# 0. 먼저 알아두면 좋은 개념

- [Model-Based RL & World Models](Model-Based-RL-and-World-Models) — learned model로 planning한다는 뜻
- [Counterfactual Planning & Search](Counterfactual-Planning-and-Search) — rollout, horizon, beam, pruning, root preservation
- [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes) — expectation과 max의 차이
- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — probability, reliability, value
- [Critic, Support & OOD](Critic-Support-and-OOD) — search가 OOD value error를 exploit하는 문제
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — structural root dedup
- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility) — same-checkpoint OFF/ON 비교

---

# 1. 기본 아이디어

현재 state에서 가능한 행동이 세 개 있다고 하자.

```text
A
B
C
```

[Model-free DQN](Q-Learning-DQN-and-TD) Policy는:

```text
Q(S,A)
Q(S,B)
Q(S,C)
```

를 비교해 현재 action을 고를 수 있다.

Imagination은 한 단계 더 나아간다.

```text
A를 하면?
  → 가능한 미래들
      → 그 미래에서 다음 행동은?
          → 그 뒤에는?

B를 하면?
  → ...

C를 하면?
  → ...
```

그리고 **실제로 실행하는 것은 첫 행동 하나뿐**이다.

이후 real response를 관측하고 다시 planning한다. 이런 구조는 [receding-horizon / MPC](Counterfactual-Planning-and-Search)와 개념적으로 연결된다.

---

# 2. Imagination은 Learning인가 Planning인가?

두 개념을 분리해야 한다.

```text
Learning
→ real experience로 Policy/Prophecy/Critic parameter를 변경

Planning
→ 현재 학습된 model을 사용해 행동 전에 계산
```

Current main AASSR protocol에서 Imagination은 두 번째에 해당한다.

```text
Imagined branch
→ current decision 계산
→ persistent real replay truth로 자동 승격하지 않음
```

이 경계는 [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation)에서 중요하게 다룬다.

---

# 3. 왜 단순 `n × k` 나무 설명만으로 부족한가?

초기 직관:

```text
n개의 행동 후보
×
k depth
```

는 틀리지는 않지만 current planner의 핵심 의미를 놓친다.

실제로는 다음 문제가 더 중요하다.

1. [환경의 stochastic outcome과 agent의 decision을 구분](Chance-and-Decision-Nodes)해야 한다.
2. 각 stochastic outcome에는 probability mass가 있다.
3. [Prophecy reliability](Calibration)가 낮으면 branch를 믿으면 안 된다.
4. [Critic OOD](Critic-Support-and-OOD)라면 큰 value가 나와도 override하면 안 된다.
5. concrete aliases가 많아도 같은 [structural root](Relational-Representation-and-Generalization)를 반복 계산하면 안 된다.
6. 깊은 rollout이 실패해도 실제 legal root action을 잃으면 안 된다.
7. planning depth가 깊어질수록 [compounding model error](Model-Based-RL-and-World-Models)가 커진다.

즉 current Imagination은 **확률적 planning semantics + reliability constraints + structural computation**의 조합이다.

---

# 4. Chance node와 Decision node

이 구분이 핵심이다.

## Chance node

이미 action을 선택한 뒤 environment outcome이 갈린다.

```text
행동 A
  |-- 0.7 → 정상 진행
  |-- 0.2 → 403
  `-- 0.1 → 429
```

Agent가 이 결과 중 하나를 고를 수 없으므로:

```math
V_{chance}=\sum_i p_iV_i
```

를 사용한다.

## Decision node

Predicted state에서 다음 action은 agent가 선택할 수 있다.

```text
predicted S'
   |-- action B
   |-- action C
   `-- action D
```

최적 continuation을 가정하면:

```math
V_{decision}=\max_aV(S',a)
```

이다.

더 자세히: [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)

---

# 5. 왜 environment outcome에 `max`를 쓰면 안 되는가?

다음 action A를 보자.

```text
10% → success +1
90% → failure -1
```

좋은 outcome만 고르면:

```text
V(A)=+1
```

처럼 보인다.

하지만 expected return은:

```math
0.1(1)+0.9(-1)=-0.8
```

이다.

Environment randomness에 `max`를 쓰면 agent가 실제로 통제할 수 없는 jackpot outcome을 선택할 수 있는 것처럼 계산한다.

이것이 [optimistic stochastic backup](Chance-and-Decision-Nodes) 오류다.

---

# 6. Planning tree

개념적 구조:

```text
real state S0
  |
  +-- root action A
  |      |
  |      +-- p11 → S11
  |      |          |
  |      |          +-- decision B
  |      |          +-- decision C
  |      |
  |      `-- p12 → S12
  |
  +-- root action D
  |      `-- ...
  |
  `-- root action E
         `-- ...
```

Tree는:

```text
Decision → Chance → Decision → Chance → ...
```

가 번갈아 나타난다.

---

# 7. Prophecy는 어떤 역할인가?

[Prophecy](Prophecy)는 각 action 뒤의 stochastic public future distribution을 만든다.

```text
(S,A)
 ↓
Prophecy
 ↓
[(S1',p1), (S2',p2), ...]
```

Imagination은 이 predicted states를 tree node로 사용한다.

Prophecy가 틀리면 planner가 아무리 수학적으로 올바른 backup을 해도 잘못된 미래를 최적화할 수 있다.

이를 [model exploitation](Model-Based-RL-and-World-Models) 문제라고 볼 수 있다.

---

# 8. Critic은 왜 필요한가?

Planner depth를 무한히 늘릴 수 없다.

어느 depth에서는 rollout을 멈추고 그 이후의 장기 sparse return을 추정해야 한다.

```text
S0 → Ŝ1 → Ŝ2 → Ŝ3
                   |
                   v
                 Critic
```

Current [Critic](Critic)은 relational [GRU](GRU-and-Sequence-Models) 기반 discounted sparse-return estimator다.

즉:

```text
Prophecy
→ future state

Critic
→ future value
```

다.

---

# 9. Planning horizon / depth

Depth를 늘리면 더 먼 consequence를 볼 수 있다.

하지만 비용도 증가한다.

```text
Benefit
→ longer-horizon reasoning

Cost
→ more model calls
→ more branching
→ more OOD states
→ compounding model error
```

그래서:

```text
deeper = always better
```

가 아니다.

[Counterfactual Planning & Search](Counterfactual-Planning-and-Search)에서 horizon trade-off를 더 자세히 설명한다.

---

# 10. Branching factor

한 decision node에 action `b`개가 있고 각 action에 outcome `m`개가 있다면 naive tree는 매우 빠르게 커진다.

대략:

```text
(b × m)^depth
```

형태의 combinatorial growth를 생각할 수 있다.

실제 planner는:

- beam width
- outcome sample count
- pruning
- structural dedup
- batching

등으로 계산량을 제한한다.

---

# 11. Root preservation

깊은 branch가 unreliable하거나 prune되어도 **실제로 가능한 root action 자체가 사라지면 안 된다.**

예:

```text
root A
→ depth 1 prediction usable
→ depth 2에서 reliability 부족
```

잘못된 구현:

```text
depth 2 failure
→ A root 전체 삭제
```

보수적 구현:

```text
depth 2 expansion 중단
→ 이미 계산한 shallower A value 유지
```

이것이 [root preservation](Counterfactual-Planning-and-Search)이다.

---

# 12. Beam search와 pruning

모든 branch를 유지하기 어렵기 때문에 일부만 유지할 수 있다.

```text
Depth 1 → 100 candidates
          ↓ top/valid 24
Depth 2 → expand
          ↓ top/valid 24
```

하지만 pruning 기준이 잘못되면 유용한 branch가 초기에 사라질 수 있다.

특히:

```text
reliability가 낮아서 expansion을 멈춤
```

과:

```text
value가 낮아서 선택하지 않음
```

은 의미가 다르다.

---

# 13. Structural root deduplication

실제 action surface에는 concrete ID만 다른 action이 매우 많을 수 있다.

```text
GET route-12
GET route-31
GET route-44
...
```

하지만 [relational representation](Relational-Representation-and-Generalization)에서는 같은 action structure일 수 있다.

```text
172 concrete roots
       ↓ structural grouping
17 relational roots
```

같은 structural root의 Prophecy/Critic 계산을 한 번만 수행하고 concrete aliases에 결과를 fan-out할 수 있다.

```text
compute identity   = relational
execution identity = concrete
```

이다.

---

# 14. 왜 concrete identity를 끝까지 버리면 안 되는가?

Planner가:

```text
catalog-like route request
```

라는 structural decision을 고르더라도 실제 environment는:

```text
GET /route_31
```

같은 concrete action을 요구한다.

따라서 structural dedup은 **계산 공유**이지 action identity 병합이 아니다.

이 구분은 [Relational Representation & Generalization](Relational-Representation-and-Generalization)에서 핵심적으로 다룬다.

---

# 15. Prophecy reliability gate

World model은 완벽하지 않다.

먼저:

```text
이 prediction을 믿을 real holdout evidence가 있는가?
```

를 본다.

[Calibration](Calibration) reliability가 부족하면 aggressive override를 허용하지 않는다.

중요:

```text
reliability
!=
value
```

이다.

Prediction이 매우 reliable한 실패 branch일 수도 있다.

---

# 16. Global coverage gate

현재 action surface 전체에서 Prophecy가 거의 모르는 상태라면 planner를 통째로 비활성화할 수 있다.

```text
model coverage < threshold
→ Imagination ineligible
→ Policy fallback
```

이는 "새로운 state를 절대 탐색하지 않는다"는 뜻이 아니라:

> **모르는 world model을 이용해 실제 Policy를 override하지 않는다.**

는 뜻이다.

---

# 17. Per-root reliability gate

Global coverage가 충분해도 특정 root는 low reliability일 수 있다.

```text
root A reliability 0.8
root B reliability 0.1
root C reliability 0.7
```

B가 Critic value는 높더라도 prediction 자체가 unreliable하면 final override candidate에서 제외할 수 있다.

---

# 18. Policy root도 reliable해야 하는 이유

Alternative와 Policy를 비교하려면 둘 다 같은 수준의 model evidence가 필요하다.

```text
V_alt - V_policy
```

에서 `V_policy`의 underlying prediction이 unreliable하면 advantage가 의미 없을 수 있다.

그래서 current gate는 Policy branch도 reliability를 요구한다.

---

# 19. Local Critic support gate

Prophecy prediction이 reliable해도 [Critic](Critic)이 predicted state/action region을 본 적 없을 수 있다.

```text
Prophecy reliable
Critic OOD
→ 잘 예측한 미래를 잘못 평가
```

그래서 [local Critic support](Critic-Support-and-OOD)를 확인한다.

```text
Policy root supported?
Candidate root supported?
     |
     +-- yes/yes → value comparison
     `-- otherwise → Policy fallback
```

---

# 20. 왜 global `critic_ready`만으로 부족한가?

```text
critic_ready = True
```

는 Critic 전체가 어느 정도 학습됐다는 뜻이다.

하지만 current query가 higher-level unseen region이면:

```text
local support = low
```

일 수 있다.

Search는 이런 OOD artifact를 적극적으로 고를 수 있기 때문에 local evidence가 필요하다.

자세히: [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 21. Intervention advantage

Planner가 찾은 best reliable/supported root와 Policy root를 비교한다.

```math
\Delta V=V_{candidate}-V_{policy}
```

Candidate가 다르더라도 `ΔV`가 너무 작으면 noise일 수 있다.

그래서 fixed intervention margin `m`을 둔다.

```math
\Delta V\ge m
```

일 때만 실제 switch를 허용한다.

Margin은 **reward shaping이 아니라 decision threshold**다.

---

# 22. 왜 margin이 필요한가?

Value estimator에는 noise가 있다.

```text
Policy value    = 0.501
Candidate value = 0.503
```

같은 작은 차이로 매번 action을 바꾸면 planner가 unstable할 수 있다.

Margin은 작은 value noise에 대한 robustness 역할을 한다.

하지만 너무 크면 useful candidate도 막으므로 [hyperparameter ablation](Ablation-Benchmarking-and-Reproducibility)이 필요하다.

---

# 23. Intervention accounting

다음 단계는 서로 다르다.

```text
1. plan generated
2. alternative preferred
3. switch candidate
4. reliability/support gate passed
5. margin passed
6. actual executed action changed
```

진짜 intervention은 **6번**이다.

Candidate가 잠깐 생겼다가 gate에서 취소된 것을 intervention으로 세면 planner 효과를 과대평가한다.

따라서 current diagnostic은:

- plan count
- switch candidate count
- suppressed count
- final intervention count
- changed-action count

을 구분한다.

---

# 24. Same-checkpoint OFF/ON 비교

현재 Imagination의 순수 marginal effect를 측정하는 핵심 protocol:

```text
one training run
       ↓
frozen AASSR checkpoint
    /          \
OFF eval      ON eval
```

Training 중부터 Imagination intervention을 켜면 training trajectory가 달라진다.

```text
OFF-trained model
vs
ON-trained model
```

은 planner 효과뿐 아니라 data-distribution 효과까지 섞인다.

그래서 current main comparison은 [same-checkpoint evaluation](Ablation-Benchmarking-and-Reproducibility)을 사용한다.

---

# 25. 왜 imagined experience로 Policy를 바로 학습시키지 않는가?

Model-generated transition을 real truth처럼 학습하면:

```text
world-model error
→ imagined experience
→ Policy/Critic update
→ error self-amplification
```

이 가능하다.

다른 [model-based RL](Model-Based-RL-and-World-Models) 알고리즘에서는 imagined learning을 정당하게 사용할 수 있지만, AASSR current main experiment는 **planning effect를 깨끗하게 분리**하기 위해 persistent Policy update를 막는다.

---

# 26. Receding-horizon execution

Planner가 depth 4 future를 계산했다고 해서 네 행동을 그대로 실행하는 것이 아니다.

```text
Plan:
A0 → A1 → A2 → A3

실제:
A0만 실행
→ real response 관측
→ 다시 planning
```

이렇게 하면 predicted rollout과 real outcome의 차이를 다음 step에서 바로 반영할 수 있다.

[Model Predictive Control](Counterfactual-Planning-and-Search)과 개념적으로 닮은 부분이다.

---

# 27. 과거 2k diagnostic이 보여준 것

Repaired run에서는 Imagination이 더 이상 inert하지 않고 실제로 plan과 intervention을 만들었다.

하지만 핵심 교훈은:

```text
행동을 바꿀 수 있다
!=
더 좋은 행동을 고른다
```

였다.

여러 intervention이 `403/404/429` 같은 bad public outcome으로 이어졌고 direct success-producing intervention은 확인되지 않았다.

즉 병목이:

```text
planner inactivity
```

에서:

```text
planner decision quality
```

로 이동했다.

---

# 28. 이 실패에서 배운 것

수리 방향은 다음처럼 분해됐다.

```text
문제 1: decision-critical public status가 representation에서 약함
→ Relational State v3 + status supervision

문제 2: model reliability와 outcome probability 혼동
→ semantic calibration

문제 3: Critic이 higher-level OOD에서 value를 extrapolate
→ local Critic support

문제 4: 같은 relational root를 지나치게 많이 계산
→ structural root dedup

문제 5: accounting이 candidate와 execution을 혼동
→ final executed intervention만 count
```

중요한 연구 태도:

> **코드 repair가 들어갔다는 사실과 최종 performance improvement는 같은 주장이 아니다.**

후자는 새 benchmark로 다시 검증해야 한다.

---

# 29. Failure mode: Model error exploitation

Planner는 많은 candidate 중 가장 높은 value를 찾는다.

그 과정에서 world model의 작은 오류를 찾아낼 수 있다.

```text
실제로는 mediocre한 action
but model predicts rare amazing future
→ planner selects it
```

대응:

- stochastic probability mass
- [Calibration](Calibration)
- status-aware model
- controlled horizon

---

# 30. Failure mode: OOD Critic exploitation

Prophecy future가 plausible해도 Critic이 해당 region에서 근거 없는 high value를 낼 수 있다.

대응:

- [local Critic support](Critic-Support-and-OOD)
- fail-closed fallback

---

# 31. Failure mode: Over-pruning

Reliability/beam/pruning이 너무 강하면 좋은 root가 사라질 수 있다.

대응:

- root preservation
- separate root eligibility vs deep expansion
- pass/suppression metric 분석

---

# 32. Failure mode: Under-pruning

모든 branch를 유지하면:

- compute 폭발
- tiny-probability branches 증가
- OOD state 증가

가 가능하다.

대응:

- beam width
- structural dedup
- batching
- probability-aware branch management

---

# 33. Failure mode: Planner inertia

모든 gate가 너무 보수적이거나 Critic value가 모두 비슷하면 intervention이 0이 된다.

가능한 원인:

- Prophecy coverage 부족
- Calibration sample 부족
- Critic sparse target starvation
- local support 부족
- intervention margin 과도함

따라서:

```text
intervention = 0
```

만 보고 planner 코드가 죽었다고 결론내리면 안 된다.

Gate reason별 diagnostic이 필요하다.

---

# 34. Failure mode: Overactive planner

반대로 intervention이 많다고 좋은 것도 아니다.

```text
interventions ↑
errors ↑
success unchanged
```

일 수 있다.

최종 목표는 **intervention count 최대화**가 아니라 **Policy보다 좋은 action을 근거 있게 선택**하는 것이다.

---

# 35. Failure mode: Probability/reliability/value mixing

하나의 score에:

```text
outcome probability
confidence
Critic value
support
```

를 모두 더하거나 곱하면 해석이 어려워진다.

Current design은 역할을 분리한다.

```text
Outcome probability
→ chance expectation

Prediction reliability
→ Prophecy gate

Critic value
→ task return ranking

Local support
→ Critic evidence gate

Advantage margin
→ final switch threshold
```

이 구분은 [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability)에서 전체적으로 설명한다.

---

# 36. Imagination과 Skill

[Skill](Skills)은 여러 primitive action을 relational macro처럼 재사용할 수 있다.

Planner가 Skill을 action 후보로 다룰 때 그 내부 primitive sequence에도 stochastic future가 존재할 수 있다.

Current Skill Prophecy는 여러 stochastic outcome을 작은 beam으로 유지한다.

관련 배경: [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

---

# 37. Imagination과 Curriculum

쉬운 level에서는 Prophecy/Critic support가 충분하지만 higher level에서는 OOD가 될 수 있다.

```text
L0/L1 training frontier
     ↓
L2/L3 imagined state
→ reliability/support 부족
```

따라서 curriculum progression과 Imagination quality는 강하게 연결된다.

관련 페이지: [Curriculum Learning](Curriculum-Learning)

---

# 38. Imagination compute 최적화

Planning은 expensive하다.

Current-generation에는 다음과 같은 engineering optimization이 중요하다.

- depth-batched Prophecy
- batched Critic scoring
- structural root dedup
- cache reuse
- GPU-friendly tensor path

이러한 최적화는 **planning semantics를 바꾸지 않고 같은 계산을 더 효율적으로 실행**하는 것을 목표로 한다.

관련 기초: [Neural Networks & Optimization](Neural-Networks-and-Optimization)

---

# 39. Imagination을 평가할 때 볼 metric

## Planning activity

- plan count
- nodes expanded
- maximum depth reached
- structural roots / concrete roots

## Gate behavior

- global coverage failures
- low-reliability roots
- policy-root reliability failures
- local support failures
- insufficient-advantage suppressions

## Intervention behavior

- switch candidates
- final interventions
- changed actions
- direct success interventions
- bad-status interventions

## Final task metric

- same-checkpoint OFF success
- same-checkpoint ON success
- paired scenario improvements/regressions

[Proxy metric과 final metric](Ablation-Benchmarking-and-Reproducibility)을 분리한다.

---

# 40. 연구 가설

```text
H1. Prophecy가 usable stochastic future를 제공하는가?
H2. chance/decision backup이 probabilistically 올바르게 작동하는가?
H3. structural dedup이 semantics를 보존하면서 compute를 줄이는가?
H4. reliability gate가 model-error override를 줄이는가?
H5. local Critic support가 OOD value override를 줄이는가?
H6. intervention margin이 noise-driven switch를 줄이는가?
H7. 위 gate들이 너무 보수적이어서 useful intervention을 모두 막지는 않는가?
H8. 최종적으로 같은 frozen checkpoint에서 Full이 no-Imagination보다 실제 success를 높이는가?
```

H1~H7은 mechanism/diagnostic이고 H8이 최종 planner-benefit claim이다.

---

# 41. 관련 코드 읽는 순서

```text
current_entrypoint.py
        ↓
current_planner.py
        ↓
Prophecy / current status model
        ↓
current_semantic_calibration.py
        ↓
current_return_critic.py
        ↓
current_critic_support.py
        ↓
current_confidence_gate.py
        ↓
structural root dedup / decision optimization
```

---

# 42. 한 문장 요약

> **Imagination은 stochastic world model의 미래를 chance expectation과 decision max로 전개한 뒤, prediction reliability·Critic support·value advantage가 모두 충분할 때만 실제 Policy 행동을 바꾸는 fail-closed counterfactual planner다.**

---

다음으로 읽기:

- **[Prophecy](Prophecy)**
- **[Calibration](Calibration)**
- **[Critic](Critic)**
- **[Counterfactual Planning & Search](Counterfactual-Planning-and-Search)**
- **[Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
