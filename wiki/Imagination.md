# Imagination — 가상 미래 탐색

[Imagination(가상 미래 탐색)](Imagination)은 AASSR의 **[counterfactual planner](Counterfactual-Planning-and-Search)** 다.

핵심 질문은 다음과 같다.

> **실제 행동을 하기 전에 [Prophecy](Prophecy)가 예측한 여러 미래를 몇 단계 전개해 보면, 현재 [Policy](Policy)가 고른 행동보다 더 나은 첫 행동을 선택할 수 있는가?**

AASSR에서 [Imagination](Imagination)은 [현재(current)](Current-Status) main [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility) 기준으로 [모델이 상상한(imagined)](Research-Jargon-Guide) [데이터(data)](Terminology-Guide)를 사실처럼 학습시키는 장치가 아니라 **실행 전 [계획(planning)](Counterfactual-Planning-and-Search) 장치**다.

> [!**중요**]
> 현재 핵심 구현: `src/aassr_v2/current_planner.py`  
> 신뢰도 [판정 관문(gate)](Terminology-Guide): `src/aassr_v2/current_confidence_gate.py`  
> [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD): `src/aassr_v2/current_critic_support.py`

---

# 0. 먼저 알아두면 좋은 개념

- [Model-Based RL & World Models](Model-Based-RL-and-World-Models) — [학습된(learned)](Neural-Networks-and-Optimization) [학습 모델(model)](Terminology-Guide)로 계획한다는 뜻
- [Counterfactual Planning & Search](Counterfactual-Planning-and-Search) — [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search), [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search), [유망 후보만 남기는 빔 탐색(beam)](Counterfactual-Planning-and-Search), [유망하지 않은 탐색 가지를 제거하는 가지치기(pruning)](Counterfactual-Planning-and-Search), [탐색의 첫 행동(root)](Imagination) [의미 보존(preservation)](Ablation-Benchmarking-and-Reproducibility)
- [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes) — [확률 기댓값(expectation)](Chance-and-Decision-Nodes)과 max의 차이
- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — [확률(probability)](Stochasticity-Uncertainty-and-Probability), [신뢰도(reliability)](Calibration), [가치(value)](Value-Functions-and-Bellman-Equation)
- [Critic, Support & OOD](Critic-Support-and-OOD) — [탐색(search)](Counterfactual-Planning-and-Search)가 [학습 분포 밖(OOD)](Critic-Support-and-OOD) 가치 [오차(error)](Loss-Functions-and-Class-Imbalance)를 exploit하는 문제
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — [구조 기반(structural)](Relational-Representation-and-Generalization) 탐색의 첫 행동 [중복 계산 제거(dedup)](Reproduction)
- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility) — [같은 체크포인트(same-checkpoint)](Experiments) OFF/ON 비교

---

# 1. 기본 아이디어

현재 [상태(state)](State-Representation)에서 가능한 행동이 세 개 있다고 하자.

```text
A
B
C
```

[Model-free DQN](Q-Learning-DQN-and-TD) [Policy(정책 모델)](Policy)는:

```text
Q(S,A)
Q(S,B)
Q(S,C)
```

를 비교해 현재 [행동(action)](Reinforcement-Learning)을 고를 수 있다.

[Imagination](Imagination)은 한 단계 더 나아간다.

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

이후 [실제 환경에서 관측된(real)](Research-Jargon-Guide) [응답(response)](State-Representation)를 관측하고 다시 계획한다. 이런 구조는 [receding-horizon / MPC](Counterfactual-Planning-and-Search)와 개념적으로 연결된다.

---

# 2. Imagination은 Learning인가 Planning인가?

두 개념을 분리해야 한다.

```text
Learning
→ real experience로 Policy/Prophecy/Critic parameter를 변경

Planning
→ 현재 학습된 model을 사용해 행동 전에 계산
```

[현재(Current)](Current-Status) main AASSR 실험 규칙에서 [Imagination](Imagination)은 두 번째에 해당한다.

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

는 틀리지는 않지만 현재 [계획기(planner)](Counterfactual-Planning-and-Search)의 핵심 의미를 놓친다.

실제로는 다음 문제가 더 중요하다.

1. [환경의 stochastic outcome과 agent의 decision을 구분](Chance-and-Decision-Nodes)해야 한다.
2. 각 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)에는 [확률 질량(probability mass)](Stochasticity-Uncertainty-and-Probability)가 있다.
3. [Prophecy reliability](Calibration)가 낮으면 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)를 믿으면 안 된다.
4. [Critic OOD](Critic-Support-and-OOD)라면 큰 가치가 나와도 [기본 행동 덮어쓰기(override)](Imagination)하면 안 된다.
5. [실제 개체를 구분하는(concrete)](State-Representation) aliases가 많아도 같은 [structural root](Relational-Representation-and-Generalization)를 반복 계산하면 안 된다.
6. 깊은 가상 미래 전개이 실패해도 실제 [현재 허용된(legal)](Terminology-Guide) 탐색의 첫 행동 행동을 잃으면 안 된다.
7. 계획 [탐색 깊이(depth)](Counterfactual-Planning-and-Search)가 깊어질수록 [compounding model error](Model-Based-RL-and-World-Models)가 커진다.

즉 현재 [Imagination](Imagination)은 **확률적 계획 [의미 규칙(semantics)](State-Representation) + 신뢰도 constraints + 구조 기반 computation**의 조합이다.

---

# 4. Chance node와 Decision node

이 구분이 핵심이다.

## Chance node

이미 행동을 선택한 뒤 [환경(environment)](Reinforcement-Learning) 환경 결과이 갈린다.

```text
행동 A
  |-- 0.7 → 정상 진행
  |-- 0.2 → 403
  `-- 0.1 → 429
```

[에이전트(Agent)](Reinforcement-Learning)가 이 결과 중 하나를 고를 수 없으므로:

```math
V_{chance}=\sum_i p_iV_i
```

를 사용한다.

## Decision node

Predicted 상태에서 다음 행동은 [에이전트(agent)](Reinforcement-Learning)가 선택할 수 있다.

```text
predicted S'
   |-- action B
   |-- action C
   `-- action D
```

최적 [계속 진행되는 상태(continuation)](Chance-and-Decision-Nodes)을 가정하면:

```math
V_{decision}=\max_aV(S',a)
```

이다.

더 자세히: [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)

---

# 5. 왜 environment outcome에 `max`를 쓰면 안 되는가?

다음 행동 A를 보자.

```text
10% → success +1
90% → failure -1
```

좋은 환경 결과만 고르면:

```text
V(A)=+1
```

처럼 보인다.

하지만 [확률을 고려해 기대되는(expected)](Chance-and-Decision-Nodes) [누적 보상(return)](Value-Functions-and-Bellman-Equation)은:

```math
0.1(1)+0.9(-1)=-0.8
```

이다.

[환경(Environment)](Reinforcement-Learning) [무작위성(randomness)](Stochasticity-Uncertainty-and-Probability)에 `max`를 쓰면 에이전트가 실제로 통제할 수 없는 jackpot 환경 결과을 선택할 수 있는 것처럼 계산한다.

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

[Prophecy](Prophecy)는 각 행동 뒤의 확률적 [공개된(public)](State-Representation) [미래(future)](Counterfactual-Planning-and-Search) [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)을 만든다.

```text
(S,A)
 ↓
Prophecy
 ↓
[(S1',p1), (S2',p2), ...]
```

[Imagination](Imagination)은 이 [예측된(predicted)](Terminology-Guide) states를 [탐색 트리(tree)](Counterfactual-Planning-and-Search) [탐색 트리의 한 지점(node)](Chance-and-Decision-Nodes)로 사용한다.

[Prophecy(미래 예측 모델)](Prophecy)가 틀리면 계획기가 아무리 수학적으로 올바른 [미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)을 해도 잘못된 미래를 최적화할 수 있다.

이를 [model exploitation](Model-Based-RL-and-World-Models) 문제라고 볼 수 있다.

---

# 8. Critic은 왜 필요한가?

[계획기(Planner)](Counterfactual-Planning-and-Search) 탐색 깊이를 무한히 늘릴 수 없다.

어느 탐색 깊이에서는 가상 미래 전개을 멈추고 그 이후의 장기 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) 누적 보상을 추정해야 한다.

```text
S0 → Ŝ1 → Ŝ2 → Ŝ3
                   |
                   v
                 Critic
```

현재 [Critic](Critic)은 [관계 기반(relational)](Relational-Representation-and-Generalization) [GRU](GRU-and-Sequence-Models) 기반 [미래 보상을 시간에 따라 할인한(discounted)](Value-Functions-and-Bellman-Equation) sparse-누적 보상 [값을 추정하는 모델(estimator)](Terminology-Guide)다.

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

[Counterfactual Planning & Search](Counterfactual-Planning-and-Search)에서 미래 탐색 범위 [한쪽을 얻으면 다른 쪽을 잃는 상충 관계(trade-off)](Terminology-Guide)를 더 자세히 설명한다.

---

# 10. Branching factor

한 [행동 선택 노드(decision node)](Chance-and-Decision-Nodes)에 행동 `b`개가 있고 각 행동에 환경 결과 `m`개가 있다면 naive 탐색 트리는 매우 빠르게 커진다.

대략:

```text
(b × m)^depth
```

형태의 combinatorial growth를 생각할 수 있다.

실제 계획기는:

- 빔 탐색 width
- 환경 결과 [표본(sample)](Ablation-Benchmarking-and-Reproducibility) [횟수(count)](Terminology-Guide)
- 가지치기
- 구조 기반 중복 제거
- [묶음 처리(batching)](Reproduction)

등으로 계산량을 제한한다.

---

# 11. Root preservation

깊은 결과 경로가 [신뢰하기 어려운(unreliable)](Calibration)하거나 prune되어도 **실제로 가능한 탐색의 첫 행동 행동 자체가 사라지면 안 된다.**

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

모든 결과 경로를 유지하기 어렵기 때문에 일부만 유지할 수 있다.

```text
Depth 1 → 100 candidates
          ↓ top/valid 24
Depth 2 → expand
          ↓ top/valid 24
```

하지만 가지치기 기준이 잘못되면 유용한 결과 경로가 초기에 사라질 수 있다.

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

실제 행동 [현재 선택 가능한 영역(surface)](Terminology-Guide)에는 실제 개체를 구분하는 ID만 다른 행동이 매우 많을 수 있다.

```text
GET route-12
GET route-31
GET route-44
...
```

하지만 [relational representation](Relational-Representation-and-Generalization)에서는 같은 행동 [구조(structure)](Research-Architecture)일 수 있다.

```text
172 concrete roots
       ↓ structural grouping
17 relational roots
```

같은 구조 기반 탐색의 첫 행동의 [Prophecy](Prophecy)/[Critic(미래 가치 평가기)](Critic) 계산을 한 번만 수행하고 실제 개체를 구분하는 aliases에 결과를 fan-out할 수 있다.

```text
compute identity   = relational
execution identity = concrete
```

이다.

---

# 14. 왜 concrete identity를 끝까지 버리면 안 되는가?

계획기가:

```text
catalog-like route request
```

라는 구조 기반 [의사결정(decision)](Chance-and-Decision-Nodes)을 고르더라도 실제 환경는:

```text
GET /route_31
```

같은 [실제 실행 행동(concrete action)](State-Representation)을 요구한다.

따라서 구조 기반 중복 제거은 **계산 공유**이지 행동 [식별 방식(identity)](State-Representation) 병합이 아니다.

이 구분은 [Relational Representation & Generalization](Relational-Representation-and-Generalization)에서 핵심적으로 다룬다.

---

# 15. Prophecy reliability gate

[세계(World)](Model-Based-RL-and-World-Models) 학습 모델은 완벽하지 않다.

먼저:

```text
이 prediction을 믿을 real holdout evidence가 있는가?
```

를 본다.

[Calibration](Calibration) 신뢰도가 부족하면 aggressive 기본 행동 덮어쓰기를 허용하지 않는다.

중요:

```text
reliability
!=
value
```

이다.

Prediction이 매우 reliable한 실패 결과 경로일 수도 있다.

---

# 16. Global coverage gate

현재 행동 선택 가능 영역 전체에서 [Prophecy](Prophecy)가 거의 모르는 상태라면 계획기를 통째로 비활성화할 수 있다.

```text
model coverage < threshold
→ Imagination ineligible
→ Policy fallback
```

이는 "새로운 상태를 절대 탐색하지 않는다"는 뜻이 아니라:

> **모르는 [세계 모델(world model)](Model-Based-RL-and-World-Models)을 이용해 실제 [Policy](Policy)를 기본 행동 덮어쓰기하지 않는다.**

는 뜻이다.

---

# 17. Per-root reliability gate

Global [데이터가 어느 영역까지 포함하는지(coverage)](Critic-Support-and-OOD)가 충분해도 특정 탐색의 첫 행동는 low 신뢰도일 수 있다.

```text
root A reliability 0.8
root B reliability 0.1
root C reliability 0.7
```

B가 [Critic](Critic) 가치는 높더라도 [예측(prediction)](Terminology-Guide) 자체가 신뢰하기 어려운하면 [최종(final)](Ablation-Benchmarking-and-Reproducibility) 기본 행동 덮어쓰기 [선택 후보(candidate)](Terminology-Guide)에서 제외할 수 있다.

---

# 18. Policy root도 reliable해야 하는 이유

Alternative와 [Policy](Policy)를 비교하려면 둘 다 같은 수준의 학습 모델 [증거(evidence)](Evidence-Matrix)가 필요하다.

```text
V_alt - V_policy
```

에서 `V_policy`의 underlying 예측이 신뢰하기 어려운하면 [다른 선택보다 나은 정도(advantage)](Value-Functions-and-Bellman-Equation)가 의미 없을 수 있다.

그래서 현재 판정 관문는 [Policy](Policy) 결과 경로도 신뢰도를 요구한다.

---

# 19. Local Critic support gate

[Prophecy](Prophecy) 예측이 reliable해도 [Critic](Critic)이 예측된 상태/행동 [상태 공간의 영역(region)](Critic-Support-and-OOD)을 본 적 없을 수 있다.

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

는 [Critic](Critic) 전체가 어느 정도 학습됐다는 뜻이다.

하지만 현재 [조회 또는 질의(query)](Terminology-Guide)가 [여러 기본 행동을 묶는 상위 수준(higher-level)](Hierarchical-RL-and-Skills) [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 영역이면:

```text
local support = low
```

일 수 있다.

Search는 이런 [OOD](Critic-Support-and-OOD) artifact를 적극적으로 고를 수 있기 때문에 [현재 주변에 한정된 국소적(local)](Critic-Support-and-OOD) 증거가 필요하다.

자세히: [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 21. Intervention advantage

계획기가 찾은 best reliable/supported 탐색의 첫 행동와 [Policy](Policy) 탐색의 첫 행동를 비교한다.

```math
\Delta V=V_{candidate}-V_{policy}
```

Candidate가 다르더라도 `ΔV`가 너무 작으면 [잡음(noise)](Stochasticity-Uncertainty-and-Probability)일 수 있다.

그래서 [고정된(fixed)](Ablation-Benchmarking-and-Reproducibility) [실제 행동 개입(intervention)](Imagination) [최소 차이 기준(margin)](Imagination) `m`을 둔다.

```math
\Delta V\ge m
```

일 때만 실제 [행동 전환(switch)](Imagination)를 허용한다.

Margin은 **[보상(reward)](Sparse-Reward-and-Credit-Assignment) [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)이 아니라 의사결정 [판정 기준값(threshold)](Terminology-Guide)**다.

---

# 22. 왜 margin이 필요한가?

Value 값을 추정하는 모델에는 잡음가 있다.

```text
Policy value    = 0.501
Candidate value = 0.503
```

같은 작은 차이로 매번 행동을 바꾸면 계획기가 unstable할 수 있다.

Margin은 작은 가치 잡음에 대한 robustness 역할을 한다.

하지만 너무 크면 useful 선택 후보도 막으므로 [hyperparameter ablation](Ablation-Benchmarking-and-Reproducibility)이 필요하다.

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

진짜 실제 행동 개입은 **6번**이다.

Candidate가 잠깐 생겼다가 판정 관문에서 취소된 것을 실제 행동 개입으로 세면 계획기 효과를 과대평가한다.

따라서 현재 [진단 실험(diagnostic)](Evidence-Matrix)은:

- [계획(plan)](Counterfactual-Planning-and-Search) 횟수
- 행동 전환 선택 후보 횟수
- suppressed 횟수
- 최종 실제 행동 개입 횟수
- changed-행동 횟수

을 구분한다.

---

# 24. Same-checkpoint OFF/ON 비교

현재 [Imagination](Imagination)의 순수 [다른 조건이 같을 때의 추가 기여(marginal)](Ablation-Benchmarking-and-Reproducibility) [효과(effect)](Ablation-Benchmarking-and-Reproducibility)를 측정하는 핵심 실험 규칙:

```text
one training run
       ↓
frozen AASSR checkpoint
    /          \
OFF eval      ON eval
```

[학습(Training)](Reinforcement-Learning) 중부터 [Imagination](Imagination) 실제 행동 개입을 켜면 [학습(training)](Terminology-Guide) [경험 경로(trajectory)](Reinforcement-Learning)가 달라진다.

```text
OFF-trained model
vs
ON-trained model
```

은 계획기 효과뿐 아니라 data-distribution 효과까지 섞인다.

그래서 현재 main [비교(comparison)](Ablation-Benchmarking-and-Reproducibility)은 [same-checkpoint evaluation](Ablation-Benchmarking-and-Reproducibility)을 사용한다.

---

# 25. 왜 imagined experience로 Policy를 바로 학습시키지 않는가?

Model-generated [상태 전이(transition)](MDP-and-POMDP)을 실제 [환경 내부의 실제값(truth)](Causality-Leakage-and-Evaluation)처럼 학습하면:

```text
world-model error
→ imagined experience
→ Policy/Critic update
→ error self-amplification
```

이 가능하다.

다른 [model-based RL](Model-Based-RL-and-World-Models) 알고리즘에서는 가상 [학습(learning)](Reinforcement-Learning)을 정당하게 사용할 수 있지만, AASSR 현재 main [실험(experiment)](Experiments)는 **계획 효과를 깨끗하게 분리**하기 위해 [에피소드가 끝나도 유지되는(persistent)](Knowledge) [Policy](Policy) [학습 갱신(update)](Neural-Networks-and-Optimization)를 막는다.

---

# 26. Receding-horizon execution

계획기가 탐색 깊이 4 미래를 계산했다고 해서 네 행동을 그대로 실행하는 것이 아니다.

```text
Plan:
A0 → A1 → A2 → A3

실제:
A0만 실행
→ real response 관측
→ 다시 planning
```

이렇게 하면 예측된 가상 미래 전개과 실제 환경 결과의 차이를 다음 [단계(step)](Terminology-Guide)에서 바로 반영할 수 있다.

[Model Predictive Control](Counterfactual-Planning-and-Search)과 개념적으로 닮은 부분이다.

---

# 27. 과거 2k diagnostic이 보여준 것

Repaired [실험 실행(run)](Reproduction)에서는 [Imagination](Imagination)이 더 이상 inert하지 않고 실제로 계획과 실제 행동 개입을 만들었다.

하지만 핵심 교훈은:

```text
행동을 바꿀 수 있다
!=
더 좋은 행동을 고른다
```

였다.

여러 실제 행동 개입이 `403/404/429` 같은 bad 공개된 환경 결과으로 이어졌고 [직접적인(direct)](Terminology-Guide) [실제로 성공을 만들어내는(success-producing)](Experiments) 실제 행동 개입은 확인되지 않았다.

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

> **코드 [문제 수정(repair)](Development-History)가 들어갔다는 사실과 최종 [성능(performance)](Ablation-Benchmarking-and-Reproducibility) improvement는 같은 주장이 아니다.**

후자는 새 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)로 다시 검증해야 한다.

---

# 29. Failure mode: Model error exploitation

계획기는 많은 선택 후보 중 가장 높은 가치를 찾는다.

그 과정에서 세계 모델의 작은 오류를 찾아낼 수 있다.

```text
실제로는 mediocre한 action
but model predicts rare amazing future
→ planner selects it
```

대응:

- 확률적 확률 질량
- [Calibration](Calibration)
- [상태 코드까지 고려하는(status-aware)](Calibration) 학습 모델
- controlled 미래 탐색 범위

---

# 30. Failure mode: OOD Critic exploitation

[Prophecy](Prophecy) 미래가 plausible해도 [Critic](Critic)이 해당 영역에서 근거 없는 high 가치를 낼 수 있다.

대응:

- [local Critic support](Critic-Support-and-OOD)
- [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) [기본 경로로 돌아가기(fallback)](Imagination)

---

# 31. Failure mode: Over-pruning

Reliability/빔 탐색/가지치기이 너무 강하면 좋은 탐색의 첫 행동가 사라질 수 있다.

대응:

- 탐색의 첫 행동 보존
- separate 탐색의 첫 행동 eligibility vs deep expansion
- [검사를 통과(pass)](Ablation-Benchmarking-and-Reproducibility)/[후보 억제(suppression)](ASEQ) [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility) 분석

---

# 32. Failure mode: Under-pruning

모든 결과 경로를 유지하면:

- [계산(compute)](Reproduction) 폭발
- tiny-probability [갈라진 결과 경로(branches)](Chance-and-Decision-Nodes) 증가
- [OOD](Critic-Support-and-OOD) 상태 증가

가 가능하다.

대응:

- 빔 탐색 width
- 구조 기반 중복 제거
- 묶음 처리
- probability-aware 결과 경로 management

---

# 33. Failure mode: Planner inertia

모든 판정 관문가 너무 보수적이거나 [Critic](Critic) 가치가 모두 비슷하면 실제 행동 개입이 0이 된다.

가능한 원인:

- [Prophecy](Prophecy) 데이터 포함 범위 부족
- [Calibration(예측 신뢰도 보정)](Calibration) 표본 부족
- [Critic](Critic) 희소한 [대상 또는 학습 목표값(target)](Terminology-Guide) starvation
- [국소 데이터 근거(local support)](Critic-Support-and-OOD) 부족
- 실제 행동 개입 최소 차이 기준 과도함

따라서:

```text
intervention = 0
```

만 보고 계획기 코드가 죽었다고 결론내리면 안 된다.

Gate reason별 진단 실험이 필요하다.

---

# 34. Failure mode: Overactive planner

반대로 실제 행동 개입이 많다고 좋은 것도 아니다.

```text
interventions ↑
errors ↑
success unchanged
```

일 수 있다.

최종 목표는 **실제 행동 개입 횟수 최대화**가 아니라 **[Policy](Policy)보다 좋은 행동을 근거 있게 선택**하는 것이다.

---

# 35. Failure mode: Probability/reliability/value mixing

하나의 [평가 점수(score)](Terminology-Guide)에:

```text
outcome probability
confidence
Critic value
support
```

를 모두 더하거나 곱하면 해석이 어려워진다.

현재 [설계(design)](Design-Rationale)은 역할을 분리한다.

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

[Skill](Skills)은 여러 [더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills) 행동을 관계 기반 [여러 행동을 묶은 상위 행동(macro)](Hierarchical-RL-and-Skills)처럼 재사용할 수 있다.

계획기가 [Skill(성공 절차 재사용)](Skills)을 행동 후보로 다룰 때 그 내부 기본 행동 단위 [순서열(sequence)](GRU-and-Sequence-Models)에도 확률적 미래가 존재할 수 있다.

현재 [Skill](Skills) [Prophecy](Prophecy)는 여러 확률적 환경 결과을 작은 빔 탐색으로 유지한다.

관련 배경: [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

---

# 37. Imagination과 Curriculum

쉬운 [난이도 단계(level)](Curriculum-Learning)에서는 [Prophecy](Prophecy)/가치 평가 데이터 근거가 충분하지만 [더 높은 단계(higher)](Curriculum-Learning) 난이도 단계에서는 [OOD](Critic-Support-and-OOD)가 될 수 있다.

```text
L0/L1 training frontier
     ↓
L2/L3 imagined state
→ reliability/support 부족
```

따라서 [난이도 조절 학습(curriculum)](Curriculum-Learning) progression과 [Imagination](Imagination) [품질(quality)](Ablation-Benchmarking-and-Reproducibility)는 강하게 연결된다.

관련 페이지: [Curriculum Learning](Curriculum-Learning)

---

# 38. Imagination compute 최적화

Planning은 expensive하다.

Current-generation에는 다음과 같은 engineering [최적화(optimization)](Neural-Networks-and-Optimization)이 중요하다.

- depth-batched [Prophecy](Prophecy)
- batched [Critic](Critic) scoring
- 구조 기반 탐색의 첫 행동 중복 제거
- cache reuse
- GPU-friendly tensor [경로(path)](Counterfactual-Planning-and-Search)

이러한 최적화는 **계획 의미 규칙를 바꾸지 않고 같은 계산을 더 효율적으로 실행**하는 것을 목표로 한다.

관련 기초: [Neural Networks & Optimization](Neural-Networks-and-Optimization)

---

# 39. Imagination을 평가할 때 볼 metric

## Planning activity

- 계획 횟수
- nodes expanded
- maximum 탐색 깊이 [도달한(reached)](Curriculum-Learning)
- 구조 기반 roots / 실제 개체를 구분하는 roots

## Gate behavior

- [전체 범위(global)](Terminology-Guide) 데이터 포함 범위 failures
- low-reliability roots
- policy-root 신뢰도 failures
- 국소 데이터 근거 failures
- insufficient-advantage suppressions

## Intervention behavior

- 행동 전환 [선택 후보(candidates)](Terminology-Guide)
- 최종 실제 행동 개입s
- changed 행동s
- 직접적인 [성공(success)](Terminology-Guide) 실제 행동 개입s
- bad-status 실제 행동 개입s

## Final task metric

- 같은 체크포인트 OFF 성공
- 같은 체크포인트 ON 성공
- paired [실험 시나리오(scenario)](Experiments) improvements/[회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)s

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

H1~H7은 [작동 원리(mechanism)](Evidence-Matrix)/진단 실험이고 H8이 최종 planner-benefit [연구 주장(claim)](Evidence-Matrix)이다.

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

> **[Imagination](Imagination)은 확률적 세계 모델의 미래를 [환경의 확률 분기(chance)](Chance-and-Decision-Nodes) 확률 기댓값과 의사결정 max로 전개한 뒤, [예측 신뢰도(prediction reliability)](Calibration)·가치 평가 데이터 근거·가치 상대적 이점가 모두 충분할 때만 실제 [Policy](Policy) 행동을 바꾸는 근거가 부족하면 보수적으로 거부하는 [실제로 하지 않은 경우를 가정하는 반사실적(counterfactual)](Counterfactual-Planning-and-Search) 계획기다.**

---

다음으로 읽기:

- **[Prophecy](Prophecy)**
- **[Calibration](Calibration)**
- **[Critic](Critic)**
- **[Counterfactual Planning & Search](Counterfactual-Planning-and-Search)**
- **[Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
