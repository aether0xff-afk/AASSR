# Imagination — 가상 미래 탐색

[Imagination(가상 미래 탐색)](Imagination)은 AASSR의 **[counterfactual planner](Counterfactual-Planning-and-Search)** 다.

핵심 질문은 다음과 같다.

> **실제 행동을 하기 전에 [Prophecy](Prophecy)가 예측한 여러 미래를 몇 단계 전개해 보면, 현재 [Policy](Policy)가 고른 행동보다 더 나은 첫 행동을 선택할 수 있는가?**

AASSR에서 [Imagination](Imagination)은 [현재(current)](Current-Status) main [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility) 기준으로 [모델이 상상한(imagined)](Research-Jargon-Guide) data를 사실처럼 학습시키는 장치가 아니라 **실행 전 [계획(planning)](Counterfactual-Planning-and-Search) 장치**다.

> [!IMPORTANT]
> 현재 핵심 구현: `src/aassr_v2/current_planner.py`  
> 신뢰도 [판정 관문(gate)](Terminology-Guide): `src/aassr_v2/current_confidence_gate.py`  
> [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD): `src/aassr_v2/current_critic_support.py`

---

# 0. 먼저 알아두면 좋은 개념

- [Model-Based RL & World Models](Model-Based-RL-and-World-Models) — learned [학습 모델(model)](Terminology-Guide)로 계획한다는 뜻
- [Counterfactual Planning & Search](Counterfactual-Planning-and-Search) — rollout, horizon, beam, pruning, [탐색의 첫 행동(root)](Imagination) preservation
- [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes) — [확률 기댓값(expectation)](Chance-and-Decision-Nodes)과 max의 차이
- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — [확률(probability)](Stochasticity-Uncertainty-and-Probability), [신뢰도(reliability)](Calibration), [가치(value)](Value-Functions-and-Bellman-Equation)
- [Critic, Support & OOD](Critic-Support-and-OOD) — search가 [학습 분포 밖(OOD)](Critic-Support-and-OOD) 가치 error를 exploit하는 문제
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — [구조 기반(structural)](Relational-Representation-and-Generalization) 탐색의 첫 행동 dedup
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

Current main AASSR 실험 규칙에서 [Imagination](Imagination)은 두 번째에 해당한다.

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
6. 깊은 rollout이 실패해도 실제 legal 탐색의 첫 행동 행동을 잃으면 안 된다.
7. 계획 depth가 깊어질수록 [compounding model error](Model-Based-RL-and-World-Models)가 커진다.

즉 현재 [Imagination](Imagination)은 **확률적 계획 semantics + 신뢰도 constraints + 구조 기반 computation**의 조합이다.

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

최적 continuation을 가정하면:

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

하지만 expected [누적 보상(return)](Value-Functions-and-Bellman-Equation)은:

```math
0.1(1)+0.9(-1)=-0.8
```

이다.

[환경(Environment)](Reinforcement-Learning) randomness에 `max`를 쓰면 에이전트가 실제로 통제할 수 없는 jackpot 환경 결과을 선택할 수 있는 것처럼 계산한다.

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

[Prophecy](Prophecy)는 각 행동 뒤의 확률적 [공개된(public)](State-Representation) future [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)을 만든다.

```text
(S,A)
 ↓
Prophecy
 ↓
[(S1',p1), (S2',p2), ...]
```

[Imagination](Imagination)은 이 predicted states를 tree node로 사용한다.

[Prophecy(미래 예측 모델)](Prophecy)가 틀리면 계획기가 아무리 수학적으로 올바른 [미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)을 해도 잘못된 미래를 최적화할 수 있다.

이를 [model exploitation](Model-Based-RL-and-World-Models) 문제라고 볼 수 있다.

---

# 8. Critic은 왜 필요한가?

Planner depth를 무한히 늘릴 수 없다.

어느 depth에서는 rollout을 멈추고 그 이후의 장기 sparse 누적 보상을 추정해야 한다.

```text
S0 → Ŝ1 → Ŝ2 → Ŝ3
                   |
                   v
                 Critic
```

Current [Critic](Critic)은 [관계 기반(relational)](Relational-Representation-and-Generalization) [GRU](GRU-and-Sequence-Models) 기반 discounted sparse-누적 보상 [값을 추정하는 모델(estimator)](Terminology-Guide)다.

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

한 [행동 선택 노드(decision node)](Chance-and-Decision-Nodes)에 행동 `b`개가 있고 각 행동에 환경 결과 `m`개가 있다면 naive tree는 매우 빠르게 커진다.

대략:

```text
(b × m)^depth
```

형태의 combinatorial growth를 생각할 수 있다.

실제 계획기는:

- beam width
- 환경 결과 sample count
- pruning
- 구조 기반 dedup
- [묶음 처리(batching)](Reproduction)

등으로 계산량을 제한한다.

---

# 11. Root preservation

깊은 결과 경로가 unreliable하거나 prune되어도 **실제로 가능한 탐색의 첫 행동 행동 자체가 사라지면 안 된다.**

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

하지만 pruning 기준이 잘못되면 유용한 결과 경로가 초기에 사라질 수 있다.

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

실제 행동 surface에는 실제 개체를 구분하는 ID만 다른 행동이 매우 많을 수 있다.

```text
GET route-12
GET route-31
GET route-44
...
```

하지만 [relational representation](Relational-Representation-and-Generalization)에서는 같은 행동 structure일 수 있다.

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

Planner가:

```text
catalog-like route request
```

라는 구조 기반 decision을 고르더라도 실제 환경는:

```text
GET /route_31
```

같은 [실제 실행 행동(concrete action)](State-Representation)을 요구한다.

따라서 구조 기반 dedup은 **계산 공유**이지 행동 [식별 방식(identity)](State-Representation) 병합이 아니다.

이 구분은 [Relational Representation & Generalization](Relational-Representation-and-Generalization)에서 핵심적으로 다룬다.

---

# 15. Prophecy reliability gate

World 학습 모델은 완벽하지 않다.

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

현재 행동 surface 전체에서 [Prophecy](Prophecy)가 거의 모르는 상태라면 계획기를 통째로 비활성화할 수 있다.

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

B가 [Critic](Critic) 가치는 높더라도 [예측(prediction)](Terminology-Guide) 자체가 unreliable하면 final 기본 행동 덮어쓰기 [선택 후보(candidate)](Terminology-Guide)에서 제외할 수 있다.

---

# 18. Policy root도 reliable해야 하는 이유

Alternative와 [Policy](Policy)를 비교하려면 둘 다 같은 수준의 학습 모델 [증거(evidence)](Evidence-Matrix)가 필요하다.

```text
V_alt - V_policy
```

에서 `V_policy`의 underlying 예측이 unreliable하면 advantage가 의미 없을 수 있다.

그래서 현재 판정 관문는 [Policy](Policy) 결과 경로도 신뢰도를 요구한다.

---

# 19. Local Critic support gate

[Prophecy](Prophecy) 예측이 reliable해도 [Critic](Critic)이 predicted 상태/행동 region을 본 적 없을 수 있다.

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

하지만 현재 query가 higher-level [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) region이면:

```text
local support = low
```

일 수 있다.

Search는 이런 [OOD](Critic-Support-and-OOD) artifact를 적극적으로 고를 수 있기 때문에 local 증거가 필요하다.

자세히: [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 21. Intervention advantage

Planner가 찾은 best reliable/supported 탐색의 첫 행동와 [Policy](Policy) 탐색의 첫 행동를 비교한다.

```math
\Delta V=V_{candidate}-V_{policy}
```

Candidate가 다르더라도 `ΔV`가 너무 작으면 noise일 수 있다.

그래서 fixed [실제 행동 개입(intervention)](Imagination) [최소 차이 기준(margin)](Imagination) `m`을 둔다.

```math
\Delta V\ge m
```

일 때만 실제 switch를 허용한다.

Margin은 **[보상(reward)](Sparse-Reward-and-Credit-Assignment) shaping이 아니라 decision [판정 기준값(threshold)](Terminology-Guide)**다.

---

# 22. 왜 margin이 필요한가?

Value 값을 추정하는 모델에는 noise가 있다.

```text
Policy value    = 0.501
Candidate value = 0.503
```

같은 작은 차이로 매번 행동을 바꾸면 계획기가 unstable할 수 있다.

Margin은 작은 가치 noise에 대한 robustness 역할을 한다.

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

- plan count
- switch 선택 후보 count
- suppressed count
- final 실제 행동 개입 count
- changed-행동 count

을 구분한다.

---

# 24. Same-checkpoint OFF/ON 비교

현재 [Imagination](Imagination)의 순수 marginal effect를 측정하는 핵심 실험 규칙:

```text
one training run
       ↓
frozen AASSR checkpoint
    /          \
OFF eval      ON eval
```

Training 중부터 [Imagination](Imagination) 실제 행동 개입을 켜면 [학습(training)](Terminology-Guide) trajectory가 달라진다.

```text
OFF-trained model
vs
ON-trained model
```

은 계획기 효과뿐 아니라 data-distribution 효과까지 섞인다.

그래서 현재 main [비교(comparison)](Ablation-Benchmarking-and-Reproducibility)은 [same-checkpoint evaluation](Ablation-Benchmarking-and-Reproducibility)을 사용한다.

---

# 25. 왜 imagined experience로 Policy를 바로 학습시키지 않는가?

Model-generated [상태 전이(transition)](MDP-and-POMDP)을 실제 truth처럼 학습하면:

```text
world-model error
→ imagined experience
→ Policy/Critic update
→ error self-amplification
```

이 가능하다.

다른 [model-based RL](Model-Based-RL-and-World-Models) 알고리즘에서는 가상 learning을 정당하게 사용할 수 있지만, AASSR 현재 main experiment는 **계획 effect를 깨끗하게 분리**하기 위해 persistent [Policy](Policy) update를 막는다.

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

이렇게 하면 predicted rollout과 실제 환경 결과의 차이를 다음 step에서 바로 반영할 수 있다.

[Model Predictive Control](Counterfactual-Planning-and-Search)과 개념적으로 닮은 부분이다.

---

# 27. 과거 2k diagnostic이 보여준 것

Repaired run에서는 [Imagination](Imagination)이 더 이상 inert하지 않고 실제로 plan과 실제 행동 개입을 만들었다.

하지만 핵심 교훈은:

```text
행동을 바꿀 수 있다
!=
더 좋은 행동을 고른다
```

였다.

여러 실제 행동 개입이 `403/404/429` 같은 bad 공개된 환경 결과으로 이어졌고 direct success-producing 실제 행동 개입은 확인되지 않았다.

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

후자는 새 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)로 다시 검증해야 한다.

---

# 29. Failure mode: Model error exploitation

Planner는 많은 선택 후보 중 가장 높은 가치를 찾는다.

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
- controlled horizon

---

# 30. Failure mode: OOD Critic exploitation

[Prophecy](Prophecy) future가 plausible해도 [Critic](Critic)이 해당 region에서 근거 없는 high 가치를 낼 수 있다.

대응:

- [local Critic support](Critic-Support-and-OOD)
- [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) [기본 경로로 돌아가기(fallback)](Imagination)

---

# 31. Failure mode: Over-pruning

Reliability/beam/pruning이 너무 강하면 좋은 탐색의 첫 행동가 사라질 수 있다.

대응:

- 탐색의 첫 행동 preservation
- separate 탐색의 첫 행동 eligibility vs deep expansion
- pass/suppression [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility) 분석

---

# 32. Failure mode: Under-pruning

모든 결과 경로를 유지하면:

- compute 폭발
- tiny-probability [갈라진 결과 경로(branches)](Chance-and-Decision-Nodes) 증가
- [OOD](Critic-Support-and-OOD) 상태 증가

가 가능하다.

대응:

- beam width
- 구조 기반 dedup
- 묶음 처리
- probability-aware 결과 경로 management

---

# 33. Failure mode: Planner inertia

모든 판정 관문가 너무 보수적이거나 [Critic](Critic) 가치가 모두 비슷하면 실제 행동 개입이 0이 된다.

가능한 원인:

- [Prophecy](Prophecy) 데이터 포함 범위 부족
- [Calibration(예측 신뢰도 보정)](Calibration) sample 부족
- [Critic](Critic) sparse target starvation
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

최종 목표는 **실제 행동 개입 count 최대화**가 아니라 **[Policy](Policy)보다 좋은 행동을 근거 있게 선택**하는 것이다.

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

[Skill](Skills)은 여러 primitive 행동을 관계 기반 macro처럼 재사용할 수 있다.

Planner가 [Skill(성공 절차 재사용)](Skills)을 행동 후보로 다룰 때 그 내부 primitive sequence에도 확률적 future가 존재할 수 있다.

Current [Skill](Skills) [Prophecy](Prophecy)는 여러 확률적 환경 결과을 작은 beam으로 유지한다.

관련 배경: [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

---

# 37. Imagination과 Curriculum

쉬운 level에서는 [Prophecy](Prophecy)/가치 평가 데이터 근거가 충분하지만 higher level에서는 [OOD](Critic-Support-and-OOD)가 될 수 있다.

```text
L0/L1 training frontier
     ↓
L2/L3 imagined state
→ reliability/support 부족
```

따라서 [난이도 조절 학습(curriculum)](Curriculum-Learning) progression과 [Imagination](Imagination) quality는 강하게 연결된다.

관련 페이지: [Curriculum Learning](Curriculum-Learning)

---

# 38. Imagination compute 최적화

Planning은 expensive하다.

Current-generation에는 다음과 같은 engineering optimization이 중요하다.

- depth-batched [Prophecy](Prophecy)
- batched [Critic](Critic) scoring
- 구조 기반 탐색의 첫 행동 dedup
- cache reuse
- GPU-friendly tensor path

이러한 최적화는 **계획 semantics를 바꾸지 않고 같은 계산을 더 효율적으로 실행**하는 것을 목표로 한다.

관련 기초: [Neural Networks & Optimization](Neural-Networks-and-Optimization)

---

# 39. Imagination을 평가할 때 볼 metric

## Planning activity

- plan count
- nodes expanded
- maximum depth reached
- 구조 기반 roots / 실제 개체를 구분하는 roots

## Gate behavior

- global 데이터 포함 범위 failures
- low-reliability roots
- policy-root 신뢰도 failures
- 국소 데이터 근거 failures
- insufficient-advantage suppressions

## Intervention behavior

- switch [선택 후보(candidates)](Terminology-Guide)
- final 실제 행동 개입s
- changed 행동s
- direct [성공(success)](Terminology-Guide) 실제 행동 개입s
- bad-status 실제 행동 개입s

## Final task metric

- 같은 체크포인트 OFF 성공
- 같은 체크포인트 ON 성공
- paired scenario improvements/[회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)s

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

H1~H7은 mechanism/진단 실험이고 H8이 최종 planner-benefit [연구 주장(claim)](Evidence-Matrix)이다.

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

> **[Imagination](Imagination)은 확률적 세계 모델의 미래를 chance 확률 기댓값과 decision max로 전개한 뒤, [예측 신뢰도(prediction reliability)](Calibration)·가치 평가 데이터 근거·가치 advantage가 모두 충분할 때만 실제 [Policy](Policy) 행동을 바꾸는 근거가 부족하면 보수적으로 거부하는 counterfactual 계획기다.**

---

다음으로 읽기:

- **[Prophecy](Prophecy)**
- **[Calibration](Calibration)**
- **[Critic](Critic)**
- **[Counterfactual Planning & Search](Counterfactual-Planning-and-Search)**
- **[Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
