# State Representation

AASSR current-generation의 transfer 학습기는 **response-causal relational public state v3**를 사용한다.

이 페이지의 핵심 질문은 다음이다.

> **정답 identity나 hidden simulator state를 주지 않으면서도, 이름이 바뀐 새로운 scenario에서 같은 문제 구조를 알아볼 수 있는 상태 표현을 만들 수 있는가?**

이 질문은 일반적으로 [state와 observation의 차이](MDP-and-POMDP), [partial observability](MDP-and-POMDP), [representation learning](Relational-Representation-and-Generalization), [invariance](Relational-Representation-and-Generalization), [generalization](Relational-Representation-and-Generalization), [data leakage](Causality-Leakage-and-Evaluation) 문제와 연결된다.

> [!IMPORTANT]
> 현재 manifest 계약: `response-causal-relational-public-state-v3+latest-http-status`  
> 핵심 구현: `src/aassr_v2/current_relational_state_v3.py`

---

# 0. 먼저 알아두면 좋은 개념

- [MDP and POMDP](MDP-and-POMDP) — true state, observation, Markov property, state aliasing
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — permutation, invariance, memorization vs transfer
- [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation) — hidden simulator state를 learner input에 넣으면 왜 안 되는가?
- [Neural Networks & Optimization](Neural-Networks-and-Optimization) — feature vector, one-hot encoding, normalization
- [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment) — representation이 long-horizon learning에 미치는 영향

---

# 1. 왜 state representation이 연구 질문인가?

[강화학습](Reinforcement-Learning)에서 Policy가 아무리 강해도 입력 representation이 잘못되면 transfer가 어렵다.

예를 들어 training에서:

```text
route-12 = useful catalog-like route
```

였다고 하자.

unseen seed에서 같은 역할이:

```text
route-31 = useful catalog-like route
```

로 바뀌면 concrete ID 중심 learner는 두 상황을 별개로 볼 수 있다.

AASSR은 이름보다 **공개적으로 관측한 역할과 관계 구조**를 transfer representation의 핵심으로 사용한다.

이것은 [permutation invariance](Relational-Representation-and-Generalization)를 노리는 inductive bias다.

---

# 2. True state, Observation, Representation

세 층을 구분하는 것이 중요하다.

```text
Hidden true simulator state
        ↓ observation process
Public observation
        ↓ feature encoding
Relational State v3
        ↓
Policy / Prophecy / Critic / Skill
```

[POMDP](MDP-and-POMDP)에서는 hidden true state `S_t` 전체를 agent가 보지 못하고 observation `O_t`만 받는다.

Representation은 그 observation을 learner가 사용할 feature로 바꾼 것이다.

```text
observation contract
!=
representation format
```

이다.

Representation이 relational하다고 해서 hidden simulator 정보를 새로 볼 수 있는 것은 아니다.

---

# 3. Markov property와 representation

이론적인 [MDP](MDP-and-POMDP) state는 현재 정보만으로 다음 state distribution을 충분히 결정할 수 있는 [Markov property](MDP-and-POMDP)를 가진다.

하지만 learner의 representation이 중요한 정보를 버리면:

```text
실제 상황 A ─┐
             ├→ 같은 representation R
실제 상황 B ─┘
```

가 생길 수 있다.

A와 B에서 future dynamics나 optimal action이 다르면 **state aliasing**이다.

즉 abstraction은 transfer를 도울 수 있지만 너무 강하면 Markov sufficiency를 해칠 수 있다.

---

# 4. 무엇을 볼 수 있는가?

current pentest runtime은 실제 response에서 인과적으로 관측 가능한 public information을 사용한다.

예:

- 발견된 route/profile/object 관계
- 현재 legal action surface
- session / CSRF 존재처럼 실제 response를 통해 확인한 상태
- self-counted request usage
- self-observed workflow progress
- latest public HTTP status

이 정보들은 agent가 실제 interaction history에서 얻을 수 있는 public signal이다.

---

# 5. 무엇을 의도적으로 숨기는가?

learner에게 직접 주지 않는 정보의 예:

- hidden curriculum level
- exact hidden workflow depth
- exact hidden audit / lockout pressure
- exact hidden session countdown
- hidden rate-limit distance
- 정답 route/profile/object identity
- future state

핵심 원칙:

> **모델이 추론하거나 예측해야 할 정보를 simulator 내부에서 바로 꺼내 observation으로 주지 않는다.**

이것은 [privileged-information leakage](Causality-Leakage-and-Evaluation)를 막는 기본 규칙이다.

---

# 6. 두 종류의 identity

AASSR에서는 identity를 하나로 통일하지 않는다.

## Concrete semantic identity

사용처:

- [ASEQ](ASEQ)
- episode-local exact repetition
- concrete cycle detection
- 실제 environment action execution

```text
route-12 != route-31
```

실제 서로 다른 대상을 구분해야 하기 때문이다.

## Relational transfer identity

사용처:

- [Policy](Policy)
- [Prophecy](Prophecy)
- [Critic](Critic)
- [Skill](Skills)
- Relational DQN baseline
- DreamerV3 relational adapter

```text
route-12 -> catalog-like role
route-31 -> catalog-like role

=> same relational structure
```

이것은 [relational inductive bias](Relational-Representation-and-Generalization)다.

---

# 7. 왜 둘 중 하나만 쓰면 안 되는가?

## Concrete only

```text
identifier rename
-> state identity 전부 변경
-> memorization
-> unseen transfer 약화
```

## Relational only

```text
같은 역할의 서로 다른 concrete entity
-> 같은 대상으로 오인
-> 실제 실행 / self-loop 판정 오류
```

그래서 AASSR은:

```text
학습/transfer/compute: relational
실행/정확한 반복 판정: concrete
```

를 분리한다.

이 설계는 [compute identity와 execution identity](Counterfactual-Planning-and-Search)를 나누는 Imagination 구조와도 연결된다.

---

# 8. Abstraction의 trade-off

Representation을 더 추상화하면:

```text
Concrete detail ↓
→ transfer 가능성 ↑
→ state space 공유 ↑
```

가 가능하다.

하지만 동시에:

```text
important public distinction ↓
→ state aliasing ↑
→ Policy/Prophecy target conflict ↑
```

가 가능하다.

AASSR Relational State v3는 이 trade-off에서 **latest public HTTP status처럼 decision-critical한 공개 차이는 명시적으로 다시 보존**하는 방향으로 발전했다.

---

# 9. Relational state v3의 구조

current v3는 기존 relational v2 descriptor 뒤에 **latest public HTTP status channel**을 추가한다.

현재 코드 기준:

```text
v2 relational descriptor : 35 dimensions
latest status channel     :  8 dimensions
------------------------------------------
v3 descriptor             : 43 dimensions
```

status channel은 다음 public status vocabulary의 [one-hot/categorical representation](Neural-Networks-and-Optimization)이다.

```text
200 / 302 / 400 / 401 / 403 / 404 / 409 / 429
```

---

# 10. 왜 latest HTTP status가 필요했는가?

이전 relational state에서는 전체 semantic structure는 비슷하게 표현하면서도 최근 response의 `403/404/429` 같은 public signal을 잃을 수 있었다.

2026-08-11 Imagination diagnostic에서는 semantic prediction metric이 높게 보여도 실제 override가 이러한 오류 status로 이어지는 문제가 관찰됐다.

즉:

```text
구조적으로 비슷함
!=
decision-critical public outcome까지 같음
```

이었다.

v3는 latest status를 명시적으로 보존해 이 blind spot을 줄인다.

이 사례는 일반적인 **representation abstraction이 중요한 state variable을 지워 state aliasing을 만든 사례**로 볼 수 있다.

---

# 11. Status는 hidden 위험 신호가 아니다

중요한 방법론 경계다.

AASSR이 보는 것은 실제 response로 공개된 HTTP-like status다.

```text
latest observed 403
```

을 쓰는 것은 허용된다.

반면 simulator 내부의:

```text
lockout까지 정확히 1회 남음
hidden audit pressure = 0.93
```

같은 값은 learner에게 직접 주지 않는다.

따라서 status-aware representation은 hidden safety oracle을 추가하는 것이 아니다.

이 차이는 [public observation과 privileged information](Causality-Leakage-and-Evaluation)의 차이다.

---

# 12. 왜 status는 scalar가 아니라 categorical인가?

HTTP status code의 숫자 차이는 task semantics의 거리와 일치하지 않는다.

```text
403과 404의 숫자 차이 = 1
```

이라고 해서 두 상태의 의미가 연속적인 numeric distance 1만큼 다르다는 뜻은 아니다.

그래서 mutually exclusive [categorical feature/target](Loss-Functions-and-Class-Imbalance)으로 다루는 것이 더 자연스럽다.

---

# 13. Status vector를 어떻게 얻는가?

current implementation은 public status의 명시적 metadata/fact/vector channel에서 latest status를 복원한다.

우선순위에 따라 이미 relational prediction이 가진 status probabilities를 사용할 수도 있고, 실제 public `last_status` fact 또는 raw public observation channel에서 읽을 수도 있다.

어느 경로든 hidden audit/session state를 읽지 않는 것이 contract다.

---

# 14. Predicted relational state decode

[Prophecy](Prophecy)는 relational descriptor 자체를 예측한다.

v3 decode는:

```text
predicted base relational semantics
+
predicted legal action mask
+
predicted terminal class
+
predicted status probabilities
```

를 다시 planner가 사용할 `StateSnapshot` 형태로 복원한다.

예측된 latest status는 predicted fact/metadata에도 일관되게 반영된다.

즉 world model이 사용하는 latent/feature representation과 planner가 사용하는 action/state protocol 사이에 명시적인 decoder가 있다.

---

# 15. Legal action surface도 state의 일부인가?

AASSR planning에서는 현재 state에서 어떤 action이 실제로 가능한지가 매우 중요하다.

일반적으로 state-dependent action set을:

```math
\mathcal{A}(s)\subseteq\mathcal{A}
```

처럼 생각할 수 있다.

[Prophecy](Prophecy)가 next state를 예측하면서 legal action mask도 예측하는 이유다.

State vector가 비슷해도 legal actions가 다르면 planner에게는 다른 state일 수 있다.

---

# 16. Semantic score v3

World-model [Calibration](Calibration)에서는 단순 vector distance 하나만 보지 않는다.

current v3 semantic score는 개념적으로 다음 네 종류의 correctness를 함께 본다.

```text
base relational semantics
legal action mask
latest HTTP status
terminal class
```

현재 코드의 가중 구조는:

```text
base semantic quality : 0.35
legal-mask quality    : 0.25
status match          : 0.30
terminal match        : 0.10
```

이다.

이 수치는 reward가 아니라 **Prophecy prediction validation metric**이다.

[training loss와 evaluation metric](Loss-Functions-and-Class-Imbalance)을 구분해야 한다.

---

# 17. 왜 status 비중이 꽤 큰가?

과거 diagnostic에서 전체 semantic similarity가 높아도 status error가 실제 decision quality를 망칠 수 있다는 evidence가 나왔기 때문이다.

따라서 calibration metric이 단순 "대부분 비슷하다"만 보지 않고 decision-critical public response를 명시적으로 반영한다.

단, status match를 agent task reward에 더하는 것은 아니다.

```text
status metric weight
!=
reward shaping
```

이다.

---

# 18. 누가 v3 representation을 쓰는가?

current contract 설치 후 핵심 transfer consumer가 v3로 rebound된다.

대표적으로:

- Policy state encoding
- Prophecy relational codec/model
- semantic calibration/evaluator
- Critic/support 관련 relational state key
- DreamerV3 relational adapter

따라서 baseline과 AASSR 비교에서 relational representation 계약을 최대한 일관되게 유지한다.

이것은 [fair benchmarking](Ablation-Benchmarking-and-Reproducibility)에 중요하다.

---

# 19. Raw DQN과 Relational DQN 비교가 중요한 이유

AASSR Full이 raw DQN보다 좋아도 그 차이가 전부 Imagination 때문이라고 할 수 없다.

Representation 자체의 효과가 있을 수 있기 때문이다.

그래서:

```text
dqn_raw
   |
   | state/action representation만 relational로 변경
   v
dqn_relational
```

을 독립 control로 둔다.

이 비교는 [ablation study](Ablation-Benchmarking-and-Reproducibility)의 대표 예다.

---

# 20. State와 Knowledge의 경계

현재 public state에는 이미 실제 response에서 관측한 많은 사실이 포함된다.

[KnowledgeStore](Knowledge)는 그와 별도로 provenance와 causal timing을 가진 explicit episode context를 관리한다.

```text
State
= 현재 공개 상황 representation

Knowledge
= 어떤 response에서 언제 알게 되었는지까지 관리하는 explicit context
```

같은 사실을 무분별하게 두 경로에서 중복 주입하지 않도록 current Prophecy는 context path를 보수적으로 다룬다.

---

# 21. State와 ASEQ의 경계

Policy/Prophecy는 relational state를 쓰지만 [ASEQ](ASEQ)는 exact repetition을 판정해야 한다.

따라서 ASEQ까지 같은 relational identity로 뭉치면:

```text
서로 다른 route지만 같은 역할
-> 같은 S라고 오인
-> 정상 행동을 self-loop로 막음
```

이 생길 수 있다.

그래서 concrete semantic state와 relational state를 동시에 유지한다.

---

# 22. State와 Critic support

[Critic local support](Critic-Support-and-OOD)는 query state/action이 real training distribution 근처인지 판단해야 한다.

Raw concrete ID distance를 쓰면 unseen rename 자체를 OOD로 잘못 볼 수 있다.

그래서 support distance도 public relational structure를 중심으로 구성한다.

즉 relational representation은 Policy/Prophecy transfer뿐 아니라 **OOD evidence 정의**에도 영향을 준다.

---

# 23. State와 Skill transfer

[Skill](Skills)은 성공한 concrete trajectory를 relational action template로 저장한다.

새 seed에서 같은 structural state/action relationship을 찾아 concrete action으로 rebind한다.

따라서 Skill transfer도 State Representation의 역할/관계 정의에 의존한다.

관련 배경: [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

---

# 24. State representation과 Curriculum

[Curriculum](Curriculum-Learning) level이 올라가면 새로운 state/action distribution이 나타날 수 있다.

Relational representation이 잘 설계되면 낮은 level에서 배운 구조를 higher level에 공유할 수 있다.

하지만 higher level에서 새로운 decision-critical variable이 생기는데 descriptor가 이를 표현하지 못하면 state aliasing이 다시 발생할 수 있다.

즉 curriculum transfer 실패는 Policy 문제뿐 아니라 representation 문제일 수도 있다.

---

# 25. Failure mode: Identifier memorization

Concrete ID에 의존해 unseen rename transfer 실패.

대응:

- relational role representation
- unseen identifier permutation benchmark

---

# 26. Failure mode: Over-abstraction

서로 다른 실제 대상을 너무 강하게 같은 state로 압축.

대응:

- concrete semantic identity를 실행/ASEQ에 별도 유지
- decision-critical public channel을 descriptor에 보존

---

# 27. Failure mode: Decision-critical channel loss

전체 구조는 유지하지만 latest status 같은 중요한 public signal을 버림.

대응:

- Relational State v3
- status-aware Prophecy / calibration

---

# 28. Failure mode: Hidden-state leakage

Simulator 내부 정답/압력을 representation에 포함해 benchmark shortcut 발생.

대응:

- response-causal public observation contract
- [privileged-information audit](Causality-Leakage-and-Evaluation)

---

# 29. Failure mode: Representation drift

Policy, Prophecy, Critic, baseline이 서로 다른 relational definition을 쓰면 비교가 깨진다.

대응:

- current contract 설치
- manifest source of truth
- CI/regression validation

---

# 30. Failure mode: Feature-scale distortion

Count feature와 binary/public probability feature의 scale이 크게 다르면 neural optimization에 영향을 줄 수 있다.

대응:

- bounded normalization
- descriptor contract test

관련 기초: [Neural Networks & Optimization](Neural-Networks-and-Optimization)

---

# 31. State representation을 어떻게 평가하는가?

Representation 자체에는 단일 accuracy가 없다.

대신 다음 downstream/diagnostic을 본다.

- raw DQN vs relational DQN unseen success
- identifier permutation consistency
- equivalent-role action score consistency
- same relational structure의 Prophecy prediction consistency
- concrete ASEQ false-positive rate
- latest status preservation accuracy
- hidden-state leakage regression test
- higher-level state aliasing diagnostic

Representation quality는 결국 여러 learner의 [generalization](Relational-Representation-and-Generalization)에 미치는 영향으로 평가된다.

---

# 32. 연구 가설

```text
H1. relational representation이 raw representation보다 unseen transfer에 유리한가?
H2. concrete/relational identity 분리가 self-loop 정확도와 transfer를 동시에 지키는가?
H3. latest public status를 추가하면 Prophecy/calibration의 decision-critical 오류가 줄어드는가?
H4. hidden simulator state 없이도 충분한 문제 구조를 표현할 수 있는가?
H5. 같은 v3 contract를 Policy/Prophecy/Critic/baseline에 적용하면 비교가 더 공정해지는가?
H6. v3가 higher-level curriculum에서 state aliasing을 충분히 줄이는가?
```

---

# 33. 관련 코드

```text
src/aassr_v2/current_relational_state_v3.py
  - latest_status_vector
  - relational_state_descriptor_v3
  - relational_state_vector_v3
  - decode_relational_state_v3
  - semantic_prediction_score_v3
  - install_status_aware_relational_contract

src/aassr_v2/current_manifest.py
  - active observation / policy-state contract
```

---

# 34. 한 문장 요약

> **Relational State v3는 hidden 정답을 추가하는 표현이 아니라, public observation에서 concrete name의 불필요한 차이는 줄이되 decision-critical public status와 실행에 필요한 concrete identity는 별도 경로로 보존하는 transfer representation이다.**

---

다음으로 읽기:

- **[Research Architecture](Research-Architecture)**
- **[ASEQ](ASEQ)**
- **[Policy](Policy)**
- **[Knowledge](Knowledge)**
- **[Prophecy](Prophecy)**
- **[Calibration](Calibration)**
- **[Concept Index](Concept-Index)**
