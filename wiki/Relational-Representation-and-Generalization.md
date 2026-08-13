# Relational Representation and Generalization

AASSR은 concrete identifier 자체를 외우는 대신 **역할, 관계, 구조를 표현하는 [관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)**을 사용한다.

이 페이지는 다음 개념을 연결한다.

```text
representation
invariance
permutation
relational inductive bias
generalization
transfer
memorization
state aliasing
```

---

# 1. Representation이란?

Machine learning model은 현실의 상태를 그대로 이해하는 것이 아니라 **입력 feature [표현(representation)](Relational-Representation-and-Generalization)**을 받는다.

```text
Environment situation
      ↓ encoding
Feature vector / structure
      ↓
Learner
```

같은 실제 상황도 어떻게 encode하느냐에 따라 learner가 전혀 다르게 볼 수 있다.

---

# 2. Raw representation

Raw 표현은 concrete identifier나 원래 입력 위치를 강하게 보존할 수 있다.

예:

```text
route-12
profile-7
object-3
```

이런 값이 그대로 feature identity가 되면 model은 특정 ID와 value를 연결해 외울 수 있다.

훈련/평가에서 ID가 그대로 유지되면 성능이 높게 보일 수 있다.

하지만 이름만 바뀌면:

```text
route-31
profile-2
object-9
```

새로운 상태처럼 보일 수 있다.

---

# 3. Memorization과 Generalization

## Memorization

훈련 sample의 구체적인 패턴 자체를 기억한다.

## Generalization

훈련에서 보지 못한 새로운 sample에서도 학습한 규칙/구조를 적용한다.

AASSR [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)에서 중요한 질문:

> [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)가 바뀌어 concrete name이 permutation되어도 같은 문제 구조를 알아볼 수 있는가?

이다.

---

# 4. Permutation

Permutation은 원소의 이름/순서를 재배치하는 것이다.

예:

```text
Training:
route A = id 12
route B = id 31

Evaluation:
route A = id 44
route B = id 7
```

구조는 같은데 label만 바뀌었다.

Learner가 ID 자체에 의존하면 성능이 무너질 수 있다.

---

# 5. Invariance

어떤 변환을 해도 표현/output이 본질적으로 같게 유지되면 invariance라고 한다.

예를 들어 identifier permutation에 대해 invariant한 표현은:

```text
route-12가 catalog-like role
route-31가 catalog-like role
```

을 같은 구조로 encode할 수 있다.

AASSR 관계 기반 표현은 **난수 시드-renaming/permutation에 대한 [전이(transfer)](Relational-Representation-and-Generalization) invariance**를 노린다.

---

# 6. Equivariance

Invariance와 함께 자주 나오는 개념이다.

- Invariant: input을 특정 방식으로 바꿔도 output이 그대로
- Equivariant: input 변환에 대응하여 output도 같은 규칙으로 변함

Graph neural network나 set model에서 중요한 개념이다.

AASSR current relational descriptor는 explicit role/count/[행동(action)](Reinforcement-Learning)-feature 기반 구조이며, 모든 부분을 formal equivariant network로 구현한다고 주장하는 것은 아니다.

하지만 **concrete naming permutation에 덜 민감한 inductive bias**를 만드는 것이 핵심이다.

---

# 7. Relational representation

Relational 표현은 객체의 이름보다 **객체들 사이의 관계나 역할**을 표현한다.

Raw:

```text
route-12
profile-5
object-9
```

Relational:

```text
catalog-like route
current-user profile role
target-object role
```

처럼 표현할 수 있다.

AASSR의 current descriptor는 public facts와 role distributions, known entity counts, 행동 structure, latest status 등을 조합한다.

관련 페이지:

- [State Representation](State-Representation)

---

# 8. Inductive bias

**Inductive bias**는 learner가 제한된 data에서 어떤 종류의 규칙을 더 쉽게 배우도록 만드는 사전 구조/가정이다.

Relational 표현은:

> 이름 자체보다 관계 구조가 task에서 더 중요할 것이다.

라는 inductive bias를 준다.

이 bias가 맞으면 전이가 좋아질 수 있다.

틀리면 중요한 identity 정보를 잃을 수 있다.

---

# 9. Abstraction

Concrete state의 세부 정보를 줄이고 중요한 구조만 남기는 것을 abstr행동이라고 볼 수 있다.

```text
Concrete state
  ↓ abstraction
Relational state
```

장점:

- state space 압축
- 전이
- sample sharing

위험:

- 서로 다른 중요한 상황을 같은 state로 합침

---

# 10. State aliasing

서로 다른 실제 상황이 같은 표현으로 mapping되는 현상이다.

```text
Situation A ─┐
             ├→ Representation R
Situation B ─┘
```

A와 B에서 optimal 행동이 다르면 learner는 모순된 signal을 받는다.

AASSR Relational State v3에서 latest public status를 추가한 것도 **과도한 abstr행동으로 decision-critical 차이를 잃는 문제**를 줄이기 위한 수리다.

---

# 11. Concrete identity가 여전히 필요한 이유

Transfer learner는 relational identity를 쓰더라도 실제 [환경(environment)](Reinforcement-Learning)는 [실제 실행 행동(concrete action)](State-Representation)을 요구한다.

```text
"catalog-like route를 요청"
```

만으로는 실제 request를 실행할 수 없다.

마지막에는:

```text
GET /route-31
```

같은 실제 실행 행동이 필요하다.

그래서 AASSR은:

```text
learning/compute identity = relational
execution identity        = concrete
```

를 분리한다.

---

# 12. ASEQ는 왜 concrete semantic identity를 쓰는가?

[제자리 반복(Self-loop)](ASEQ) 판정에서는 서로 다른 concrete entity를 같은 것으로 합치면 위험하다.

```text
S(route-12) → action → S(route-31)
```

이 실제 진행인데 relational role만 같다고:

```text
S → A → S
```

로 오인할 수 있다.

그래서 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ) exact repetition은 concrete semantic identity를 유지한다.

관련 페이지:

- [ASEQ](ASEQ)
- [State Representation](State-Representation)

---

# 13. Relational action representation

State뿐 아니라 행동도 relational하게 표현해야 전이가 된다.

예:

```text
request route-12
request route-31
```

을 concrete signature로만 보면 다르다.

하지만:

```text
request + catalog-like-target
```

같은 feature로 encode하면 structural similarity를 공유할 수 있다.

AASSR의 [Policy(정책 모델)](Policy), [Prophecy(미래 예측 모델)](Prophecy), [Critic(미래 가치 평가기)](Critic), [Skill(성공 절차 재사용)](Skills)이 relational 행동 key를 활용한다.

---

# 14. Structural root deduplication

Planning에서는 같은 relational 행동 structure를 가진 concrete aliases가 많을 수 있다.

```text
A1 ─┐
A2 ─┼→ same relational root
A3 ─┘
```

Expensive [Prophecy](Prophecy)/[Critic](Critic) computation을 한 번만 하고 결과를 aliases에 공유할 수 있다.

이것은 **계산 최적화**지만 관계 기반 표현이 있어야 가능하다.

관련 페이지:

- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)
- [Imagination](Imagination)

---

# 15. Transfer learning

Transfer learning은 한 task/distribution에서 배운 지식을 다른 관련 task/distribution에 재사용하는 것이다.

AASSR에서는:

```text
Training seeds
→ relational structure 학습
→ unseen seeds
```

의 전이가 핵심이다.

같은 표준 비교 실험 template에서 opaque identifiers를 바꾸어 concrete memorization을 어렵게 한다.

---

# 16. In-distribution generalization과 OOD

[일반화(Generalization)](Relational-Representation-and-Generalization)에도 정도가 있다.

```text
ID rename만 다른 unseen seed
→ structural in-family generalization

완전히 새로운 relation/dynamics
→ stronger OOD shift
```

Relational 표현이 identifier permutation에는 강해도 완전히 새로운 dynamics에 자동으로 일반화하는 것은 아니다.

이 한계를 명시해야 한다.

관련 페이지:

- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 17. Relational representation과 POMDP

Abstr행동이 강해지면 hidden-state aliasing이 커질 수 있다.

즉 전이와 Markov sufficiency 사이에 trade-off가 있다.

```text
더 많은 concrete detail
→ Markov information 보존
→ memorization 위험

더 강한 abstraction
→ transfer
→ state aliasing 위험
```

AASSR은 public latest status 같은 decision-critical channel을 별도 보존하여 이 trade-off를 조정한다.

관련 페이지:

- [MDP and POMDP](MDP-and-POMDP)

---

# 18. Relational representation과 World Model

[Prophecy](Prophecy)가 relational state를 예측하면 future concrete ID를 정확히 생성하지 않아도 **역할 구조의 변화**를 예측할 수 있다.

장점:

- [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) ID에 대한 전이
- output space 구조화

하지만 실제 행동 surface를 reconstruct해야 하므로 legal mask/decode contract가 중요하다.

관련 페이지:

- [Prophecy](Prophecy)
- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)

---

# 19. Relational representation과 Critic

[Critic](Critic)도 concrete ID를 암기하면 학습 중 보지 못한 branch value 전이가 약해질 수 있다.

Relational [상태 전이(transition)](MDP-and-POMDP) features를 사용하면:

```text
구체적 이름은 다름
하지만 비슷한 역할 구조
→ value pattern transfer 가능
```

을 노릴 수 있다.

하지만 [학습 분포 밖(OOD)](Critic-Support-and-OOD) extrapolation은 여전히 가능하므로 [국소 데이터 근거(local support)](Critic-Support-and-OOD)가 필요하다.

---

# 20. Relational Skill

성공 sequence를 실제 실행 행동 signature로 저장하면 새 난수 시드에서 재사용하기 어렵다.

AASSR [Skill](Skills)은:

```text
successful concrete trajectory
→ relational action templates
→ unseen state의 concrete actions에 rebind
```

한다.

관련 페이지:

- [Hierarchical RL and Skills](Hierarchical-RL-and-Skills)
- [Skills](Skills)

---

# 21. Representation effect를 왜 별도 baseline으로 보나?

AASSR Full이 raw [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)보다 좋아도 그 차이가 [Imagination(가상 미래 탐색)](Imagination) 때문인지 관계 기반 표현 때문인지 알 수 없다.

그래서:

```text
dqn_raw
→ dqn_relational
```

비교가 필요하다.

이 차이는 표현 effect를 분리한다.

그 다음:

```text
dqn_relational
→ AASSR no-Imagination
→ AASSR Full
```

로 다른 구성요소 효과를 본다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 22. Representation leakage

Relational feature를 만들 때 hidden simulator truth를 사용하면 안 된다.

예:

```text
hidden 정답 object를 알고
"target-object role" feature를 직접 생성
```

하면 표현이 사실상 oracle이 된다.

AASSR current contract는 실제 response에서 관측한 관계만 사용하도록 response-causal boundary를 둔다.

관련 페이지:

- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)

---

# 23. 핵심 오해

## "Relational이면 concrete identity를 전혀 저장하지 않는다"

아니다. 실행과 [ASEQ](ASEQ)에는 [실제 개체 구분(concrete identity)](State-Representation)가 필요하다.

## "Permutation invariance면 모든 unseen task에 일반화한다"

아니다. 이름 재배치에는 강할 수 있지만 dynamics 자체가 바뀌면 별도 [OOD](Critic-Support-and-OOD) 문제다.

## "Abstraction은 강할수록 좋다"

아니다. decision-critical information을 지우면 state aliasing이 커진다.

## "Relational DQN은 AASSR Full과 같다"

아니다. 표현 효과만 분리하는 model-free [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)이다.

---

# 24. 다음으로 읽기

- [State Representation](State-Representation)
- [MDP and POMDP](MDP-and-POMDP)
- [Critic, Support and OOD](Critic-Support-and-OOD)
- [Skills](Skills)
- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

관련 색인: **[Concept Index](Concept-Index)**