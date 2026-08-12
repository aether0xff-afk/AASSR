# Hierarchical Reinforcement Learning and Skills

**Hierarchical Reinforcement Learning(HRL)** 은 긴 문제를 여러 시간 규모의 행동 단위로 나누는 강화학습 연구 방향이다.

AASSR의 [Skills](Skills)는 이 문제와 연결되지만, 사람이 정답 macro를 미리 제공하는 방식이 아니라 **반복 성공한 실제 ASeq를 relational template로 승격**한다.

---

# 1. Primitive action

Environment가 직접 받아들이는 가장 기본적인 action을 primitive action이라고 하자.

예:

```text
request route
login
request object
submit state change
```

긴 문제를 primitive만으로 풀면 매번 여러 step sequence를 다시 구성해야 한다.

---

# 2. Temporal abstraction

여러 primitive action을 하나의 고수준 행동 단위로 묶는 것을 temporal abstraction이라고 볼 수 있다.

```text
Primitive:
A1 → A2 → A3 → A4

High-level:
Skill X
```

고수준 planner는 `Skill X`를 하나의 선택처럼 다룰 수 있다.

---

# 3. Macro action

정해진 action sequence를 하나의 macro로 묶을 수 있다.

```text
Macro M = [A1,A2,A3]
```

장점:

- 긴 sequence 재사용
- planning horizon 축소

하지만 concrete ID를 그대로 macro에 넣으면 transfer가 약하다.

AASSR은 raw macro보다 relational template를 사용한다.

---

# 4. Option framework

Hierarchical RL에서 유명한 개념 중 하나가 **option**이다.

일반적으로 option은:

```math
(I,\pi,\beta)
```

로 표현할 수 있다.

- `I`: initiation set, 시작 가능한 상태
- `π`: option 내부 policy
- `β`: termination condition

즉 단순 고정 sequence보다 일반적인 temporally extended action이다.

AASSR Skill이 고전적인 option framework와 동일한 구현이라는 뜻은 아니다.

하지만 "여러 primitive를 재사용 가능한 고수준 행동으로 만든다"는 연구 배경은 연결된다.

---

# 5. Skill

넓은 RL 문맥에서 skill은 재사용 가능한 행동 패턴/subpolicy를 의미할 수 있다.

AASSR current Skill은 더 구체적이다.

```text
실제 성공 trajectory
   ↓
relational ASeq template 추출
   ↓
반복 성공 evidence
   ↓
promotion
   ↓
현재 action surface에 concrete rebinding
```

관련 페이지:

- [Skills](Skills)
- [ASEQ](ASEQ)

---

# 6. 왜 Skill이 sample efficiency를 높일 수 있나?

이미 여러 번 성공한 sequence를 매번 random exploration으로 다시 발견할 필요가 없어진다.

```text
처음:
A1 → A2 → A3 → A4

다음:
Skill X
```

긴 horizon의 일부를 재사용하면 higher-level exploration이 가능해진다.

---

# 7. 사람이 Skill을 넣어주는 방법

가장 쉬운 방식:

```text
researcher knows correct sequence
→ macro/skill로 직접 제공
```

실용적으로는 유용할 수 있다.

하지만 AASSR 연구에서는 "정답 수행 과정을 인간이 미리 주입하지 않는다"는 원칙이 중요하다.

그래서 current Skill은 **실제 agent 성공 experience에서만 promotion**된다.

---

# 8. Skill discovery

Skill을 자동으로 발견하는 문제를 skill discovery라고 한다.

가능한 접근:

- 자주 반복되는 subtrajectory
- bottleneck state
- diversity objective
- eigenoptions
- unsupervised skill discovery
- goal-conditioned behavior

AASSR은 그중 **goal completion에 실제로 반복 기여한 relational ASeq**를 promotion하는 좁고 auditable한 방식을 사용한다.

---

# 9. Promotion threshold

한 번 우연히 성공한 sequence를 바로 skill로 만들면 noise를 고정할 수 있다.

```text
1회 성공
→ 우연일 수 있음

반복 성공
→ 더 강한 evidence
```

그래서 promotion threshold가 필요하다.

Threshold가 낮으면 빠르게 skill이 생기지만 false promotion 위험이 커진다.

높으면 안정적이지만 sample이 많이 필요하다.

---

# 10. Relational Skill

Concrete sequence:

```text
request route-12
login profile-4
request object-7
```

를 그대로 저장하면 unseen seed에서 ID가 바뀌었을 때 쓸 수 없다.

Relational template:

```text
request [catalog-like route]
login [credential-bearing profile role]
request [target-like object role]
```

처럼 구조를 저장하면 새 concrete action에 rebind할 수 있다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)

---

# 11. Initiation condition과 AASSR Skill

고전 option의 initiation set처럼, AASSR Skill도 모든 state에서 실행 가능한 것은 아니다.

각 template step에 맞는 concrete legal action이 현재 action surface에 있어야 한다.

```text
현재 state
→ template T0 matching action 있음?
→ 실행
→ 다음 state
→ template T1 matching action 있음?
```

중간에 matching primitive가 없으면 unavailable해진다.

---

# 12. Open-loop macro의 위험

고정 sequence를 실제 outcome을 확인하지 않고 끝까지 실행하면 위험하다.

```text
A1 실행
→ 예상과 다른 state
→ 그래도 A2,A3 강제 실행
```

AASSR Skill execution/prediction은 current state에 맞는 concrete primitive를 step마다 resolve하는 구조를 가져, 단순 raw script replay와 차이가 있다.

---

# 13. Skill과 stochasticity

Skill 내부 action 하나마다 여러 outcome이 가능하면 전체 Skill 결과도 여러 branch가 된다.

```text
Skill = A1,A2

A1
 ├→ S1a
 │    └→ A2 → ...
 └→ S1b
      └→ A2 → ...
```

각 step에서 best outcome 하나만 선택하면 stochastic risk를 잃을 수 있다.

AASSR current Skill Prophecy는 여러 outcome을 작은 beam으로 유지한다.

---

# 14. Outcome mass와 Skill reliability

Skill branch에서도:

```text
outcome mass
!=
prediction reliability
```

다.

Sequence가 길어질수록 primitive prediction reliability가 누적되어 전체 Skill confidence가 낮아질 수 있다.

동시에 stochastic outcome mass도 branch별로 별도로 추적해야 한다.

관련 페이지:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 15. Skill beam

Primitive마다 `M`개의 stochastic outcome이 있고 Skill 길이가 `L`이면 naive branch 수는 `M^L`로 늘어날 수 있다.

그래서 일부 branch만 유지한다.

```text
candidate branches 생성
→ 중요한 mass/reliability branch 유지
→ retained mass renormalize
```

이는 exact planning의 근사다.

---

# 16. Hierarchy의 장점

- 긴 horizon 압축
- 반복되는 해결 구조 재사용
- higher-level planning 가능
- unseen concrete ID에 relational transfer 가능

---

# 17. Hierarchy의 위험

## Premature abstraction

잘못된 sequence를 skill로 만들어 반복.

## Skill domination

높은 estimated value의 skill만 계속 선택해 primitive exploration이 사라짐.

## Context mismatch

훈련에서는 성공했던 skill이 새 state에서는 전제가 다름.

## Error compounding

긴 skill rollout에서 Prophecy error 누적.

---

# 18. Skill과 Curriculum

쉬운 환경에서 발견한 Skill이 높은 난도로 transfer되면 curriculum progression을 도울 수 있다.

하지만:

```text
낮은 level success skill
→ 높은 level에서도 sufficient?
```

는 자동으로 성립하지 않는다.

AASSR의 transfer bottleneck에서 바로 이런 질문이 중요하다.

---

# 19. Skill과 창의성

Skill 재사용은 창의성과 동일하지 않다.

```text
기존 성공 sequence를 재사용
```

과:

```text
새로운 유효한 solution path를 구성
```

은 다르다.

AASSR의 장기 연구 질문 중 "인간이 미리 준 경로와 다른 해결 과정을 만들 수 있는가?"를 분석하려면 Skill reuse와 novel composition을 분리해야 한다.

---

# 20. Skill ablation

Skill의 효과를 보려면:

```text
same base agent
Skill OFF
vs
Skill ON
```

같은 control이 필요하다.

함께 볼 metric:

- skill promotion count
- skill execution count
- successful skill execution
- failed/unavailable skill
- unseen seed rebinding rate
- primitive-only success와 비교

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 21. AASSR 연결 요약

```text
Real successful ASeq
       ↓
Relational templates
       ↓
Repeated evidence
       ↓
Skill promotion
       ↓
New state에서 concrete rebind
       ↓
Policy / Prophecy / Imagination candidate
```

---

# 22. 다음으로 읽기

- [Skills](Skills)
- [ASEQ](ASEQ)
- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [Prophecy](Prophecy)
- [Research Questions](Research-Questions)

관련 색인: **[Concept Index](Concept-Index)**