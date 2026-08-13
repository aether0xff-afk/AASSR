# Hierarchical Reinforcement Learning and Skills

**Hierarchical [강화학습(Reinforcement Learning)](Reinforcement-Learning)(HRL)** 은 긴 문제를 여러 시간 규모의 행동 단위로 나누는 강화학습 연구 방향이다.

AASSR의 [Skills](Skills)는 이 문제와 연결되지만, 사람이 정답 [여러 행동을 묶은 상위 행동(macro)](Hierarchical-RL-and-Skills)를 미리 제공하는 방식이 아니라 **반복 성공한 실제 ASeq를 [관계 기반(relational)](Relational-Representation-and-Generalization) [재사용 가능한 틀(template)](Skills)로 승격**한다.

---

# 1. Primitive action

[환경(Environment)](Reinforcement-Learning)가 직접 받아들이는 가장 기본적인 [행동(action)](Reinforcement-Learning)을 [더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills) 행동이라고 하자.

예:

```text
request route
login
request object
submit state change
```

긴 문제를 기본 행동 단위만으로 풀면 매번 여러 [단계(step)](Terminology-Guide) [순서열(sequence)](GRU-and-Sequence-Models)를 다시 구성해야 한다.

---

# 2. Temporal abstraction

여러 기본 행동 단위 행동을 하나의 고수준 행동 단위로 묶는 것을 [시간 순서를 고려하는(temporal)](GRU-and-Sequence-Models) abstr행동이라고 볼 수 있다.

```text
Primitive:
A1 → A2 → A3 → A4

High-level:
Skill X
```

고수준 [계획기(planner)](Counterfactual-Planning-and-Search)는 `Skill X`를 하나의 선택처럼 다룰 수 있다.

---

# 3. Macro action

정해진 행동 순서열를 하나의 행동 묶음로 묶을 수 있다.

```text
Macro M = [A1,A2,A3]
```

장점:

- 긴 순서열 재사용
- [계획(planning)](Counterfactual-Planning-and-Search) [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search) 축소

하지만 [실제 개체를 구분하는(concrete)](State-Representation) ID를 그대로 행동 묶음에 넣으면 [전이(transfer)](Relational-Representation-and-Generalization)가 약하다.

AASSR은 [가공하지 않은 원본(raw)](State-Representation) 행동 묶음보다 관계 기반 재사용 가능한 틀를 사용한다.

---

# 4. Option framework

Hierarchical RL에서 유명한 개념 중 하나가 **[여러 기본 행동을 묶은 상위 행동 단위(option)](Hierarchical-RL-and-Skills)**이다.

일반적으로 상위 행동 단위은:

```math
(I,\pi,\beta)
```

로 표현할 수 있다.

- `I`: initiation [집합(set)](Terminology-Guide), 시작 가능한 상태
- `π`: 상위 행동 단위 내부 [정책(policy)](Policy)
- `β`: termination [실험 조건(condition)](Ablation-Benchmarking-and-Reproducibility)

즉 단순 고정 순서열보다 일반적인 temporally extended 행동이다.

AASSR [Skill(성공 절차 재사용)](Skills)이 고전적인 상위 행동 단위 [문제 표현 틀(framework)](Terminology-Guide)와 동일한 구현이라는 뜻은 아니다.

하지만 "여러 기본 행동 단위를 재사용 가능한 고수준 행동으로 만든다"는 연구 배경은 연결된다.

---

# 5. Skill

넓은 RL 문맥에서 [재사용 가능한 기술(skill)](Skills)은 재사용 가능한 행동 패턴/subpolicy를 의미할 수 있다.

AASSR [현재(current)](Current-Status) [Skill](Skills)은 더 구체적이다.

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

이미 여러 번 성공한 순서열를 매번 [무작위(random)](Ablation-Benchmarking-and-Reproducibility) [탐색(exploration)](Exploration-and-Exploitation)으로 다시 발견할 필요가 없어진다.

```text
처음:
A1 → A2 → A3 → A4

다음:
Skill X
```

긴 미래 탐색 범위의 일부를 재사용하면 [여러 기본 행동을 묶는 상위 수준(higher-level)](Hierarchical-RL-and-Skills) 탐색이 가능해진다.

---

# 7. 사람이 Skill을 넣어주는 방법

가장 쉬운 방식:

```text
researcher knows correct sequence
→ macro/skill로 직접 제공
```

실용적으로는 유용할 수 있다.

하지만 AASSR 연구에서는 "정답 수행 과정을 인간이 미리 주입하지 않는다"는 원칙이 중요하다.

그래서 현재 [Skill](Skills)은 **실제 [에이전트(agent)](Reinforcement-Learning) 성공 [경험(experience)](Replay-Buffer-and-Episode-Boundaries)에서만 [다음 난이도로 승급(promotion)](Curriculum-Learning)**된다.

---

# 8. Skill discovery

[Skill](Skills)을 자동으로 발견하는 문제를 기술 [스스로 새로운 성공 경로를 발견하는 것(discovery)](Research-Questions)라고 한다.

가능한 접근:

- 자주 반복되는 subtrajectory
- bottleneck [상태(state)](State-Representation)
- diversity [학습 목표(objective)](Terminology-Guide)
- eigenoptions
- unsupervised 기술 발견
- goal-conditioned [행동 양상(behavior)](Experiments)

AASSR은 그중 **[최종 목표(goal)](Sparse-Reward-Problem) completion에 실제로 반복 기여한 관계 기반 ASeq**를 난이도 승급하는 좁고 auditable한 방식을 사용한다.

---

# 9. Promotion threshold

한 번 우연히 성공한 순서열를 바로 기술로 만들면 [잡음(noise)](Stochasticity-Uncertainty-and-Probability)를 고정할 수 있다.

```text
1회 성공
→ 우연일 수 있음

반복 성공
→ 더 강한 evidence
```

그래서 난이도 승급 [판정 기준값(threshold)](Terminology-Guide)가 필요하다.

Threshold가 낮으면 빠르게 기술이 생기지만 false 난이도 승급 위험이 커진다.

높으면 안정적이지만 [표본(sample)](Ablation-Benchmarking-and-Reproducibility)이 많이 필요하다.

---

# 10. Relational Skill

[실제 개체를 구분하는(Concrete)](State-Representation) 순서열:

```text
request route-12
login profile-4
request object-7
```

를 그대로 저장하면 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)에서 ID가 바뀌었을 때 쓸 수 없다.

[관계 기반(Relational)](Relational-Representation-and-Generalization) 재사용 가능한 틀:

```text
request [catalog-like route]
login [credential-bearing profile role]
request [target-like object role]
```

처럼 구조를 저장하면 새 [실제 실행 행동(concrete action)](State-Representation)에 rebind할 수 있다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)

---

# 11. Initiation condition과 AASSR Skill

고전 상위 행동 단위의 initiation 집합처럼, AASSR [Skill](Skills)도 모든 상태에서 실행 가능한 것은 아니다.

각 재사용 가능한 틀 단계에 맞는 실제 개체를 구분하는 [현재 허용된(legal)](Terminology-Guide) 행동이 현재 행동 [현재 선택 가능한 영역(surface)](Terminology-Guide)에 있어야 한다.

```text
현재 state
→ template T0 matching action 있음?
→ 실행
→ 다음 state
→ template T1 matching action 있음?
```

중간에 matching 기본 행동 단위가 없으면 [현재 사용할 수 없는(unavailable)](Terminology-Guide)해진다.

---

# 12. Open-loop macro의 위험

고정 순서열를 실제 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)을 확인하지 않고 끝까지 실행하면 위험하다.

```text
A1 실행
→ 예상과 다른 state
→ 그래도 A2,A3 강제 실행
```

AASSR [Skill](Skills) [실제 실행(execution)](Research-Jargon-Guide)/[예측(prediction)](Terminology-Guide)은 현재 상태에 맞는 실제 개체를 구분하는 기본 행동 단위를 단계마다 resolve하는 구조를 가져, 단순 원본 script [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)와 차이가 있다.

---

# 13. Skill과 stochasticity

[Skill](Skills) 내부 행동 하나마다 여러 환경 결과이 가능하면 전체 [Skill](Skills) 결과도 여러 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)가 된다.

```text
Skill = A1,A2

A1
 ├→ S1a
 │    └→ A2 → ...
 └→ S1b
      └→ A2 → ...
```

각 단계에서 best 환경 결과 하나만 선택하면 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) risk를 잃을 수 있다.

AASSR 현재 [Skill](Skills) [Prophecy(미래 예측 모델)](Prophecy)는 여러 환경 결과을 작은 [유망 후보만 남기는 빔 탐색(beam)](Counterfactual-Planning-and-Search)으로 유지한다.

---

# 14. Outcome mass와 Skill reliability

[Skill](Skills) 결과 경로에서도:

```text
outcome mass
!=
prediction reliability
```

다.

Sequence가 길어질수록 기본 행동 단위 [예측 신뢰도(prediction reliability)](Calibration)가 누적되어 전체 [Skill](Skills) [예측 신뢰 정도(confidence)](Calibration)가 낮아질 수 있다.

동시에 확률적 환경 결과 [확률 질량(mass)](Stochasticity-Uncertainty-and-Probability)도 결과 경로별로 별도로 추적해야 한다.

관련 페이지:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 15. Skill beam

Primitive마다 `M`개의 확률적 환경 결과이 있고 [Skill](Skills) 길이가 `L`이면 naive 결과 경로 수는 `M^L`로 늘어날 수 있다.

그래서 일부 결과 경로만 유지한다.

```text
candidate branches 생성
→ 중요한 mass/reliability branch 유지
→ retained mass renormalize
```

이는 [정확히 동일한(exact)](ASEQ) 계획의 근사다.

---

# 16. Hierarchy의 장점

- 긴 미래 탐색 범위 압축
- 반복되는 해결 구조 재사용
- 상위 수준 계획 가능
- 학습 중 보지 못한 실제 개체를 구분하는 ID에 관계 기반 전이 가능

---

# 17. Hierarchy의 위험

## Premature abstraction

잘못된 순서열를 기술로 만들어 반복.

## Skill domination

높은 estimated [가치(value)](Value-Functions-and-Bellman-Equation)의 기술만 계속 선택해 기본 행동 단위 탐색이 사라짐.

## Context mismatch

훈련에서는 성공했던 기술이 새 상태에서는 전제가 다름.

## Error compounding

긴 기술 [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search)에서 [Prophecy](Prophecy) [오차(error)](Loss-Functions-and-Class-Imbalance) 누적.

---

# 18. Skill과 Curriculum

쉬운 환경에서 발견한 [Skill](Skills)이 높은 난도로 전이되면 [난이도 조절 학습(curriculum)](Curriculum-Learning) progression을 도울 수 있다.

하지만:

```text
낮은 level success skill
→ 높은 level에서도 sufficient?
```

는 자동으로 성립하지 않는다.

AASSR의 전이 bottleneck에서 바로 이런 질문이 중요하다.

---

# 19. Skill과 창의성

[Skill](Skills) 재사용은 창의성과 동일하지 않다.

```text
기존 성공 sequence를 재사용
```

과:

```text
새로운 유효한 solution path를 구성
```

은 다르다.

AASSR의 장기 연구 질문 중 "인간이 미리 준 경로와 다른 해결 과정을 만들 수 있는가?"를 분석하려면 [Skill](Skills) reuse와 novel composition을 분리해야 한다.

---

# 20. Skill ablation

[Skill](Skills)의 효과를 보려면:

```text
same base agent
Skill OFF
vs
Skill ON
```

같은 [효과를 비교하기 위한 대조 조건(control)](Ablation-Benchmarking-and-Reproducibility)이 필요하다.

함께 볼 [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility):

- 기술 난이도 승급 [횟수(count)](Terminology-Guide)
- 기술 실행 횟수
- successful 기술 실행
- failed/현재 사용 불가 기술
- 학습 중 보지 못한 난수 시드 [새 문제의 실제 객체에 다시 연결하는 것(rebinding)](Skills) [비율(rate)](Terminology-Guide)
- primitive-only [성공(success)](Terminology-Guide)와 비교

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