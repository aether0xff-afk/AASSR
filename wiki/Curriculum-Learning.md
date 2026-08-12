# Curriculum Learning

**Curriculum Learning**은 처음부터 가장 어려운 문제만 주는 대신, 학습 가능한 쉬운 문제에서 시작해 점차 난도를 높이는 방법이다.

AASSR에서는 희소 보상 환경에서 **최초 성공 experience를 스스로 발견할 수 있게 하면서도 정답 trajectory를 직접 주입하지 않는 방법**으로 curriculum을 사용한다.

---

# 1. 왜 Curriculum이 필요한가?

아주 어려운 sparse-reward environment에서는 random exploration으로 성공을 단 한 번도 찾지 못할 수 있다.

```text
Hard task
→ reward 0
→ reward 0
→ reward 0
→ ...
→ training signal 거의 없음
```

쉬운 task에서 먼저 성공 구조를 경험하면 value/model/skill이 학습을 시작할 수 있다.

---

# 2. 인간 학습과의 직관

보통 복잡한 문제를 처음부터 전부 풀기보다:

```text
기본 문제
→ 중간 문제
→ 복잡한 문제
```

순서로 연습한다.

Curriculum learning도 비슷한 직관을 machine learning에 적용한다.

---

# 3. Difficulty level

Task family를 난도 level로 나눌 수 있다.

```text
Level 0
Level 1
Level 2
Level 3
...
```

난도는 다음과 같은 구조 변화로 만들 수 있다.

- 더 긴 required horizon
- 더 많은 distractor action
- 더 복잡한 dependency
- 더 강한 partial observability
- 더 큰 irreversible-risk structure

AASSR benchmark의 실제 difficulty contract는 [Experiments](Experiments)와 current environment code를 source of truth로 봐야 한다.

---

# 4. Fixed curriculum

미리 정한 schedule대로 난도를 올린다.

```text
10k steps Level 0
10k steps Level 1
10k steps Level 2
```

간단하지만 learner가 아직 준비되지 않았거나 이미 너무 쉬워졌는데도 schedule이 고정된다.

---

# 5. Adaptive curriculum

Agent performance에 따라 난도를 자동으로 바꾼다.

```text
성공 안정화
→ 승급

성능 붕괴
→ 강등
```

AASSR current research에서는 자동 promotion/demotion 구조를 사용해 쉬운 환경 성공과 높은 난도의 transfer를 연결하려 한다.

---

# 6. Promotion

일정 performance criterion을 충족하면 더 어려운 level로 이동한다.

예:

```text
최근 N episode success 충분
→ L0 → L1
```

Promotion criterion이 너무 느슨하면 우연한 성공 하나로 어려운 level로 가서 학습이 붕괴할 수 있다.

너무 엄격하면 쉬운 level에 오래 머문다.

---

# 7. Demotion

높은 level에서 성능이 크게 떨어지면 쉬운 level로 돌아갈 수 있다.

```text
L2 도달
→ success 거의 없음
→ L1으로 강등
```

AASSR pilot에서 실제로 자동 승급과 강등이 관찰된 적이 있다.

Curriculum 자체가 동작했다는 evidence와 final unseen transfer가 성공했다는 claim은 분리해야 한다.

---

# 8. Curriculum과 Reward Shaping의 차이

두 방법 모두 학습을 쉽게 만들 수 있지만 방식이 다르다.

## Reward shaping

같은 task 안에서 intermediate reward를 추가한다.

```text
subgoal → +0.2
```

## Curriculum

External reward는 그대로 두고 **task distribution의 난도**를 조절한다.

```text
쉬운 task → 어려운 task
```

AASSR은 sparse reward contract를 유지하기 위해 후자를 사용한다.

관련 페이지:

- [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 9. Guided trajectory와도 다르다

Curriculum:

```text
문제를 쉽게 함
하지만 어떤 action을 해야 하는지는 agent가 찾음
```

Guided trajectory:

```text
정답 action sequence 자체를 제공
```

AASSR current main protocol은 curriculum을 허용하지만 oracle/guided success trajectory injection을 피한다.

---

# 10. Easy task가 너무 쉬우면 생기는 문제

Level 0이 사실상 random action으로도 쉽게 성공하면 의미 있는 learning structure를 만들지 못할 수 있다.

좋은 easy level은:

```text
random에는 쉽지 않음
learned exploration에는 가능
```

정도의 난도를 가지는 것이 좋다.

---

# 11. Transfer bottleneck

Curriculum에서 가장 중요한 질문은:

> 쉬운 level에서 배운 것이 더 어려운 level에 실제로 transfer되는가?

이다.

```text
L0 success
→ L1 success?
→ L2 success?
```

AASSR 연구에서는 최초 성공 discovery 병목을 해결한 뒤 **higher-level transfer failure**가 주요 병목으로 나타난 적이 있다.

---

# 12. Catastrophic forgetting

어려운 level만 계속 학습하면 쉬운 level에서 배운 행동을 잊을 수 있다.

이를 막기 위해 mixed replay나 이전 level sample 유지 같은 방법을 사용할 수 있다.

AASSR에서는 쉬운 난도 경험을 replay에 유지하는지 여부가 curriculum transfer에 영향을 줄 수 있다.

---

# 13. Distribution shift

Level이 올라가면 state/action distribution이 바뀐다.

```text
L0 distribution
→ L1 distribution
→ L2 distribution
```

Prophecy와 Critic이 L0/L1에만 익숙하면 higher level에서 OOD가 될 수 있다.

관련 페이지:

- [Critic, Support & OOD](Critic-Support-and-OOD)
- [Relational Representation & Generalization](Relational-Representation-and-Generalization)

---

# 14. Curriculum과 World Model

쉬운 level transition만으로 학습한 world model이 higher-level dynamics를 정확히 예측한다고 보장할 수 없다.

따라서 curriculum 승급 직후:

- Prophecy reliability
- status prediction
- legal-mask prediction

을 따로 확인해야 한다.

관련 페이지:

- [Prophecy](Prophecy)
- [Calibration](Calibration)

---

# 15. Curriculum과 Critic

Critic도 쉬운 level success/failure return에만 학습되어 higher-level imagined state에서 extrapolate할 수 있다.

이것이 local support gate와 연결된다.

```text
Global Critic trained
!=
Higher-level current state supported
```

---

# 16. Curriculum과 Skill

쉬운 level의 성공 sequence를 Skill로 승격하면 higher level에서 긴 primitive sequence를 압축할 수 있다.

하지만 higher level의 prerequisites가 다르면 Skill이 그대로 작동하지 않을 수 있다.

관련 페이지:

- [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)
- [Skills](Skills)

---

# 17. Curriculum leakage

난도 level 자체가 hidden task structure를 직접 알려주는 feature가 되면 문제가 될 수 있다.

예:

```text
observation에 current hidden level = 3 제공
```

이 값이 실제 response에서 알 수 없는 simulator metadata라면 learner shortcut이 된다.

AASSR은 hidden curriculum metadata를 public observation에 직접 주지 않는 방향을 사용한다.

관련 페이지:

- [Causality, Leakage & Evaluation](Causality-Leakage-and-Evaluation)

---

# 18. Curriculum metric

단순 최종 success뿐 아니라:

- first success transition
- level promotion time
- maximum level reached
- demotion count
- per-level success
- replay composition
- transfer after promotion

등을 함께 볼 수 있다.

---

# 19. Curriculum ablation

Curriculum이 실제로 도움이 되는지 보려면:

```text
Fixed hard training
vs
Adaptive curriculum
```

같은 비교를 할 수 있다.

하지만 environment interactions와 total transition budget을 공정하게 맞춰야 한다.

관련 페이지:

- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 20. Automatic curriculum이 정답을 알려주는가?

자동으로 난도를 조절하는 것 자체는 정답 action을 알려주는 것이 아니다.

다만 promotion rule이 hidden goal progress를 너무 자세히 사용하거나 다음 정답 stage를 직접 expose하면 indirect guidance가 될 수 있다.

어떤 signal로 difficulty를 조절하는지 명시해야 한다.

---

# 21. AASSR 연구 단계에서의 의미

AASSR의 과거 병목은 대략 다음처럼 이동했다.

```text
최초 성공 experience가 없음
        ↓ curriculum / exploration repair
쉬운 level에서는 자율 성공 발견
        ↓
higher-level transfer 실패
        ↓
representation / Prophecy / Critic / Imagination reliability 문제 분석
```

즉 curriculum은 전체 문제의 끝이 아니라 **학습이 시작될 수 있는 frontier를 만들어주는 장치**다.

---

# 22. 다음으로 읽기

- [Sparse Reward Problem](Sparse-Reward-Problem)
- [Research Questions](Research-Questions)
- [Exploration & Exploitation](Exploration-and-Exploitation)
- [Critic, Support & OOD](Critic-Support-and-OOD)
- [Experiments](Experiments)

관련 색인: **[Concept Index](Concept-Index)**