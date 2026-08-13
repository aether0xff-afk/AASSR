# Curriculum Learning

**[난이도 조절 학습(Curriculum)](Curriculum-Learning) Learning**은 처음부터 가장 어려운 문제만 주는 대신, 학습 가능한 쉬운 문제에서 시작해 점차 난도를 높이는 방법이다.

AASSR에서는 희소 보상 환경에서 **최초 성공 [경험(experience)](Replay-Buffer-and-Episode-Boundaries)를 스스로 발견할 수 있게 하면서도 정답 [경험 경로(trajectory)](Reinforcement-Learning)를 직접 주입하지 않는 방법**으로 [난이도 조절 학습(curriculum)](Curriculum-Learning)을 사용한다.

---

# 1. 왜 Curriculum이 필요한가?

아주 어려운 sparse-[보상(reward)](Sparse-Reward-and-Credit-Assignment) [환경(environment)](Reinforcement-Learning)에서는 [무작위(random)](Ablation-Benchmarking-and-Reproducibility) [탐색(exploration)](Exploration-and-Exploitation)으로 성공을 단 한 번도 찾지 못할 수 있다.

```text
Hard task
→ reward 0
→ reward 0
→ reward 0
→ ...
→ training signal 거의 없음
```

쉬운 [연구 과제(task)](Sparse-Reward-Problem)에서 먼저 성공 구조를 경험하면 [가치(value)](Value-Functions-and-Bellman-Equation)/[학습 모델(model)](Terminology-Guide)/[재사용 가능한 기술(skill)](Skills)이 학습을 시작할 수 있다.

---

# 2. 인간 학습과의 직관

보통 복잡한 문제를 처음부터 전부 풀기보다:

```text
기본 문제
→ 중간 문제
→ 복잡한 문제
```

순서로 연습한다.

난이도 조절 학습 [학습(learning)](Reinforcement-Learning)도 비슷한 직관을 machine 학습에 적용한다.

---

# 3. Difficulty level

Task family를 난도 [난이도 단계(level)](Curriculum-Learning)로 나눌 수 있다.

```text
Level 0
Level 1
Level 2
Level 3
...
```

난도는 다음과 같은 구조 변화로 만들 수 있다.

- 더 긴 required [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search)
- 더 많은 distractor [행동(action)](Reinforcement-Learning)
- 더 복잡한 dependency
- 더 강한 [부분 관측(partial observability)](MDP-and-POMDP)
- 더 큰 irreversible-risk [구조(structure)](Research-Architecture)

AASSR [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)의 실제 [난이도(difficulty)](Curriculum-Learning) [명세(contract)](Current-Status)는 [Experiments](Experiments)와 [현재(current)](Current-Status) 환경 code를 [최종 기준(source of truth)](Current-Status)로 봐야 한다.

---

# 4. Fixed curriculum

미리 정한 [학습 진행 스케줄(schedule)](Curriculum-Learning)대로 난도를 올린다.

```text
10k steps Level 0
10k steps Level 1
10k steps Level 2
```

간단하지만 [학습 주체(learner)](Terminology-Guide)가 아직 준비되지 않았거나 이미 너무 쉬워졌는데도 학습 스케줄이 고정된다.

---

# 5. Adaptive curriculum

[에이전트(Agent)](Reinforcement-Learning) [성능(performance)](Ablation-Benchmarking-and-Reproducibility)에 따라 난도를 자동으로 바꾼다.

```text
성공 안정화
→ 승급

성능 붕괴
→ 강등
```

AASSR 현재 [연구(research)](Research-Questions)에서는 자동 [다음 난이도로 승급(promotion)](Curriculum-Learning)/demotion 구조를 사용해 쉬운 환경 성공과 높은 난도의 [전이(transfer)](Relational-Representation-and-Generalization)를 연결하려 한다.

---

# 6. Promotion

일정 성능 criterion을 충족하면 더 어려운 난이도 단계로 이동한다.

예:

```text
최근 N episode success 충분
→ L0 → L1
```

Promotion criterion이 너무 느슨하면 우연한 성공 하나로 어려운 난이도 단계로 가서 학습이 붕괴할 수 있다.

너무 엄격하면 쉬운 난이도 단계에 오래 머문다.

---

# 7. Demotion

높은 난이도 단계에서 성능이 크게 떨어지면 쉬운 난이도 단계로 돌아갈 수 있다.

```text
L2 도달
→ success 거의 없음
→ L1으로 강등
```

AASSR pilot에서 실제로 자동 승급과 강등이 관찰된 적이 있다.

난이도 조절 학습 자체가 동작했다는 [증거(evidence)](Evidence-Matrix)와 [최종(final)](Ablation-Benchmarking-and-Reproducibility) [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 전이가 성공했다는 [연구 주장(claim)](Evidence-Matrix)은 분리해야 한다.

---

# 8. Curriculum과 Reward Shaping의 차이

두 방법 모두 학습을 쉽게 만들 수 있지만 방식이 다르다.

## Reward shaping

같은 연구 과제 안에서 [중간(intermediate)](Sparse-Reward-and-Credit-Assignment) 보상를 추가한다.

```text
subgoal → +0.2
```

## Curriculum

External 보상는 그대로 두고 **연구 과제 [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)의 난도**를 조절한다.

```text
쉬운 task → 어려운 task
```

AASSR은 [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment) 명세를 유지하기 위해 후자를 사용한다.

관련 페이지:

- [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 9. Guided trajectory와도 다르다

난이도 조절 학습:

```text
문제를 쉽게 함
하지만 어떤 action을 해야 하는지는 agent가 찾음
```

Guided 경험 경로:

```text
정답 action sequence 자체를 제공
```

AASSR 현재 main [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)은 난이도 조절 학습을 허용하지만 [정답을 알고 있는 기준(oracle)](Ablation-Benchmarking-and-Reproducibility)/[정답 경로로 유도된(guided)](Causality-Leakage-and-Evaluation) [성공(success)](Terminology-Guide) 경험 경로 injection을 피한다.

---

# 10. Easy task가 너무 쉬우면 생기는 문제

Level 0이 사실상 무작위 행동으로도 쉽게 성공하면 의미 있는 학습 구조를 만들지 못할 수 있다.

좋은 easy 난이도 단계은:

```text
random에는 쉽지 않음
learned exploration에는 가능
```

정도의 난도를 가지는 것이 좋다.

---

# 11. Transfer bottleneck

난이도 조절 학습에서 가장 중요한 질문은:

> 쉬운 난이도 단계에서 배운 것이 더 어려운 난이도 단계에 실제로 전이되는가?

이다.

```text
L0 success
→ L1 success?
→ L2 success?
```

AASSR 연구에서는 최초 성공 [스스로 새로운 성공 경로를 발견하는 것(discovery)](Research-Questions) 병목을 해결한 뒤 **[여러 기본 행동을 묶는 상위 수준(higher-level)](Hierarchical-RL-and-Skills) 전이 [실패(failure)](Replay-Buffer-and-Episode-Boundaries)**가 주요 병목으로 나타난 적이 있다.

---

# 12. Catastrophic forgetting

어려운 난이도 단계만 계속 학습하면 쉬운 난이도 단계에서 배운 행동을 잊을 수 있다.

이를 막기 위해 mixed [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)나 이전 난이도 단계 [표본(sample)](Ablation-Benchmarking-and-Reproducibility) 유지 같은 방법을 사용할 수 있다.

AASSR에서는 쉬운 난도 경험을 경험 재사용에 유지하는지 여부가 난이도 조절 학습 전이에 영향을 줄 수 있다.

---

# 13. Distribution shift

Level이 올라가면 [상태(state)](State-Representation)/행동 분포이 바뀐다.

```text
L0 distribution
→ L1 distribution
→ L2 distribution
```

[Prophecy(미래 예측 모델)](Prophecy)와 [Critic(미래 가치 평가기)](Critic)이 L0/L1에만 익숙하면 [더 높은 단계(higher)](Curriculum-Learning) 난이도 단계에서 [학습 분포 밖(OOD)](Critic-Support-and-OOD)가 될 수 있다.

관련 페이지:

- [Critic, Support & OOD](Critic-Support-and-OOD)
- [Relational Representation & Generalization](Relational-Representation-and-Generalization)

---

# 14. Curriculum과 World Model

쉬운 난이도 단계 [상태 전이(transition)](MDP-and-POMDP)만으로 학습한 [세계 모델(world model)](Model-Based-RL-and-World-Models)이 상위 수준 [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)를 정확히 예측한다고 보장할 수 없다.

따라서 난이도 조절 학습 승급 직후:

- [Prophecy](Prophecy) [신뢰도(reliability)](Calibration)
- [상태 코드(status)](Terminology-Guide) [예측(prediction)](Terminology-Guide)
- [가능 행동 마스크(legal-mask)](Prophecy) 예측

을 따로 확인해야 한다.

관련 페이지:

- [Prophecy](Prophecy)
- [Calibration](Calibration)

---

# 15. Curriculum과 Critic

[Critic](Critic)도 쉬운 난이도 단계 성공/실패 [누적 보상(return)](Value-Functions-and-Bellman-Equation)에만 학습되어 상위 수준 [모델이 상상한(imagined)](Research-Jargon-Guide) 상태에서 extrapolate할 수 있다.

이것이 [국소 데이터 근거(local support)](Critic-Support-and-OOD) [판정 관문(gate)](Terminology-Guide)와 연결된다.

```text
Global Critic trained
!=
Higher-level current state supported
```

---

# 16. Curriculum과 Skill

쉬운 난이도 단계의 성공 [순서열(sequence)](GRU-and-Sequence-Models)를 [Skill(성공 절차 재사용)](Skills)로 승격하면 더 높은 난이도 단계에서 긴 [더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills) 순서열를 압축할 수 있다.

하지만 더 높은 난이도 단계의 prerequisites가 다르면 [Skill](Skills)이 그대로 작동하지 않을 수 있다.

관련 페이지:

- [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)
- [Skills](Skills)

---

# 17. Curriculum leakage

난도 난이도 단계 자체가 [숨겨진(hidden)](MDP-and-POMDP) 연구 과제 구조를 직접 알려주는 [학습에 사용하는 특징(feature)](Terminology-Guide)가 되면 문제가 될 수 있다.

예:

```text
observation에 current hidden level = 3 제공
```

이 값이 실제 [응답(response)](State-Representation)에서 알 수 없는 [환경 시뮬레이터(simulator)](MDP-and-POMDP) [부가 정보(metadata)](State-Representation)라면 학습 주체 [정답 정보를 우회적으로 이용하는 지름길(shortcut)](Causality-Leakage-and-Evaluation)이 된다.

AASSR은 숨겨진 난이도 조절 학습 부가 정보를 [공개된(public)](State-Representation) [관측(observation)](MDP-and-POMDP)에 직접 주지 않는 방향을 사용한다.

관련 페이지:

- [Causality, Leakage & Evaluation](Causality-Leakage-and-Evaluation)

---

# 18. Curriculum metric

단순 최종 성공뿐 아니라:

- first 성공 상태 전이
- 난이도 단계 난이도 승급 [시간(time)](Terminology-Guide)
- maximum 난이도 단계 [도달한(reached)](Curriculum-Learning)
- demotion [횟수(count)](Terminology-Guide)
- per-level 성공
- 경험 재사용 composition
- 전이 after 난이도 승급

등을 함께 볼 수 있다.

---

# 19. Curriculum ablation

난이도 조절 학습이 실제로 도움이 되는지 보려면:

```text
Fixed hard training
vs
Adaptive curriculum
```

같은 비교를 할 수 있다.

하지만 환경 inter행동s와 total 상태 전이 [실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility)을 공정하게 맞춰야 한다.

관련 페이지:

- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 20. Automatic curriculum이 정답을 알려주는가?

자동으로 난도를 조절하는 것 자체는 정답 행동을 알려주는 것이 아니다.

다만 난이도 승급 [규칙(rule)](Terminology-Guide)이 숨겨진 [최종 목표(goal)](Sparse-Reward-Problem) [진행도(progress)](Terminology-Guide)를 너무 자세히 사용하거나 다음 정답 stage를 직접 expose하면 indirect guidance가 될 수 있다.

어떤 [학습 신호(signal)](Information-Theory-and-Intrinsic-Motivation)로 난이도를 조절하는지 명시해야 한다.

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

즉 난이도 조절 학습은 전체 문제의 끝이 아니라 **학습이 시작될 수 있는 frontier를 만들어주는 장치**다.

---

# 22. 다음으로 읽기

- [Sparse Reward Problem](Sparse-Reward-Problem)
- [Research Questions](Research-Questions)
- [Exploration & Exploitation](Exploration-and-Exploitation)
- [Critic, Support & OOD](Critic-Support-and-OOD)
- [Experiments](Experiments)

관련 색인: **[Concept Index](Concept-Index)**