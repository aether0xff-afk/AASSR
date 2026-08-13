# Causality, Leakage and Fair Evaluation

AASSR 연구에서 성능 수치보다 먼저 지켜야 하는 것이 **정보의 시간 방향과 평가 공정성**이다.

이 페이지는 다음을 설명한다.

```text
causality
data leakage
target leakage
hindsight leakage
privileged information
train/eval contamination
same-checkpoint comparison
```

---

# 1. Causality가 왜 중요한가?

온라인 [에이전트(agent)](Reinforcement-Learning)는 현재 시점에 실제로 알고 있는 정보만으로 행동해야 한다.

```text
과거 관측
  ↓
현재 decision
  ↓
현재 action
  ↓
미래 response
```

미래 [응답(response)](State-Representation)에서 얻은 정보를 과거 [의사결정(decision)](Chance-and-Decision-Nodes) [입력(input)](Terminology-Guide)에 사용하면 실제 온라인 상황에서는 불가능한 성능을 만든다.

---

# 2. Data leakage

**Data [정보 누출(leakage)](Causality-Leakage-and-Evaluation)**는 원래 [학습 모델(model)](Terminology-Guide)이 사용할 수 없어야 하는 정보가 [학습(training)](Terminology-Guide)/[평가(evaluation)](Ablation-Benchmarking-and-Reproducibility) 입력에 들어가 성능을 부풀리는 현상을 넓게 말한다.

예:

- [검사 또는 테스트(test)](Ablation-Benchmarking-and-Reproducibility) [정답 범주 표시(label)](Loss-Functions-and-Class-Imbalance)의 일부가 [학습에 사용하는 특징(feature)](Terminology-Guide)에 포함
- 평가 [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility) 정답을 학습에 사용
- [미래(future)](Counterfactual-Planning-and-Search) [관측(observation)](MDP-and-POMDP)이 [현재(current)](Current-Status) 입력에 포함

Leakage가 있으면 높은 성능이 실제 [일반화(generalization)](Relational-Representation-and-Generalization) 능력을 의미하지 않는다.

---

# 3. Target leakage

Prediction [대상 또는 학습 목표값(target)](Terminology-Guide)과 직접적으로 연결된 미래/정답 정보가 학습 특징에 들어가는 경우다.

예:

```text
목표: 다음 status 예측
input feature에 이미 next_status 포함
```

모델은 [예측(prediction)](Terminology-Guide)을 한 것이 아니라 정답을 읽은 것이다.

AASSR에서 [숨겨진(hidden)](MDP-and-POMDP) [환경 시뮬레이터(simulator)](MDP-and-POMDP) 미래/정답 [상태(state)](State-Representation)를 [학습 주체(learner)](Terminology-Guide) [표현(representation)](Relational-Representation-and-Generalization)에 넣지 않는 이유다.

---

# 4. Hindsight leakage

행동 후 알게 된 정보를 마치 행동 전에 알고 있었던 것처럼 사용하는 정보 누출다.

AASSR [Knowledge(에피소드 지식)](Knowledge) 예:

```text
K_t
 ↓
predict(S_t,A_t)
 ↓
A_t 실행
 ↓
response에서 K_new 획득
```

잘못된 처리:

```text
K_new를 predict(S_t,A_t)의 input에 넣음
```

이것은 [경험이 들어올 때마다 갱신하는 온라인 방식(online)](Neural-Networks-and-Optimization) 시점에서는 불가능하다.

관련 페이지:

- [Knowledge](Knowledge)

---

# 5. Hindsight learning과 Hindsight leakage는 다르다

일부 RL 알고리즘은 의도적으로 [경험 경로(trajectory)](Reinforcement-Learning)를 사후 재해석한다.

예: [최종 목표(goal)](Sparse-Reward-Problem) relabeling.

그런 알고리즘은 **학습 규칙 자체가 [결과를 본 뒤 얻은 사후 정보(hindsight)](Causality-Leakage-and-Evaluation)를 허용하도록 명시되어 있다.**

반면 정보 누출는:

```text
실제 decision 시점에는 없었던 정보를
있는 것처럼 모델 입력/평가에 몰래 사용
```

하는 문제다.

둘을 혼동하면 안 된다.

---

# 6. Privileged information

Simulator가 내부적으로 아는 정보 중 실제 에이전트가 볼 수 없는 것을 privileged [정보(information)](Information-Theory-and-Intrinsic-Motivation)이라고 볼 수 있다.

예:

- 숨겨진 [난이도 조절 학습(curriculum)](Curriculum-Learning) [난이도 단계(level)](Curriculum-Learning)
- [정확히 동일한(exact)](ASEQ) 숨겨진 [복구할 수 없는 실패 잠금(lockout)](Replay-Buffer-and-Episode-Boundaries) [환경 내부의 숨은 압박 값(pressure)](Causality-Leakage-and-Evaluation)
- 정확히 동일한 숨겨진 [한 번의 접속 세션(session)](Terminology-Guide) [남은 횟수 카운트다운(countdown)](Causality-Leakage-and-Evaluation)
- correct 대상/목표값 [식별 방식(identity)](State-Representation)

Researcher가 debugging용으로 이 값을 볼 수는 있다.

하지만 학습 주체 입력에 넣으면 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)가 쉬워질 수 있다.

---

# 7. Public observation contract

AASSR은 [실제 응답에서 원인 순서를 지키는(response-causal)](Causality-Leakage-and-Evaluation) [공개된(public)](State-Representation) 정보만 학습 주체가 사용하도록 관측 [명세(contract)](Current-Status)를 둔다.

허용 예:

- 실제로 관측한 [가장 최근의(latest)](Current-Status) HTTP [상태 코드(status)](Terminology-Guide)
- 실제로 발견한 route/profile/object relation
- 현재 [현재 허용된(legal)](Terminology-Guide) [행동(action)](Reinforcement-Learning) [현재 선택 가능한 영역(surface)](Terminology-Guide)

금지 예:

- 다음 행동이 성공할지 미리 알려주는 숨겨진 flag
- 실패 잠금까지 남은 정확한 숨겨진 횟수

관련 페이지:

- [State Representation](State-Representation)

---

# 8. Debug information과 Learner information

실험 로그에는 [숨은 환경 상태(hidden state)](MDP-and-POMDP)를 기록할 수도 있다.

```text
Debug log
→ 연구자가 failure 원인 분석
```

하지만:

```text
Learner observation
→ agent decision input
```

과는 분리해야 한다.

로그에 존재한다는 이유만으로 학습 모델 학습 특징로 사용하면 정보 누출가 된다.

---

# 9. Cross-episode leakage

이전 [한 번의 문제 풀이 구간(episode)](Terminology-Guide)에서 알게 된 [실제 개체를 구분하는(concrete)](State-Representation) 정답을 새 한 번의 문제 풀이 구간에 그대로 들고 가면 난수 시드 [전이(transfer)](Relational-Representation-and-Generalization)가 오염될 수 있다.

```text
Episode A에서 route-12가 target임을 앎
Episode B에서 ID permutation됨
그런데 이전 concrete memory를 정답처럼 사용
```

AASSR 현재 [Knowledge](Knowledge)는 [현재 에피소드 안에서만 유지되는(episode-local)](Knowledge) 명세를 기본으로 한다.

---

# 10. Replay leakage와 Knowledge

Replay buffer는 과거 학습 [경험(experience)](Replay-Buffer-and-Episode-Boundaries)를 저장한다.

Learner가 [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)에서 statistical pattern을 학습하는 것은 정상이다.

하지만 경험 재사용에 있는 실제 개체를 구분하는 [실제로 관측한 사실(fact)](Causality-Leakage-and-Evaluation)를 현재 한 번의 문제 풀이 구간의 [명시적인(explicit)](Causality-Leakage-and-Evaluation) [이미 알려진(known)](Terminology-Guide) 실제 관측 사실처럼 행동 선택 가능 영역에 직접 주입하면 다른 의미가 된다.

```text
Replay
= 학습 데이터

Knowledge
= 현재 episode에서 실제로 알고 있는 사실
```

관련 페이지:

- [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)
- [Knowledge](Knowledge)

---

# 11. Imagined fact와 Real fact

[세계(World)](Model-Based-RL-and-World-Models) 학습 모델이 예측한 사실은 실제 관측이 아니다.

```text
Prophecy: "다음에 token을 얻을 것 같다"
```

와:

```text
Real response: token 획득
```

은 구분해야 한다.

Imagined 실제 관측 사실를 [에피소드가 끝나도 유지되는(persistent)](Knowledge) [실제 사실에 근거한(factual)](Causality-Leakage-and-Evaluation) [기억(memory)](GRU-and-Sequence-Models)에 [실제 환경에서 관측된(real)](Research-Jargon-Guide) [환경 내부의 실제값(truth)](Causality-Leakage-and-Evaluation)로 기록하면 학습 모델 hallucination이 [지식(knowledge)](Knowledge)가 된다.

AASSR 원칙:

> 상상은 [계획(planning)](Counterfactual-Planning-and-Search)에 사용하고, 실제 사실 기반 [학습(learning)](Reinforcement-Learning)의 근거는 실제 [상태 전이(transition)](MDP-and-POMDP)으로 유지한다.

---

# 12. Model-generated training data의 위험

[모델 기반 강화학습(Model-based RL)](Model-Based-RL-and-World-Models)에서는 [모델이 상상한(imagined)](Research-Jargon-Guide) [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search)을 [정책(policy)](Policy)/[가치(value)](Value-Functions-and-Bellman-Equation) 학습에 사용할 수도 있다.

그 자체가 잘못은 아니다.

하지만 학습 모델 [편향(bias)](Ablation-Benchmarking-and-Reproducibility)가 있으면:

```text
model error
→ imagined training data
→ policy가 error를 학습
```

이 생길 수 있다.

AASSR 현재 main [비교(comparison)](Ablation-Benchmarking-and-Reproducibility)은 [Imagination(가상 미래 탐색)](Imagination)의 **계획 [다른 조건이 같을 때의 추가 기여(marginal)](Ablation-Benchmarking-and-Reproducibility) [효과(effect)](Ablation-Benchmarking-and-Reproducibility)**를 보기 위해 학습 [실제 행동 개입(intervention)](Imagination)/가상 [Policy(정책 모델)](Policy) [학습 갱신(update)](Neural-Networks-and-Optimization)를 제한한다.

---

# 13. Train/Test contamination

Test 난수 시드 또는 [최종(final)](Ablation-Benchmarking-and-Reproducibility) 표준 비교 실험 정보를 학습 모델 development에 반복적으로 사용하면 테스트가 사실상 [검증(validation)](Ablation-Benchmarking-and-Reproducibility) [집합(set)](Terminology-Guide)이 된다.

```text
Test 결과 확인
→ hyperparameter 조정
→ 같은 Test 재사용
→ 다시 조정
```

이 과정을 반복하면 최종 일반화 성능이라고 부르기 어렵다.

그래서 development, 검증, [결과를 미리 보지 않는 비공개 평가(blind)](Ablation-Benchmarking-and-Reproducibility) 평가의 구분이 필요하다.

---

# 14. Seed leakage

[무작위(Random)](Ablation-Benchmarking-and-Reproducibility) 난수 시드는 단순 RNG 숫자뿐 아니라 표준 비교 실험 식별 방식 partition 역할을 할 수 있다.

[학습(Training)](Reinforcement-Learning) 난수 시드와 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 평가 난수 시드를 명확히 분리해야 한다.

```text
Train seeds
≠
Unseen seeds
```

또 난수 시드별 opaque [식별자(identifier)](State-Representation) [이름 순서를 바꾸는 순열(permutation)](Relational-Representation-and-Generalization)이 있다면 동일 난수 시드를 반복 tuning에 쓰는 것이 어떤 정보 노출을 만들 수 있는지 고려해야 한다.

---

# 15. Same-checkpoint comparison

AASSR [Imagination](Imagination) 효과를 보려면:

```text
one training run
      ↓
checkpoint freeze
   /        \
OFF eval   ON eval
```

을 해야 한다.

잘못된 비교:

```text
OFF model 따로 training
ON model 따로 training
```

이 경우 성능 차이가 [계획기(planner)](Counterfactual-Planning-and-Search) 때문인지 학습 경험 경로 차이 때문인지 모른다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 16. Evaluation 중 학습

OFF 평가 후 ON 평가 사이에:

- 경험 재사용 학습 갱신
- 학습 모델 [기울기(gradient)](Neural-Networks-and-Optimization) 학습 갱신
- [예측 신뢰도 보정(calibration)](Calibration) reference 변경

등이 일어나면 같은 [체크포인트(checkpoint)](Reproduction) 비교가 아니다.

따라서 평가 [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)에서 어떤 stateful 학습 갱신가 가능한지 [공정성과 구현을 점검하는 감사(audit)](Causality-Leakage-and-Evaluation)해야 한다.

---

# 17. Calibration leakage

[Calibration(예측 신뢰도 보정)](Calibration) [검증용 분리 데이터(holdout)](Calibration)을 [세계 모델(world model)](Model-Based-RL-and-World-Models) 학습에 다시 사용하면 [신뢰도(reliability)](Calibration) [추정값(estimate)](Value-Functions-and-Bellman-Equation)가 낙관적으로 변할 수 있다.

또 live 한 번의 문제 풀이 구간 [Knowledge](Knowledge)를 과거 검증용 분리 데이터 상태 전이에 넣으면 사후 정보 [문맥 정보(context)](GRU-and-Sequence-Models) 정보 누출가 생길 수 있다.

그래서 AASSR 현재 예측 신뢰도 보정은 context-free/[학습을 멈춘(frozen)](Ablation-Benchmarking-and-Reproducibility) 검증용 분리 데이터 경계를 중요하게 본다.

관련 페이지:

- [Calibration](Calibration)

---

# 18. Oracle

[정답을 알고 있는 기준(Oracle)](Ablation-Benchmarking-and-Reproducibility)은 환경이 풀 수 있는지 확인하거나 upper bound를 보기 위해 정답 정보를 사용하는 특별한 [효과를 비교하기 위한 대조 조건(control)](Ablation-Benchmarking-and-Reproducibility)일 수 있다.

중요:

```text
Oracle evaluation
→ benchmark validation용

Agent training
→ oracle action injection 금지
```

정답을 아는 기준 [성공(success)](Terminology-Guide)를 학습 주체 성공처럼 보고하면 안 된다.

---

# 19. Guided trajectory

Researcher가 정답 [순서열(sequence)](GRU-and-Sequence-Models) 일부를 학습 경험로 넣으면 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [탐색(exploration)](Exploration-and-Exploitation) 문제가 크게 쉬워질 수 있다.

이 자체가 imitation/demonstration 학습 연구에서는 정당하다.

하지만 AASSR의 질문이 **[정답 경로로 유도된(guided)](Causality-Leakage-and-Evaluation) 경험 경로 없이 스스로 최초 성공을 찾는가**이므로 현재 main [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)에서는 제외한다.

---

# 20. Reward leakage

Hidden [진행도(progress)](Terminology-Guide) [변수(variable)](Terminology-Guide)을 [보상(reward)](Sparse-Reward-and-Credit-Assignment) [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)에 사용하면 학습 주체 입력에는 없더라도 [학습 목표(objective)](Terminology-Guide)를 통해 정답 구조가 전달될 수 있다.

예:

```text
hidden stage 2 진입 → +0.3
```

Learner가 숨겨진 stage를 직접 보지 않아도 보상가 그 정보를 알려준다.

AASSR이 희소한 [환경이 주는 외부(external)](Terminology-Guide) 보상를 좁게 유지하는 이유다.

관련 페이지:

- [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 21. Feature engineering과 leakage의 경계

Feature engineering 자체는 문제가 아니다.

예:

```text
observed status one-hot
```

는 공개된 정보을 유용하게 표현한 것이다.

문제는 학습 특징를 만들 때 숨겨진 환경 내부 실제값를 사용했는가다.

```text
observed public relation → relational feature  O
hidden target role       → feature             X
```

---

# 22. Causal timing table

| 정보 | 현재 행동 전에 사용 가능? |
|---|---|
| 이전 응답에서 관측한 상태 코드 | O |
| 이전 응답에서 발견한 route | O |
| 현재 행동 후 나올 상태 코드 | X |
| 숨겨진 접속 세션 남은 횟수 | X |
| 현재 한 번의 문제 풀이 구간의 실제 [Knowledge](Knowledge) | O |
| 가상 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)에서만 생긴 실제 관측 사실 | 실제 행동 입력으로 확정 사용 X |

---

# 23. Fair comparison과 compute

공정성은 정보뿐 아니라 [실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility)에도 적용된다.

비교 모델이:

- 상태 전이 실험 예산
- [환경(environment)](Reinforcement-Learning) inter행동 수
- 학습 난수 시드 수
- 평가 한 번의 문제 풀이 구간 수

에서 크게 다르면 성능 차이 해석이 어렵다.

Model-based method는 추가 [계산(compute)](Reproduction)를 많이 사용할 수 있으므로 wall [시간(time)](Terminology-Guide)/계산도 별도 보고하는 것이 좋다.

---

# 24. Diagnostic와 final claim

Development 중 작은 [진단 실험(diagnostic)](Evidence-Matrix)에서 문제가 고쳐졌다고 해서 최종 표준 비교 실험 성능 향상이 입증된 것은 아니다.

```text
unit/regression test
→ 코드 계약 확인

2k diagnostic
→ failure mechanism 확인

full benchmark
→ 성능 claim
```

이 층을 분리해야 한다.

---

# 25. AASSR의 주요 anti-leakage 원칙

```text
1. hidden simulator truth를 learner input에 주지 않는다.
2. action 후 Knowledge를 action 전 prediction에 넣지 않는다.
3. imagined fact를 real fact처럼 저장하지 않는다.
4. Oracle/guided success trajectory를 main agent training에 넣지 않는다.
5. sparse external reward에 hidden subgoal을 넣지 않는다.
6. no-Imagination/Full은 same frozen checkpoint로 비교한다.
7. final unseen benchmark를 development tuning과 분리한다.
```

---

# 26. 다음으로 읽기

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)
- [Knowledge](Knowledge)
- [State Representation](State-Representation)
- [Calibration](Calibration)
- [Experiments](Experiments)

관련 색인: **[Concept Index](Concept-Index)**