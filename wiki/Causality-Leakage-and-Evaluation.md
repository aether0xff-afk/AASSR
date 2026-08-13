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

미래 response에서 얻은 정보를 과거 decision input에 사용하면 실제 온라인 상황에서는 불가능한 성능을 만든다.

---

# 2. Data leakage

**Data leakage**는 원래 model이 사용할 수 없어야 하는 정보가 training/evaluation input에 들어가 성능을 부풀리는 현상을 넓게 말한다.

예:

- test label의 일부가 feature에 포함
- evaluation [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility) 정답을 training에 사용
- future [관측(observation)](MDP-and-POMDP)이 current input에 포함

Leakage가 있으면 높은 성능이 실제 [일반화(generalization)](Relational-Representation-and-Generalization) 능력을 의미하지 않는다.

---

# 3. Target leakage

Prediction target과 직접적으로 연결된 미래/정답 정보가 feature에 들어가는 경우다.

예:

```text
목표: 다음 status 예측
input feature에 이미 next_status 포함
```

모델은 prediction을 한 것이 아니라 정답을 읽은 것이다.

AASSR에서 hidden simulator future/정답 state를 learner [표현(representation)](Relational-Representation-and-Generalization)에 넣지 않는 이유다.

---

# 4. Hindsight leakage

행동 후 알게 된 정보를 마치 행동 전에 알고 있었던 것처럼 사용하는 leakage다.

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

이것은 online 시점에서는 불가능하다.

관련 페이지:

- [Knowledge](Knowledge)

---

# 5. Hindsight learning과 Hindsight leakage는 다르다

일부 RL 알고리즘은 의도적으로 trajectory를 사후 재해석한다.

예: goal relabeling.

그런 알고리즘은 **학습 규칙 자체가 hindsight를 허용하도록 명시되어 있다.**

반면 leakage는:

```text
실제 decision 시점에는 없었던 정보를
있는 것처럼 모델 입력/평가에 몰래 사용
```

하는 문제다.

둘을 혼동하면 안 된다.

---

# 6. Privileged information

Simulator가 내부적으로 아는 정보 중 실제 에이전트가 볼 수 없는 것을 privileged information이라고 볼 수 있다.

예:

- hidden [난이도 조절 학습(curriculum)](Curriculum-Learning) level
- exact hidden lockout pressure
- exact hidden session countdown
- correct target identity

Researcher가 debugging용으로 이 값을 볼 수는 있다.

하지만 learner input에 넣으면 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)가 쉬워질 수 있다.

---

# 7. Public observation contract

AASSR은 response-causal public information만 learner가 사용하도록 관측 contract를 둔다.

허용 예:

- 실제로 관측한 latest HTTP status
- 실제로 발견한 route/profile/object relation
- 현재 legal [행동(action)](Reinforcement-Learning) surface

금지 예:

- 다음 행동이 성공할지 미리 알려주는 hidden flag
- lockout까지 남은 정확한 hidden 횟수

관련 페이지:

- [State Representation](State-Representation)

---

# 8. Debug information과 Learner information

실험 로그에는 hidden state를 기록할 수도 있다.

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

로그에 존재한다는 이유만으로 model feature로 사용하면 leakage가 된다.

---

# 9. Cross-episode leakage

이전 episode에서 알게 된 concrete 정답을 새 episode에 그대로 들고 가면 난수 시드 [전이(transfer)](Relational-Representation-and-Generalization)가 오염될 수 있다.

```text
Episode A에서 route-12가 target임을 앎
Episode B에서 ID permutation됨
그런데 이전 concrete memory를 정답처럼 사용
```

AASSR current [Knowledge](Knowledge)는 episode-local contract를 기본으로 한다.

---

# 10. Replay leakage와 Knowledge

Replay buffer는 과거 training experience를 저장한다.

Learner가 replay에서 statistical pattern을 학습하는 것은 정상이다.

하지만 replay에 있는 concrete fact를 현재 episode의 explicit known fact처럼 행동 surface에 직접 주입하면 다른 의미가 된다.

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

World model이 예측한 사실은 실제 관측이 아니다.

```text
Prophecy: "다음에 token을 얻을 것 같다"
```

와:

```text
Real response: token 획득
```

은 구분해야 한다.

Imagined fact를 persistent factual memory에 real truth로 기록하면 model hallucination이 knowledge가 된다.

AASSR 원칙:

> 상상은 planning에 사용하고, factual learning의 근거는 real [상태 전이(transition)](MDP-and-POMDP)으로 유지한다.

---

# 12. Model-generated training data의 위험

[모델 기반 강화학습(Model-based RL)](Model-Based-RL-and-World-Models)에서는 imagined rollout을 policy/value training에 사용할 수도 있다.

그 자체가 잘못은 아니다.

하지만 model bias가 있으면:

```text
model error
→ imagined training data
→ policy가 error를 학습
```

이 생길 수 있다.

AASSR current main comparison은 [Imagination(가상 미래 탐색)](Imagination)의 **planning marginal effect**를 보기 위해 training [실제 행동 개입(intervention)](Imagination)/imagined [Policy(정책 모델)](Policy) update를 제한한다.

---

# 13. Train/Test contamination

Test 난수 시드 또는 final 표준 비교 실험 정보를 model development에 반복적으로 사용하면 test가 사실상 validation set이 된다.

```text
Test 결과 확인
→ hyperparameter 조정
→ 같은 Test 재사용
→ 다시 조정
```

이 과정을 반복하면 final 일반화 성능이라고 부르기 어렵다.

그래서 development, validation, blind evaluation의 구분이 필요하다.

---

# 14. Seed leakage

Random 난수 시드는 단순 RNG 숫자뿐 아니라 표준 비교 실험 identity partition 역할을 할 수 있다.

Training 난수 시드와 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) evaluation 난수 시드를 명확히 분리해야 한다.

```text
Train seeds
≠
Unseen seeds
```

또 난수 시드별 opaque identifier permutation이 있다면 동일 난수 시드를 반복 tuning에 쓰는 것이 어떤 정보 노출을 만들 수 있는지 고려해야 한다.

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

이 경우 성능 차이가 planner 때문인지 training trajectory 차이 때문인지 모른다.

관련 페이지:

- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 16. Evaluation 중 학습

OFF 평가 후 ON 평가 사이에:

- replay update
- model gradient update
- calibration reference 변경

등이 일어나면 같은 [체크포인트(checkpoint)](Reproduction) 비교가 아니다.

따라서 evaluation mode에서 어떤 stateful update가 가능한지 audit해야 한다.

---

# 17. Calibration leakage

[Calibration(예측 신뢰도 보정)](Calibration) [검증용 분리 데이터(holdout)](Calibration)을 [세계 모델(world model)](Model-Based-RL-and-World-Models) training에 다시 사용하면 reliability estimate가 낙관적으로 변할 수 있다.

또 live episode [Knowledge](Knowledge)를 과거 검증용 분리 데이터 상태 전이에 넣으면 hindsight context leakage가 생길 수 있다.

그래서 AASSR current calibration은 context-free/frozen 검증용 분리 데이터 경계를 중요하게 본다.

관련 페이지:

- [Calibration](Calibration)

---

# 18. Oracle

Oracle은 환경이 풀 수 있는지 확인하거나 upper bound를 보기 위해 정답 정보를 사용하는 특별한 control일 수 있다.

중요:

```text
Oracle evaluation
→ benchmark validation용

Agent training
→ oracle action injection 금지
```

Oracle success를 learner success처럼 보고하면 안 된다.

---

# 19. Guided trajectory

Researcher가 정답 sequence 일부를 training experience로 넣으면 sparse [탐색(exploration)](Exploration-and-Exploitation) 문제가 크게 쉬워질 수 있다.

이 자체가 imitation/demonstration learning 연구에서는 정당하다.

하지만 AASSR의 질문이 **guided trajectory 없이 스스로 최초 성공을 찾는가**이므로 current main protocol에서는 제외한다.

---

# 20. Reward leakage

Hidden progress variable을 [보상(reward)](Sparse-Reward-and-Credit-Assignment) shaping에 사용하면 learner input에는 없더라도 objective를 통해 정답 구조가 전달될 수 있다.

예:

```text
hidden stage 2 진입 → +0.3
```

Learner가 hidden stage를 직접 보지 않아도 보상가 그 정보를 알려준다.

AASSR이 sparse external 보상를 좁게 유지하는 이유다.

관련 페이지:

- [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 21. Feature engineering과 leakage의 경계

Feature engineering 자체는 문제가 아니다.

예:

```text
observed status one-hot
```

는 public information을 유용하게 표현한 것이다.

문제는 feature를 만들 때 hidden truth를 사용했는가다.

```text
observed public relation → relational feature  O
hidden target role       → feature             X
```

---

# 22. Causal timing table

| 정보 | 현재 행동 전에 사용 가능? |
|---|---|
| 이전 response에서 관측한 status | O |
| 이전 response에서 발견한 route | O |
| 현재 행동 후 나올 status | X |
| hidden session countdown | X |
| current episode의 real [Knowledge](Knowledge) | O |
| imagined branch에서만 생긴 fact | real 행동 input으로 확정 사용 X |

---

# 23. Fair comparison과 compute

공정성은 정보뿐 아니라 budget에도 적용된다.

비교 모델이:

- 상태 전이 budget
- [환경(environment)](Reinforcement-Learning) inter행동 수
- training 난수 시드 수
- evaluation episode 수

에서 크게 다르면 성능 차이 해석이 어렵다.

Model-based method는 추가 compute를 많이 사용할 수 있으므로 wall time/compute도 별도 보고하는 것이 좋다.

---

# 24. Diagnostic와 final claim

Development 중 작은 diagnostic에서 문제가 고쳐졌다고 해서 final 표준 비교 실험 성능 향상이 입증된 것은 아니다.

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