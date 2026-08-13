# Replay Buffer and Episode Boundaries

이 페이지는 **[경험(experience)](Replay-Buffer-and-Episode-Boundaries) [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)**와 **[한 번의 문제 풀이 구간(episode)](Terminology-Guide) [경계(boundary)](Replay-Buffer-and-Episode-Boundaries)**, 그리고 `terminal`, `failure`, `truncation`, `reset`, `TD bootstrap`이 왜 서로 다른 개념인지 설명한다.

AASSR의 학습 메커니즘 수정에서 실제로 중요한 버그가 있었던 부분이기도 하다.

---

# 1. Replay Buffer란?

Replay buffer는 과거 경험 [상태 전이(transition)](MDP-and-POMDP)을 저장하는 메모리다.

전형적인 항목:

```text
(state, action, reward, next_state, terminal)
```

또는 더 많은 [부가 정보(metadata)](State-Representation)를 포함할 수 있다.

[DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)은 현재 바로 직전에 일어난 경험만 학습하지 않고 buffer에서 minibatch를 뽑아 여러 번 재사용한다.

---

# 2. 왜 Replay를 쓰는가?

## Sample reuse

한 번 얻은 실제 상태 전이을 여러 [기울기(gradient)](Neural-Networks-and-Optimization) [학습 갱신(update)](Neural-Networks-and-Optimization)에 재사용할 수 있다.

## Temporal correlation 완화

실제 [경험 경로(trajectory)](Reinforcement-Learning)는:

```text
S0 → S1 → S2 → S3
```

처럼 이웃 [표본(sample)](Ablation-Benchmarking-and-Reproducibility)끼리 매우 비슷할 수 있다.

Replay에서 무작위로 섞어 뽑으면 minibatch correlation을 줄일 수 있다.

## Off-policy learning

[Q-learning](Q-Learning-DQN-and-TD)은 과거 [행동 양상(behavior)](Experiments) [정책(policy)](Policy)가 만든 경험도 재사용할 수 있다.

---

# 3. Replay는 Knowledge와 다르다

AASSR에서는 두 메모리를 구분해야 한다.

```text
Replay Buffer
= learner가 학습에 재사용하는 과거 transition dataset

Knowledge Store
= 현재 episode에서 실제 response를 통해 지금 알고 있는 explicit facts
```

과거 경험 재사용에 어떤 route가 있었다고 해서 새 한 번의 문제 풀이 구간의 [에이전트(agent)](Reinforcement-Learning)가 그 [실제 개체를 구분하는(concrete)](State-Representation) route를 현재 알고 있다고 취급하면 안 된다.

관련 페이지:

- [Knowledge](Knowledge)
- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)

---

# 4. Episode boundary란?

한 경험 경로가 더 이상 다음 [관측(observation)](MDP-and-POMDP)과 자연스럽게 이어지지 않는 경계다.

```text
Episode A
S0 → S1 → S2
            X boundary
Episode B
T0 → T1 → ...
```

이 경계가 있는데도 [학습 주체(learner)](Terminology-Guide)가:

```text
S2 → T0
```

를 같은 한 번의 문제 풀이 구간의 정상 상태 전이처럼 해석하면 잘못된 [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries)이 생길 수 있다.

---

# 5. Terminal

**[에피소드 종료(Terminal)](Replay-Buffer-and-Episode-Boundaries) [상태(state)](State-Representation)**는 [연구 과제(task)](Sparse-Reward-Problem) [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models) 관점에서 한 번의 문제 풀이 구간가 끝난 상태다.

예:

```text
success
true irreversible failure
```

에피소드 종료 상태 전이에서는 일반적으로 이후 [가치(value)](Value-Functions-and-Bellman-Equation)를 이어붙이지 않는다.

```math
y=r
```

---

# 6. Truncation

**Truncation**은 연구 과제 자체가 자연스럽게 끝난 것이 아니라 외부 규칙이나 제한 때문에 경험 경로가 끊긴 경우다.

예:

- [시간(time)](Terminology-Guide) [제한(limit)](Terminology-Guide)
- 상태 전이 cap
- [데이터(data)](Terminology-Guide) collection [실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility)

일반 RL에서는 [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries) 이후 underlying 연구 과제가 계속될 수 있으므로 true [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries)과 구분한다.

하지만 실제 구현에서 [환경(environment)](Reinforcement-Learning)가 즉시 [환경 초기화(reset)](Replay-Buffer-and-Episode-Boundaries)되어 **다음 관측이 완전히 새 한 번의 문제 풀이 구간라면**, 경험 재사용 연결을 그대로 다음 상태 가치 이어받기할 수는 없다.

---

# 7. Failure와 Truncation은 reward semantics가 다르다

AASSR의 대표 계약:

```text
true failure  → -1
truncation    →  0
```

왜냐하면 상태 전이 cap이나 rate-limit administrative 환경 초기화을 연구 과제 실패와 동일한 `-1`로 바꾸면 원래 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [학습 목표(objective)](Terminology-Guide)를 변형할 수 있기 때문이다.

그렇다고 외부 제한 종료 뒤 새 한 번의 문제 풀이 구간 상태까지 다음 상태 가치 이어받기해야 한다는 뜻은 아니다.

---

# 8. Reward boundary와 Bootstrap boundary

이 페이지의 핵심 구분이다.

```text
Reward semantics
= 이 transition의 task 결과가 무엇인가?

Bootstrap boundary
= next_state value를 이 transition target에 이어붙여도 되는가?
```

예:

```text
stalled reset
reward = 0
bootstrap = stop
```

이 조합이 가능하다.

즉:

```text
reward 0
!=
non-terminal bootstrap
```

이다.

---

# 9. 잘못된 bootstrap 예시

Episode A가 [진전 없이 반복하다 멈춘(stalled)](ASEQ)되어 환경 초기화됐다고 하자.

```text
Episode A 마지막 state S
  ↓ reset
Episode B 첫 state T
```

잘못 저장:

```text
(S, A, reward=0, next_state=T, terminal=False)
```

그러면 [Q-러닝(Q-learning)](Q-Learning-DQN-and-TD) [대상 또는 학습 목표값(target)](Terminology-Guide):

```math
y=0+\gamma\max_{a'}Q(T,a')
```

가 된다.

즉 **새 한 번의 문제 풀이 구간 B의 가치가 한 번의 문제 풀이 구간 A의 마지막 행동에 보상처럼 연결된다.**

이것은 [인과적으로 공정한(causal)](Causality-Leakage-and-Evaluation) 경험 경로가 아니다.

---

# 10. AASSR에서 발견된 training mismatch

과거 [사람의 정답 경로 없이 자율적인(autonomous)](Research-Questions) [학습(training)](Terminology-Guide) [경로(path)](Counterfactual-Planning-and-Search)에서는 환경 초기화이 일어나도 에피소드 종료 판단이 `next_state.available_actions` 같은 조건에 과도하게 의존해 **환경 초기화 상태 전이이 non-에피소드 종료 경험 재사용로 들어갈 수 있는 문제**가 있었다.

수리 방향:

```text
stall / rate-limit / transition-cap reset
→ reward는 0 유지
→ explicit episode boundary 설정
→ TD bootstrap만 차단
```

즉 [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment) [명세(contract)](Current-Status)는 유지하면서 학습 연결만 고쳤다.

---

# 11. Why not reward -1?

Reset을 경험했으니 `-1`을 주면 간단해 보인다.

하지만 그러면 에이전트는 원래 연구 과제 [실패(failure)](Replay-Buffer-and-Episode-Boundaries)가 아니라 **실험 runner의 administrative 제한**를 연구 과제 학습 목표로 학습할 수 있다.

```text
true task failure
!=
transition budget exhausted
```

따라서 AASSR은 [보상(reward)](Sparse-Reward-and-Credit-Assignment) [의미 규칙(semantics)](State-Representation)와 경계 의미 규칙를 분리한다.

---

# 12. Terminal mask

[DQN](Q-Learning-DQN-and-TD) 대상/목표값을 일반적으로:

```math
y=r+\gamma(1-d)\max_{a'}Q(s',a')
```

로 쓸 수 있다.

여기서 `d=1`이면 다음 상태 가치 이어받기을 끊는다.

AASSR에서 중요한 것은 이 `d`가 단순히 `reward == -1`인지 확인하는 값이 아니라 **경험 경로 continuity 명세**를 나타내야 한다는 것이다.

---

# 13. Success boundary

성공 `+1`로 한 번의 문제 풀이 구간가 끝나면:

```text
reward = +1
bootstrap = stop
```

이다.

에피소드 종료 뒤의 새 한 번의 문제 풀이 구간 가치를 더하면 성공 직전 [Q값(Q-value)](Value-Functions-and-Bellman-Equation)가 `+1`보다 더 큰 이상한 대상/목표값을 받을 수 있다.

---

# 14. True failure boundary

실제 irreversible 실패:

```text
reward = -1
bootstrap = stop
```

이다.

여기서 `-1`은 연구 과제 의미이며, 다음 상태 가치 이어받기 stop은 한 번의 문제 풀이 구간 continuity 의미다.

둘이 동시에 발생하지만 개념적으로는 별개의 축이다.

---

# 15. Stall

**Stall**은 에이전트가 의미 있는 진행을 만들지 못한 채 한 번의 문제 풀이 구간 운영 규칙에 걸린 상태일 수 있다.

AASSR [진단 실험(diagnostic)](Evidence-Matrix)에서는 [성공(success)](Terminology-Guide)/실패/외부 제한 종료/진전 없이 멈춘를 분리해 집계한다.

왜냐하면:

```text
0% success
```

만 보면 모든 한 번의 문제 풀이 구간가 실제 실패했는지, 그냥 움직이지 못했는지 알 수 없기 때문이다.

---

# 16. Rate limit reset

Rate 제한에 걸렸다는 [공개된(public)](State-Representation) [상태 코드(status)](Terminology-Guide)는 [의사결정에 중요한(decision-critical)](Calibration) 정보일 수 있다.

하지만 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility) runner가 그 뒤 한 번의 문제 풀이 구간를 환경 초기화하는 방식과 **연구 과제 보상**는 구분해야 한다.

AASSR [현재(current)](Current-Status) [표현(representation)](Relational-Representation-and-Generalization)은 [가장 최근의(latest)](Current-Status) 공개된 HTTP 상태 코드를 상태에 보존하지만, [숨겨진(hidden)](MDP-and-POMDP) [정확히 동일한(exact)](ASEQ) [남은 횟수 카운트다운(countdown)](Causality-Leakage-and-Evaluation)/[환경 내부의 숨은 압박 값(pressure)](Causality-Leakage-and-Evaluation)를 직접 학습 주체에 주지 않는다.

관련 페이지:

- [State Representation](State-Representation)
- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)

---

# 17. Replay sampling bias

Replay [데이터 묶음(dataset)](Ablation-Benchmarking-and-Reproducibility)이 특정 [행동(action)](Reinforcement-Learning)/[환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)에 과도하게 치우치면 학습 주체도 그 분포에 크게 영향을 받는다.

예:

```text
200 status 95%
403/429   5%
```

희귀 실패/상태 코드가 [학습 모델(model)](Terminology-Guide) 학습에서 묻힐 수 있다.

AASSR [Prophecy(미래 예측 모델)](Prophecy)에서는 [상태 코드까지 고려하는(status-aware)](Calibration)/balanced 학습을 별도로 사용한다.

관련 페이지:

- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)
- [Prophecy](Prophecy)

---

# 18. Holdout과 Replay

[Calibration(예측 신뢰도 보정)](Calibration)에서는 경험 재사용의 일부를 **[검증용 분리 데이터(holdout)](Calibration)**으로 분리해 [세계 모델(world model)](Model-Based-RL-and-World-Models) [신뢰도(reliability)](Calibration)를 평가할 수 있다.

```text
training replay
→ model update

holdout replay
→ model correctness/reliability 평가
```

같은 데이터를 학습과 검증에 완전히 겹쳐 쓰면 [예측 신뢰도 보정(calibration)](Calibration)이 지나치게 낙관적일 수 있다.

관련 페이지:

- [Calibration](Calibration)
- [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

---

# 19. Real transition과 imagined transition

AASSR의 기본 원칙:

```text
Replay factual data
= real environment transition
```

[Imagination(가상 미래 탐색)](Imagination)이 만든 [예측된(predicted)](Terminology-Guide) 상태 전이을 실제 경험 재사용와 같은 [환경 내부의 실제값(truth)](Causality-Leakage-and-Evaluation)로 저장해 학습 주체를 업데이트하면 학습 모델 [오차(error)](Loss-Functions-and-Class-Imbalance)가 자기증폭될 수 있다.

따라서 현재 핵심 비교에서는 [모델이 상상한(imagined)](Research-Jargon-Guide) 경험와 [실제 환경에서 관측된(real)](Research-Jargon-Guide) [증거(evidence)](Evidence-Matrix)를 분리한다.

관련 페이지:

- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)

---

# 20. Critic replay

AASSR [Critic(미래 가치 평가기)](Critic)은 단순 `(S,A,S')` 하나뿐 아니라 경험 경로 [후속 구간(suffix)](GRU-and-Sequence-Models)를 학습 example로 만든다.

```text
S0 → S1 → S2 → S3 → terminal

S0...
S1...
S2...
S3...
```

각 후속 구간를 [과거 기억을 0으로 초기화한(zero-memory)](GRU-and-Sequence-Models) [탐색의 첫 행동(root)](Imagination)로 학습해 현재 [의사결정(decision)](Chance-and-Decision-Nodes) [지점(point)](Terminology-Guide)에서의 [과거 정보를 이어가는 순환형(recurrent)](GRU-and-Sequence-Models) [학습된 모델로 값을 계산하는 추론(inference)](Neural-Networks-and-Optimization)와 맞춘다.

관련 페이지:

- [Critic](Critic)
- [GRU and Sequence Models](GRU-and-Sequence-Models)

---

# 21. Replay에서 꼭 기록해야 할 것

연구 재현성을 위해 상태 전이마다 가능한 한 다음 의미가 분리되어야 한다.

- 상태
- 행동
- [다음(next)](Terminology-Guide) 상태
- [환경이 주는 외부(external)](Terminology-Guide) 보상
- 성공/실패 환경 결과
- 에피소드 종료 여부
- 외부 제한 종료 여부
- 환경 초기화 reason
- 한 번의 문제 풀이 구간 id
- [정보의 출처 기록(provenance)](Knowledge)/[과정을 추적한 기록(trace)](Development-History) id

모든 구현이 같은 schema를 써야 한다는 뜻은 아니지만, **분석 시 의미를 복원할 수 있어야 한다.**

---

# 22. 핵심 오해

## "Reward가 0이면 그냥 non-terminal이다"

아니다. 환경 초기화 경계에서도 보상는 0일 수 있다.

## "Truncation은 failure다"

항상 아니다. 연구 과제 실패가 아니라 실험 운영상 끊긴 것일 수 있다.

## "Terminal은 reward가 -1 또는 +1일 때만 true다"

구현 계약에 따라 다르다. 중요한 것은 [미래(future)](Counterfactual-Planning-and-Search) 다음 상태 가치 이어받기이 인과적으로 공정한하게 이어질 수 있는지다.

## "Replay에 있으면 agent가 그 사실을 현재 알고 있다"

아니다. Replay는 학습 주체의 [학습 데이터(training data)](Terminology-Guide)이고 [Knowledge(에피소드 지식)](Knowledge)는 현재 한 번의 문제 풀이 구간의 [명시적인(explicit)](Causality-Leakage-and-Evaluation) [이미 알려진(known)](Terminology-Guide) facts다.

---

# 23. 다음으로 읽기

- [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD)
- [Knowledge](Knowledge)
- [Critic](Critic)
- [Calibration](Calibration)
- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)

관련 색인: **[Concept Index](Concept-Index)**