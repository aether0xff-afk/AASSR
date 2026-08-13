# Critic — 미래 가치 평가기

[Critic(미래 가치 평가기)](Critic)은 AASSR의 [Imagination](Imagination)에서 **예측된 미래가 실제 [sparse task objective](Sparse-Reward-and-Credit-Assignment) 관점에서 얼마나 가치 있는지** 평가한다.

현재 [Critic](Critic)은 [관계 기반(relational)](Relational-Representation-and-Generalization) [GRU](GRU-and-Sequence-Models) 기반이며, 실제 [한 번의 문제 풀이 구간(episode)](Terminology-Guide)에서 얻은 [discounted sparse return](Value-Functions-and-Bellman-Equation)을 학습한다.

> [!IMPORTANT]
> 현재 manifest 계약: `relational-gru-discounted-sparse-return+zero-memory-decision-suffixes+batched-train-v3`  
> 핵심 구현: `src/aassr_v2/current_return_critic.py`  
> [학습 분포 밖(OOD)](Critic-Support-and-OOD) [데이터 근거(support)](Critic-Support-and-OOD): `src/aassr_v2/current_critic_support.py`

---

# 0. 먼저 알아두면 좋은 개념

- [Value Functions & Bellman Equation](Value-Functions-and-Bellman-Equation) — [누적 보상(return)](Value-Functions-and-Bellman-Equation), discount factor, [가치(value)](Value-Functions-and-Bellman-Equation)
- [GRU & Sequence Models](GRU-and-Sequence-Models) — recurrent [숨은 환경 상태(hidden state)](MDP-and-POMDP), sequence encoding, suffix [학습(training)](Terminology-Guide)
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) identifier [전이(transfer)](Relational-Representation-and-Generalization)
- [Critic, Support & OOD](Critic-Support-and-OOD) — interpolation, extrapolation, [국소 데이터 근거(local support)](Critic-Support-and-OOD)
- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — 가치와 [신뢰도(reliability)](Calibration)/데이터 근거의 차이
- [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries) — [성공(success)](Terminology-Guide)/[실패(failure)](Replay-Buffer-and-Episode-Boundaries)/[외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)과 target semantics

---

# 1. 연구 질문

> **실제 성공과 실패가 드문 환경에서 학습한 누적 보상 [학습 모델(model)](Terminology-Guide)이 [Imagination(가상 미래 탐색)](Imagination) branch의 장기 가치를 구분할 수 있는가? 그리고 현재 [상태(state)](State-Representation)/[행동(action)](Reinforcement-Learning)에서 그 값을 믿을 실제 근거가 있는지 구분할 수 있는가?**

[Prophecy](Prophecy)는:

```text
무슨 일이 일어날까?
```

를 예측한다.

[Critic](Critic)은:

```text
그 미래는 최종 task return 관점에서 얼마나 좋은가?
```

를 평가한다.

즉:

```text
transition prediction
!=
value prediction
```

이다.

---

# 2. Reward와 Critic target

[Critic](Critic)은 사람이 만든 intermediate score를 학습하지 않는다.

기본 external outcome:

```text
success       +1
true failure  -1
truncation     0
ordinary       0
```

따라서 [Critic](Critic)은 **실제 sparse task 누적 보상**을 미래 sequence에 연결한다.

이것은 [reward shaping](Sparse-Reward-and-Credit-Assignment)과 다르다.

```text
route 발견 +0.2
```

같은 hand-crafted intermediate [보상(reward)](Sparse-Reward-and-Credit-Assignment)를 target에 추가하는 구조가 아니다.

---

# 3. Reward와 Return은 다르다

현재 보상가 `0`이어도 몇 단계 뒤 성공한다면 현재 상태의 장기 가치는 양수일 수 있다.

```text
S0 → 0
S1 → 0
S2 → 0
S3 → +1
```

[누적 보상(Return)](Value-Functions-and-Bellman-Equation):

```math
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
```

AASSR [Critic](Critic)은 이런 **장기 sparse-누적 보상 structure**를 학습한다.

자세히: [Value Functions & Bellman Equation](Value-Functions-and-Bellman-Equation)

---

# 4. Discounted return

어떤 decision point에서 [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries)까지 `n` [상태 전이(transition)](MDP-and-POMDP)s가 남았고 최종 보상만 존재한다고 단순화하면:

```math
G_s=R_{final}\gamma^{n-1}
```

이다.

따라서 같은 `+1` 성공라도 더 빠른 성공은 더 큰 discounted 누적 보상을 가질 수 있다.

반대로 true 실패 `-1`도 거리에 따라 discount된다.

`γ`가 왜 필요한지는 [discount factor](Value-Functions-and-Bellman-Equation)에서 설명한다.

---

# 5. 왜 별도 Critic이 필요한가?

[Policy(정책 모델)](Policy) [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)도 `Q(s,a)`를 학습한다.

그런데 왜 [Critic](Critic)이 또 필요한가?

```text
Policy DQN
→ 현재 real state에서 primitive action의 model-free external Q

Critic
→ predicted/imagined transition sequence의 long-horizon sparse return 평가
```

즉 두 가치 estimator의 **[입력(input)](Terminology-Guide) [명세(contract)](Current-Status)와 사용 위치**가 다르다.

[Policy](Policy)는 기본 행동 선택기이고 [Critic](Critic)은 [planner leaf/branch evaluator](Counterfactual-Planning-and-Search)에 가깝다.

---

# 6. 왜 GRU인가?

단일 `(state, action)`만으로는 최근 상태 전이 flow가 가진 문맥을 충분히 표현하지 못할 수 있다.

[POMDP](MDP-and-POMDP)에서는 과거 history가 [숨겨진(hidden)](MDP-and-POMDP) condition을 추론하는 데 도움이 되기도 한다.

[GRU](GRU-and-Sequence-Models)는 sequence를 숨겨진 [표현(representation)](Relational-Representation-and-Generalization)으로 압축한다.

```text
(S0,A0,S1)
(S1,A1,S2)
(S2,A2,S3)
       |
       v
      GRU
       |
       v
predicted sparse return
```

---

# 7. GRU hidden state는 Knowledge와 같은가?

아니다.

```text
GRU hidden state
= learned latent sequence memory

KnowledgeStore
= 실제 response에서 획득한 explicit fact + provenance
```

[Knowledge](Knowledge)는 어떤 fact를 언제 알았는지 추적할 수 있지만 [GRU(게이트 순환 유닛)](GRU-and-Sequence-Models) 숨겨진 vector는 직접 해석하기 어렵다.

둘은 서로 다른 memory mechanism이다.

---

# 8. Zero-memory planning 문제

[Imagination](Imagination)은 한 번의 문제 풀이 구간 시작점에서만 호출되지 않는다.

```text
real trajectory
S0 → S1 → S2 → S3
           ^
           여기서 planning 시작 가능
```

그런데 [Critic](Critic)이 학습에서 항상 한 번의 문제 풀이 구간 시작부터 숨겨진 memory를 누적했다고 하자.

```text
training:
h0 → h1 → h2 → h3
```

실제 [계획기(planner)](Counterfactual-Planning-and-Search)가 `S2`에서 갑자기 시작하면 `h2`를 갖고 있지 않을 수 있다.

```text
inference:
zero hidden + S2
```

Training과 inference의 recurrent-state 명세가 달라진다.

이 문제는 [GRU & Sequence Models](GRU-and-Sequence-Models)에서 일반적으로 설명한다.

---

# 9. Decision suffix training

이를 맞추기 위해 현재 [Critic](Critic)은 한 real 한 번의 문제 풀이 구간에서 모든 decision suffix를 만든다.

```text
S0 → S1 → S2 → S3 → terminal

suffix 0: S0 → S1 → S2 → S3
suffix 1: S1 → S2 → S3
suffix 2: S2 → S3
suffix 3: S3
```

각 suffix는 **zero recurrent memory**에서 시작하는 독립 학습 example이 된다.

따라서 어떤 [현재(current)](Current-Status) decision 상태에서 계획기가 시작해도 학습 condition과 더 잘 맞는다.

---

# 10. Prefix마다 root-return target

한 suffix 안에서 sequence가 점점 길어질 때 현재 design은 해당 suffix의 [탐색의 첫 행동(root)](Imagination) sparse-누적 보상 target을 유지하도록 학습한다.

개념적으로:

```text
suffix: S1 → S2 → S3 → success
root target = γ²

prefix [S1]          → γ²
prefix [S1,S2]       → γ²
prefix [S1,S2,S3]    → γ²
```

목표는 [Critic](Critic)이 **현재 [계획(planning)](Counterfactual-Planning-and-Search) 탐색의 첫 행동에서 시작했을 때의 future 누적 보상**을 sequence context가 늘어나면서 계속 추정하도록 만드는 것이다.

---

# 11. Critic input과 relational identity

[Critic](Critic)은 concrete route/profile/object 이름 자체보다 [relational transition feature](Relational-Representation-and-Generalization)를 사용한다.

목표:

```text
training seed의 route-12에서 배운 가치 구조
        ↓
unseen seed의 같은 역할 route-31에 transfer
```

이것은 [ASEQ](ASEQ)가 concrete semantic [식별 방식(identity)](State-Representation)를 유지하는 이유와 반대처럼 보일 수 있다.

하지만 목적이 다르다.

```text
ASEQ
→ exact real self-loop 판정
→ concrete identity 중요

Critic
→ unseen structural value transfer
→ relational identity 중요
```

---

# 12. Function approximation의 장점과 위험

Neural [Critic](Critic)은 학습 sample과 비슷한 입력에 generalize할 수 있다.

하지만 학습 데이터 근거 밖에서도 항상 숫자를 출력한다.

```text
in-distribution
→ interpolation 가능

out-of-distribution
→ 근거 없는 extrapolation 가능
```

이 위험은 [Critic, Support & OOD](Critic-Support-and-OOD)의 핵심 주제다.

---

# 13. Confidence는 Critic value feature가 아니다

현재 confidence-gate repair의 중요한 원칙이다.

[Prophecy(미래 예측 모델)](Prophecy) 신뢰도가 [Critic](Critic) 입력에 들어가면 [신경망(network)](Neural-Networks-and-Optimization)가 다음 shortcut을 배울 수 있다.

```text
confidence 높음
→ value 높음
```

하지만:

```text
예측을 정확히 믿을 수 있음
!=
그 예측된 미래가 좋은 미래임
```

이다.

그래서 현재 구조는 기존 tensor shape를 유지하되 confidence feature slot을 constant로 중립화한다.

```text
Critic ranking = sparse-return value
Prophecy reliability = eligibility gate
```

자세한 개념 구분: [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability)

---

# 14. Global `critic_ready`의 한계

[Critic](Critic)이 충분한 gradient update를 했다는 global flag가 있다고 하자.

```text
critic_ready = True
```

이것은:

```text
Critic 전체가 어느 정도 학습됨
```

을 의미할 뿐:

```text
현재 query state/action이 training support 안에 있음
```

을 의미하지 않는다.

즉:

```text
global readiness
!=
local empirical support
```

다.

---

# 15. Local Critic support

현재는 real [Critic](Critic) 학습 상태 전이s를 이용해 query 상태/행동이 실제 학습 region과 얼마나 가까운지 판단한다.

질문은 하나다.

> **이 [Critic](Critic) 가치를 비교에 사용할 real 학습 [증거(evidence)](Evidence-Matrix)가 충분한가?**

```text
Policy root support 충분?
Candidate root support 충분?
        |
        +-- 둘 다 yes → value comparison 가능
        `-- 하나라도 no → fail closed, Policy 유지
```

---

# 16. Support distance

Support distance는 raw ID의 단순 Euclidean distance가 아니라 **[공개된(public)](State-Representation) 관계 기반 structural region**의 차이를 본다.

비교 요소 예:

- workflow progress
- known route/profile/object counts
- observed role distributions
- object-related 공개된 facts
- latest observed 공개된 [상태 코드(status)](Terminology-Guide)

중요한 점:

```text
Critic predicted value 자체는 support distance에 쓰지 않음
```

그래야:

```text
비슷한 value를 냈으니 지원된다
```

는 순환 논리를 피할 수 있다.

---

# 17. Support confidence

현재 구현은 nearest real sample distance와 sample count를 함께 반영하는 형태다.

개념적 intuition:

```math
support
\approx
e^{-4d_{nearest}}
\times
\frac{N}{N+4}
```

정확한 현재 parameter는 code를 [최종 기준(source of truth)](Current-Status)로 확인한다.

의미는:

```text
아주 가까운 sample 1개
→ 완전한 support라고 보지 않음

sample 많지만 전부 멂
→ 역시 낮은 support
```

이다.

---

# 18. Support는 value가 아니다

또 하나의 중요한 분리다.

```text
support 높음
!=
좋은 행동
```

실패 branch도 [학습 데이터(training data)](Terminology-Guide)가 많으면 데이터 근거가 높을 수 있다.

Support는:

```text
이 Critic estimate를 비교에 사용할 empirical basis가 있는가?
```

만 판단한다.

즉 [Calibration](Calibration)과 마찬가지로 **증거 [판정 관문(gate)](Terminology-Guide)**이지 보상/가치 bonus가 아니다.

---

# 19. Planner에서의 역할

[Imagination](Imagination) 탐색의 첫 행동 [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility):

```text
Policy A    : V = 0.1
Candidate B : V = 0.6
```

값만 보면 B가 좋아 보인다.

그런데:

```text
support(A)=0.8
support(B)=0.2
```

라면 `V(B)=0.6`은 [OOD extrapolation](Critic-Support-and-OOD)일 수 있다.

Current design은 이런 경우 B [기본 행동 덮어쓰기(override)](Imagination)를 취소하고 [Policy](Policy)를 유지한다.

---

# 20. Prophecy Calibration과 Critic Support 비교

| 값 | 질문 |
|---|---|
| [Prophecy](Prophecy) [결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability) | 이 future outcome이 발생할 mass는? |
| [Prophecy](Prophecy) 신뢰도 | 상태 전이 [예측(prediction)](Terminology-Guide)을 믿을 수 있나? |
| [Critic](Critic) 가치 | 이 future의 sparse 누적 보상은? |
| [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD) | 그 가치를 믿을 real 학습 증거가 있나? |

AASSR 현재 repair의 중요한 철학은 **서로 다른 의미를 하나의 confidence score로 합치지 않는 것**이다.

---

# 21. 왜 Planner가 Critic OOD를 더 위험하게 만드는가?

Search는 많은 candidate 중 가장 높은 가치를 찾는다.

```text
branch 1 → 0.1
branch 2 → 0.2
branch 3 → 0.3
branch 4 → 2.8  ← OOD artifact
```

Optimization은 우연한 예측 error를 적극적으로 선택할 수 있다.

이것이 [model exploitation / optimizer's curse](Model-Based-RL-and-World-Models)와 연결된다.

따라서 계획에서 국소 데이터 근거가 특히 중요하다.

---

# 22. 과거 2k diagnostic에서 왜 중요했는가?

Repaired [Imagination](Imagination)은 이전에는 inert했지만 이후 실제로 행동을 바꾸게 됐다.

그러나 higher-level region에서 **real [Critic](Critic) 학습 데이터 근거가 부족한 상태**에서도 branch 가치를 갈라냈고, 많은 기본 행동 덮어쓰기가 실제 error/상태 코드로 이어졌다.

이 결과가 보여준 것:

```text
Critic can discriminate values
!=
Critic is trustworthy here
```

이다.

[국소 데이터 근거(Local support)](Critic-Support-and-OOD) 판정 관문는 바로 이 실패 mode를 겨냥한다.

---

# 23. Sparse target starvation

Success/실패 trajectory가 너무 적으면 [Critic](Critic) target 대부분이 `0` 근처일 수 있다.

```text
ordinary 0
ordinary 0
ordinary 0
...
```

그러면 branch 간 누적 보상 discrimination이 약해진다.

결과:

- 거의 모든 탐색의 첫 행동 가치가 비슷함
- [Imagination](Imagination) advantage가 [최소 차이 기준(margin)](Imagination)을 넘지 못함
- [실제 행동 개입(intervention)](Imagination)이 0에 가까워질 수 있음

이 문제의 뿌리는 [sparse reward / credit assignment](Sparse-Reward-and-Credit-Assignment)에 있다.

---

# 24. Critic training과 Replay

[Critic](Critic)은 real trajectory를 기반으로 suffix 학습 examples를 만든다.

이때 [episode boundary](Replay-Buffer-and-Episode-Boundaries)가 정확해야 한다.

새 한 번의 문제 풀이 구간 상태가 이전 한 번의 문제 풀이 구간 suffix에 붙으면 실제로 존재하지 않는 누적 보상 target이 만들어질 수 있다.

```text
real trajectory continuity
```

가 [Critic](Critic) factual basis의 핵심이다.

---

# 25. Critic training loss

Current [Critic](Critic)은 누적 보상 [회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)에 [Smooth L1/Huber 계열 loss](Loss-Functions-and-Class-Imbalance)를 사용할 수 있다.

또 [gradient clipping](Neural-Networks-and-Optimization)으로 recurrent 학습 안정성을 높인다.

이 [학습 손실(loss)](Loss-Functions-and-Class-Imbalance)는:

```text
Critic parameter optimization objective
```

이지 [환경(environment)](Reinforcement-Learning) 보상 자체가 아니다.

---

# 26. Batched training/inference

[Imagination](Imagination) tree에는 많은 branch가 있다.

[Critic](Critic)을 branch마다 scalar GPU call로 실행하면 overhead가 매우 커질 수 있다.

```text
branch1 → GPU
branch2 → GPU
branch3 → GPU
...
```

Current hardware path는 여러 branch를 한 batch로 평가한다.

```text
[branch1, branch2, branch3, ...]
            ↓
           GPU
```

이것은 [semantics-preserving optimization](Neural-Networks-and-Optimization)이며 [Critic](Critic) target 정의를 바꾸지 않는다.

---

# 27. Failure modes

## 27.1 Sparse target starvation

**원인:** 성공/실패 real trajectory 부족.  
**결과:** 모든 가치가 비슷함.  
**대응:** 더 많은 real frontier experience, [난이도 조절 학습(curriculum)](Curriculum-Learning)/전이 개선.

## 27.2 OOD extrapolation

**원인:** 학습 모델이 학습 데이터 근거 밖에서도 숫자를 출력.  
**결과:** 계획기가 unrealistically high candidate를 선택.  
**대응:** 국소 데이터 근거 [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD).

## 27.3 Recurrent-state mismatch

**원인:** full 한 번의 문제 풀이 구간 숨겨진 memory로 학습하지만 현재 decision은 zero memory.  
**대응:** every-decision suffix 학습.

## 27.4 Confidence leakage

**원인:** [Prophecy](Prophecy) 신뢰도가 [Critic](Critic) 입력에 가치 feature로 들어감.  
**대응:** confidence-independent encoding.

## 27.5 Over-conservative support

**원인:** 데이터 근거 threshold가 너무 높음.  
**결과:** useful novel candidate도 전부 막음.  
**대응:** 실제 행동 개입-quality [구성요소 제거 비교(ablation)](Ablation-Benchmarking-and-Reproducibility)으로 threshold 검증.

---

# 28. Critic을 평가할 때 볼 metric

## Model-level

- 누적 보상 회귀 검증 error
- positive/negative/zero target 분포
- sequence-length별 error
- decision-suffix coverage

## Support-level

- 국소 데이터 근거 pass rate
- nearest-support distance
- sample count
- suppressed [OOD](Critic-Support-and-OOD) candidate count

## Planner-level

- candidate 가치 gap
- 데이터 근거 때문에 취소된 실제 행동 개입
- supported 실제 행동 개입 error rate

## Agent-level

- no-[Imagination](Imagination) 대비 Full 성공
- direct success-producing 실제 행동 개입
- bad-status 실제 행동 개입 감소 여부

높은 [Critic](Critic) 회귀 검증 score가 final 성공를 자동 보장하지 않는다.

[Proxy metric과 task metric](Ablation-Benchmarking-and-Reproducibility)을 분리한다.

---

# 29. 연구 가설

```text
H1. real sparse return만으로 Critic이 branch value를 분리할 수 있는가?
H2. decision suffix training이 arbitrary planning-root 평가를 개선하는가?
H3. relational input이 unseen seed transfer에 도움이 되는가?
H4. local support gate가 OOD intervention error를 줄이는가?
H5. support gate가 useful generalization까지 과도하게 막지는 않는가?
H6. supported Critic ranking이 same-checkpoint Imagination success를 실제로 높이는가?
```

H1~H4가 좋아도 H6은 별도로 검증해야 한다.

---

# 30. 관련 코드

```text
src/aassr_v2/current_return_critic.py
  - ReturnAwareHardwareRelationalGRUBranchCritic

src/aassr_v2/current_critic_support.py
  - local support replay
  - semantic support distance
  - fail-closed override gate

src/aassr_v2/current_confidence_gate.py
  - confidence-independent Critic encoder
```

---

# 31. 한 문장 요약

> **[Critic](Critic)은 [Prophecy](Prophecy)가 만든 미래를 실제 sparse 누적 보상으로 평가하되, 그 숫자가 학습 증거 밖의 extrapolation인지 국소 데이터 근거로 다시 확인하는 sequence-value 학습 모델이다.**

---

다음으로 읽기:

- **[Calibration](Calibration)**
- **[Imagination](Imagination)**
- **[Prophecy](Prophecy)**
- **[GRU & Sequence Models](GRU-and-Sequence-Models)**
- **[Critic, Support & OOD](Critic-Support-and-OOD)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
