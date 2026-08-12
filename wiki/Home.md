# AASSR Wiki

> **AASSR (An Agent for Solving Sparse Reward problem)**는 중간 보상이 거의 없고, 가능한 행동이 많으며, 환경을 완전히 관찰할 수 없는 상황에서 **경험을 구조화하고 미래를 예측한 뒤 실제 행동 전에 여러 가능성을 비교하는 강화학습 연구 시스템**이다.

> [!IMPORTANT]
> 이 위키는 `main`의 **current-generation runtime**을 기준으로 작성한다. 과거 AASSR v0.4, 초기 Prophecy/Imagination, effect-composition 계열은 연구 역사와 재현을 위해 남아 있지만 현행 구조와 섞어서 설명하지 않는다.

---

# 이 위키는 어떻게 읽는가?

이 위키는 단순한 코드 설명서가 아니다.

각 주제를 다음 깊이로 연결한다.

```text
왜 이 문제가 필요한가?
        ↓
연구 질문은 무엇인가?
        ↓
어떤 설계로 답하려 하는가?
        ↓
수학 / 알고리즘은 무엇인가?
        ↓
실제 코드는 어떻게 구현되는가?
        ↓
어떤 실험이 그 주장을 검증하는가?
        ↓
무엇이 실패했고 무엇이 아직 미검증인가?
```

처음 보는 사람은 위쪽 연구 흐름만 읽고, 구현을 검증하거나 재현하려는 사람은 각 메커니즘의 기술 계층까지 내려갈 수 있다.

---

# 30초 요약

```mermaid
flowchart LR
    O[Public Observation] --> R[Relational State]
    O --> S[Concrete Semantic State]
    R --> P[Policy]
    S --> A[ASEQ]
    O --> K[Knowledge]
    P --> C[Candidate Actions]
    A --> C
    C --> W[Prophecy]
    K --> W
    R --> W
    W --> CAL[Calibration]
    W --> I[Imagination]
    CAL --> I
    I --> CR[Critic]
    CR --> SUP[Local Support]
    SUP --> G[Override Gate]
    G --> ACT[Concrete Action]
    ACT --> E[Environment]
    E --> O
```

핵심 흐름:

1. **관측** 가능한 정보만 본다.
2. **Relational state**로 이름이 바뀐 환경에서도 구조를 알아보려 한다.
3. **ASEQ**로 실제 `(S,A,S')` 경험과 self-loop를 다룬다.
4. **Policy**가 sparse-return 기반 기본 행동을 선택한다.
5. **Knowledge**가 현재 episode에서 실제 response로 알아낸 사실의 causal context를 보존한다.
6. **Prophecy**가 행동 후 가능한 미래의 확률 분포를 예측한다.
7. **Calibration**이 그 예측을 믿어도 되는지 real holdout으로 확인한다.
8. **Imagination**이 chance와 decision을 구분해 여러 단계의 미래를 계산한다.
9. **Critic**이 sparse return 관점에서 미래 가치를 평가한다.
10. **Local support gate**가 OOD Critic extrapolation을 막는다.
11. 반복 성공한 구조는 **Skill**의 relational template로 재사용될 수 있다.
12. 모든 조건이 충분할 때만 Policy 행동을 실제로 override한다.

---

# 연구부터 읽기

AASSR을 연구 프로젝트로 이해하려면 다음 순서를 권장한다.

1. **[Sparse Reward Problem](Sparse-Reward-Problem)**  
   AASSR이 정확히 어떤 문제를 풀려고 하는가?

2. **[Research Questions](Research-Questions)**  
   핵심 질문을 어떤 하위 가설로 나눴는가?

3. **[Research Architecture](Research-Architecture)**  
   각 연구 질문이 실제 current-generation 설계로 어떻게 연결되는가?

4. **[Design Rationale](Design-Rationale)**  
   왜 relational, mixture, calibration, chance expectation, local support 같은 선택을 했는가?

5. **[Experiments](Experiments)**  
   어떤 control과 ablation으로 각 효과를 분리하는가?

6. **[Current Status](Current-Status)**  
   지금까지 무엇이 확인됐고 무엇이 아직 주장 불가능한가?

---

# 기술적으로 깊게 읽기

각 메커니즘은 별도 deep-dive로 연결한다.

| 주제 | 핵심 질문 |
|---|---|
| **[ASEQ](ASEQ)** | 실제 `(S,A,S')` 중 어떤 반복만 막아야 하는가? |
| **[Policy](Policy)** | sparse reward와 information value를 섞지 않고 행동을 어떻게 평가하는가? |
| **[Knowledge](Knowledge)** | 무엇을 언제 알았으며 hindsight leak을 어떻게 막는가? |
| **[Prophecy](Prophecy)** | stochastic multimodal future를 어떻게 예측하는가? |
| **[Calibration](Calibration)** | outcome probability와 model reliability를 어떻게 분리하는가? |
| **[Critic](Critic)** | imagined future의 sparse return과 local support를 어떻게 평가하는가? |
| **[Imagination](Imagination)** | chance expectation과 agent decision max를 어떻게 연결하는가? |
| **[Skills](Skills)** | 성공 ASeq를 concrete ID 암기 없이 어떻게 재사용하는가? |
| **[Core Architecture](Core-Architecture)** | 이 모든 것이 실제 current runtime에서 어떻게 연결되는가? |

실행과 재현은 **[Reproduction](Reproduction)**, 용어는 **[Glossary](Glossary)** 에서 본다.

---

# 핵심 연구 질문

AASSR의 가장 큰 질문은 다음과 같다.

> **중간 보상이 거의 없는 환경에서 에이전트가 정답 경로나 중간 목표를 사람이 주입하지 않아도 경험 구조와 미래 예측을 이용해 스스로 장기 행동 과정을 만들 수 있는가?**

이를 현재 실험 가능한 질문으로 나누면:

```text
희소 보상만으로 최초 성공을 발견할 수 있는가?
        ↓
관계 표현이 unseen transfer를 개선하는가?
        ↓
ASEQ가 진전 없는 반복을 줄이는가?
        ↓
Prophecy가 usable future distribution을 학습하는가?
        ↓
Calibration이 잘못된 예측을 걸러내는가?
        ↓
Critic이 실제 sparse return으로 미래 가치를 구분하는가?
        ↓
Imagination이 같은 Policy보다 더 좋은 행동을 만들 수 있는가?
        ↓
성공 구조를 Skill로 transfer할 수 있는가?
        ↓
AASSR 전체가 강한 baseline보다 나은가?
```

---

# 현재 source of truth

현재 실행 세대 이름:

```text
aassr-current-generation-v2
```

단일 source of truth:

```text
src/aassr_v2/current_manifest.py
```

현재 주요 component:

| Layer | Current contract |
|---|---|
| Observation | relational public state v3 + latest HTTP status |
| ASEQ | semantic self-loop empirical v3 |
| Policy | relational-invariant DQN + information residual |
| Prophecy | relational conditional-mixture ensemble v5, status-balanced |
| Calibration | semantic probability holdout calibration v3, status-aware |
| Knowledge | episode-local response knowledge context |
| Imagination | structural compute dedup + probability chance / decision tree |
| Critic | relational GRU discounted sparse-return |
| Critic support | local real-training support, fail-closed |
| Skill | relational ASEQ template |
| Training Imagination | disabled for same-checkpoint comparison |

---

# 현재 실험 구조

최종 current-generation comparison의 기본 축은 다음과 같다.

```text
dqn_raw
   |
   | representation effect
   v
dqn_relational
   |
   | AASSR stack beyond representation
   v
aassr_current_no_imagination
   |
   | Imagination marginal effect
   v
aassr_current_full
```

추가 model-based baseline으로 official pinned DreamerV3 relational adapter를 비교한다.

AASSR의 OFF/ON 비교는 반드시 **하나의 frozen checkpoint**에서 수행한다.

```text
one training run
      |
      v
frozen checkpoint
   /          \
OFF eval    ON eval
```

---

# 현재 연구 상태를 읽을 때 주의할 점

AASSR에는 여러 세대의 실험 결과가 존재한다.

따라서 다음을 구분해야 한다.

```text
과거 mechanism evidence
current code validation
reduced diagnostic
final performance benchmark
```

과거 실험에서 ASEQ가 self-loop를 줄였다는 evidence가 있어도 그것이 곧 current Full AASSR의 최종 성공률은 아니다.

마찬가지로 current code에 repair가 들어갔다고 해서 새 장기 실험에서 성능 향상이 검증된 것도 아니다.

이 경계는 **[Experiments](Experiments)** 와 **[Current Status](Current-Status)** 에서 명시한다.

---

# 연구 역사

AASSR은 여러 실패와 구조 변경을 거쳐 현재 세대로 왔다.

과거 구현과 실험 흐름은 **[Development History](Development-History)** 에서 따로 관리한다.

과거 코드는 재현 가능성을 위해 저장소에 남아 있지만 current runtime의 구성 요소로 자동 해석하면 안 된다.

---

# 빠른 경로

**처음 보는 사람**  
[AASSR in 5 Minutes](AASSR-in-5-Minutes) → [Research Questions](Research-Questions) → [Research Architecture](Research-Architecture)

**왜 이렇게 설계했는지 보고 싶은 사람**  
[Sparse Reward Problem](Sparse-Reward-Problem) → [Design Rationale](Design-Rationale)

**연구 결과를 보고 싶은 사람**  
[Experiments](Experiments) → [Current Status](Current-Status)

**구현을 검증하고 싶은 사람**  
[Core Architecture](Core-Architecture) → [Policy](Policy) → [Prophecy](Prophecy) → [Calibration](Calibration) → [Critic](Critic) → [Imagination](Imagination)

**기억/재사용 계층을 보고 싶은 사람**  
[ASEQ](ASEQ) → [Knowledge](Knowledge) → [Skills](Skills)

---

**Current source of truth:** `src/aassr_v2/current_manifest.py`
