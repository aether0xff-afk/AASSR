# GRU and Sequence Models

AASSR의 [현재(current)](Current-Status) [Critic(미래 가치 평가기)](Critic)은 **[GRU(게이트 순환 유닛)](GRU-and-Sequence-Models)(Gated Recurrent Unit)** 기반 [순서열(sequence)](GRU-and-Sequence-Models) [학습 모델(model)](Terminology-Guide)을 사용한다.

이 페이지는 RNN, [숨은 환경 상태(hidden state)](MDP-and-POMDP), [GRU](GRU-and-Sequence-Models) [판정 관문(gate)](Terminology-Guide), 순서열 [학습용 수치 표현으로 바꾸는 인코딩(encoding)](State-Representation), [과거 기억을 0으로 초기화한(zero-memory)](GRU-and-Sequence-Models) [학습된 모델로 값을 계산하는 추론(inference)](Neural-Networks-and-Optimization), [후속 구간(suffix)](GRU-and-Sequence-Models) [학습(training)](Terminology-Guide)을 설명한다.

---

# 1. 왜 sequence model이 필요한가?

현재 [상태(state)](State-Representation) 하나만으로 과거 [상태 전이(transition)](MDP-and-POMDP) 흐름을 충분히 요약할 수 없는 경우가 있다.

```text
S0 → S1 → S2 → S3
```

현재 `S3`의 의미가 그 전에 어떤 경로를 거쳤는지에 따라 달라질 수 있다.

특히 [POMDP](MDP-and-POMDP)에서는 [기록(history)](Development-History)가 숨은 환경 상태를 추론하는 데 도움이 된다.

Sequence 학습 모델은 여러 timestep 정보를 하나의 internal [표현(representation)](Relational-Representation-and-Generalization)으로 압축한다.

---

# 2. Feed-forward network와 Recurrent network

## Feed-forward

```text
x_t → network → y_t
```

각 [입력(input)](Terminology-Guide)을 독립적으로 처리한다.

## Recurrent

```text
x_t + h_{t-1}
      ↓
 recurrent network
      ↓
 h_t, y_t
```

이전 숨은 환경 상태 `h_{t-1}`가 현재 계산에 들어간다.

---

# 3. Hidden state

RNN의 숨은 환경 상태는 과거 순서열 정보를 압축한 내부 벡터다.

```math
h_t=f(x_t,h_{t-1})
```

이상적으로는 현재 [예측(prediction)](Terminology-Guide)에 필요한 과거 정보를 `h_t`에 보존한다.

하지만 실제로 어떤 정보를 유지할지는 학습으로 결정된다.

---

# 4. Vanilla RNN의 문제

긴 순서열를 학습할 때 [기울기(gradient)](Neural-Networks-and-Optimization)가 너무 작아지거나 커지는 문제가 있다.

- vanishing 기울기
- exploding 기울기

이 때문에 오래 전 정보를 학습하기 어려울 수 있다.

LSTM과 [GRU](GRU-and-Sequence-Models)는 [조건부 통과 판단(gating)](Terminology-Guide) [작동 원리(mechanism)](Evidence-Matrix)을 사용해 이 문제를 완화하려는 re[현재 구조(current architecture)](Current-Status)다.

---

# 5. GRU

[GRU](GRU-and-Sequence-Models)는 대표적으로:

- [학습 갱신(update)](Neural-Networks-and-Optimization) 판정 관문
- [환경 초기화(reset)](Replay-Buffer-and-Episode-Boundaries) 판정 관문

를 사용한다.

일반적인 형태의 한 예:

```math
z_t=\sigma(W_zx_t+U_zh_{t-1})
```

```math
r_t=\sigma(W_rx_t+U_rh_{t-1})
```

```math
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1}))
```

```math
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
```

구현 convention에 따라 식의 계수 방향은 다를 수 있지만 핵심은 판정 관문가 **과거 정보를 얼마나 유지/갱신할지** 조절한다는 것이다.

---

# 6. Update gate

Update 판정 관문는 기존 숨은 환경 상태와 새 [선택 후보(candidate)](Terminology-Guide) 상태를 얼마나 섞을지 조절한다.

직관적으로:

```text
z 작음 → 과거 memory 더 유지
z 큼   → 새 정보로 더 많이 갱신
```

정확한 해석은 구현 수식 convention에 따라 달라질 수 있다.

---

# 7. Reset gate

Reset 판정 관문는 새 선택 후보를 계산할 때 과거 숨은 환경 상태의 어느 부분을 얼마나 사용할지 조절한다.

과거 정보 중 현재 입력 처리에 불필요한 부분을 줄일 수 있다.

---

# 8. Sequence encoding

AASSR [Critic](Critic)은 [경험 경로(trajectory)](Reinforcement-Learning)의 [관계 기반(relational)](Relational-Representation-and-Generalization) 상태 전이 정보를 순서열로 받아 [숨겨진(hidden)](MDP-and-POMDP) 표현을 만들고 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [누적 보상(return)](Value-Functions-and-Bellman-Equation)을 예측한다.

개념적으로:

```text
Transition 0
    ↓
Transition 1
    ↓
Transition 2
    ↓
GRU hidden state
    ↓
Return prediction
```

관련 페이지:

- [Critic](Critic)

---

# 9. Sequence length

더 긴 순서열를 주면 더 많은 기록를 볼 수 있다.

하지만:

- [계산(compute)](Reproduction) 증가
- padding/[묶음 처리(batching)](Reproduction) 복잡성
- irrelevant 기록 가능

이 생긴다.

AASSR에서는 [계획(planning)](Counterfactual-Planning-and-Search) [탐색의 첫 행동(root)](Imagination)가 [한 번의 문제 풀이 구간(episode)](Terminology-Guide) 중간 어디에서든 나타날 수 있다는 점이 더 중요한 문제다.

---

# 10. Hidden state mismatch

[학습(Training)](Reinforcement-Learning)에서는 항상 한 번의 문제 풀이 구간 시작부터 [GRU](GRU-and-Sequence-Models)를 돌렸다고 하자.

```text
S0 → S1 → S2 → S3
h0   h1   h2   h3
```

그런데 실제 [Imagination(가상 미래 탐색)](Imagination)은 `S2`에서 갑자기 시작한다.

[계획기(Planner)](Counterfactual-Planning-and-Search)가 과거 [과거 정보를 이어가는 순환형(recurrent)](GRU-and-Sequence-Models) [기억(memory)](GRU-and-Sequence-Models) `h2`를 가지고 있지 않다면:

```text
S2 + zero hidden state
```

로 [Critic](Critic)을 평가하게 된다.

학습과 추론 조건이 다르다.

이를 recurrent-state [서로 맞지 않는 불일치(mismatch)](Causality-Leakage-and-Evaluation)라고 볼 수 있다.

---

# 11. Zero-memory inference

[현재(Current)](Current-Status) [의사결정(decision)](Chance-and-Decision-Nodes) [지점(point)](Terminology-Guide)에서 과거 [Critic](Critic) 숨은 환경 상태를 명시적으로 전달하지 않고:

```text
h_0 = zeros
```

에서 현재 후속 구간 [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)을 시작할 수 있다.

그렇다면 학습도 이 조건을 포함해야 한다.

---

# 12. Decision suffix training

Episode:

```text
S0 → S1 → S2 → S3 → terminal
```

에서 다음 학습 sequences를 만든다.

```text
[S0,S1,S2,S3]
[S1,S2,S3]
[S2,S3]
[S3]
```

각 순서열를 **zero 숨은 환경 상태에서 시작**하게 학습한다.

따라서 어느 의사결정 상태에서 계획이 시작되더라도 학습 [명세(contract)](Current-Status)와 더 잘 맞는다.

AASSR 현재 [Critic](Critic)의 중요한 설계다.

---

# 13. Suffix target

각 후속 구간의 시작점에서 [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) 희소한 누적 보상까지의 [미래 보상을 시간에 따라 할인한(discounted)](Value-Functions-and-Bellman-Equation) 누적 보상을 [대상 또는 학습 목표값(target)](Terminology-Guide)으로 둘 수 있다.

예:

```text
S2 → S3 → success +1
```

이면 탐색의 첫 행동 `S2`의 대상/목표값은 [미래 보상의 할인율(discount)](Value-Functions-and-Bellman-Equation) [실험에서 바꾸어 보는 요인(factor)](Ablation-Benchmarking-and-Reproducibility)에 따라 `γ` 계열이 된다.

관련 페이지:

- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)
- [Critic](Critic)

---

# 14. Prefix training within suffix

한 후속 구간 안에서도 여러 prefix를 학습 example로 사용할 수 있다.

```text
[S1]
[S1,S2]
[S1,S2,S3]
```

현재 AASSR [Critic](Critic)은 계획 탐색의 첫 행동 관점의 sparse-누적 보상 대상/목표값을 순서열 prefixes에 학습하도록 설계되어 있다.

이 부분은 일반적인 sequence-to-one [회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility)보다 AASSR 계획 명세에 맞춘 특수한 학습 구조다.

---

# 15. Padding과 batching

Batch 안의 순서열 길이가 다르면 shorter 순서열에 padding을 넣을 수 있다.

```text
seq A: x1 x2 x3 x4
seq B: y1 y2 PAD PAD
```

실제 학습 모델은 순서열 length/[가능/불가능을 표시하는 마스크(mask)](Terminology-Guide)를 이용해 padding이 숨겨진 학습 갱신에 의미 있는 [데이터(data)](Terminology-Guide)처럼 들어가지 않도록 해야 한다.

AASSR 현재 hardware [경로(path)](Counterfactual-Planning-and-Search)는 [Critic](Critic)의 많은 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes) 평가을 [여러 입력 묶음(batch)](Reproduction) 처리해 GPU 효율을 높인다.

---

# 16. Batched inference

[Imagination](Imagination) [탐색 트리(tree)](Counterfactual-Planning-and-Search)에서는 여러 결과 경로 [Critic](Critic) [평가 점수(score)](Terminology-Guide)가 한꺼번에 필요하다.

Scalar 호출:

```text
branch 1 → GPU
branch 2 → GPU
branch 3 → GPU
...
```

보다:

```text
[branch1, branch2, branch3, ...]
            ↓ one batch
           GPU
```

가 효율적이다.

이 최적화는 [Critic](Critic) [가치(value)](Value-Functions-and-Bellman-Equation) [의미 규칙(semantics)](State-Representation)를 바꾸지 않고 [실제 실행(execution)](Research-Jargon-Guide) overhead를 줄이는 목적이다.

---

# 17. GRU와 Partial Observability

Recurrent [신경망(network)](Neural-Networks-and-Optimization)는 past [관측(observation)](MDP-and-POMDP) 기록를 숨은 환경 상태에 압축하여 POMDP에서 도움이 될 수 있다.

하지만:

```text
GRU 사용
!=
POMDP 완전 해결
```

이다.

필요한 숨겨진 [정보(information)](Information-Theory-and-Intrinsic-Motivation)이 관측 기록에 전혀 나타나지 않거나 학습이 충분하지 않으면 복원할 수 없다.

---

# 18. GRU와 relational representation

AASSR [Critic](Critic)의 입력은 [실제 개체를 구분하는(concrete)](State-Representation) identifiers보다 관계 기반 상태 전이 [학습에 사용하는 특징(features)](Terminology-Guide)를 사용한다.

따라서 순환형 기억도:

```text
route-12라는 이름
```

보다는:

```text
어떤 관계 구조의 transition sequence였는가
```

를 학습하도록 유도된다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)

---

# 19. GRU와 Prophecy의 차이

과거 AASSR에는 [GRU](GRU-and-Sequence-Models) 기반 [Prophecy(미래 예측 모델)](Prophecy) 계열도 있었지만 [현재 세대(current-generation)](Current-Status)의 [현재 활성(active)](Current-Status) [Prophecy](Prophecy)는 관계 기반 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration) 구조로 발전했다.

현재 [GRU](GRU-and-Sequence-Models)를 보면 무조건 [Prophecy](Prophecy)라고 생각하면 안 된다.

```text
Current Prophecy
→ stochastic relational world model

Current Critic
→ relational GRU sparse-return model
```

[최종 기준(source of truth)](Current-Status)는 `current_manifest.py`다.

---

# 20. Gradient clipping

Recurrent 학습 모델은 exploding 기울기 위험이 있을 수 있어 기울기 norm clipping을 사용할 수 있다.

개념:

```text
gradient norm이 threshold 초과
→ 크기 제한
```

AASSR [Critic](Critic) 학습에서도 안정성을 위해 clipping을 사용할 수 있다.

---

# 21. Hidden state는 Knowledge인가?

아니다.

```text
GRU hidden state
= neural latent sequence memory

KnowledgeStore
= explicit real-response fact + provenance
```

둘은 역할과 해석 가능성이 다르다.

[Knowledge(에피소드 지식)](Knowledge)는 어떤 사실을 언제 알았는지 명시적으로 추적할 수 있다.

[GRU](GRU-and-Sequence-Models) 숨은 환경 상태는 학습된 [직접 관측되지 않는 잠재 표현(latent)](GRU-and-Sequence-Models) [수치 벡터(vector)](Neural-Networks-and-Optimization)라 직접 의미를 해석하기 어렵다.

관련 페이지:

- [Knowledge](Knowledge)

---

# 22. Failure modes

## Training/inference hidden-state mismatch

Episode-start 숨은 환경 상태로만 학습했는데 mid-episode 기억 0 초기화에서 평가.

## Sequence truncation

중요한 오래 전 기록가 순서열에서 잘림.

## Overfitting

특정 학습 경험 경로 pattern을 암기.

## OOD sequence

새 관계 기반 상태 전이 combination에서 잘못된 누적 보상 예측.

관련 페이지:

- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 23. AASSR 연결 요약

```text
Real episode transitions
       ↓
모든 decision suffix 생성
       ↓
각 suffix를 zero hidden state에서 시작
       ↓
Relational GRU Critic training
       ↓
Imagination branch sequence 평가
       ↓
Local support gate
```

---

# 24. 다음으로 읽기

- [Critic](Critic)
- [Critic, Support and OOD](Critic-Support-and-OOD)
- [MDP and POMDP](MDP-and-POMDP)
- [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)

관련 색인: **[Concept Index](Concept-Index)**