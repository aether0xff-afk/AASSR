# 실험 설계와 결과 (Experiments)

이 페이지는 AASSR의 **실험 [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility), 비교 조건, [증거(evidence)](Evidence-Matrix) [난이도 단계(level)](Curriculum-Learning), [과거 기록(historical)](Development-History) [진단 실험(diagnostic)](Evidence-Matrix), [현재(current)](Current-Status) [연구 주장(claim)](Evidence-Matrix) [경계(boundary)](Replay-Buffer-and-Episode-Boundaries)**를 정리한다.

> [!**중요**]
> AASSR 저장소에는 여러 세대의 결과가 함께 남아 있다. 이 페이지에서는 숫자를 반드시 다음 중 하나로 분류한다.
>
> ```text
> [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility) [검증(validation)](Ablation-Benchmarking-and-Reproducibility)
> [작동 원리(mechanism)](Evidence-Matrix) 진단 실험
> 과거 기록 root-cause 진단 실험
> [현재 세대(current-generation)](Current-Status) reduced 검증
> multi-[난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility) 표준 비교 실험
> [최종 비공개 평가(final blind)](Ablation-Benchmarking-and-Reproducibility)ed [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)
> ```
>
> 서로 다른 세대의 숫자를 한 표에 섞어 “성능 추세”처럼 해석하지 않는다.

전체 연구 질문과 가설: [Research Questions](Research-Questions)  
RQ별 변수·지표·연구 주장 상태: [Evidence Matrix](Evidence-Matrix)

---

## 목차

1. [실험 철학](#1-실험-철학)
2. [보상과 관측 계약](#2-보상과-관측-계약)
3. [Benchmark validation](#3-benchmark-validation)
4. [RQ1 — Autonomous first success](#4-rq1--autonomous-first-success)
5. [RQ2 — Raw vs Relational](#5-rq2--raw-vs-relational)
6. [RQ3 — ASEQ self-loop](#6-rq3--aseq-self-loop)
7. [RQ4/RQ5 — Prophecy & Calibration](#7-rq4rq5--prophecy--calibration)
8. [RQ6 — Critic local support](#8-rq6--critic-local-support)
9. [RQ7 — Imagination same-checkpoint](#9-rq7--imagination-same-checkpoint)
10. [RQ8 — Five-condition final suite](#10-rq8--five-condition-final-suite)
11. [RQ9 — Skill](#11-rq9--skill)
12. [통계와 보고 형식](#12-통계와-보고-형식)
13. [최종 claim gate](#13-최종-claim-gate)

---

# 1. 실험 철학

AASSR은 여러 메커니즘이 결합된 시스템이다.

```text
Relational Representation
+ ASEQ
+ Policy
+ Knowledge
+ Prophecy
+ Calibration
+ Critic
+ Local Support
+ Imagination
+ Skills
```

따라서 [전체 AASSR 조건(Full)](Experiments) AASSR 하나만 돌려 [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)과 비교하면 **왜 차이가 났는지 알 수 없다.**

연구 설계는 다음 순서로 효과를 분리한다.

```text
dqn_raw
  ↓ representation effect
dqn_relational
  ↓ AASSR stack beyond representation
aassr_current_no_imagination
  ↓ Imagination marginal effect
aassr_current_full
```

그리고 외부 [환경 모델을 사용하는(model-based)](Model-Based-RL-and-World-Models) family [비교(comparison)](Ablation-Benchmarking-and-Reproducibility):

```text
dreamerv3_relational
↔
aassr_current_full
```

관련 개념: [Ablation](Ablation-Benchmarking-and-Reproducibility), [Confounder](Ablation-Benchmarking-and-Reproducibility), [Same-checkpoint evaluation](Causality-Leakage-and-Evaluation)

---

# 2. 보상과 관측 계약

## 2.1 External reward

현재 pentest 계열 [연구 과제(task)](Sparse-Reward-Problem) [보상(reward)](Sparse-Reward-and-Credit-Assignment):

```text
proof success       +1
true failure        -1
stall                0
rate-limit trunc.    0
transition-cap       0
ordinary transition  0
```

다음은 연구 과제 보상로 사용하지 않는다.

- [정답 경로로 유도된(guided)](Causality-Leakage-and-Evaluation) [진행도(progress)](Terminology-Guide) [평가 점수(score)](Terminology-Guide)
- [정답을 알고 있는 기준(oracle)](Ablation-Benchmarking-and-Reproducibility) route proximity
- [숨겨진(hidden)](MDP-and-POMDP) [대상 또는 학습 목표값(target)](Terminology-Guide) proximity
- [중간(intermediate)](Sparse-Reward-and-Credit-Assignment) proof hint
- 사람이 만든 성공 [행동(action)](Reinforcement-Learning) [순서열(sequence)](GRU-and-Sequence-Models)

즉 [sparse reward](Sparse-Reward-and-Credit-Assignment) 자체를 유지한다.

## 2.2 Internal information signal

[Policy](Policy)의 [정보 가치 잔차(information residual)](Policy)은 [환경이 주는 외부(external)](Terminology-Guide) 보상와 별도다.

```text
external task reward
!=
internal information-value signal
```

따라서 정보 가치 잔차을 “중간 보상 [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)”으로 보고 [성공(success)](Terminology-Guide) 보상에 더한 값으로 해석하면 안 된다.

## 2.3 Observation boundary

Learner는 [response-causal](Causality-Leakage-and-Evaluation) [공개된(public)](State-Representation) [정보(information)](Information-Theory-and-Intrinsic-Motivation)만 사용한다.

허용 예:

- 실제 [응답(response)](State-Representation)에서 관측한 [가장 최근의(latest)](Current-Status) HTTP-like [상태 코드(status)](Terminology-Guide)
- 발견된 route/profile/object relation
- 현재 [현재 허용된(legal)](Terminology-Guide) 행동 [현재 선택 가능한 영역(surface)](Terminology-Guide)
- 응답에서 획득한 [한 번의 접속 세션(session)](Terminology-Guide)/CSRF [실제로 관측한 사실(fact)](Causality-Leakage-and-Evaluation)

금지 예:

- 숨겨진 대상/목표값 [식별 방식(identity)](State-Representation)
- [정확히 동일한(exact)](ASEQ) 숨겨진 [공정성과 구현을 점검하는 감사(audit)](Causality-Leakage-and-Evaluation) [환경 내부의 숨은 압박 값(pressure)](Causality-Leakage-and-Evaluation)
- 정확히 동일한 숨겨진 접속 세션 [남은 횟수 카운트다운(countdown)](Causality-Leakage-and-Evaluation)
- [미래(future)](Counterfactual-Planning-and-Search) [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)
- 숨겨진 [난이도 조절 학습(curriculum)](Curriculum-Learning) [정답 범주 표시(label)](Loss-Functions-and-Class-Imbalance)을 [학습 주체(learner)](Terminology-Guide) [상태(state)](State-Representation)로 직접 주입

---

# 3. Benchmark validation

AASSR은 실제 외부 시스템을 공격하지 않고 **safe in-process HTTP [의사결정(decision)](Chance-and-Decision-Nodes) lab**을 사용한다.

```mermaid
flowchart LR
    E[Entry] --> D[Discovery]
    D --> L[Login / Session]
    L --> O[Object Candidates]
    O --> A[Authorization Boundary]
    A --> C[CSRF / Workflow]
    C --> P[Proof]
```

환경은 다음과 같은 공개된 inter행동 [구조(structure)](Research-Architecture)를 모사한다.

- `200/302/400/401/403/404/409/429`
- redirect
- 접속 세션
- CSRF
- object authorization
- workflow prerequisites
- 감사 / [복구할 수 없는 실패 잠금(lockout)](Replay-Buffer-and-Episode-Boundaries)
- [비율(rate)](Terminology-Guide) [제한(limit)](Terminology-Guide)
- 접속 세션 expiration
- decoy routes
- 난수 시드마다 바뀌는 opaque identifiers

실제 [신경망(network)](Neural-Networks-and-Optimization) socket, shell, 환경이 주는 외부 대상/목표값은 사용하지 않는다.

## 3.1 난도 validation

40 평가 난수 시드s에서 표준 비교 실험 검증 비교 기준:

| Tier | [정답을 알고 있는 기준(Oracle)](Ablation-Benchmarking-and-Reproducibility) | [무작위(Random)](Ablation-Benchmarking-and-Reproducibility) | Browse-first | Response-guided | Abstract Q |
|---|---:|---:|---:|---:|---:|
| Easy | 100.0% | 0.0% | 0.3% | **100.0%** | 0.0% |
| Medium | 100.0% | 0.0% | 0.0% | **30.0%** | 0.0% |
| Hard | 100.0% | 0.0% | 0.0% | **20.0%** | 0.0% |

이 결과의 의미:

```text
Oracle 100%
→ task가 구조적으로 불가능하지 않음

Random ~0%
→ 무작위 행동만으로 쉽게 풀리지 않음

Response-guided degradation
→ 난도 차이가 존재
```

이 결과는 **AASSR 우위 증거가 아니다.** [표준 비교 실험(Benchmark)](Ablation-Benchmarking-and-Reproducibility)가 [에이전트(agent)](Reinforcement-Learning) 비교에 사용할 수 있는지 검증한 것이다.

---

# 4. RQ1 — Autonomous first success

질문:

> 정답 경로 유도 [경험 경로(trajectory)](Reinforcement-Learning)와 형태 조정 보상 없이 [실제 환경에서 관측된(real)](Research-Jargon-Guide) 성공 [경험(experience)](Replay-Buffer-and-Episode-Boundaries)가 발생하는가?

관련: [Research Questions — RQ1](Research-Questions#rq1--희소-보상만으로-최초-성공을-발견할-수-있는가)

## 관측해야 할 값

- first proof [상태 전이(transition)](MDP-and-POMDP) index
- total [학습(training)](Terminology-Guide) proofs
- 난이도 조절 학습 [다음 난이도로 승급(promotion)](Curriculum-Learning) / demotion
- frontier exposure
- [진전 없이 반복하다 멈춘(stalled)](ASEQ) episodes

## 중요한 fairness rule

[난이도 조절 학습(Curriculum)](Curriculum-Learning)은 [난이도(difficulty)](Curriculum-Learning) exposure를 조절할 수 있지만 **정답 행동을 주면 안 된다.**

```text
허용
Easy → Medium → Hard exposure

금지
“이 상태에서는 login을 눌러라”
“이 object가 정답이다”
```

## 현재 evidence 해석

과거 [사람의 정답 경로 없이 자율적인(autonomous)](Research-Questions) pilot과 [특정 범위에 집중한(focused)](Experiments) 학습에서 human-written 성공 순서열 없이 proof가 발생한 사례가 있다.

이것은:

> “자율적인 proof [스스로 새로운 성공 경로를 발견하는 것(discovery)](Research-Questions)가 가능하다”

라는 좁은 증거다.

이것만으로:

> “AASSR이 비교 기준보다 sample-efficient하다”

를 말할 수는 없다.

---

# 5. RQ2 — Raw vs Relational

질문:

> [Relational Representation](Relational-Representation-and-Generalization)이 [실제 개체를 구분하는(concrete)](State-Representation) ID 중심 [표현(representation)](Relational-Representation-and-Generalization)보다 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) [전이(transfer)](Relational-Representation-and-Generalization)를 개선하는가?

## 핵심 control

```text
dqn_raw
vs
dqn_relational
```

가능한 한 표현 외 요소를 고정한다.

| 고정 | 변경 |
|---|---|
| 보상 | 표현 |
| 상태 전이 [실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility) | 표현 |
| [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) 학습 실험 규칙 | 표현 |
| eval 난수 시드s | 표현 |
| 난이도 조절 학습 exposure | 표현 |

## Primary metrics

- 학습 중 보지 못한 성공
- tier별 성공
- [학습 진행의 도달 기준점(milestone)](Curriculum-Learning) [도달(reach)](Curriculum-Learning)
- 진전 없이 멈춘 비율
- mean requests

## Secondary diagnostics

- 상태/행동 collision 비율
- 표현 [같은 구조를 가리키는 다른 이름(alias)](State-Representation) frequency
- rename [이름 순서를 바꾸는 순열(permutation)](Relational-Representation-and-Generalization) consistency

> [!NOTE]
> [현재(Current)](Current-Status) [State Representation v3](State-Representation)은 가장 최근의 공개된 HTTP 상태 코드를 보존한다. 따라서 과거 [관계 기반(relational)](Relational-Representation-and-Generalization) v2와 현재 v3 결과를 같은 표현 [실험 조건(condition)](Ablation-Benchmarking-and-Reproducibility)으로 취급하면 안 된다.

---

# 6. RQ3 — ASEQ self-loop

질문:

> [의미 기준(semantic)](State-Representation) `S → A → S` 증거를 사용하면 진전 없이 멈춘 [행동 양상(behavior)](Experiments)를 줄일 수 있는가?

## 6.1 No-retraining diagnostic

과거 L1 학습 중 보지 못한에서 3 [체크포인트(checkpoint)](Reproduction)s × 8 난수 시드s:

```text
raw greedy stalled       24 / 24
exact ASEQ stalled        0 / 24
```

성공:

| 체크포인트 | [가공하지 않은 원본(raw)](State-Representation) [현재 추정값이 가장 큰 행동만 고르는 탐욕 선택(greedy)](Exploration-and-Exploitation) | 정확히 동일한 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ) |
|---|---:|---:|
| L2 first [도달한(reached)](Curriculum-Learning) | 0/8 | 2/8 |
| L2 pre-demotion | 0/8 | 7/8 |
| post-demotion retrained | 0/8 | 5/8 |

이 결과는 **[ASEQ](ASEQ) [잘못된 행동을 제한하는 보호 규칙(guard)](ASEQ)의 작동 원리 진단 실험**이다.

## 6.2 Consistent retraining diagnostic

학습과 평가 모두 정확히 동일한 [ASEQ](ASEQ) [규칙(rule)](Terminology-Guide)을 사용한 집중형 [실험 실행(run)](Reproduction):

| 학습 [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration) | 학습 successes | L0 | L1 | L2 |
|---|---:|---:|---:|---:|
| [구버전 호환 코드(legacy)](Development-History) filter | 29 | 15 | 14 | 0 |
| 정확히 동일한 [ASEQ](ASEQ) | **50** | **30** | **19** | **1** |

Unseen 평가:

| trained with | L0 | L1 | L2 |
|---|---:|---:|---:|
| 구버전 filter | 1/8 | 1/8 | 0/8 |
| 정확히 동일한 [ASEQ](ASEQ) | **8/8** | **7/8** | **1/8** |

제한:

- [연구(research)](Research-Questions) 난수 시드 1개
- 학습 중 보지 못한 난수 시드 8개
- 집중형 L0~L2 [실험(experiment)](Experiments)
- 현재 전체 AASSR 조건 [최종(final)](Ablation-Benchmarking-and-Reproducibility) 표준 비교 실험가 아님

따라서 이 숫자를 현재 AASSR leaderboard에 넣지 않는다.

---

# 7. RQ4/RQ5 — Prophecy & Calibration

현재 [Prophecy](Prophecy) [명세(contract)](Current-Status):

```text
relational-conditional-mixture-ensemble-v5-status-balanced
```

현재 [Calibration](Calibration) 명세:

```text
semantic-probability-holdout-calibration-v3-status-aware
```

두 모듈은 하나의 [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)으로 평가하면 안 된다.

## 7.1 Prophecy prediction metrics

### State structure
- 의미 기준 [상태를 요약한 표현(descriptor)](State-Representation) [오차(error)](Loss-Functions-and-Class-Imbalance) / [유사도(similarity)](Critic-Support-and-OOD)
- top-k 환경 결과 의미 기준 [품질(quality)](Ablation-Benchmarking-and-Reproducibility)

### Action surface
- [가능 행동 마스크(legal-mask)](Prophecy) [정확도(accuracy)](Ablation-Benchmarking-and-Reproducibility)
- legal-slot precision / [놓치지 않고 찾아낸 비율인 재현율(recall)](Ablation-Benchmarking-and-Reproducibility)

### Terminal
- [현재 활성(active)](Current-Status) / 성공 / [실패(failure)](Replay-Buffer-and-Episode-Boundaries) / [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries) [범주(class)](Loss-Functions-and-Class-Imbalance) 정확도
- [드문(rare)](Loss-Functions-and-Class-Imbalance) [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) 재현율

### Public status
- [범주형(categorical)](Loss-Functions-and-Class-Imbalance) 상태 코드 정확도
- 상태 코드 confusion matrix
- 드문 `403/404/429` 재현율

### Multimodality
- [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration) [구성요소(component)](Research-Architecture) [사용량(usage)](Terminology-Guide)
- 환경 결과 [확률 질량(mass)](Stochasticity-Uncertainty-and-Probability) [수치 범위를 맞추는 정규화(normalization)](Neural-Networks-and-Optimization)
- multiple [실제 관측 경험에 근거한(empirical)](Ablation-Benchmarking-and-Reproducibility) 환경 결과 [의미 보존(preservation)](Ablation-Benchmarking-and-Reproducibility)

## 7.2 Calibration metrics

Prediction 품질와 [신뢰도(reliability)](Calibration) 품질는 다르다.

필요한 진단 실험:

```text
reliability bucket
→ actual holdout prediction quality
```

예:

| Reliability bucket | Observed 품질 |
|---|---:|
| 0.9–1.0 | ? |
| 0.8–0.9 | ? |
| 0.7–0.8 | ? |
| ... | ... |

가능하면 [확률을 고려해 기대되는(expected)](Chance-and-Decision-Nodes) [예측 신뢰도 보정(calibration)](Calibration) 오차류의 summary뿐 아니라 **[의사결정에 중요한(decision-critical)](Calibration) [정보 채널(channel)](Causality-Leakage-and-Evaluation)별 신뢰도**를 같이 본다.

## 7.3 왜 이게 필요해졌나?

2026-08-11 진단 실험에서 전체 의미 기준 품질는 높게 보였지만 [계획기(planner)](Counterfactual-Planning-and-Search) [실제 행동 개입(intervention)](Imagination)은 많은 `403/404/429` 오류를 만들었다.

전체 [탐색의 첫 행동(root)](Imagination) cause는 별도 보관한다.

→ [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)

---

# 8. RQ6 — Critic local support

[Critic](Critic)은 실제 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) [누적 보상(return)](Value-Functions-and-Bellman-Equation)을 학습한다.

하지만 다음은 다르다.

```text
Critic has trained globally
!=
current state/action is supported by real training data
```

## 핵심 ablation

가능한 비교:

```text
same checkpoint
same planner
same Prophecy

A: local support gate OFF
B: local support gate ON
```

단, [판정 관문(gate)](Terminology-Guide) OFF가 위험한 [실제 데이터 근거가 부족한(unsupported)](Critic-Support-and-OOD) 행동을 허용할 수 있으므로 진단 실험 [환경(environment)](Reinforcement-Learning)와 실패 accounting을 엄격히 유지한다.

## Primary mechanism metrics

- [데이터 근거(support)](Critic-Support-and-OOD) [검사를 통과(pass)](Ablation-Benchmarking-and-Reproducibility) 비율
- 데이터 근거 reject 비율
- 근거 부족 high-value [선택 후보(candidate)](Terminology-Guide) [횟수(count)](Terminology-Guide)
- 실제 행동 개입 오차 비율
- successful 실제 행동 개입 비율
- 계획기 activity after 판정 관문

## Fail-closed의 두 실패 모드

### 너무 느슨함

```text
OOD value extrapolation
→ planner exploit
```

### 너무 엄격함

```text
모든 root unsupported
→ intervention 0
→ planner inert
```

따라서 **안전성과 계획기 usability를 동시에 측정**한다.

---

# 9. RQ7 — Imagination same-checkpoint

가장 중요한 [인과적으로 공정한(causal)](Causality-Leakage-and-Evaluation) 실험 중 하나다.

## 9.1 Rule

```text
one AASSR training run
        ↓
frozen checkpoint
     /             \
OFF                   ON
Policy-only       Imagination
```

OFF와 ON을 따로 재학습하면 hard 비교 실패다.

## 9.2 왜 training-time Imagination을 끄는가?

현재 [주요(primary)](Research-Questions) marginal-effect 비교에서 training-time 계획기가 [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries) [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)까지 바꾸면:

```text
planner effect
+ training trajectory effect
+ replay effect
```

가 섞인다.

그래서 현재 manifest는:

```text
training_imagination = disabled-same-checkpoint
```

계약을 둔다.

## 9.3 Planner metrics

### Opportunity
- [계획(plan)](Counterfactual-Planning-and-Search) 횟수
- 탐색의 첫 행동 횟수
- [구조 기반(structural)](Relational-Representation-and-Generalization) 탐색의 첫 행동 횟수

### Gate
- [신뢰하기 어려운(unreliable)](Calibration) suppressions
- 근거 부족 suppressions
- insufficient-advantage suppressions

### Intervention
- [행동 전환(switch)](Imagination) [선택 후보(candidates)](Terminology-Guide)
- 최종 실제 행동 개입s
- changed 행동s
- [직접적인(direct)](Terminology-Guide) [실제로 성공을 만들어내는(success-producing)](Experiments) 실제 행동 개입s
- bad-status 실제 행동 개입s

### Final task
- 성공
- true 실패
- 진전 없이 멈춘
- 외부 제한 종료
- 도달 기준점 도달

### Runtime
- wall [시간(time)](Terminology-Guide)
- [세계 모델(world-model)](Model-Based-RL-and-World-Models) [모델 호출 횟수(calls)](Reproduction)
- [Critic(미래 가치 평가기)](Critic) 호출 횟수
- [중복 계산 제거(dedup)](Reproduction) ratio

## 9.4 Historical 2026-08-11 result

과거 진단 실험:

```text
no-Imagination 4/20
Full           4/20
interventions  86
bad-status     58/86
```

이 숫자는 현재 v5/[상태 코드까지 고려하는(status-aware)](Calibration)/local-support [구조(architecture)](Research-Architecture)의 최종 result가 아니다.

상세: [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)

---

# 10. RQ8 — Five-condition final suite

최종 현재 세대 비교 row:

| Condition | [표현(Representation)](Relational-Representation-and-Generalization) | Model-based 구성요소 | 역할 |
|---|---|---|---|
| `dqn_raw` | 원본 현재 | none | corrected [환경 예측 모델 없이 직접 학습하는(model-free)](Reinforcement-Learning) 비교 기준 |
| `dqn_relational` | 관계 기반 현재 | none | 표현 [구성요소 제거 비교(ablation)](Ablation-Benchmarking-and-Reproducibility) |
| `dreamerv3_relational` | 관계 기반 현재 | [공식 구현(official)](Experiments) [DreamerV3(외부 세계 모델 강화학습 비교군)](Experiments) | 환경이 주는 외부 모델 기반 비교 기준 |
| `aassr_current_no_imagination` | 관계 기반 현재 | AASSR models, 계획기 OFF | non-[Imagination(가상 미래 탐색)](Imagination) AASSR stack |
| `aassr_current_full` | 관계 기반 현재 | AASSR 계획기 ON | [Imagination](Imagination) [다른 조건이 같을 때의 추가 기여(marginal)](Ablation-Benchmarking-and-Reproducibility) [효과(effect)](Ablation-Benchmarking-and-Reproducibility) |

## 10.1 AASSR OFF/ON checkpoint 수

AASSR는 체크포인트 하나다.

```text
Raw DQN checkpoint
Relational DQN checkpoint
DreamerV3 checkpoint
AASSR checkpoint
```

그 AASSR 체크포인트를 OFF/ON 두 평가 결과 유형로 사용한다.

## 10.2 Fair sample budget

과학적 [표본(sample)](Ablation-Benchmarking-and-Reproducibility) 실험 예산은 **실제 [더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills) 환경 행동**으로 센다.

Imagined [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search) [단계(step)](Terminology-Guide)은 환경 표본로 세지 않는다.

다만 [계산(compute)](Reproduction) cost는 별도 [실행 구조(runtime)](Current-Status) 평가지표으로 반드시 보고한다.

## 10.3 DreamerV3 fairness boundary

[DreamerV3](Experiments) 비교 기준은 upstream 알고리즘을 AASSR에 유리하도록 개조하는 것이 아니라, pinned 공식 implementation을 현재 [관측(observation)](MDP-and-POMDP)/행동 interface에 [서로 다른 입력·행동 형식을 연결하는 변환기(adapter)](Experiments)로 연결하는 방향을 사용한다.

최종 result에는 반드시:

- upstream pin
- preset
- JAX platform
- train ratio
- dtype
- 변환 어댑터 명세

를 기록한다.

---

# 11. RQ9 — Skill

[Skill](Skills)은 주요 five-condition suite와 별도 작동 원리 실험로 보는 편이 해석이 쉽다.

질문:

> repeated successful 실제 ASeq를 관계 기반 [재사용 가능한 틀(template)](Skills)로 승격하면 학습 중 보지 못한 난수 시드에서 primitive-only보다 재사용 효율이 좋아지는가?

## 비교 후보

```text
primitive-only
vs
concrete macro
vs
relational Skill
```

## 지표

- [Skill(성공 절차 재사용)](Skills) 난이도 승급 횟수
- 난이도 승급 precision
- 실제 개체를 구분하는 [새 문제의 실제 객체에 다시 연결하는 것(rebinding)](Skills) 성공
- [현재 사용할 수 없는(unavailable)](Terminology-Guide) 기본 행동 단위 비율
- [Skill](Skills) completion 성공
- 상태 전이s saved
- [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) 가상 미래 전개 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes) survival
- 기본 행동 단위 [탐색(exploration)](Exploration-and-Exploitation) [후보 억제(suppression)](ASEQ)

[Skill](Skills)이 잘 작동해도 그것이 곧 “창의성”은 아니다.

관련: [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

---

# 12. 통계와 보고 형식

최종 [성능(performance)](Ablation-Benchmarking-and-Reproducibility) table은 평균 성공률 하나로 끝내지 않는다.

## 12.1 최소 보고 단위

```text
research seed
×
tier / evaluation pool
×
condition
```

각 cell에서:

- 성공 횟수 / denominator
- true 실패
- 진전 없이 멈춘
- 외부 제한 종료
- mean/median requests
- 실행 구조

을 보존한다.

## 12.2 Aggregate

가능하면:

- mean across 연구 난수 시드s
- standard deviation
- 난수 시드-level 원본 values
- binomial 성공 [불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability) 또는 적절한 [예측 신뢰 정도(confidence)](Calibration) interval

을 함께 보고한다.

작은 `n`에서 소수점 둘째 자리까지 정밀한 차이를 과해석하지 않는다.

## 12.3 Mechanism metric과 final metric 분리

```text
Prophecy accuracy
Critic support pass
Planner intervention
```

은 **작동 원리 평가지표**이다.

```text
proof success
true failure
```

은 **연구 과제 평가지표**이다.

Mechanism 평가지표이 좋아졌다고 최종 성공가 자동으로 좋아진 것은 아니다.

---

# 13. 최종 claim gate

다음 표현은 증거 난이도 단계을 충족하기 전까지 사용하지 않는다.

```text
“AASSR이 DQN보다 우수하다.”
“AASSR이 DreamerV3보다 우수하다.”
“Imagination이 성능을 향상시킨다.”
“Relational representation이 일반화를 유의미하게 개선한다.”
```

각 문장을 사용하려면 대응되는 RQ의 현재 세대 controlled 증거가 있어야 한다.

Claim 상태는 [Evidence Matrix](Evidence-Matrix)와 [Current Status](Current-Status)에 기록한다.

---

## 결과 분류 예시

```text
24/24 stall → 0/24
= ASEQ mechanism diagnostic

2026-08-11 4/20 vs 4/20, 86 interventions
= historical Imagination root-cause diagnostic

current Prophecy v5 manifest
= architecture contract

future five-condition multi-seed aggregate
= current performance evidence

final unseen blind set
= final claim evidence
```

---

## 다음으로 읽기

- [Research Questions](Research-Questions)
- [Evidence Matrix](Evidence-Matrix)
- [Current Status](Current-Status)
- [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)
- [Reproduction](Reproduction)
