# 연구 질문-증거 연결표 (Evidence Matrix)

이 페이지는 AASSR의 **연구 질문 → 가설 → 비교 조건 → 측정 지표 → 현재 [증거(evidence)](Evidence-Matrix) → 주장 가능한 범위**를 한 장에서 연결한다.

> [!IMPORTANT]
> 이 표는 성능 숫자를 모아놓은 leaderboard가 아니다. 각 숫자가 **무엇을 검증하는지**, 그리고 **무엇까지는 아직 말할 수 없는지**를 추적하기 위한 연구 지도다.

---

## 0. 읽는 법

AASSR 위키에서는 다음을 서로 다른 증거 level로 취급한다.

```text
구현 존재
  ↓
unit / regression contract
  ↓
mechanism diagnostic
  ↓
reduced current-generation validation
  ↓
multi-seed benchmark
  ↓
final blinded evaluation
```

위 단계는 서로 대체되지 않는다.

예를 들어:

```text
Imagination 코드가 동작함
!=
Imagination이 실제 행동을 바꿈
!=
Imagination이 성공률을 높임
!=
여러 seed에서 안정적으로 개선함
```

관련 개념: [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility), [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation)

---

# RQ1. 희소 보상만으로 최초 성공을 발견할 수 있는가?

## 질문

> [guided trajectory](Causality-Leakage-and-Evaluation), oracle [행동(action)](Reinforcement-Learning) injection, intermediate [reward shaping](Sparse-Reward-and-Credit-Assignment) 없이도 에이전트가 실제 성공 경험을 스스로 만들 수 있는가?

## 가설

**H1:** [Curriculum Learning](Curriculum-Learning), [exploration](Exploration-and-Exploitation), [Policy](Policy), [ASEQ](ASEQ)를 이용하면 sparse external [보상(reward)](Sparse-Reward-and-Credit-Assignment)만으로 최초 proof가 발생할 수 있다.

**H0:** 성공 trajectory를 직접 주입하지 않으면 학습 예산 안에서 성공 experience가 발생하지 않는다.

## 독립변수

- autonomous [학습 주체(learner)](Terminology-Guide) configuration
- [난이도 조절 학습(curriculum)](Curriculum-Learning) schedule / promotion-demotion rule
- [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ) guard 사용 여부

## 종속변수

- first-proof [상태 전이(transition)](MDP-and-POMDP) index
- [학습(training)](Terminology-Guide) [성공(success)](Terminology-Guide) count
- difficulty level reached
- stalled [한 번의 문제 풀이 구간(episode)](Terminology-Guide) rate

## 통제해야 할 것

- external 보상 [명세(contract)](Current-Status)
- real 상태 전이 budget
- scenario [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility) pool
- [숨겨진(hidden)](MDP-and-POMDP) information access

## 현재 evidence

과거 autonomous pilot과 exact-[ASEQ](ASEQ) focused run에서 **성공 경험 자체가 사람의 정답 행동 sequence 없이 발생할 수 있음**은 관측됐다.

## 지금 주장 가능한 것

> AASSR 계열 autonomous 학습은 적어도 일부 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility) 설정에서 sparse [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) 보상만으로 실제 성공 experience를 생성할 수 있다.

## 아직 주장하면 안 되는 것

- 모든 난도에서 안정적으로 최초 성공을 발견한다.
- 난이도 조절 학습 없이도 같은 성능을 낸다.
- 현재 Full AASSR이 다른 [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)보다 sample-efficient하다.

관련 페이지: [Sparse Reward Problem](Sparse-Reward-Problem), [Curriculum Learning](Curriculum-Learning), [Experiments](Experiments)

---

# RQ2. Relational representation이 unseen transfer를 개선하는가?

## 질문

> concrete identifier를 외우는 대신 역할과 관계를 표현하면 이름이 바뀐 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) scenario에서 더 잘 일반화하는가?

관련 개념: [Relational Representation & Generalization](Relational-Representation-and-Generalization), [State Representation](State-Representation)

## 가설

**H1:** 동일한 학습 budget에서 `dqn_relational`이 `dqn_raw`보다 학습 중 보지 못한 난수 시드에서 높은 성공률 또는 더 나은 milestone reach를 보인다.

**H0:** [관계 기반 표현(relational representation)](Relational-Representation-and-Generalization)은 raw [표현(representation)](Relational-Representation-and-Generalization)보다 학습 중 보지 못한 performance를 개선하지 않는다.

## 핵심 ablation

```text
dqn_raw
   ↓
   representation만 변경
dqn_relational
```

## 독립변수

- [상태(state)](State-Representation)/행동 표현

## 종속변수

- 학습 중 보지 못한 성공 rate
- milestone reach
- stalled rate
- mean 상태 전이s / requests

## 고정 변수

- [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) [구조(architecture)](Research-Architecture)와 optimizer 계열
- 보상
- 학습 budget
- 난이도 조절 학습 exposure
- [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility) 난수 시드s

## 현재 상태

Relational 표현 자체는 [현재 실행 구조(current runtime)](Current-Status)에서 사용 중이지만, **[현재 세대(current-generation)](Current-Status) 최종 multi-난수 시드 표현 [구성요소 제거 비교(ablation)](Ablation-Benchmarking-and-Reproducibility)은 최종 suite 증거로 별도 보고해야 한다.**

## 주장 경계

[표현(Representation)](Relational-Representation-and-Generalization)이 rename invariance를 구현한다는 code-level 사실과, 학습 중 보지 못한 성공를 유의미하게 높인다는 performance [연구 주장(claim)](Evidence-Matrix)은 구분한다.

---

# RQ3. ASEQ가 진전 없는 self-loop를 줄이는가?

## 질문

> 실제 [ASEQ](ASEQ) `(S,A,S')` 중 `S → A → S`로 반복되는 semantic [제자리 반복(self-loop)](ASEQ)만 억제하면 탐색 정체가 줄어드는가?

## 가설

**H1:** exact [ASEQ](ASEQ) guard는 greedy policy의 stalled 한 번의 문제 풀이 구간를 줄이며, 진행하는 동일 행동 반복은 막지 않는다.

**H0:** [ASEQ](ASEQ) guard는 stalled rate를 줄이지 못하거나 필요한 반복 행동까지 막아 성능을 해친다.

## 핵심 diagnostic

과거 L1 학습 중 보지 못한 [진단 실험(diagnostic)](Evidence-Matrix):

```text
raw greedy stalled       24 / 24
exact ASEQ stalled        0 / 24
```

이 결과는 **제자리 반복 suppression [메커니즘 증거(mechanism evidence)](Evidence-Matrix)**다.

## 추가로 봐야 할 지표

- 성공 rate
- stalled rate
- suppressed-행동 count
- false suppression: `S → A → S'`, `S' != S`인데 막힌 횟수

## 지금 주장 가능한 것

> 관측된 semantic `S → A → S` 반복을 이용하는 exact [ASEQ](ASEQ) guard가 특정 학습 중 보지 못한 진단 실험에서 deterministic stall을 제거했다.

## 아직 주장하면 안 되는 것

> [ASEQ](ASEQ) 하나만으로 전체 AASSR의 최종 성능이 높아진다.

---

# RQ4. Prophecy는 usable stochastic world model인가?

## 질문

> 현재 [공개 관측 상태(public state)](State-Representation), 행동, causal [Knowledge(에피소드 지식)](Knowledge)로부터 [계획기(planner)](Counterfactual-Planning-and-Search)가 사용할 수 있는 next-outcome distribution을 학습하는가?

관련 페이지: [Prophecy](Prophecy), [Model-Based RL & World Models](Model-Based-RL-and-World-Models), [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

## current contract

현재 `main`의 `current_manifest.py` 기준 [Prophecy(미래 예측 모델)](Prophecy)는:

```text
relational-conditional-mixture-ensemble-v5-status-balanced
```

이며 다음을 예측하는 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [세계 모델(world model)](Model-Based-RL-and-World-Models) 계열이다.

- [관계 기반(relational)](Relational-Representation-and-Generalization) next descriptor
- latest [공개된(public)](State-Representation) HTTP [상태 코드(status)](Terminology-Guide)
- [가능 행동 마스크(legal-action mask)](Prophecy)
- [현재 활성(active)](Current-Status) / 성공 / [실패(failure)](Replay-Buffer-and-Episode-Boundaries) / [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries) outcome
- mixture [결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability)

## 가설

**H1:** [검증용 분리 데이터(holdout)](Calibration) real 상태 전이s에서 [Prophecy](Prophecy)가 decision-relevant next-state structure와 outcome distribution을 충분히 보존한다.

**H0:** [예측(prediction)](Terminology-Guide) error 또는 mode collapse 때문에 [계획(planning)](Counterfactual-Planning-and-Search)에 사용할 수 없다.

## 핵심 지표

- probability-weighted semantic quality
- top-k semantic quality
- legal-mask accuracy
- 에피소드 종료-class accuracy
- 공개된 HTTP-status [범주형(categorical)](Loss-Functions-and-Class-Imbalance) accuracy
- mixture mode coverage
- rare-status recall
- calibration quality

## 중요한 금지 해석

```text
semantic similarity가 높음
!=
planner가 안전하게 사용 가능
```

2026-08-11 [과거 기록(historical)](Development-History) 진단 실험은 이 차이를 보여준 대표 사례다. 자세한 내용은 [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11).

---

# RQ5. Calibration은 prediction reliability를 decision에 쓸 수 있게 만드는가?

## 질문

> world-model 결과 확률와 [학습 모델(model)](Terminology-Guide) [신뢰도(reliability)](Calibration)를 분리했을 때, unreliable 예측을 실제 [기본 행동 덮어쓰기(override)](Imagination) 전에 걸러낼 수 있는가?

관련 페이지: [Calibration](Calibration), [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability)

## 핵심 구분

```text
Outcome probability
= 환경 결과가 발생할 확률

Prediction reliability
= 그 probability / future prediction을 믿어도 되는 정도
```

## 가설

**H1:** 검증용 분리 데이터-calibrated 신뢰도 [판정 관문(gate)](Terminology-Guide)를 적용하면 prediction-error-heavy [실제 행동 개입(intervention)](Imagination)을 줄이면서 유용한 계획 opportunity는 유지한다.

## 측정 지표

- 판정 관문 pass rate
- 판정 관문 reject rate
- accepted vs rejected branch error
- 신뢰도 bucket별 observed 예측 quality
- 실제 행동 개입 error rate

## 현재 상태

현재 실행 구조은 `semantic-probability-holdout-calibration-v3-status-aware` 계약을 사용한다. 이것은 구현 명세이며, 최종 causal performance effect는 [현재(current)](Current-Status) reduced/final 구성요소 제거 비교으로 검증해야 한다.

---

# RQ6. Critic value를 local support 없이 믿어도 되는가?

## 질문

> [Critic](Critic)이 global 학습을 받았더라도 현재 imagined 상태/행동이 학습 distribution 밖이라면 그 [가치(value)](Value-Functions-and-Bellman-Equation)를 믿어도 되는가?

답해야 하는 두 질문은 다르다.

```text
Critic ready?
= 모델이 전반적으로 학습되었나?

Locally supported?
= 지금 이 state/action 주변에 real training evidence가 있나?
```

관련 페이지: [Critic, Support & OOD](Critic-Support-and-OOD)

## 가설

**H1:** local real-training [데이터 근거(support)](Critic-Support-and-OOD) 판정 관문를 추가하면 unsupported high-value 기본 행동 덮어쓰기를 억제한다.

**H0:** [국소 데이터 근거(local support)](Critic-Support-and-OOD)는 실제 행동 개입 quality에 영향을 주지 않거나 지나치게 보수적으로 계획기를 비활성화한다.

## 지표

- 국소 데이터 근거 pass/fail
- 데이터 근거 distance distribution
- unsupported 실제 행동 개입 count
- supported 실제 행동 개입 outcome quality
- 계획기 실제 행동 개입 count
- false-closed rate

## historical evidence

2026-08-11 진단 실험에서 학습 successes는 낮은 난이도 조절 학습 level에 집중되었지만 [Imagination(가상 미래 탐색)](Imagination) 기본 행동 덮어쓰기는 높은 학습 중 보지 못한 level에서 대량 발생했다. 이 결과는 국소 데이터 근거 판정 관문를 설계하게 만든 **root-cause 증거**다.

---

# RQ7. Imagination의 marginal effect는 무엇인가?

## 질문

> 같은 학습된 AASSR [체크포인트(checkpoint)](Reproduction)에서 계획기를 켰을 때 [Policy(정책 모델)](Policy)-only보다 실제 첫 행동과 최종 성능이 좋아지는가?

관련 페이지: [Imagination](Imagination), [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)

## 반드시 지켜야 하는 비교

```text
one AASSR training run
        ↓
frozen checkpoint
      /            \
planner OFF      planner ON
```

OFF/ON을 따로 학습하면 [Imagination](Imagination)의 causal marginal effect를 분리할 수 없다.

## H1

`aassr_current_full`이 동일 체크포인트의 `aassr_current_no_imagination`보다 학습 중 보지 못한 성공 / 실패 trade-off에서 개선된다.

## H0

계획기 ON/OFF 차이가 없거나 계획기가 오류를 늘린다.

## 지표

### Final task
- 성공
- true 실패
- stalled
- 외부 제한 종료
- milestone reach

### Planner mechanism
- plans
- switch candidates
- 신뢰도 rejects
- 데이터 근거 rejects
- final 실제 행동 개입s
- changed 행동s
- direct-success 실제 행동 개입s
- bad-status 실제 행동 개입s

### Runtime
- wall time
- [Prophecy](Prophecy) calls
- [Critic(미래 가치 평가기)](Critic) calls
- structural [탐색의 첫 행동(root)](Imagination) dedup ratio

## historical warning

2026-08-11의 `4/20 vs 4/20`, 86 실제 행동 개입 결과는 **현재 repaired 구조의 최종 성능 결과가 아니다.** 그것은 잘못된 실제 행동 개입 원인을 찾은 과거 기록 진단 실험이다.

---

# RQ8. AASSR 전체가 strong baseline보다 나은가?

## 질문

> 동일한 [관측(observation)](MDP-and-POMDP)/보상/budget/평가 [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)에서 현재 AASSR은 model-free 및 model-based 비교 기준보다 더 높은 장기 문제 해결 성능을 보이는가?

## current comparison chain

```text
dqn_raw
  ↓ representation effect
dqn_relational
  ↓ AASSR non-Imagination stack
aassr_current_no_imagination
  ↓ Imagination marginal effect
aassr_current_full
```

외부 model-based family 비교:

```text
dreamerv3_relational
↔
aassr_current_full
```

## 핵심 조건

- 같은 real 상태 전이 budget
- 같은 train/eval 난수 시드 실험 규칙
- 같은 external [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)
- 숨겨진 information 없음
- [최종 비공개 평가(final blind)](Ablation-Benchmarking-and-Reproducibility) set 사전 미사용
- AASSR OFF/ON [같은 체크포인트(same checkpoint)](Experiments)

## 최종 지표

- 성공 rate by tier
- aggregate 성공
- true 실패
- stalled/외부 제한 종료
- mean requests
- [실행 구조(runtime)](Current-Status)
- 난수 시드 variance

## 현재 주장 상태

> **Pending.** 현재 세대 full multi-난수 시드/final-blind 증거가 완료되기 전에는 “AASSR이 [DQN](Q-Learning-DQN-and-TD)/[DreamerV3(외부 세계 모델 강화학습 비교군)](Experiments)보다 우수하다”고 쓰지 않는다.

---

# RQ9. Skill은 성공 구조를 transfer 가능한 macro로 재사용하는가?

## 질문

> 반복 성공한 real ASeq를 [relational template](Relational-Representation-and-Generalization)로 승격하면 concrete ID가 달라진 학습 중 보지 못한 scenario에서도 재사용할 수 있는가?

관련 페이지: [Skills](Skills), [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

## 가설

**H1:** 관계 기반 [Skill(성공 절차 재사용)](Skills)은 raw concrete macro보다 학습 중 보지 못한 rebinding 성공률이 높고 primitive-only search cost를 줄인다.

## 지표

- promotion count
- rebinding 성공
- unavailable primitive rate
- skill-completion 성공
- primitive-only 대비 상태 전이 saving
- 확률적 rollout 실패

## 현재 상태

[Skill](Skills)은 현재 실행 구조에 구현되어 있지만 전체 final 표준 비교 실험의 primary performance 연구 주장과는 별도 증거로 다룬다.

---

# 장기 질문: Creativity

> 에이전트가 사람이 제공한 정답 trajectory나 이미 저장된 [Skill](Skills)을 그대로 복제하지 않고도 **새로운 유효한 해결 경로**를 반복적으로 만들어내는가?

이 질문은 현재 primary 표준 비교 실험보다 한 단계 뒤에 둔다.

먼저 필요한 조건:

```text
autonomous success
→ unseen transfer
→ reliable planning
→ path diversity analysis
```

Creativity는 단순 행동 diversity가 아니다. 유효한 목표 달성 경로 중 **학습 solution structure와 실질적으로 다른 경로**인지 정의와 [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)이 필요하다.

---

# 한눈에 보는 상태

| RQ | 핵심 비교 / 증거 | 현재 상태 | 지금 가능한 주장 |
|---|---|---|---|
| RQ1 최초 성공 | autonomous sparse-보상 학습 | 부분 증거 | 일부 setting에서 autonomous proof 가능 |
| RQ2 관계 기반 [전이(transfer)](Relational-Representation-and-Generalization) | raw vs 관계 기반 [DQN](Q-Learning-DQN-and-TD) | final 현재 suite 필요 | 표현 명세는 현재 활성 |
| RQ3 [ASEQ](ASEQ) | exact 제자리 반복 guard 진단 실험 | 메커니즘 증거 있음 | observed 제자리 반복 stall 억제 |
| RQ4 [Prophecy](Prophecy) | 검증용 분리 데이터 world-model [평가지표(metrics)](Ablation-Benchmarking-and-Reproducibility) | 현재 활성 + [검증(validation)](Ablation-Benchmarking-and-Reproducibility) 진행 | 확률적 v5 명세 현재 활성 |
| RQ5 [Calibration(예측 신뢰도 보정)](Calibration) | 신뢰도 판정 관문 audit | 현재 활성 + 검증 진행 | [상태 코드까지 고려하는(status-aware)](Calibration) calibration 현재 활성 |
| RQ6 [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD) | 국소 데이터 근거 구성요소 제거 비교 | 현재 활성 + 검증 진행 | [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) 데이터 근거 판정 관문 현재 활성 |
| RQ7 [Imagination](Imagination) | [같은 체크포인트(same-checkpoint)](Experiments) OFF vs ON | final performance 미확정 | 계획기 semantics 현재 활성 |
| RQ8 전체 성능 | five-condition + blind | pending | 우월성 주장 금지 |
| RQ9 [Skill](Skills) | primitive vs 관계 기반 skill | 제한적 증거 | mechanism experimental |
| Creativity | path novelty analysis | future | primary 연구 주장 아님 |

---

## 다음으로 읽기

- [Research Questions](Research-Questions)
- [Experiments](Experiments)
- [Current Status](Current-Status)
- [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)
- [Reproduction](Reproduction)
