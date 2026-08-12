# Evidence Matrix

이 페이지는 AASSR의 **연구 질문 → 가설 → 비교 조건 → 측정 지표 → 현재 evidence → 주장 가능한 범위**를 한 장에서 연결한다.

> [!IMPORTANT]
> 이 표는 성능 숫자를 모아놓은 leaderboard가 아니다. 각 숫자가 **무엇을 검증하는지**, 그리고 **무엇까지는 아직 말할 수 없는지**를 추적하기 위한 연구 지도다.

---

## 0. 읽는 법

AASSR 위키에서는 다음을 서로 다른 evidence level로 취급한다.

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

> [guided trajectory](Causality-Leakage-and-Evaluation), oracle action injection, intermediate [reward shaping](Sparse-Reward-and-Credit-Assignment) 없이도 에이전트가 실제 성공 경험을 스스로 만들 수 있는가?

## 가설

**H1:** [Curriculum Learning](Curriculum-Learning), [exploration](Exploration-and-Exploitation), [Policy](Policy), [ASEQ](ASEQ)를 이용하면 sparse external reward만으로 최초 proof가 발생할 수 있다.

**H0:** 성공 trajectory를 직접 주입하지 않으면 학습 예산 안에서 성공 experience가 발생하지 않는다.

## 독립변수

- autonomous learner configuration
- curriculum schedule / promotion-demotion rule
- ASEQ guard 사용 여부

## 종속변수

- first-proof transition index
- training success count
- difficulty level reached
- stalled episode rate

## 통제해야 할 것

- external reward contract
- real transition budget
- scenario seed pool
- hidden information access

## 현재 evidence

과거 autonomous pilot과 exact-ASEQ focused run에서 **성공 경험 자체가 사람의 정답 action sequence 없이 발생할 수 있음**은 관측됐다.

## 지금 주장 가능한 것

> AASSR 계열 autonomous training은 적어도 일부 benchmark 설정에서 sparse terminal reward만으로 실제 성공 experience를 생성할 수 있다.

## 아직 주장하면 안 되는 것

- 모든 난도에서 안정적으로 최초 성공을 발견한다.
- curriculum 없이도 같은 성능을 낸다.
- 현재 Full AASSR이 다른 baseline보다 sample-efficient하다.

관련 페이지: [Sparse Reward Problem](Sparse-Reward-Problem), [Curriculum Learning](Curriculum-Learning), [Experiments](Experiments)

---

# RQ2. Relational representation이 unseen transfer를 개선하는가?

## 질문

> concrete identifier를 외우는 대신 역할과 관계를 표현하면 이름이 바뀐 unseen scenario에서 더 잘 일반화하는가?

관련 개념: [Relational Representation & Generalization](Relational-Representation-and-Generalization), [State Representation](State-Representation)

## 가설

**H1:** 동일한 training budget에서 `dqn_relational`이 `dqn_raw`보다 unseen seed에서 높은 성공률 또는 더 나은 milestone reach를 보인다.

**H0:** relational representation은 raw representation보다 unseen performance를 개선하지 않는다.

## 핵심 ablation

```text
dqn_raw
   ↓
   representation만 변경
dqn_relational
```

## 독립변수

- state/action representation

## 종속변수

- unseen success rate
- milestone reach
- stalled rate
- mean transitions / requests

## 고정 변수

- DQN architecture와 optimizer 계열
- reward
- training budget
- curriculum exposure
- evaluation seeds

## 현재 상태

Relational representation 자체는 current runtime에서 사용 중이지만, **current-generation 최종 multi-seed representation ablation은 최종 suite evidence로 별도 보고해야 한다.**

## 주장 경계

Representation이 rename invariance를 구현한다는 code-level 사실과, unseen success를 유의미하게 높인다는 performance claim은 구분한다.

---

# RQ3. ASEQ가 진전 없는 self-loop를 줄이는가?

## 질문

> 실제 [ASEQ](ASEQ) `(S,A,S')` 중 `S → A → S`로 반복되는 semantic self-loop만 억제하면 탐색 정체가 줄어드는가?

## 가설

**H1:** exact ASEQ guard는 greedy policy의 stalled episode를 줄이며, 진행하는 동일 action 반복은 막지 않는다.

**H0:** ASEQ guard는 stalled rate를 줄이지 못하거나 필요한 반복 행동까지 막아 성능을 해친다.

## 핵심 diagnostic

과거 L1 unseen diagnostic:

```text
raw greedy stalled       24 / 24
exact ASEQ stalled        0 / 24
```

이 결과는 **self-loop suppression mechanism evidence**다.

## 추가로 봐야 할 지표

- success rate
- stalled rate
- suppressed-action count
- false suppression: `S → A → S'`, `S' != S`인데 막힌 횟수

## 지금 주장 가능한 것

> 관측된 semantic `S → A → S` 반복을 이용하는 exact ASEQ guard가 특정 unseen diagnostic에서 deterministic stall을 제거했다.

## 아직 주장하면 안 되는 것

> ASEQ 하나만으로 전체 AASSR의 최종 성능이 높아진다.

---

# RQ4. Prophecy는 usable stochastic world model인가?

## 질문

> 현재 public state, action, causal Knowledge로부터 planner가 사용할 수 있는 next-outcome distribution을 학습하는가?

관련 페이지: [Prophecy](Prophecy), [Model-Based RL & World Models](Model-Based-RL-and-World-Models), [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

## current contract

현재 `main`의 `current_manifest.py` 기준 Prophecy는:

```text
relational-conditional-mixture-ensemble-v5-status-balanced
```

이며 다음을 예측하는 stochastic world model 계열이다.

- relational next descriptor
- latest public HTTP status
- legal-action mask
- active / success / failure / truncation outcome
- mixture outcome probability

## 가설

**H1:** holdout real transitions에서 Prophecy가 decision-relevant next-state structure와 outcome distribution을 충분히 보존한다.

**H0:** prediction error 또는 mode collapse 때문에 planning에 사용할 수 없다.

## 핵심 지표

- probability-weighted semantic quality
- top-k semantic quality
- legal-mask accuracy
- terminal-class accuracy
- public HTTP-status categorical accuracy
- mixture mode coverage
- rare-status recall
- calibration quality

## 중요한 금지 해석

```text
semantic similarity가 높음
!=
planner가 안전하게 사용 가능
```

2026-08-11 historical diagnostic은 이 차이를 보여준 대표 사례다. 자세한 내용은 [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11).

---

# RQ5. Calibration은 prediction reliability를 decision에 쓸 수 있게 만드는가?

## 질문

> world-model outcome probability와 model reliability를 분리했을 때, unreliable prediction을 실제 override 전에 걸러낼 수 있는가?

관련 페이지: [Calibration](Calibration), [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability)

## 핵심 구분

```text
Outcome probability
= 환경 결과가 발생할 확률

Prediction reliability
= 그 probability / future prediction을 믿어도 되는 정도
```

## 가설

**H1:** holdout-calibrated reliability gate를 적용하면 prediction-error-heavy intervention을 줄이면서 유용한 planning opportunity는 유지한다.

## 측정 지표

- gate pass rate
- gate reject rate
- accepted vs rejected branch error
- reliability bucket별 observed prediction quality
- intervention error rate

## 현재 상태

current runtime은 `semantic-probability-holdout-calibration-v3-status-aware` 계약을 사용한다. 이것은 구현 contract이며, 최종 causal performance effect는 current reduced/final ablation으로 검증해야 한다.

---

# RQ6. Critic value를 local support 없이 믿어도 되는가?

## 질문

> [Critic](Critic)이 global training을 받았더라도 현재 imagined state/action이 training distribution 밖이라면 그 value를 믿어도 되는가?

답해야 하는 두 질문은 다르다.

```text
Critic ready?
= 모델이 전반적으로 학습되었나?

Locally supported?
= 지금 이 state/action 주변에 real training evidence가 있나?
```

관련 페이지: [Critic, Support & OOD](Critic-Support-and-OOD)

## 가설

**H1:** local real-training support gate를 추가하면 unsupported high-value override를 억제한다.

**H0:** local support는 intervention quality에 영향을 주지 않거나 지나치게 보수적으로 planner를 비활성화한다.

## 지표

- local support pass/fail
- support distance distribution
- unsupported intervention count
- supported intervention outcome quality
- planner intervention count
- false-closed rate

## historical evidence

2026-08-11 diagnostic에서 training successes는 낮은 curriculum level에 집중되었지만 Imagination override는 높은 unseen level에서 대량 발생했다. 이 결과는 local support gate를 설계하게 만든 **root-cause evidence**다.

---

# RQ7. Imagination의 marginal effect는 무엇인가?

## 질문

> 같은 학습된 AASSR checkpoint에서 planner를 켰을 때 Policy-only보다 실제 첫 행동과 최종 성능이 좋아지는가?

관련 페이지: [Imagination](Imagination), [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes)

## 반드시 지켜야 하는 비교

```text
one AASSR training run
        ↓
frozen checkpoint
      /            \
planner OFF      planner ON
```

OFF/ON을 따로 학습하면 Imagination의 causal marginal effect를 분리할 수 없다.

## H1

`aassr_current_full`이 동일 checkpoint의 `aassr_current_no_imagination`보다 unseen success / failure trade-off에서 개선된다.

## H0

planner ON/OFF 차이가 없거나 planner가 오류를 늘린다.

## 지표

### Final task
- success
- true failure
- stalled
- truncation
- milestone reach

### Planner mechanism
- plans
- switch candidates
- reliability rejects
- support rejects
- final interventions
- changed actions
- direct-success interventions
- bad-status interventions

### Runtime
- wall time
- Prophecy calls
- Critic calls
- structural root dedup ratio

## historical warning

2026-08-11의 `4/20 vs 4/20`, 86 intervention 결과는 **현재 repaired architecture의 최종 성능 결과가 아니다.** 그것은 잘못된 intervention 원인을 찾은 historical diagnostic이다.

---

# RQ8. AASSR 전체가 strong baseline보다 나은가?

## 질문

> 동일한 observation/reward/budget/evaluation protocol에서 current AASSR은 model-free 및 model-based baseline보다 더 높은 장기 문제 해결 성능을 보이는가?

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

- 같은 real transition budget
- 같은 train/eval seed protocol
- 같은 external sparse reward
- hidden information 없음
- final blind set 사전 미사용
- AASSR OFF/ON same checkpoint

## 최종 지표

- success rate by tier
- aggregate success
- true failure
- stalled/truncation
- mean requests
- runtime
- seed variance

## 현재 주장 상태

> **Pending.** current-generation full multi-seed/final-blind evidence가 완료되기 전에는 “AASSR이 DQN/DreamerV3보다 우수하다”고 쓰지 않는다.

---

# RQ9. Skill은 성공 구조를 transfer 가능한 macro로 재사용하는가?

## 질문

> 반복 성공한 real ASeq를 [relational template](Relational-Representation-and-Generalization)로 승격하면 concrete ID가 달라진 unseen scenario에서도 재사용할 수 있는가?

관련 페이지: [Skills](Skills), [Hierarchical RL & Skills](Hierarchical-RL-and-Skills)

## 가설

**H1:** relational Skill은 raw concrete macro보다 unseen rebinding 성공률이 높고 primitive-only search cost를 줄인다.

## 지표

- promotion count
- rebinding success
- unavailable primitive rate
- skill-completion success
- primitive-only 대비 transition saving
- stochastic rollout failure

## 현재 상태

Skill은 current runtime에 구현되어 있지만 전체 final benchmark의 primary performance claim과는 별도 evidence로 다룬다.

---

# 장기 질문: Creativity

> 에이전트가 사람이 제공한 정답 trajectory나 이미 저장된 Skill을 그대로 복제하지 않고도 **새로운 유효한 해결 경로**를 반복적으로 만들어내는가?

이 질문은 현재 primary benchmark보다 한 단계 뒤에 둔다.

먼저 필요한 조건:

```text
autonomous success
→ unseen transfer
→ reliable planning
→ path diversity analysis
```

Creativity는 단순 action diversity가 아니다. 유효한 목표 달성 경로 중 **training solution structure와 실질적으로 다른 경로**인지 정의와 metric이 필요하다.

---

# 한눈에 보는 상태

| RQ | 핵심 비교 / evidence | 현재 상태 | 지금 가능한 주장 |
|---|---|---|---|
| RQ1 최초 성공 | autonomous sparse-reward training | 부분 evidence | 일부 setting에서 autonomous proof 가능 |
| RQ2 relational transfer | raw vs relational DQN | final current suite 필요 | representation contract는 active |
| RQ3 ASEQ | exact self-loop guard diagnostic | mechanism evidence 있음 | observed self-loop stall 억제 |
| RQ4 Prophecy | holdout world-model metrics | active + validation 진행 | stochastic v5 contract active |
| RQ5 Calibration | reliability gate audit | active + validation 진행 | status-aware calibration active |
| RQ6 Critic support | local support ablation | active + validation 진행 | fail-closed support gate active |
| RQ7 Imagination | same-checkpoint OFF vs ON | final performance 미확정 | planner semantics active |
| RQ8 전체 성능 | five-condition + blind | pending | 우월성 주장 금지 |
| RQ9 Skill | primitive vs relational skill | 제한적 evidence | mechanism experimental |
| Creativity | path novelty analysis | future | primary claim 아님 |

---

## 다음으로 읽기

- [Research Questions](Research-Questions)
- [Experiments](Experiments)
- [Current Status](Current-Status)
- [Historical Imagination Diagnostic — 2026-08-11](Historical-Imagination-Diagnostic-2026-08-11)
- [Reproduction](Reproduction)
