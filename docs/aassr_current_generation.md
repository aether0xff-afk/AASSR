# AASSR Current-Generation 아키텍처

이 문서는 `main` 브랜치의 **현행 AASSR 기술 구조**를 설명합니다.

과거 v0.4, Imagination v2, Neural Delta 중심 실험, GridPush/ToolGrid 등은 재현성을 위해 저장소에 남아 있지만 현재 실행 경로의 component가 아닙니다.

현행 구성의 최종 source of truth는 다음 파일입니다.

```text
src/aassr_v2/current_manifest.py
```

`LEGACY_COMPONENTS_ACTIVE`는 비어 있어야 합니다.

## Public entrypoint

Package-level pentest builder도 현재 standalone runtime을 가리킵니다.

```python
from aassr_v2 import build_pentest_aassr_core

agent = build_pentest_aassr_core(
    seed=7,
    train_transitions=10_000,
    use_imagination=True,
    device="cuda",
)
```

명시적인 canonical 경로는 다음입니다.

```python
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
```

이 두 current pentest 경로는 동일한 현행 runtime을 가리킵니다.

## Active stack

현재 `current_manifest.py` 기준 구성은 다음과 같습니다.

| 계층 | 현행 구현 |
|---|---|
| Observation | response-causal relational public state v3 + latest HTTP status |
| ASEQ | semantic self-loop empirical v3 |
| Policy | relational-invariant DQN + information residual |
| Policy state | relational public structural v3 + latest HTTP status |
| Policy action | relational role features |
| Prophecy | relational conditional-mixture ensemble v5, status-balanced |
| Status objective | class-balanced categorical public HTTP status |
| Calibration | semantic probability holdout calibration v3, status-aware |
| Information evaluator | semantic relational probability-aware v3 |
| Knowledge | episode-local response Knowledge context |
| Imagination | structural-compute dedup + probabilistic chance/decision tree |
| Critic | relational GRU discounted sparse-return Critic |
| Critic support | local real-training support, fail-closed |
| Skill | relational ASEQ template |
| Goal | external final goal + relational skill promotion |
| Hardware | DQN + relational world model + return GRU Critic on same device |
| Training Imagination | disabled for same-checkpoint comparison |

과거 effect-composition 경로는 현재 relational world model에 의해 대체되어 비활성화되어 있습니다.

## Observation contract

현행 learner가 사용하는 것은 **공개적으로 관측 가능한 정보**입니다.

Relational state v3는 최신 HTTP response status를 보존합니다.

반면 다음 정보는 learner에게 직접 주지 않습니다.

- hidden curriculum level
- hidden workflow depth
- exact hidden session countdown
- hidden audit / lockout pressure
- hidden rate-limit distance
- 정답 route/profile/object identity

Concrete identifier는 실제 환경 실행에는 존재하지만 transfer representation의 핵심 lookup key로 사용하지 않습니다.

## 두 종류의 identity

AASSR에는 목적이 다른 두 identity가 존재합니다.

### Concrete semantic identity

ASEQ와 실제 episode 내부의 반복 판별에 사용합니다.

서로 다른 concrete route/profile/object는 같은 episode 안에서 구분됩니다.

### Relational transfer identity

다음에 사용합니다.

- Policy
- Prophecy
- Critic
- Skill
- Relational DQN baseline
- DreamerV3 relational adapter

Seed가 바뀌어 concrete identifier 이름이 달라져도 관측된 역할 구조가 같으면 같은 관계 표현으로 일반화할 수 있도록 합니다.

## Policy

현행 Policy는 relational-invariant DQN입니다.

핵심 목적은 다음과 같습니다.

```text
concrete 이름 암기
        ↓ 배제
관계 / 역할 / 공개 상태 기반 행동 평가
```

DQN은 accelerator-aware batch path를 사용하며, Bellman target 계산에서 불필요한 per-row host synchronization을 줄인 구현을 사용합니다.

## Prophecy

현재 world model은 **relational conditional-mixture ensemble v5**입니다.

이전의 단순 mean prediction과 달리 여러 가능한 미래가 있을 때 합법적인 multimodal outcome을 하나의 평균 상태로 붕괴시키지 않도록 설계되어 있습니다.

예측 대상에는 다음 구조가 포함됩니다.

- relational descriptor v3
- latest public HTTP status
- legal action mask
- active / success / failure / truncation 상태
- mixture outcome probability

### Status supervision

HTTP status는 서로 배타적인 categorical public outcome으로 학습합니다.

Class imbalance는 관측 빈도를 이용해 일반적으로 보정하며 특정 status code의 의미를 사람이 직접 지정하지 않습니다.

즉 다음은 허용되는 일반 학습 메커니즘입니다.

```text
드물게 관측된 class
→ 학습에서 완전히 묻히지 않도록 frequency balance
```

하지만 다음과 같은 규칙은 없습니다.

```text
403 → 위험하므로 피한다
429 → 무조건 행동을 바꾼다
```

## Calibration과 confidence

Prophecy confidence는 행동의 가치를 올려주는 bonus가 아닙니다.

**예측을 믿어도 되는지 확인하는 reliability signal**로만 사용합니다.

Imagination이 Policy와 다른 행동을 선택하려면 적어도 현재 비교에 필요한 예측이 충분히 신뢰 가능해야 합니다.

불확실하면 Policy 행동을 유지합니다.

## Critic

Critic은 실제 sparse return을 학습합니다.

```text
success       +1
true failure  -1
otherwise      0
```

현행 Critic은 relational GRU 기반이며 discounted sparse-return target을 사용합니다.

### Local Critic support gate

Critic이 값을 출력할 수 있다는 이유만으로 어떤 상태에서도 그 값을 신뢰하지 않습니다.

실제 training transition에서 현재 state/action 근처를 충분히 경험했는지 확인하는 local support gate가 별도로 있습니다.

```text
충분한 real-training support
        ↓
Critic 비교 허용

support 부족 / OOD
        ↓
Imagination override 취소
        ↓
Policy 행동 유지
```

이 support는 reward도 아니고 hidden level 정보도 아닙니다.

## Imagination

Imagination은 Prophecy의 확률적 미래를 사용하여 여러 단계의 counterfactual branch를 평가합니다.

Chance outcome과 agent의 다음 decision은 구분해서 처리합니다.

```text
현재 행동
   │
   ├─ 환경의 여러 가능한 결과       ← probability / expectation
   │       │
   │       └─ 다음 agent decision   ← 선택 가능한 행동 비교
   │
   └─ ...
```

### Structural root dedup

Concrete action이 여러 개 존재해도 relational structure가 같은 root라면 비싼 Prophecy/Critic 계산을 공유할 수 있습니다.

그러나 실제 환경에 실행되는 마지막 행동은 concrete action입니다.

즉:

```text
계산: structural alias 공유
실행: concrete action 보존
```

## Intervention accounting

과거 진단에서는 confidence gate가 후보를 승인한 뒤 Critic-support gate가 최종적으로 취소했는데도 intervention counter가 먼저 증가하는 계측 문제가 있었습니다.

현재는 **모든 gate를 통과하고 실제 executed action이 Policy action과 달라진 경우만 intervention으로 집계**합니다.

따라서 다음은 구분됩니다.

- switch candidate
- suppressed switch
- final intervention
- changed executed action

## ASEQ

ASEQ는 실제 transition `(S, A, S')`입니다.

현행 self-loop 억제는 좁게 유지합니다.

```text
S → A → S
```

같은 semantic state에서 같은 행동을 했는데 실제 상태 진전 없이 다시 같은 상태로 돌아오는 패턴이 반복될 때만 억제합니다.

```text
S → A → S'
S' != S
```

처럼 실제 상태가 바뀌는 반복은 일반적으로 허용합니다.

## Hardware execution

`--device cuda` 또는 `--device cuda:0`은 current AASSR의 주요 신경망 경로에 적용됩니다.

주요 accelerator-aware 경로:

- Policy state/action pair batch scoring
- Prophecy depth batching
- Critic branch batching
- DQN fused Bellman target
- calibration batch prediction
- root structural compute dedup

TF32는 기본 CUDA 경로에서 허용되며 stricter float32가 필요하면 canonical CLI의 `--no-tf32`를 사용할 수 있습니다.

Target GPU 확인:

```powershell
python scripts/check_current_generation_hardware.py --device cuda:0
```

## 실험 조건

Canonical PyTorch current-generation runner는 다음 네 결과를 직접 비교합니다.

1. `dqn_raw`
2. `dqn_relational`
3. `aassr_current_no_imagination`
4. `aassr_current_full`

Raw DQN과 Relational DQN은 representation 차이를 분리하기 위한 control입니다.

AASSR은 하나의 checkpoint만 학습합니다.

```text
AASSR training
      │
      ▼
frozen checkpoint
  ┌───────┴────────┐
  ▼                ▼
no-Imagination     Full
```

두 AASSR evaluation 사이에 재학습이 일어나면 안 됩니다.

## DreamerV3 비교

Official DreamerV3 control은 별도 JAX runtime에서 실행됩니다.

AASSR 저장소는 current relational observation과 dynamic-action adapter를 제공하지만, Dreamer 알고리즘 자체의 actor/critic/world-model 학습 코드는 pinned upstream 구현을 사용합니다.

Dreamer는 fixed relational categorical action vocabulary와 현재 legal-action mask를 사용합니다.

자세한 계약은 다음 문서를 봅니다.

```text
docs/dreamerv3_current_baseline.md
```

Dreamer smoke는 API와 step accounting을 검증하는 용도이며 그 자체가 성능 benchmark 결과는 아닙니다.

## 최근 2k 진단에서 확인된 상태

최근 reduced CUDA run은 최종 성능 claim이 아니라 병목 진단입니다.

```text
training budget: 2048 real transitions
training successes: 32
no-Imagination: 8/20
Full:           8/20
L0:             4/4 success
L1:             4/4 success
L2:             4/4 true failure
final intervention: 0
```

핵심 관찰은 **training 중 L2 transition이 0개였다는 것**입니다.

따라서 현재 reduced-run에서 Policy/Prophecy/Critic이 L2에서 동시에 OOD가 되는 것은 작은 budget에서 frontier 경험이 부족한 현상으로 설명될 가능성이 있습니다.

별도 넓은 real-transition holdout에서는 희귀 public status가 실제 학습 데이터에 충분히 존재하면 world model이 이를 학습할 수 있음을 확인했습니다.

그래서 현재 다음 질문은:

> **transition budget을 키우기만 해도 curriculum이 자연스럽게 다음 frontier로 이동하는가?**

입니다.

## 다음 대형 실험에서 고정할 것

다음 scaling 실험에서는 다음을 바꾸지 않습니다.

- observation contract
- Policy
- Prophecy
- status objective
- Critic
- confidence gate
- Critic support gate
- ASEQ
- curriculum rule

변경하는 것은 **real transition budget**입니다.

이렇게 해야 2k의 L1 정체가 단순 sample-efficiency 문제인지 구조적인 curriculum/transfer 문제인지 구분할 수 있습니다.

## Current CI contract

현행 current-generation gate는 다음을 확인합니다.

- active builder와 no-legacy-component 계약
- relational state v3 / public status
- multimodal mixture prediction
- Policy/Prophecy/Critic batching
- cache와 hardware 경로
- planner scalar/batch 의미 보존
- confidence gate
- local Critic support fail-closed
- structural root dedup
- final intervention accounting
- exact training budget
- frozen evaluation
- real 192-transition learning smoke
- Dreamer relational fairness contract

큰 실험은 이 gate가 초록인 commit을 기준으로 실행해야 합니다.

## 관련 문서

- [`../README.md`](../README.md) — 저장소 전체 개요와 빠른 시작
- [`CURRENT_RUNTIME.md`](CURRENT_RUNTIME.md) — 현행/진단/역사 경로 안내
- [`dreamerv3_current_baseline.md`](dreamerv3_current_baseline.md) — DreamerV3 비교 계약
- [`aassr_v040_architecture.md`](aassr_v040_architecture.md) — **과거 v0.4 재현 문서**

옛 `main` 전체 상태는 다음 archive 브랜치에 보존되어 있습니다.

```text
archive/pre-current-main-2026-08-12
```
