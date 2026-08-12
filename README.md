# AASSR

AASSR은 **희소 보상(sparse reward)** 환경에서 자율적으로 문제를 해결하는 에이전트를 연구하기 위한 강화학습 아키텍처입니다.

현재 `main` 브랜치는 과거 v0.4 통합본이 아니라, 최근 CUDA 실험과 검증에 사용된 **current-generation AASSR**를 기준으로 합니다.

> 이 저장소에는 과거 실험과 이전 세대 구현도 재현성을 위해 남아 있습니다.  
> **현행 모델은 README의 버전 문자열이나 옛 실험 파일명이 아니라 `current_manifest.py`와 `current_entrypoint.py`를 기준으로 판단합니다.**

## 현재 연구 질문

AASSR이 풀고자 하는 핵심 질문은 다음과 같습니다.

> **정답 행동이나 중간 보상을 직접 주기 어려운 환경에서, 에이전트가 실제 경험으로 환경을 이해하고 미래를 예측하여 더 복잡한 문제로 일반화할 수 있는가?**

현재 연구에서는 특히 다음을 봅니다.

- 희소한 최종 보상만으로 자율 학습이 가능한가?
- 낮은 난도에서 배운 전략이 더 높은 난도로 전이되는가?
- Prophecy가 실제 환경의 다음 상태와 위험을 충분히 일반화하는가?
- Imagination이 Policy보다 더 좋은 행동을 **안전하게** 선택할 수 있는가?
- 성능 향상이 task-specific 인간 규칙이 아니라 일반적인 학습 메커니즘에서 나오는가?

## 현행 구조

현재 pentest 연구 경로의 canonical builder는 하나입니다.

```python
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core

agent = build_current_pentest_aassr_core(
    seed=7,
    train_transitions=10_000,
    use_imagination=True,
    device="cuda",
)
```

전체 흐름은 대략 다음과 같습니다.

```text
실제 공개 관측
    │
    ▼
Relational State v3
    │
    ├──────────────► Policy (DQN)
    │                    │
    │                    ▼
    │                기본 행동 후보
    │
    └──────────────► Prophecy
                         │
                         ▼
                  가능한 미래 예측
                         │
                         ▼
                    Imagination
                         │
                    Critic 평가
                         │
              Confidence / OOD gate
                         │
              ┌──────────┴──────────┐
              │                     │
       충분히 신뢰 가능        불확실 / OOD
              │                     │
              ▼                     ▼
       행동 변경 가능          Policy 행동 유지
              │                     │
              └──────────┬──────────┘
                         ▼
                    실제 환경 실행
                         │
                         ▼
                  실제 transition 학습
```

### Policy

현재 Policy는 **relational-invariant DQN**을 사용합니다. 구체적인 route/profile/object ID 자체보다 구조적 역할과 공개된 상태 정보를 사용하도록 설계되어 있습니다.

### Prophecy

현재 Prophecy는 **status-balanced conditional-mixture relational world model**입니다.

주요 특징:

- relational public state v3 사용
- 최신 공개 HTTP status 보존
- 여러 가능한 미래를 하나의 평균 상태로 뭉개지 않는 mixture 구조
- 희귀 public status가 데이터 불균형 때문에 사라지지 않도록 frequency-balanced categorical objective 사용
- 특정 `403`, `404`, `429`에 대한 수동 규칙은 없음

### Imagination

Prophecy가 예측한 미래를 여러 단계 탐색하여 Policy 행동보다 나은 후보가 있는지 비교합니다.

하지만 Imagination이 항상 Policy를 덮어쓰는 것은 아닙니다. 현재는 다음 두 안전장치를 모두 통과해야 실제 행동이 바뀝니다.

- 예측 신뢰도(confidence) gate
- 실제 training support 기반 Critic OOD gate

따라서 학습하지 않은 영역에서는 **fail-closed**, 즉 기존 Policy 행동을 유지합니다.

### Critic

Critic은 실제 희소 반환 `{-1, 0, +1}`을 기준으로 미래 행동의 값을 평가합니다.

Critic support는 추가 보상이나 정답 힌트가 아니라, **현재 판단이 실제 학습 데이터 범위 안에 있는지 확인하는 신뢰성 gate**로만 사용합니다.

### ASEQ

ASEQ는 실제 transition `(S, A, S')`을 이용합니다.

현재 반복 억제는 매우 좁게 적용됩니다.

```text
S → A → S
```

처럼 **실제로 상태가 변하지 않는 동일 semantic self-loop가 반복되는 경우만** 억제합니다.

`S → A → S'`, `S' != S`처럼 상태가 실제로 바뀌는 반복 행동은 막지 않습니다.

## 관측과 인간 개입 원칙

현재 실험에서는 다음과 같은 hidden 정보가 learner에게 들어가지 않도록 유지합니다.

- hidden curriculum level
- hidden workflow depth
- exact hidden session countdown
- hidden audit / lockout distance
- hidden rate-limit distance
- 정답 route/profile/object

또한 다음과 같은 task-specific 개입은 사용하지 않습니다.

- `403이면 이 행동을 피하라` 같은 규칙
- 성공 trajectory 주입
- 정답 action 주입
- 특정 실패 status에 대한 shaping reward
- evaluation 정답을 이용한 action filtering

학습 보상은 기본적으로 다음 외부 sparse return을 사용합니다.

```text
성공          +1
실제 실패     -1
stall/truncation/rate-limit  0
```

## 현재 실험 프로토콜

AASSR의 **Policy-only(no-Imagination)**와 **Full Imagination** 비교는 동일한 학습 checkpoint를 사용합니다.

```text
한 번 학습
   │
   ▼
frozen checkpoint
   ├────────► no-Imagination 평가
   └────────► Full Imagination 평가
```

평가 중에는 학습 파라미터를 변경하지 않습니다.

또한 training 중 Imagination은 이 same-checkpoint 비교의 공정성을 위해 비활성화되어 있습니다.

## 최근 축소 진단 결과

최근 2,048 real-transition CUDA 실험은 **최종 성능 실험이 아니라 병목 진단용 reduced run**입니다.

관측된 핵심 결과:

- training success: `32`
- L0뿐 아니라 L1에서도 training success 발생
- frozen evaluation:
  - no-Imagination: `8 / 20`
  - Full: `8 / 20`
- L0: `4 / 4` 성공
- L1: `4 / 4` 성공
- L2: `4 / 4` true failure
- 최종 실제 Imagination intervention: `0`

중요한 점은 2k training 동안 에이전트가 **L2 training transition을 경험하지 못했다는 것**입니다.

따라서 현재 병목은 단순히 “Prophecy가 특정 HTTP status를 표현하지 못한다”라기보다,

> **작은 transition budget에서 다음 frontier의 경험 자체가 충분하지 않아 Policy, Prophecy, Critic이 동시에 OOD가 되는 문제**

에 더 가깝습니다.

별도의 넓은 real-transition holdout에서는 희귀 status도 충분한 실제 training support가 있을 때 학습 가능한 것을 확인했습니다.

따라서 다음 큰 실험에서는 curriculum 규칙을 먼저 바꾸지 않고, **현재 알고리즘을 그대로 고정한 채 transition budget만 증가**시켜 frontier가 자연스럽게 이동하는지 확인합니다.

## 현재 실험 실행

### 설치

개발 환경:

```bash
python -m pip install -e ".[dev]"
```

CUDA 경로까지 사용할 경우:

```bash
python -m pip install -e ".[dev,gpu]"
```

CUDA 확인:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Canonical current-generation 실험

```bash
python scripts/run_pentest_current_generation_main.py \
  --output-dir runs/current_generation/seed-7 \
  --research-seed 7 \
  --transition-budget 10000 \
  --block-target 512 \
  --device cuda
```

이 runner는 다음 조건을 실행합니다.

1. Raw DQN
2. Relational DQN
3. Current AASSR 학습
4. 동일 frozen AASSR checkpoint의 no-Imagination 평가
5. 동일 frozen AASSR checkpoint의 Full Imagination 평가

최종 요약은 다음 위치에 기록됩니다.

```text
<output-dir>/summary.json
```

### 상세 Imagination 진단

```text
scripts/run_repaired_imagination_final.py
```

파일명은 과거 repair 과정에서 만들어진 이름이지만, 현재는 canonical current builder를 사용합니다. 상세 decision/prediction/intervention trace가 필요할 때 사용합니다.

## 테스트

기본 테스트:

```bash
python -m compileall -q src tests scripts
pytest -q
```

현행 runtime은 GitHub Actions에서 다음을 별도로 검증합니다.

- relational state/status contract
- multimodal Prophecy
- builder/batching/cache
- Critic/planner/confidence/OOD gate
- root dedup
- DreamerV3 fairness
- real-environment learning smoke
- same-checkpoint/frozen evaluation
- 최종 실행 행동 기준 intervention accounting

## 저장소 구조

```text
AASSR/
├─ src/aassr_v2/
│  ├─ current_manifest.py       # 현행 component 계약
│  ├─ current_entrypoint.py     # 유일한 current builder
│  ├─ current_*                 # 현행 runtime 구성요소
│  └─ ...                       # 역사/공용 연구 코드
│
├─ scripts/
│  ├─ run_pentest_current_generation_main.py
│  ├─ run_repaired_imagination_final.py
│  └─ ...
│
├─ docs/
│  ├─ CURRENT_RUNTIME.md        # 현행 runtime 안내
│  └─ ...                       # 과거 실험 및 연구 문서
│
└─ tests/
```

## 현행과 과거 코드 구분

### 현행

다음 파일을 우선 source of truth로 봅니다.

1. `src/aassr_v2/current_manifest.py`
2. `src/aassr_v2/current_entrypoint.py`
3. `src/aassr_v2/pentest_current_generation_main.py`
4. `scripts/run_pentest_current_generation_main.py`
5. current-generation CI / tests

### 진단용

예:

- repaired Imagination trace
- rare-status holdout
- profiling
- targeted ablation

현행 runtime을 검사하지만 모델 정의 자체는 아닙니다.

### 역사 / 재현용

예:

- v0.4
- Imagination v2
- GridPush
- ToolGrid
- 이전 Prophecy 실험
- 과거 pentest mechanism 실험
- paper reproduction runner

이 파일들은 과거 결과 재현을 위해 남아 있으며, **현재 모델의 구성요소라는 뜻은 아닙니다.**

병합 전 옛 `main` 상태는 다음 archive 브랜치에 보존되어 있습니다.

```text
archive/pre-current-main-2026-08-12
```

## 문서 우선순위

문서와 코드가 충돌할 경우 다음 순서를 따릅니다.

```text
current_manifest.py
        ↓
current_entrypoint.py
        ↓
pentest_current_generation_main.py
        ↓
canonical CLI
        ↓
current-generation tests / CI
        ↓
문서
```

즉 **코드가 문서보다 우선**합니다.

## 현재 다음 단계

현재 가장 중요한 실험 질문은 이것입니다.

> **2k에서 보였던 L1 frontier가 단순히 작은 학습량 때문인지, 아니면 curriculum/transfer 구조 자체의 병목인지?**

이를 위해 다음 큰 실험에서는 알고리즘과 curriculum 규칙을 그대로 두고 transition budget만 늘립니다.

만약 큰 budget에서 frontier가 자연스럽게 L2/L3 이상으로 이동한다면 현재 문제는 주로 **sample efficiency**입니다.

반대로 충분히 큰 budget에서도 L1 근처에 머문다면 그때 curriculum/frontier sampling 자체를 구조적으로 재검토합니다.

---

현행 runtime에 대한 더 자세한 설명은 [`docs/CURRENT_RUNTIME.md`](docs/CURRENT_RUNTIME.md)를 참고하세요.
