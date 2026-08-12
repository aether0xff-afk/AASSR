# AASSR 현행 Runtime 안내

이 문서는 `main` 브랜치에서 사용하는 **현행 연구 runtime**의 탐색 페이지입니다.

최상위 README의 과거 버전 문자열, 오래된 실험 이름, 역사적인 workflow 이름만 보고 현재 모델을 판단하면 안 됩니다. 과거 파일들은 재현성을 위해 저장소에 남아 있으며 이전 세대를 설명할 수 있습니다.

## 무엇을 source of truth로 볼 것인가

다음 순서로 판단합니다.

1. `src/aassr_v2/current_manifest.py` — 현행 component 계약
2. `src/aassr_v2/current_entrypoint.py` — 유일한 현행 AASSR builder
3. `src/aassr_v2/pentest_current_generation_main.py` — 현재 training / frozen evaluation protocol
4. `scripts/run_pentest_current_generation_main.py` — canonical current-generation CLI
5. current-generation regression tests와 CI

설명 문서와 위 코드가 충돌한다면 **코드 경로가 우선**이며 문서가 오래된 것으로 간주합니다.

## Canonical builder

```python
from aassr_v2.current_entrypoint import build_current_pentest_aassr_core
```

`build_current_pentest_aassr_core()`가 현재 pentest AASSR의 유일한 active builder입니다.

과거에 따로 존재하던 `current_mixture_entrypoint.py`는 이제 compatibility alias입니다. 예전 repair/diagnostic script의 실행 호환성을 유지하기 위해 남아 있을 뿐, 두 번째 current runtime을 정의하지 않습니다.

## 현행 모델 구성

현재 builder는 다음 경로를 설치합니다.

- response-causal public observation contract
- 최신 공개 HTTP status를 포함한 relational public state v3
- relational-invariant hardware DQN Policy
- fully batched Prophecy / Critic planning path
- current semantic/runtime repairs
- status-balanced conditional-mixture relational Prophecy
- confidence-as-reliability-only gate
- local real-training Critic support gate
- structural root compute deduplication
- current decision optimizations

Status objective는 **일반적인 class imbalance 보정**입니다.

다음과 같은 task-specific 규칙은 없습니다.

- `403이면 피한다`
- `429이면 특정 행동을 막는다`
- hidden curriculum level을 learner에 입력한다
- 정답 trajectory를 주입한다
- 특정 성공 action을 강제로 replay한다

## 현재 실험 프로토콜

Training과 evaluation은 분리되어 있습니다.

외부 sparse return:

```text
성공                         +1
실제 실패                    -1
truncation / stall / rate-limit  0
```

핵심 공정성 조건:

- training Imagination은 same-checkpoint 비교를 위해 비활성화
- Policy-only와 Full Imagination은 **동일한 frozen AASSR checkpoint**를 사용
- evaluation 중 학습 파라미터를 업데이트하지 않음
- hidden audit pressure와 exact hidden session countdown은 계속 masking
- Critic support는 reward/value bonus가 아니라 reliability gate
- Imagination intervention은 모든 gate를 통과한 뒤 **실제 실행 행동이 바뀐 경우만** 집계

## Canonical runner

### 전체 current-generation 비교

```bash
python scripts/run_pentest_current_generation_main.py \
  --output-dir runs/current_generation/seed-7 \
  --research-seed 7 \
  --transition-budget 10000 \
  --block-target 512 \
  --device cuda
```

이 runner는 다음을 실행합니다.

1. Raw DQN training/evaluation
2. Relational DQN training/evaluation
3. Current AASSR training
4. 같은 frozen AASSR checkpoint의 no-Imagination evaluation
5. 같은 frozen AASSR checkpoint의 Full Imagination evaluation

결과는 기본적으로 다음 파일에 요약됩니다.

```text
<output-dir>/summary.json
```

### 상세 Imagination diagnostic runner

```text
scripts/run_repaired_imagination_final.py
```

파일명은 repair 과정의 역사적인 이름을 유지하고 있지만, canonical-builder 통합 이후에는 동일한 current AASSR runtime을 사용합니다.

다음이 필요할 때 사용합니다.

- decision trace
- Prophecy prediction trace
- switch candidate
- confidence gate 원인
- Critic support/OOD gate 원인
- 실제 intervention 여부
- root dedup 통계

### Rare-status diagnostic

```text
scripts/run_current_status_rare_holdout.py
```

이 파일은 development diagnostic이며 메인 성능 runner가 아닙니다.

희귀 public outcome이 충분한 real training support를 받았을 때 held-out에서 학습 가능한지 확인하기 위한 용도입니다.

## 최근 reduced-run에서 확인된 병목

최근 2,048 real-transition CUDA 진단에서는 다음이 관측되었습니다.

- training success `32`
- L0와 L1에서 성공 경험 확보
- no-Imagination `8/20`
- Full `8/20`
- L0 `4/4` 성공
- L1 `4/4` 성공
- L2 `4/4` true failure
- 실제 실행 행동 기준 Imagination intervention `0`

중요하게도 이 2k training에서는 **L2 training transition이 수집되지 않았습니다.**

따라서 현재 reduced-run의 병목은 다음처럼 해석합니다.

```text
작은 training budget
      │
      ▼
L0 / L1 중심 경험
      │
      ▼
L2 frontier 경험 부족
      │
      ├─ Policy OOD
      ├─ Prophecy OOD
      └─ Critic support 부족
             │
             ▼
      Imagination fail-closed
             │
             ▼
         L2 failure
```

별도의 더 넓은 real-transition holdout에서는 희귀 public status도 충분한 training support가 있을 때 학습 가능함을 확인했습니다.

따라서 다음 대형 실험에서는 **curriculum 규칙을 먼저 수정하지 않고 transition budget만 증가**시킵니다.

목적은 다음 두 가능성을 구분하는 것입니다.

```text
A. budget 증가 → frontier가 자연스럽게 L2/L3로 이동
   => 주 병목은 sample efficiency

B. budget 증가 → 계속 L1 근처에 정체
   => curriculum / transfer 구조 병목 가능성 증가
```

## 저장소 분류

### Active

다음 경로에서 실제 import되는 `current_*` 파일이 현행입니다.

- `current_entrypoint.py`
- `pentest_current_generation_main.py`
- canonical CLI
- active current-generation tests/CI

### Diagnostic / development

예:

- repaired Imagination trace
- rare-status holdout
- hardware profiling
- targeted ablation
- 특정 representation/Prophecy/Critic 감사 도구

이들은 현행 runtime을 검사하지만 **모델 정의 자체는 아닙니다.**

### Historical / reproduction

예:

- v0.4
- Imagination v2
- GridPush
- ToolGrid
- 이전 Prophecy 구현
- 과거 pentest training mechanism
- paper reproduction runner
- 예전 one-off workflow

이 파일들은 과거 실험 재현을 위해 남아 있습니다.

새로운 current-generation runner가 역사 파일을 import해야 한다면 그것이 명시적인 compatibility layer인지 확인해야 하며, active builder는 여전히 `build_current_pentest_aassr_core()`여야 합니다.

## 아카이브

current-generation을 `main`으로 승격하기 전 옛 `main`은 다음 브랜치에 보존되어 있습니다.

```text
archive/pre-current-main-2026-08-12
```

필요하면 이 브랜치를 기준으로 이전 저장소 상태를 그대로 확인하거나 재현할 수 있습니다.

## 큰 실험 전 체크리스트

대형 CUDA 실험은 다음 조건을 만족한 뒤 실행합니다.

1. current-generation unit/contract test가 모두 통과
2. current builder와 compatibility builder가 같은 구현으로 resolve
3. frozen evaluation이 학습 state를 변경하지 않음
4. intervention counter가 최종 실행 행동 기준으로 기록됨
5. confidence gate와 Critic support gate가 fail-closed로 동작
6. root dedup 이후에도 concrete final execution 유지
7. output summary에 `current_manifest.py`의 `CURRENT_COMPONENTS`가 기록됨
8. DreamerV3 fairness check 통과
9. real-environment smoke 통과

현재 이 조건들은 main 병합 전 current-generation gate에서 검증되었습니다.

## 다음 scaling 실험 원칙

다음 큰 실험에서는 한 번에 여러 구조를 바꾸지 않습니다.

**고정:**

- Policy 구조
- Prophecy 구조
- status objective
- confidence gate
- Critic support gate
- ASEQ 규칙
- curriculum 규칙

**변경:**

- transition budget

즉 다음 실험의 질문은 단순합니다.

> **현재 AASSR이 더 많은 실제 경험만으로 2k에서 보인 frontier 병목을 넘어설 수 있는가?**

이 질문에 답한 뒤에만 curriculum/frontier sampling 구조 변경 여부를 판단합니다.
