# AASSR

AASSR은 **희소 보상(sparse reward)** 환경에서 자율적으로 문제를 해결하는 에이전트를 연구하기 위한 강화학습 아키텍처입니다.

현재 `main` 브랜치는 최근 CUDA 실험과 검증에 사용된 **current-generation AASSR**를 기준으로 합니다.

> 과거 실험과 이전 세대 구현은 재현성을 위해 `archive/pre-current-main-2026-08-12` 브랜치에 전체 상태가 보존되어 있습니다.  
> **현행 모델은 `current_manifest.py`와 `current_entrypoint.py`를 기준으로 판단합니다.**

## 현재 연구 질문

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

- relational public state v3 사용
- 최신 공개 HTTP status 보존
- 여러 가능한 미래를 하나의 평균 상태로 뭉개지 않는 mixture 구조
- 희귀 public status가 데이터 불균형 때문에 사라지지 않도록 frequency-balanced categorical objective 사용
- 특정 `403`, `404`, `429`에 대한 수동 규칙 없음

### Imagination

Prophecy가 예측한 미래를 여러 단계 탐색하여 Policy 행동보다 나은 후보가 있는지 비교합니다.

실제 행동 변경에는 다음 안전장치를 모두 통과해야 합니다.

- 예측 신뢰도(confidence) gate
- 실제 training support 기반 Critic OOD gate

학습하지 않은 영역에서는 **fail-closed**, 즉 기존 Policy 행동을 유지합니다.

### Critic

Critic은 실제 희소 반환 `{-1, 0, +1}`을 기준으로 미래 행동의 값을 평가합니다.

Critic support는 추가 보상이나 정답 힌트가 아니라, **현재 판단이 실제 학습 데이터 범위 안에 있는지 확인하는 신뢰성 gate**로만 사용합니다.

### ASEQ

ASEQ는 실제 transition `(S, A, S')`을 이용합니다.

```text
S → A → S
```

처럼 **실제로 상태가 변하지 않는 동일 semantic self-loop가 반복되는 경우만** 억제합니다.

`S → A → S'`, `S' != S`처럼 상태가 실제로 바뀌는 반복 행동은 막지 않습니다.

## 관측과 인간 개입 원칙

learner에게 직접 주지 않는 정보:

- hidden curriculum level
- hidden workflow depth
- exact hidden session countdown
- hidden audit / lockout distance
- hidden rate-limit distance
- 정답 route/profile/object

사용하지 않는 task-specific 개입:

- `403이면 이 행동을 피하라` 같은 규칙
- 성공 trajectory 주입
- 정답 action 주입
- 특정 실패 status에 대한 shaping reward
- evaluation 정답을 이용한 action filtering

외부 sparse return:

```text
성공                         +1
실제 실패                    -1
stall / truncation / rate-limit 0
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

평가 중에는 학습 파라미터를 변경하지 않습니다. Training Imagination도 same-checkpoint 비교의 공정성을 위해 비활성화되어 있습니다.

## 최근 축소 진단 결과

최근 2,048 real-transition CUDA 실험은 **최종 성능 실험이 아니라 병목 진단용 reduced run**입니다.

- training success: `32`
- L0뿐 아니라 L1에서도 training success 발생
- no-Imagination: `8 / 20`
- Full: `8 / 20`
- L0: `4 / 4` 성공
- L1: `4 / 4` 성공
- L2: `4 / 4` true failure
- 최종 실제 Imagination intervention: `0`

2k training 동안 에이전트는 **L2 training transition을 경험하지 못했습니다.** 현재 병목은 작은 transition budget에서 다음 frontier의 경험이 부족해 Policy, Prophecy, Critic이 동시에 OOD가 되는 문제에 더 가깝습니다.

따라서 다음 큰 실험에서는 curriculum 규칙을 먼저 바꾸지 않고, **현재 알고리즘을 고정한 채 transition budget만 증가**시켜 frontier가 자연스럽게 이동하는지 확인합니다.

## 실행

### 설치

```bash
python -m pip install -e ".[dev]"
```

CUDA까지 사용할 경우:

```bash
python -m pip install -e ".[dev,gpu]"
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

이 runner는 Raw DQN, Relational DQN, current AASSR를 학습하고, 같은 frozen AASSR checkpoint에서 no-Imagination과 Full을 비교합니다.

## 저장소 구조

```text
AASSR/
├─ README.md
├─ src/aassr_v2/          # Python package
│  └─ README.md           # 현행 모듈 지도
├─ scripts/               # 실행/진단 CLI
│  └─ README.md           # canonical / diagnostic / legacy 구분
├─ docs/                  # 기술·실험 문서
│  └─ README.md           # 현행 / 과거 문서 안내
├─ configs/               # 재현용 실험 설정
│  └─ README.md
├─ tests/                 # regression / protocol tests
├─ wiki/                  # 설명용 wiki 원고
└─ .github/workflows/     # 현재 CI와 진단 workflow만 유지
   └─ README.md
```

과거 generated paper 결과와 one-off Actions workflow는 main에서 제거했으며, **`archive/pre-current-main-2026-08-12` 브랜치에 그대로 보존**되어 있습니다.

## 현행 source of truth

다음 순서로 확인하세요.

1. `src/aassr_v2/current_manifest.py`
2. `src/aassr_v2/current_entrypoint.py`
3. `src/aassr_v2/pentest_current_generation_main.py`
4. `scripts/run_pentest_current_generation_main.py`
5. current-generation CI

관련 문서:

- `docs/CURRENT_RUNTIME.md`
- `docs/aassr_current_generation.md`
- `docs/dreamerv3_current_baseline.md`

## 개발 검증

```bash
python -m compileall -q src tests scripts
pytest -q
```

핵심 CI:

- `tests.yml`
- `aassr-current-generation.yml`
- `dreamerv3-current-smoke.yml`

현재 큰 CUDA scaling 실험은 GitHub hosted CPU에서 실행하지 않습니다.
