# AASSR

AASSR은 **희소 보상(sparse reward)** 환경에서 사람이 정답 전략이나 중간 목표를 직접 넣지 않아도 에이전트가 실제 경험으로 환경을 이해하고, 미래를 예측하고, 더 나은 행동을 선택할 수 있는지를 연구하는 강화학습 아키텍처입니다.

## 2026-08-14 구조 원칙

현재 연구의 Core/Plugin 경계는 다음 원칙으로 고정합니다.

> **플러그인은 세계의 문법만 알려주고, 세계의 의미는 AASSR Core가 스스로 배운다.**

즉 Plugin은 **플레이 방법과 공개 정보의 종류**만 제공합니다. 상태 표현, 관계 해석, 행동 우선순위, 문제 해결 기억, world model, Critic, Imagination, 실패 일반화는 모두 Core의 책임입니다.

```text
실제 환경
   │
   ▼
Plugin
- 행동 문법 / 매개변수 자료형
- 공개 관찰 채널 / 자료형
- 실제 I/O 실행
- 외부 reward / 종료 신호 전달
   │
   ▼
AASSR Core
- 공개 정보 기억
- 일반 표현과 concrete 경험
- 후보 명령 생성
- Knowledge
- Policy
- Prophecy
- Calibration
- Critic
- Imagination
- ASEQ
- Skills
```

## 새 Core 사용

새 환경 독립 Core의 진입점은 다음입니다.

```python
from aassr_v2.core import build_aassr_core

core = build_aassr_core(
    plugin,
    seed=7,
    train_transitions=10_000,
    use_imagination=True,
    device="cpu",
)
```

새 구조의 source of truth:

1. `src/aassr_v2/core/manifest.py`
2. `src/aassr_v2/core/plugin_contract.py`
3. `src/aassr_v2/core/representation.py`
4. `src/aassr_v2/core/public_memory.py`
5. `src/aassr_v2/core/dqn.py` / `prophecy_model.py` / `critic.py` / `skills_core.py`
6. `src/aassr_v2/core/runtime.py`
7. `docs/decisions/2026-08-14-minimal-plugin-contract-ko.md`
8. `docs/CORE_PLUGIN_ARCHITECTURE.md`

## Plugin은 무엇을 할 수 있나?

허용:

- 행동의 종류와 형식 정의
- 관찰 데이터의 종류와 형식 정의
- Core가 선택한 행동을 실제 환경에 실행
- **현재** 공개된 결과를 그대로 전달
- 환경이 원래 주는 외부 보상/종료/중단 신호 전달
- 실제 통신에 필요한 프로토콜 상태 유지

금지:

- `state_vector`, `semantic_state_identity`, `action_structure` 정의
- 행동 후보의 가치 평가/순위화/전략적 필터링
- 후보 행동 목록 자체를 전략적으로 구성해 반환
- 이전 관찰에서 발견한 대상/링크/실패 후보를 문제 해결 지식으로 누적
- 정답/오답/target/진전 같은 과제 의미 라벨 제공
- Prophecy/world model 설치
- Critic/Imagination 점수 정의
- shaping reward 추가

새 계약에서는 Plugin이 후보 행동 목록도 반환하지 않습니다. Core가 `ActionSpec`, 공개 관찰의 자료형, 그리고 **Core가 직접 기억한 공개 값**을 이용해 후보 명령을 구성합니다.

후보가 너무 많을 때도 concrete 이름의 사전순 앞부분을 자르지 않습니다. Core가 episode 단위 seed로 bounded sampling을 수행하여 이름 자체에 대한 숨은 우선순위를 만들지 않습니다.

Plugin 제작법: `src/aassr_v2/plugins/README.md` 및 Wiki의 `Plugin-Development` 참고.

## Core의 기억 경계

문제 해결 기억은 Plugin이 아니라 Core가 가집니다.

- `CorePublicKnowledge`: 이번 episode에서 실제로 공개된 typed value와 새 evidence 기억
- concrete action 경험: 후보별 실제 시도/상태 변화/오류/외부 reward 통계
- 구조적 학습: DQN/Prophecy/Critic/Skills의 지속 학습

Concrete 후보별 경험은 기본적으로 episode-local입니다. reset 뒤 같은 문자열 ID가 다른 숨은 의미를 가질 수 있기 때문입니다. 명시적으로 `preserve_knowledge_across_episodes`를 사용하는 실험에서만 다음 episode로 넘깁니다.

## 관찰값의 시간적 종류와 ASEQ

Plugin은 관찰값의 의미가 아니라 기계적인 수명만 선언할 수 있습니다.

- `STATE`: 현재 semantic state의 일부
- `EVENT`: 한 응답/사건에서 관찰된 공개 증거
- `COUNTER`: 누적 횟수
- `MEASUREMENT`: 지연 시간처럼 매번 흔들릴 수 있는 값

`COUNTER`와 `MEASUREMENT` 변화만으로 semantic state가 달라지지 않습니다. 동일한 `EVENT` 반복도 새로운 진전으로 세지 않습니다.

ASEQ는 실제 `(S, A, S')` 경험을 저장하고 다음 경우만 억제합니다.

```text
S → A → S
```

같은 semantic state에서 동일 행동이 반복되어 실제로 상태 변화가 없었던 경우입니다.

```text
S → A → S'   (S' != S)
```

처럼 상태가 바뀌는 반복은 허용합니다. 모든 행동이 guard되면 원래 행동 집합으로 되돌아가 controller 자유도를 보존합니다.

## 실제 localhost I/O

`LocalHttpPlugin`은 loopback의 실제 HTTP 서비스와 소켓 통신합니다.

터미널 1:

```bash
python examples/local_http_lab/server.py --port 8765
```

터미널 2:

```bash
python scripts/run_local_http_core.py \
  --base-url http://127.0.0.1:8765 \
  --episodes 64 \
  --max-steps 32 \
  --device cpu \
  --output runs/local_http_core.json
```

이 플러그인은 `localhost`, `127.0.0.1`, `::1`만 허용합니다. 예제 서버는 새 Core/Plugin 경계를 확인하기 위한 작은 통제 환경이며 최종 벤치마크로 간주하지 않습니다.

CI의 28-episode 실제 loopback 학습 smoke에서는 다음 업데이트가 실제 발생했습니다.

```text
real transitions       216
DQN gradient updates    89
Prophecy updates        54
Critic updates          26
```

이 수치는 **실제 socket I/O 위에서 새 Core 학습 경로가 활성화됨**을 확인하는 구조 검증입니다. 짧은 exploration training의 성공률/평균 return은 성능 주장에 사용하지 않습니다. 해당 smoke에서 Imagination은 신뢰 조건이 아직 충족되지 않아 실제 treatment가 활성화되지 않았습니다.

## 기존 10k pentest 연구는 삭제하지 않음

`current_*`, `plugins/current_pentest.py`, pentest simulator 계열 코드는 **기존 10k 체크포인트와 과거 실험의 정확한 재현**을 위해 남겨 둡니다.

중요하게도:

- 기존 10k 수치는 새 Core의 성능 수치가 아닙니다.
- 기존 broad Plugin API는 representation과 world-model 권한까지 가졌기 때문에 새 Plugin 제작 기준으로 사용하지 않습니다.
- 과거 no-Imagination/Full 비교에서 Imagination이 실제로 실행되지 않은 run은 Imagination 성능 증거로 해석하지 않습니다.

따라서 연구 증거를 다음처럼 분리합니다.

```text
historical evidence
  = 기존 pentest-coupled runtime 결과

new architecture evidence
  = 최소 Plugin 경계 + Core 독립성 + localhost 실제 I/O/학습 경로
```

새 Core가 기존 구조보다 성능이 높다는 주장은 별도의 실험을 통과하기 전에는 하지 않습니다.

## Core 경계 감사

Core에 특정 환경 의존성이 다시 들어오는 것을 CI에서 막습니다.

```bash
python scripts/audit_core_boundary.py
```

`src/aassr_v2/core/` 안에 pentest/HTTP/route/profile/CSRF 등 특정 환경 어휘나 환경 Plugin import가 들어오면 실패합니다. 직접 import뿐 아니라 로컬 전이 의존 그래프도 검사합니다.

관련 감사 기록:

- `docs/audits/core-simulator-dependency-audit-2026-08-14.md`

## 설치와 검사

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src tests scripts examples
python scripts/audit_core_boundary.py
pytest -q
```

PyTorch Core smoke test는 CPU PyTorch가 필요합니다.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
pytest -q tests/test_core_runtime_smoke.py
```

## 저장소 구조

```text
AASSR/
├─ src/aassr_v2/
│  ├─ core/                  # 새 환경 독립 AASSR Core
│  ├─ plugins/               # 최소 환경 I/O Plugin
│  └─ current_*, pentest_*   # 과거 current-generation 재현 경로
├─ examples/local_http_lab/  # 실제 loopback HTTP 통제 환경
├─ scripts/
├─ tests/
├─ docs/
└─ wiki/
```
