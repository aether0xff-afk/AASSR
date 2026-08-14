# `aassr_v2` 소스 구조

## 새 환경 독립 Core

새 연구 아키텍처의 source of truth는 `src/aassr_v2/core/`다.

- `core/manifest.py` — Core와 Plugin의 권한 계약
- `core/plugin_contract.py` — 최소 Plugin API
- `core/representation.py` — 자료형 기반 표현과 저수준 concrete 경험 기반
- `core/public_memory.py` — Core 공개 지식, episode-local concrete 경험, 후보 생성/제한
- `core/dqn.py` — Core Policy와 DQN
- `core/prophecy_model.py` — Core Prophecy와 holdout calibration
- `core/critic.py` — signed sparse-return Critic
- `core/skills_core.py` — structural Skills
- `core/runtime.py` — 환경 독립 실행 루프와 canonical `build_aassr_core()`

Plugin은 `src/aassr_v2/plugins/`에 둔다.

- `plugins/local_http.py` — 최소 계약을 따르는 loopback 실제 HTTP 예시
- `plugins/README.md` — 새 Plugin 제작법
- `plugins/current_pentest.py` — **기존 10k runtime 재현용 broad Plugin. 새 제작 예시로 사용하지 않음.**

새 Plugin은 상태 표현, 후보 행동 목록, 의미 라벨, 문제 해결 기억, world model을 소유하지 않는다. `ActionSpec`과 공개 자료형만 제공하고 실제 I/O를 수행한다.

## 과거 current-generation 경로

`current_agent.py`, `current_entrypoint.py`, `current_generation.py`, `current_relational_state*.py`, `current_status_models.py`, `pentest_*`는 기존 10k 체크포인트와 연구 기록을 재현하기 위해 유지한다.

이 경로는 새 최소 Plugin 철학보다 더 넓은 환경 권한을 사용하므로, 새 Core 변경 시 암묵적으로 import하지 않는다.

## 경계 검사

```bash
python scripts/audit_core_boundary.py
```

`src/aassr_v2/core/`에서 특정 환경 모듈/어휘가 발견되거나 전이 import가 환경/simulator 쪽으로 들어가면 실패한다.

## 핵심 원칙

> **Plugin은 세계의 문법만 제공하고, 의미와 기억과 후보 선택은 Core가 담당한다.**
