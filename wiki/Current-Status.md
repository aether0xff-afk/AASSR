# 현재 상태

업데이트: 2026-08-14

## 30초 요약

AASSR은 최근 pentest simulator 디버깅이 Core 설계를 끌고 가는 문제를 확인했고, 연구 구조를 **Core 우선 + 최소 Plugin 계약**으로 전환 중이다.

새 원칙:

> Plugin은 플레이 방법과 공개 정보의 종류만 제공한다. 의미 발견, 관계 학습, 기억, 행동 선택, 예측과 상상은 Core가 담당한다.

## 새 canonical architecture

- `src/aassr_v2/core/plugin_contract.py` — 최소 Plugin 계약
- `src/aassr_v2/core/representation.py` — Core 소유 표현/경험 기억
- `src/aassr_v2/core/dqn.py` — Core Policy/DQN
- `src/aassr_v2/core/prophecy_model.py` — Core Prophecy/Calibration
- `src/aassr_v2/core/critic.py` — signed Critic
- `src/aassr_v2/core/skills_core.py` — structural Skills
- `src/aassr_v2/core/runtime.py` — 환경 독립 실행 루프
- `src/aassr_v2/plugins/local_http.py` — 최소 계약 예시
- `scripts/audit_core_boundary.py` — Core에 환경 의존성이 다시 들어오는지 정적 검사

## 기존 10k 결과의 위치

기존 10k checkpoint와 post-10k pentest 진단은 삭제하지 않는다. 다만 그것은 **기존 pentest-coupled runtime에 대한 historical evidence**다.

특히 10k에서 no-Imagination과 Full 결과가 같았던 것은 Imagination이 실제로 0회 실행된 조건이었으므로 "Imagination이 효과 없다"는 증거가 아니다.

## 지금 말할 수 있는 것

- 기존 broad Plugin API에는 representation/world-model 권한이 있어 새 철학과 맞지 않았다.
- 기존 current 계열 Core 코드 일부가 pentest/simulator 구현을 직접 import했다.
- 새 Core 경계에서는 이 역의존을 정적 검사로 금지한다.
- 새 loopback Plugin은 실제 HTTP 소켓을 사용하며 외부 host를 거부한다.

## 아직 말하면 안 되는 것

- 새 Core가 기존 10k보다 성능이 높다.
- simulator가 과거 실패의 모든 원인이었다.
- localhost 실험만으로 AASSR 일반성이 증명됐다.

이 세 주장은 후속 실험이 필요하다.
