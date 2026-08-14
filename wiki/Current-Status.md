# 현재 상태

업데이트: 2026-08-14

## 30초 요약

AASSR의 새 연구 구조는 **Core 우선 + 최소 Plugin 계약**으로 구현되었다. 현재 작업은 Draft PR #41에서 검증 중이며 기존 10k pentest runtime은 과거 증거 재현용으로 그대로 보존한다.

새 원칙:

> Plugin은 플레이 방법과 공개 정보의 종류만 제공한다. 의미 발견, 관계 학습, 기억, 행동 선택, 예측과 상상은 Core가 담당한다.

## 새 환경 독립 Core

- `src/aassr_v2/core/plugin_contract.py` — 최소 Plugin 권한
- `src/aassr_v2/core/representation.py` — 자료형 기반 Core 표현의 저수준 기반
- `src/aassr_v2/core/public_memory.py` — 공개 정보 기억, episode-local concrete 경험, 후보 생성
- `src/aassr_v2/core/dqn.py` — 외부 sparse reward Policy/DQN + 별도 정보 가치
- `src/aassr_v2/core/prophecy_model.py` — Core 소유 Prophecy/Calibration
- `src/aassr_v2/core/critic.py` — signed sparse-return Critic
- `src/aassr_v2/core/skills_core.py` — concrete 이름이 아닌 구조 기반 Skills
- `src/aassr_v2/core/runtime.py` — 환경 독립 실행 루프
- `src/aassr_v2/plugins/local_http.py` — 최소 계약을 따르는 loopback 실제 HTTP Plugin
- `scripts/audit_core_boundary.py` — Core의 직접/간접 환경 의존성 정적 감사

## 감사 중 추가로 잡힌 문제

구조를 만든 뒤 첫 실제 localhost smoke를 다시 보면서 두 가지를 추가로 발견해 수정했다.

1. **Plugin이 이전 페이지의 링크를 기억하고 있었다.**
   - 통신을 위한 CookieJar는 Plugin에 남겼다.
   - 이전에 무엇을 발견했는가는 문제 해결 기억이므로 Core의 `CorePublicKnowledge`로 이동했다.

2. **지연 시간과 매번 달라지는 응답 값이 가짜 semantic progress를 만들 수 있었다.**
   - `STATE`, `EVENT`, `COUNTER`, `MEASUREMENT`를 분리했다.
   - counter/measurement 변화만으로 semantic state가 바뀌지 않는다.
   - 같은 EVENT 반복도 새 진전으로 세지 않는다.

3. **후보 수 제한이 concrete 이름의 사전순 앞부분을 고르는 편향을 만들 수 있었다.**
   - 후보 제한은 Plugin이 아니라 Core가 담당한다.
   - 사전순 앞부분 절단을 제거하고 episode seed 기반 bounded sampling으로 교체했다.
   - 같은 episode의 같은 공개 증거에서는 후보 표면이 안정적으로 유지된다.

4. **concrete 후보별 경험이 episode를 넘어 남을 수 있었다.**
   - 같은 문자열 ID가 reset 뒤 다른 숨은 의미를 가질 가능성이 있으므로 기본값은 episode-local로 변경했다.
   - 명시적으로 지식 보존 실험을 선택할 때만 유지한다.

## 실제 localhost 학습 smoke

`LocalHttpPlugin`은 Python 객체로 HTTP를 흉내 내는 simulator가 아니라 실제 `127.0.0.1` 소켓을 통해 로컬 웹 서비스와 통신한다.

현재 CI smoke 조건:

```text
28 episodes
max 12 steps / episode
CPU PyTorch
실제 loopback HTTP
```

관찰된 결과:

```text
real transitions       216
DQN gradient updates    89
Prophecy updates        54
Critic updates          26
positive episodes        1
negative episodes       17
zero episodes           10
```

이 결과의 의미는 **새 Core가 실제 네트워크 I/O 위에서 Policy/Prophecy/Critic 학습 업데이트까지 실제로 수행했다**는 것이다. 성공률이나 평균 return을 성능 증거로 사용하면 안 된다. 이 smoke는 exploration 중인 짧은 구조 검증이며 성능 벤치마크가 아니다.

Imagination은 이 smoke에서도 Critic 신뢰 조건이 아직 충족되지 않아 실제 treatment가 활성화되지 않았다. 따라서 이 결과로 Imagination 성능을 판단하지 않는다.

## 기존 10k 결과의 위치

기존 10k checkpoint와 post-10k pentest 진단은 삭제하지 않는다. 그것은 **기존 pentest-coupled runtime에 대한 historical evidence**다.

특히 10k에서 no-Imagination과 Full 결과가 같았던 것은 Imagination이 실제로 0회 실행된 조건이었으므로 "Imagination이 효과 없다"는 증거가 아니다.

## 지금 말할 수 있는 것

- 새 Plugin API에는 representation/world-model/전략 필터 권한이 없다.
- Plugin은 후보 행동 목록을 제공하지 않는다.
- 공개 정보 기억과 concrete 후보 경험은 Core가 담당한다.
- 새 Core 경계의 직접/간접 simulator 의존성은 자동 검사 대상이다.
- 실제 loopback I/O에서 DQN/Prophecy/Critic 학습 업데이트가 발생했다.
- 기존 10k 재현 경로는 변경하지 않았다.

## 아직 말하면 안 되는 것

- 새 Core가 기존 10k보다 성능이 높다.
- simulator가 과거 실패의 모든 원인이었다.
- localhost smoke만으로 AASSR 일반성이 증명됐다.
- Imagination이 새 Core에서 성능을 높인다.

이 주장은 이후 별도의 통제된 실험으로 검증해야 한다.
