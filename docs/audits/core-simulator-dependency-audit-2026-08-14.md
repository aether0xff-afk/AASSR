# Core의 시뮬레이터/도메인 의존성 감사 — 2026-08-14

## 목적

AASSR이 특정 pentest 시뮬레이터를 잘 푸는 코드가 아니라 환경과 독립된 학습 시스템인지 코드 수준에서 확인했다. 판정 기준은 다음 설계 결정이다.

> 플러그인은 플레이 방법과 공개 정보의 종류만 제공하며, 의미 해석과 학습/추론은 Core가 담당한다.

## 코드로 확정된 문제

### 1. 기존 Plugin API가 표현 권한을 가지고 있었다

기존 `CurrentRepresentationBinding`은 `state_vector`, `state_key`, `semantic_state_identity`, `action_structure`, `decode_state`, `prediction_score`를 Plugin 쪽 계약으로 넘겼다.

**판정:** 새 철학과 불일치.

**수정:** 새 `core/plugin_contract.py`에서는 이 권한을 제거하고 금지 권한으로 검사한다.

### 2. 기존 Plugin이 world model을 설치할 수 있었다

기존 `CurrentRuntimePlugin`은 `install_world_model`을 제공했고 pentest plugin이 HTTP status 전용 Prophecy 구현을 설치했다.

**판정:** 새 철학과 불일치.

**수정:** 새 Plugin 계약에는 모델 설치/선택 API가 없다. Prophecy는 Core가 생성한다.

### 3. 기존 Core 계열 파일이 pentest 모듈을 직접 import했다

`current_generation.py`는 pentest action/state feature 구현을 직접 사용했고 `current_relational_state.py`는 route/profile/object 역할을 직접 해석했다.

**판정:** Core → 환경 역의존.

**수정:** 새 active Core를 `src/aassr_v2/core/`에 별도 경계로 만들고 이 디렉터리에 pentest/HTTP/route/profile/CSRF 관련 import·토큰이 들어오면 정적 감사와 테스트가 실패하도록 했다.

### 4. HTTP status가 보편 Core 표현에 들어가 있었다

`current_relational_state_v3.py`와 `current_status_models.py`는 HTTP status 전용 채널과 loss를 보편 current stack의 일부로 사용했다.

**판정:** 도메인 결합.

**수정:** 새 Core는 `CATEGORICAL` 같은 자료형만 본다. 특정 값의 의미를 수동으로 부여하지 않는다.

### 5. Agent loop가 환경 내부 속성을 직접 읽었다

기존 current agent 종료 판단은 `success`, `rate_limited`, `locked`, `failed` 같은 환경 속성을 직접 확인했다.

**판정:** 실행 제어가 도메인 상태에 결합.

**수정:** 새 Core는 표준 `reward`, `terminated`, `truncated`, `error`만 받는다. 종료 이유가 학습에 필요한 공개 정보라면 일반 observation field로 들어와야 한다.

### 6. 일반 이름의 builder가 실제로 pentest builder였다

기존 `build_current_aassr_core()`는 내부적으로 pentest 전용 agent를 만들었다.

**판정:** API 이름과 실제 소유권이 불일치.

**수정:** 새 canonical builder는 `aassr_v2.core.build_aassr_core(plugin, ...)`이며 어떤 환경 플러그인도 같은 계약으로 연결한다.

## 새 경계에서 유지한 것

- ASEQ의 정확한 `S → A → S` 반복 억제 의미
- 외부 sparse reward와 내부 정보 가치의 분리
- Policy / Prophecy / Critic / Imagination / Skills / Knowledge의 역할 분리
- 실제 transition만 학습 사실로 사용하는 원칙
- Imagination이 실제로 실행되지 않았으면 ON/OFF 성능 실험을 유효한 treatment로 보지 않는 진단

## 아직 증명되지 않은 것

이번 감사로 다음은 증명되지 않는다.

- 새 Core가 기존 10k pentest runtime보다 성능이 높다.
- localhost 환경에서 장기 의존성을 이미 안정적으로 해결한다.
- 새 일반 표현이 충분한 전이/일반화 능력을 가진다.

이것들은 구조 분리 후 별도의 실험으로 검증해야 한다.

## 연구 경로

1. 최소 Plugin 계약과 Core 경계 CI 통과
2. 실제 loopback HTTP 통신 smoke test
3. 작은 localhost sparse-reward 서비스에서 학습 동작 확인
4. 난도를 하나씩 증가시키며 Core 자체 병목 측정
5. 필요하면 simulator를 단위/회귀 테스트에만 사용
6. 충분한 근거가 쌓인 뒤 기존 10k 증거와 별도 세대로 비교
