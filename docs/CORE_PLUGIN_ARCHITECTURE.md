# AASSR Core / Plugin 경계 v2

## 결정

AASSR의 새 연구 경계는 다음 한 문장으로 고정한다.

> **플러그인은 세계의 문법만 알려주고, 세계의 의미는 AASSR Core가 스스로 배운다.**

이 문서는 `src/aassr_v2/core/`의 새 최소 플러그인 계약을 설명한다. 기존 `current_*` + `plugins/current_pentest.py` 경로는 10k 체크포인트와 과거 실험 재현을 위해 보존하지만, 새 플러그인을 만들 때 따라야 할 설계 기준이 아니다.

## 권한 경계

```text
실제 환경
   │
   │ 공개 I/O
   ▼
Plugin
   ├─ 행동 문법(ActionSpec)
   ├─ 관찰 자료형(ObservationField)
   ├─ 실제 명령 실행
   └─ 외부 reward / terminated / truncated 전달
   │
   ▼
AASSR Core
   ├─ 후보 명령 생성
   ├─ 상태/행동 표현
   ├─ Knowledge
   ├─ Policy
   ├─ Prophecy
   ├─ Calibration
   ├─ Critic
   ├─ Imagination
   ├─ ASEQ
   └─ Skills
```

플러그인은 **후보 행동 목록조차 전략적으로 반환하지 않는다.** 행동 종류와 매개변수 자료형만 선언한다. Core가 공개 관찰에서 같은 자료형의 값을 모아 실행 가능한 후보를 구성한다. 이를 통해 플러그인이 정답 후보만 남기거나 실패 후보를 제거하는 우회 경로를 차단한다.

## 플러그인이 할 수 있는 것

- 행동 이름과 매개변수의 자료형 정의
- 공개 관찰 채널 이름과 자료형 정의
- Core가 선택한 명령을 실제 환경 프로토콜로 변환하여 실행
- 환경이 실제로 반환한 공개 데이터를 손실 없이 전달
- 환경이 실제로 주는 외부 보상과 종료/중단 신호 전달
- 네트워크, 게임 API, 로봇 제어 등 입출력 프로토콜 처리

## 플러그인이 할 수 없는 것

- 상태 벡터 또는 semantic state identity 정의
- 행동 feature/representation 정의
- 후보 행동의 가치 평가, 순위화, 전략적 필터링
- `정답`, `좋은 후보`, `잘못된 후보`, `진전` 같은 과제 의미 라벨 부여
- Prophecy/world model 설치 또는 모델 종류 선택
- Critic/Imagination 점수 정의
- shaping reward 추가
- 숨은 환경 상태나 정답을 관찰값으로 변환해 노출

`validate_minimal_plugin()`은 플러그인이 `install_world_model`, `state_vector`, `action_structure`, `rank_actions` 같은 Core 권한을 노출하면 즉시 거부한다.

## Core가 표현을 소유하는 이유

기존 pentest 경로에서는 플러그인이 route/profile/object 역할과 HTTP status 전용 표현, 전용 Prophecy head까지 결정했다. 그러면 높은 성능이 나와도 "AASSR이 의미를 배운 것인지, 플러그인이 의미를 미리 정리해 준 것인지"를 분리하기 어렵다.

새 Core는 `ValueKind`와 `TemporalKind`만 받아 일반적인 표현을 만든다. 예를 들어 Core는 어떤 필드가 `ENTITY`, `TEXT`, `CATEGORICAL`, `SCALAR`인지는 알지만, 그 필드가 로그인인지 목표인지 알지 못한다.

## 기존 10k 경로의 위치

기존 `current_entrypoint.py`, `current_plugin_api.py`, `current_relational_state*.py`, `current_status_models.py`, `plugins/current_pentest.py`는 삭제하지 않는다.

이유:

1. 10k 체크포인트와 기존 실험을 정확히 재현해야 한다.
2. 구조 변경과 과거 성능 결과를 섞으면 연구 증거가 훼손된다.
3. 새 Core 성능이 검증되기 전 과거 구현을 덮어쓰지 않는다.

따라서 현재 구분은 다음과 같다.

```text
과거 증거 재현: current_* pentest runtime
새 연구 아키텍처: aassr_v2.core + 최소 Plugin
```

새 Core의 성능 우위는 아직 주장하지 않는다. 이 변경의 현재 증거는 **구조적 분리, 정적 감사, 단위/통합 테스트**까지다.
