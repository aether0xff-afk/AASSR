# AASSR Plugin 만들기

새 플러그인은 `aassr_v2.core.plugin_contract`의 **최소 계약**만 구현한다.

핵심 규칙:

> **플러그인은 세계의 문법만 알려주고, 세계의 의미는 Core가 배운다.**

Plugin은 플레이 방법과 현재 공개된 정보의 자료형을 전달하는 얇은 I/O 계층이다. 문제를 풀기 위한 기억, 의미 해석, 후보 선택은 Core의 책임이다.

## 최소 구현

```python
from aassr_v2.core import (
    ActionParameter,
    ActionSpec,
    ObservationField,
    PluginObservation,
    PluginSchema,
    PluginStepResult,
    ValueKind,
)

class MyPlugin:
    schema = PluginSchema(
        plugin_id="my-env",
        version="v1",
        observations=(
            ObservationField("screen_text", ValueKind.TEXT),
            ObservationField("objects", ValueKind.SET, item_kind=ValueKind.ENTITY),
        ),
        actions=(
            ActionSpec(
                "use",
                parameters=(ActionParameter("target", ValueKind.ENTITY),),
            ),
        ),
    )

    def reset(self, *, seed=None):
        return PluginStepResult(
            PluginObservation({"screen_text": "...", "objects": ("x", "y")})
        )

    def step(self, command):
        # command를 실제 환경 프로토콜로 실행한다.
        # 현재 공개 결과만 반환한다.
        # 의미 해석, 발견 이력, 행동 순위화, shaping reward는 넣지 않는다.
        ...
```

## 허용되는 작업

- 행동의 이름/매개변수/자료형 정의
- 관찰 채널과 자료형 정의
- 실제 환경과 통신
- **현재 응답/현재 관찰**에서 기계적으로 보이는 값을 전달
- 환경이 원래 주는 reward/terminated/truncated 전달
- 통신을 수행하는 데 필요한 프로토콜 상태 유지
  - 예: HTTP CookieJar, 연결 세션, 로봇 통신 핸들

## 금지되는 작업

- `state_vector`, `semantic_state_identity`, `action_structure` 구현
- `install_world_model` 같은 모델 주입
- "이 후보가 유망하다"는 순위/필터
- 정답/오답/target/진전 등 task 의미 라벨
- 실패 응답을 보고 후보를 플러그인에서 제거
- 내부 reward 추가
- **이전 관찰에서 발견한 대상/링크/후보를 문제 해결 지식으로 누적**

마지막 항목이 중요하다. 예를 들어 HTTP Plugin이 이전 페이지에서 본 링크를 계속 들고 있는 것은 "어떻게 HTTP 요청을 보내는가"가 아니라 "이전에 무엇을 발견했는가"라는 문제 해결 기억이다. 이런 기억은 `CorePublicKnowledge`가 담당한다. 반면 쿠키는 다음 HTTP 요청을 실제로 수행하기 위한 프로토콜 상태이므로 Plugin에 있어도 된다.

플러그인은 후보 명령 목록을 반환하지 않는다. Core가 `ActionSpec`과 공개 관찰의 자료형을 이용해 후보를 생성한다.

## 제어 신호와 관찰은 분리한다

`PluginStepResult.reward`, `terminated`, `truncated`, `error`는 Core 실행을 제어하는 표준 채널이다. 연구 harness나 프로토콜이 이 값을 별도의 전송 필드/헤더로 운반하더라도, **그 전송용 제어 값 자체를 다시 일반 observation에 중복 노출하면 안 된다.**

예를 들어 localhost HTTP 연구 harness가 `X-AASSR-Reward`라는 전용 header로 외부 reward를 운반한다면:

```text
X-AASSR-Reward
  → PluginStepResult.reward        허용
  → observation["headers"]         금지
```

일반 환경이 실제로 공개하는 다른 header는 그대로 관찰할 수 있다. 금지되는 것은 **연구 harness가 만든 제어 채널을 learner-visible world data로 재노출하는 것**이다. 이렇게 해야 reward/terminal 정보를 이름만 보고 맞히는 shortcut을 막을 수 있다.

또한 `diagnostics`는 관찰이나 제어 신호가 아니다. `external_reward`, `terminated`, `truncated` 같은 예약 제어 key를 diagnostics로 덮어쓸 수 없으며, `validate_step_result()`가 이 경계를 검사한다.

## 자료형

- `BOOLEAN`: 참/거짓
- `SCALAR`: 연속/수치 값
- `CATEGORICAL`: 범주 값
- `ENTITY`: 환경에서 다시 행동 매개변수로 사용할 수 있는 식별 가능한 값
- `TEXT`: 공개 텍스트
- `SET`: 같은 종류 값의 집합 (`item_kind` 지정 가능)
- `MAPPING`: 공개 key/value 구조
- `BYTES`: 공개 바이트 데이터

## 시간적 종류

관찰값의 **의미**가 아니라 기계적인 수명만 선언한다.

- `STATE`: 현재 세계 상태의 일부로 지속되는 공개 값
- `EVENT`: 한 응답/한 사건에서 관찰된 공개 증거
- `COUNTER`: 요청 횟수처럼 누적되는 숫자
- `MEASUREMENT`: 지연 시간처럼 측정 때마다 흔들릴 수 있는 값

Core는 `COUNTER`와 `MEASUREMENT` 변화만으로 semantic state가 달라졌다고 판단하지 않는다. `EVENT`는 새 공개 증거로 기억할 수 있지만 동일 증거 반복은 새 진전으로 세지 않는다.

## 검사

```bash
python scripts/audit_core_boundary.py
pytest -q \
  tests/test_minimal_plugin_contract.py \
  tests/test_core_episode_scope.py \
  tests/test_core_boundary_static.py \
  tests/test_local_http_plugin.py
```

실제 네트워크 예시는 `aassr_v2.plugins.local_http.LocalHttpPlugin`을 참고한다. 이 플러그인은 loopback 주소만 허용하며, 각 HTTP 응답에서 현재 공개된 데이터만 자료형 그대로 전달한다.
