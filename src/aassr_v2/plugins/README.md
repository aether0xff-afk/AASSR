# AASSR Plugin 만들기

새 플러그인은 `aassr_v2.core.plugin_contract`의 **최소 계약**만 구현한다.

핵심 규칙:

> 플러그인은 세계의 문법만 알려주고, 세계의 의미는 Core가 배운다.

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
        # 의미 해석, 행동 순위화, shaping reward는 하지 않는다.
        ...
```

## 허용되는 작업

- 행동의 이름/매개변수/자료형 정의
- 관찰 채널과 자료형 정의
- 실제 환경과 통신
- 공개 결과 전달
- 환경이 원래 주는 reward/terminated/truncated 전달

## 금지되는 작업

- `state_vector`, `semantic_state_identity`, `action_structure` 구현
- `install_world_model` 같은 모델 주입
- "이 후보가 유망하다"는 순위/필터
- 정답/오답/target/진전 등 task 의미 라벨
- 실패 응답을 보고 후보를 플러그인에서 제거
- 내부 reward 추가

플러그인은 후보 명령 목록을 반환하지 않는다. Core가 `ActionSpec`과 공개 관찰의 자료형을 이용해 후보를 생성한다.

## 자료형

- `BOOLEAN`: 참/거짓
- `SCALAR`: 연속/수치 값
- `CATEGORICAL`: 범주 값
- `ENTITY`: 환경에서 다시 행동 매개변수로 사용할 수 있는 식별 가능한 값
- `TEXT`: 공개 텍스트
- `SET`: 같은 종류 값의 집합 (`item_kind` 지정 가능)
- `MAPPING`: 공개 key/value 구조
- `BYTES`: 공개 바이트 데이터

`TemporalKind.COUNTER`와 `MEASUREMENT`는 Core의 semantic self-loop identity에서 제외된다. 이것은 플러그인이 중요도를 정하는 것이 아니라 해당 값의 **기계적 수명/성질**을 알리는 것이다.

## 검사

```bash
python scripts/audit_core_boundary.py
pytest -q tests/test_minimal_plugin_contract.py tests/test_core_boundary_static.py
```

실제 네트워크 예시는 `aassr_v2.plugins.local_http.LocalHttpPlugin`을 참고한다. 이 플러그인은 loopback 주소만 허용하며 공개 HTTP 데이터를 자료형 그대로 전달한다.
