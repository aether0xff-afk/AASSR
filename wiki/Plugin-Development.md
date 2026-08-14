# 플러그인 제작법

AASSR의 새 Plugin 원칙은 간단하다.

> **Plugin은 플레이 방법과 공개 정보의 종류만 알려준다. 의미는 Core가 학습한다.**

## Plugin이 하는 일

```text
행동 문법 선언
+ 관찰 자료형 선언
+ 기계적 값 호환성 선언
+ 실제 환경 I/O
+ 외부 보상/종료 신호 전달
```

예를 들어 웹 환경이라면 Plugin이 `요청을 보내는 방법`, `URL/본문/상태/쿠키 같은 공개 데이터 종류`를 알려줄 수 있다. 그러나 "이 URL은 로그인", "이 응답은 진전", "이 후보는 틀림" 같은 의미는 알려주면 안 된다.

## Plugin이 하지 않는 일

- 상태 벡터 작성
- 행동 feature 작성
- semantic state 정의
- 행동 후보 순위화/전략적 제거
- Prophecy/world model 설치
- Critic/Imagination 점수 작성
- shaping reward
- 정답/target/중간 목표 노출
- **이전 관찰에서 발견한 대상이나 후보를 문제 해결 지식으로 누적**

마지막 항목은 통신 상태와 구분해야 한다. HTTP 쿠키처럼 다음 요청을 실제로 수행하기 위해 필요한 프로토콜 상태는 Plugin에 둘 수 있다. 하지만 "이전 페이지에서 어떤 링크를 발견했다" 같은 탐색 기억은 Core가 보관해야 한다.

새 계약에서는 Plugin이 후보 명령 목록도 반환하지 않는다. Core가 행동 스키마와 공개 관찰값의 자료형, 그리고 Core가 직접 축적한 공개 지식을 이용해 후보를 만든다.

## `value_space`: 같은 자료형 안에서 기계적 호환성만 구분

`TEXT`, `ENTITY` 같은 자료형만으로는 실제 행동 매개변수를 안전하게 만들기 어려울 수 있다.

웹 예시:

```text
응답 HTML              TEXT / response-body
form payload template  TEXT / form-payload

현재 URL               ENTITY / url
페이지에서 발견한 URL   ENTITY / url
객체 식별자             ENTITY / object-id
```

둘 다 `TEXT` 또는 `ENTITY`라고 해서 서로 아무 슬롯에나 넣을 수 있는 것은 아니다. 그래서 관찰과 행동 매개변수에 선택적으로 `value_space`를 선언할 수 있다.

```python
ObservationField(
    "links",
    ValueKind.SET,
    item_kind=ValueKind.ENTITY,
    value_space="url",
)

ActionParameter(
    "url",
    ValueKind.ENTITY,
    value_space="url",
)
```

이 값이 알려주는 것은 **기계적인 호환성뿐**이다.

```text
"url" 값은 "url" 매개변수에 넣을 수 있다.
```

다음과 같은 전략적 의미는 절대 넣으면 안 된다.

```text
correct-url
likely-target
bad-choice
progress-route
```

그런 이름은 Plugin이 task meaning을 미리 알려주는 것이므로 AASSR의 학습 결과로 인정할 수 없다.

`value_space`를 생략한 기존 Plugin은 이전처럼 같은 `ValueKind`를 사용한다. 새 Plugin에서는 서로 다른 프로토콜 값 공간이 같은 자료형을 공유한다면 `value_space`를 쓰는 것이 안전하다.

## 제어 신호는 일반 관찰과 분리한다

`reward`, `terminated`, `truncated`, `error`는 환경에서 Core로 들어가는 표준 제어 신호다. 어떤 연구 harness가 이 값을 전송하기 위해 별도 프로토콜 필드나 header를 사용하더라도, **그 전송용 값을 다시 일반 관찰 채널에 중복 노출하면 안 된다.**

예를 들어 localhost 연구 harness의 `X-AASSR-Reward` header는 외부 reward로만 전달하고 일반 HTTP header 관찰에서는 제거한다. 반면 실제 서비스가 원래 공개하는 일반 header는 그대로 관찰할 수 있다.

이 경계를 두는 이유는 간단하다. 연구 harness가 만든 `reward`, `terminal` 표시 이름을 learner가 직접 보고 정답 shortcut으로 사용하면 Core의 학습 능력을 검증한 것이 아니기 때문이다.

`diagnostics`도 제어 신호와 별개다. diagnostics가 `external_reward`, `terminated`, `truncated` 같은 예약 key를 덮어쓸 수 없도록 계약과 테스트에서 막는다.

## 관찰값의 시간적 종류

Plugin은 관찰의 의미가 아니라 기계적인 수명만 선언할 수 있다.

- `STATE`: 현재 상태의 일부로 지속되는 공개 값
- `EVENT`: 한 사건/응답에서 관찰된 공개 증거
- `COUNTER`: 누적 횟수
- `MEASUREMENT`: 매번 흔들릴 수 있는 측정값

Core는 `COUNTER`나 `MEASUREMENT`가 달라졌다는 이유만으로 semantic state가 바뀌었다고 보지 않는다. 같은 공개 EVENT가 반복된 것도 새 진전으로 세지 않는다. 이 규칙은 ASEQ의 `S → A → S` 의미를 지키기 위해 중요하다.

## 최소 코드

```python
class MyPlugin:
    schema = PluginSchema(
        plugin_id="my-env",
        version="v1",
        observations=(
            ObservationField(
                "objects",
                ValueKind.SET,
                item_kind=ValueKind.ENTITY,
                value_space="object-id",
            ),
        ),
        actions=(
            ActionSpec(
                "use",
                (
                    ActionParameter(
                        "target",
                        ValueKind.ENTITY,
                        value_space="object-id",
                    ),
                ),
            ),
        ),
    )

    def reset(self, *, seed=None):
        return PluginStepResult(PluginObservation({"objects": (...,)}))

    def step(self, command):
        # 실제 환경에 command를 보낸 뒤 현재 공개 결과만 반환
        ...
```

자세한 개발 규격은 `src/aassr_v2/plugins/README.md`와 `docs/CORE_PLUGIN_ARCHITECTURE.md`를 기준으로 한다.
