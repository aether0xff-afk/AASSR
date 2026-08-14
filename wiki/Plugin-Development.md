# 플러그인 제작법

AASSR의 새 Plugin 원칙은 간단하다.

> **Plugin은 플레이 방법과 공개 정보의 종류만 알려준다. 의미는 Core가 학습한다.**

## Plugin이 하는 일

```text
행동 문법 선언
+ 관찰 자료형 선언
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
            ObservationField("objects", ValueKind.SET, item_kind=ValueKind.ENTITY),
        ),
        actions=(
            ActionSpec("use", (ActionParameter("target", ValueKind.ENTITY),)),
        ),
    )

    def reset(self, *, seed=None):
        return PluginStepResult(PluginObservation({"objects": (...,)}))

    def step(self, command):
        # 실제 환경에 command를 보낸 뒤 현재 공개 결과만 반환
        ...
```

자세한 개발 규격은 `src/aassr_v2/plugins/README.md`와 `docs/CORE_PLUGIN_ARCHITECTURE.md`를 기준으로 한다.
