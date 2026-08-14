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

새 계약에서는 Plugin이 후보 명령 목록도 반환하지 않는다. Core가 행동 스키마와 공개 관찰값의 자료형을 이용해 후보를 만든다.

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
        # 실제 환경에 command를 보낸 뒤 공개 결과만 반환
        ...
```

자세한 개발 규격은 `src/aassr_v2/plugins/README.md`와 `docs/CORE_PLUGIN_ARCHITECTURE.md`를 기준으로 한다.
