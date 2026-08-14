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

### 7. 첫 localhost smoke에서 Plugin 쪽에 문제 해결 기억이 남아 있었다

최초 `LocalHttpPlugin` 구현은 이전 응답에서 발견한 link를 `_known_links`에 누적했다. 통신 자체를 위해 CookieJar를 유지하는 것과 달리, "전에 무엇을 발견했는가"는 문제 해결 기억이다.

**판정:** Plugin 권한 초과.

**수정:** Plugin은 현재 응답에서 기계적으로 추출한 link/form만 반환한다. 과거에 공개된 entity를 다음 판단까지 유지하는 기능은 `CorePublicKnowledge`로 이동했다.

### 8. 첫 localhost smoke에서 volatile 값이 semantic progress처럼 보였다

최초 smoke artifact에서 실제 transition 30개에 대해 experience의 novel-outcome revision도 30번 증가했다. `latency_ms`, 매 요청마다 달라질 수 있는 header 값 같은 공개 측정치가 raw observation fingerprint에 들어가면서 동일한 문제 상태도 매번 다른 결과처럼 취급될 수 있었다.

이 구조는 ASEQ의 semantic self-loop 판단까지 약하게 만들 수 있다. "요청 횟수/시간이 달라졌다"는 이유만으로 `S → A → S`가 아닌 것처럼 보이면 사용자와 연구에서 고정한 ASEQ 의미와 맞지 않는다.

**수정:**

- `STATE`: 현재 semantic state identity에 사용
- `EVENT`: 공개 evidence로 기억하되 동일 evidence 반복은 새 진전으로 세지 않음
- `COUNTER`, `MEASUREMENT`: semantic identity와 evidence revision에서 제외
- mapping 형태 EVENT는 volatile value 전체가 아니라 공개 key 구조를 기본 evidence로 사용

이 규칙은 `core/public_memory.py`에 있고 테스트로 고정한다.

### 9. Concrete 후보별 경험이 episode를 넘어 누수될 수 있었다

새 Core의 첫 구현에서는 concrete action signature별 실제 경험 통계가 runtime 전체에 남았다. 그런데 환경 reset 뒤 같은 문자열 ID가 다른 숨은 의미를 가질 수 있는 환경에서는 이전 episode의 후보별 평가가 다음 episode에 그대로 전달될 수 있다.

**판정:** 일반 Core의 기본값으로는 너무 강한 기억이며 rename/generalization 실험을 오염시킬 수 있다.

**수정:** concrete 후보별 경험은 기본적으로 `episode-local`로 변경했다. DQN/Prophecy/Critic/Skills 같은 구조적 학습은 계속 유지하지만, concrete ID에 묶인 시도/결과 통계는 episode 시작 시 초기화한다. 명시적으로 `preserve_knowledge_across_episodes`를 선택한 실험에서만 보존한다.

### 10. 후보 수 제한이 concrete 이름의 사전순 편향을 만들 수 있었다

Core가 직접 후보를 만들도록 옮긴 뒤에도, 후보가 너무 많을 때 정렬된 concrete ID의 앞부분만 자르면 이름 자체가 선택 확률에 영향을 주게 된다. 이는 과거 signature lexicographic tie 문제를 다른 위치에서 다시 만드는 셈이다.

**판정:** Core가 후보를 소유하더라도 concrete 이름에 전략적 우선순위가 생기면 안 된다.

**수정:** bounded candidate surface는 Core가 episode/evidence 기반 seed로 표본화한다. 표본화 seed에 concrete 후보 이름을 넣지 않고 후보 수/구조만 사용한다. 같은 episode의 같은 공개 evidence에서는 후보 표면이 안정적으로 유지되며, 사전순 앞부분을 고르는 방식은 제거했다.

### 11. Plugin diagnostics가 Core 제어 신호를 덮어쓸 수 있었다

`PluginEnvironmentAdapter`의 호환 `raw` 경로에는 Core가 만든 `external_reward`, `terminated`, `truncated`와 Plugin diagnostics가 함께 들어간다. 기존 형태에서는 diagnostics의 같은 이름 key가 뒤에서 병합될 경우 Core가 만든 제어 신호를 덮어쓸 수 있었다.

이 문제는 task 의미를 해석하는 문제와 별개다. Plugin에게 디버그 정보를 허용하더라도 학습 보상과 episode 경계를 바꿀 권한까지 생기면 최소 계약이 깨진다.

**판정:** Plugin → Core 제어 경계 결함.

**수정:**

- `external_reward`, `terminated`, `truncated`를 Plugin diagnostics의 예약 금지 key로 지정
- `reward`는 유한한 실수만 허용
- `terminated`, `truncated`, `error`는 실제 bool만 허용
- `error_code`는 문자열 또는 `None`만 허용
- 별도 regression test를 CI gate에 추가

이 수정은 의미 판단을 Plugin에 주지 않고 오히려 Plugin 권한을 더 축소한다.

### 12. localhost 계측용 reward/termination header가 learner observation에도 노출되고 있었다

로컬 실제 HTTP 실험은 서버가 `X-AASSR-Reward`, `X-AASSR-Terminated`, `X-AASSR-Truncated` 같은 전용 header로 외부 reward와 episode 경계를 harness에 전달한다. 기존 `LocalHttpPlugin`은 이 값을 제어 신호로 소비하면서 동시에 일반 `headers` observation에도 그대로 넣고 있었다.

그러면 learner가 환경의 일반 공개 정보와 실험 계측용 제어 채널을 구분하지 못한다. 특히 Prophecy가 terminal을 예측할 때 harness가 붙인 이름 자체를 shortcut으로 사용할 가능성이 생긴다.

**판정:** 실제 환경 정보와 연구 harness 제어 채널의 경계 불완전.

**수정:**

- configured reward/termination/truncation header는 외부 제어 신호로만 소비
- learner가 받는 일반 `headers` observation에서는 해당 header를 제거
- 다른 실제 공개 header는 그대로 보존
- real socket regression test에서 reward/termination은 전달되지만 제어 header는 observation에 없음을 검증

이 수정은 환경의 의미를 Plugin이 해석한 것이 아니라, **연구 harness 자체가 만든 제어 채널을 관찰 데이터와 분리한 것**이다.

### 13. 같은 자료형이라는 이유만으로 서로 다른 프로토콜 값이 섞일 수 있었다

최소 Plugin 계약의 초기 버전은 `TEXT`, `ENTITY` 같은 `ValueKind`만으로 후보 매개변수를 만들었다. 하지만 실제 환경에서 같은 Python/표현 자료형이라고 같은 행동 슬롯에 넣을 수 있는 것은 아니다.

예를 들어 HTTP에서는:

```text
응답 HTML              TEXT
form payload template  TEXT
```

둘 다 `TEXT`지만 응답 HTML 전체를 POST body 후보로 사용하는 것은 기계적으로 잘못된 조합이다. `ENTITY`도 URL, object ID, 장치 ID처럼 서로 다른 프로토콜 공간이 섞일 수 있다.

**판정:** task 의미 문제가 아니라 행동 문법의 타입 체계가 너무 거칠었다.

**수정:**

- `ObservationField`와 `ActionParameter`에 선택적 `value_space` 추가
- `value_space`는 "형식상 같은 슬롯에 넣을 수 있는가"만 표현
- `url`, `form-payload`, `object-id` 같은 기계적 이름은 허용
- `correct-url`, `bad-profile`, `target-id` 같은 전략적/정답 의미는 금지
- Core 후보 생성과 `CorePublicKnowledge` 재사용이 같은 `value_space` 안에서만 일어나도록 수정
- 기존 positional 생성자의 의미가 바뀌지 않도록 새 필드를 과거 필드 뒤에 추가
- 일반 `SchemaDrivenRepresentation`과 active `MemoryBackedRepresentation` 둘 다 회귀 테스트

이 수정은 Plugin에게 행동 우선순위를 주는 것이 아니라 **플레이 가능한 값의 형식**만 더 정확히 말하게 한다.

### 14. 새 Core가 과거 Plugin framework의 `PluginOutcome`에 남아 의존하고 있었다

새 `core/representation.py`와 `core/dqn.py`는 환경 의미를 사용하지 않았지만, transition 결과 객체를 과거 `action_plugins.PluginOutcome`에서 가져오고 있었다. 새 Core/Plugin 경계를 만들고도 타입 소유권 일부가 legacy plugin framework에 남아 있었던 셈이다.

**판정:** 기능상 도메인 누출은 아니지만 새로운 Core의 소유권 경계가 불완전.

**수정:**

- `core/transition.py`에 환경 중립 `CoreTransitionOutcome` 추가
- Plugin adapter와 DQN은 Core-owned outcome만 사용
- `scripts/audit_core_boundary.py`에서 `action_plugins` direct/transitive dependency를 금지
- 기존 historical path의 `action_plugins`는 그대로 보존

CI의 Core boundary audit가 이 금지 조건에서도 통과했다.

### 15. Skill의 구조적 행동이 여러 concrete 후보와 맞으면 다시 사전순 ID를 골랐다

`CoreRelationalSkillLibrary.resolve_primitive()`는 structural template과 맞는 concrete 행동이 여러 개일 때 `min(action.signature)`를 사용했다. 이는 일반 Policy 후보 선택에서 제거했던 lexicographic concrete-ID 편향을 Skills 안에서 다시 만들었다.

**판정:** rename/generalization 철학과 불일치. 특히 성공 ASeq를 다른 episode/환경 구조에 재사용할 때 Skill이 이름 순서 때문에 다른 행동을 실행할 수 있다.

**수정:**

- ambiguous Skill grounding은 현재 **학습된 Core Policy value**로 concrete 후보를 고름
- 정확한 value tie는 concrete signature를 seed에 넣지 않는 Core-seeded symmetric tie로 처리
- runtime이 `self.policy.value`를 Skill grounding에 연결
- `ambiguous_groundings`, `value_groundings`, `symmetric_groundings` 진단 추가
- 높은 학습 가치 후보가 lexicographic 뒤에 있어도 선택되는 회귀 테스트 추가
- 동일 value에서는 여러 Core seed에서 처음 이름만 고정 선택하지 않는 회귀 테스트 추가

이 변경은 Skill에 도메인 규칙을 넣는 것이 아니라, **Core가 이미 학습한 Policy를 사용해 structural Skill을 현재 concrete 세계에 grounding**하는 것이다.

## 새 경계에서 유지한 것

- ASEQ의 정확한 `S → A → S` 반복 억제 의미
- 외부 sparse reward와 내부 정보 가치의 분리
- Policy / Prophecy / Critic / Imagination / Skills / Knowledge의 역할 분리
- 실제 transition만 학습 사실로 사용하는 원칙
- Imagination이 실제로 실행되지 않았으면 ON/OFF 성능 실험을 유효한 treatment로 보지 않는 진단

## 현재 자동 검사

새 CI는 다음을 별도 gate로 검사한다.

- Core direct/transitive environment import 차단
- 과거 `action_plugins` framework로의 Core 역의존 차단
- Plugin 권한 초과 차단
- Plugin observation에 전략적 action-candidate channel이 없음
- Plugin diagnostics가 reward/termination 제어 신호를 덮어쓰지 못함
- 비정상/비유한 reward와 잘못된 제어 flag 형식 차단
- localhost harness control header가 learner observation에 노출되지 않음
- 같은 `ValueKind`라도 다른 `value_space` 값이 행동 슬롯에서 섞이지 않음
- Core 공개 기억도 기계적 `value_space`를 보존함
- Core가 공개 typed value를 기억해 다음 후보 생성에 사용할 수 있음
- episode reset 시 Core 공개 기억이 초기화됨
- concrete 후보별 경험이 기본적으로 episode-local임
- 후보 수 제한이 concrete 이름의 사전순 앞부분을 선택하지 않음
- 같은 episode/같은 evidence에서 bounded 후보 표면이 안정적임
- Skill grounding이 lexicographic concrete ID를 우선하지 않음
- counter/measurement 변화가 semantic state를 가짜로 바꾸지 않음
- 같은 semantic `S → A → S`가 threshold 뒤 ASEQ에서 실제로 guard됨
- loopback 실제 socket I/O
- 외부 redirect 차단
- Plugin이 이전 페이지의 발견 link를 기억하지 않음

2026-08-14의 `value_space`, Core-owned transition, Skill grounding 수정 이후에도 `aassr-core-minimal-plugin`의 `boundary`와 `core-runtime-cpu` 두 job이 모두 통과했다.

## 실제 loopback 학습 smoke

구조 분리만 해 놓고 실제 학습 경로가 죽어 있는지 확인하기 위해, `127.0.0.1`의 실제 `ThreadingHTTPServer`를 띄우고 `LocalHttpPlugin`을 통해 28 episode의 CPU 학습 smoke를 실행했다.

완료된 CI run에서 관찰된 값:

```text
episodes                 28
real transitions        216
positive episodes         1
negative episodes        17
zero episodes            10
DQN gradient updates     89
Prophecy observations   181
Prophecy updates         54
Critic episodes          28
Critic transitions      216
Critic updates           26
Calibration refreshes    43
ASEQ guard events          3
```

이 결과로 주장할 수 있는 것은 제한적이다.

**확인된 것:**
- Python 객체 내부에서 HTTP를 흉내 내는 simulator가 아니라 실제 loopback socket I/O가 사용됨
- 새 최소 Plugin → Core adapter → Policy/Prophecy/Critic 학습 경로가 실제로 연결됨
- DQN, Prophecy, Critic의 parameter update가 실제 발생함
- Core가 공개 정보를 기억하고 행동 후보를 다시 구성할 수 있음

**이 결과로 주장하지 않는 것:**
- 새 Core의 성공률이 높음
- 기존 10k runtime보다 성능이 좋음
- 일반화가 증명됨
- Imagination이 성능을 높임

해당 smoke에서 Imagination은 요청되어 있었지만 Critic 신뢰 조건이 충족되지 않아 실제 planner treatment가 활성화되지 않았다. 따라서 Imagination 성능 증거로 사용하지 않는다.

## 기존 10k 경로 보존

기존 `current_*`, `plugins/current_pentest.py`, pentest simulator 계열은 삭제하거나 새 Core에 맞춰 조용히 의미를 바꾸지 않는다. 기존 10k checkpoint와 post-10k 진단을 정확히 재현하기 위한 historical path로 유지한다.

따라서 현재 연구 증거는 분리한다.

```text
historical evidence
  = 기존 pentest-coupled runtime / 10k checkpoint 결과

new architecture evidence
  = 최소 Plugin 계약 / Core 독립성 / 실제 localhost I/O 및 학습 경로
```

## 아직 증명되지 않은 것

이번 감사로 다음은 증명되지 않는다.

- 새 Core가 기존 10k pentest runtime보다 성능이 높다.
- localhost 환경에서 장기 의존성을 이미 안정적으로 해결한다.
- 새 일반 표현이 충분한 전이/일반화 능력을 가진다.
- simulator가 과거 L2 실패의 모든 원인이었다.
- 새 Core에서 Imagination의 실질적 성능 기여가 확인됐다.

이것들은 구조 분리 후 별도의 통제 실험으로 검증해야 한다.

## 다음 연구 경로

1. 최소 Plugin 계약과 Core 경계 CI를 상시 유지
2. localhost 실제 환경의 난도를 한 축씩 증가
3. Core의 representation / Knowledge / Policy / Prophecy / Critic 병목을 각각 분리해 측정
4. Imagination은 실제 planner run과 intervention이 발생한 조건에서만 ON/OFF 비교
5. simulator는 단위/회귀/고장 주입과 historical reproduction에 사용
6. 충분한 근거가 쌓인 뒤 새 Core 세대의 성능/일반화 주장을 별도 benchmark로 검증
