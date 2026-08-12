# State Representation

AASSR current-generation의 transfer 학습기는 **response-causal relational public state v3**를 사용한다.

이 페이지의 핵심 질문은 다음이다.

> **정답 identity나 hidden simulator state를 주지 않으면서도, 이름이 바뀐 새로운 scenario에서 같은 문제 구조를 알아볼 수 있는 상태 표현을 만들 수 있는가?**

> [!IMPORTANT]
> 현재 manifest 계약: `response-causal-relational-public-state-v3+latest-http-status`  
> 핵심 구현: `src/aassr_v2/current_relational_state_v3.py`

---

# 1. 왜 state representation이 연구 질문인가?

강화학습에서 Policy가 아무리 강해도 입력 representation이 잘못되면 transfer가 어렵다.

예를 들어 training에서:

```text
route-12 = useful catalog-like route
```

였다고 하자.

unseen seed에서 같은 역할이:

```text
route-31 = useful catalog-like route
```

로 바뀌면 concrete ID 중심 learner는 두 상황을 별개로 볼 수 있다.

AASSR은 이름보다 **공개적으로 관측한 역할과 관계 구조**를 transfer representation의 핵심으로 사용한다.

---

# 2. Observation contract와 representation은 다르다

먼저 환경에서 agent가 볼 수 있는 것과, 그 정보를 learner가 어떤 벡터로 표현하는지를 구분해야 한다.

```text
Environment public response
        ↓
Observation contract
        ↓
Relational representation
        ↓
Policy / Prophecy / Critic / Skill
```

Representation이 relational하다고 해서 hidden simulator 정보를 추가로 볼 수 있는 것은 아니다.

---

# 3. 무엇을 볼 수 있는가?

current pentest runtime은 실제 response에서 인과적으로 관측 가능한 public information을 사용한다.

예:

- 발견된 route/profile/object 관계
- 현재 legal action surface
- session / CSRF 존재와 같이 실제로 확인한 상태
- self-counted request usage
- self-observed workflow progress
- latest public HTTP status

---

# 4. 무엇을 의도적으로 숨기는가?

learner에게 직접 주지 않는 정보의 예:

- hidden curriculum level
- exact hidden workflow depth
- exact hidden audit / lockout pressure
- exact hidden session countdown
- hidden rate-limit distance
- 정답 route/profile/object identity
- future state

핵심 원칙:

> **모델이 추론하거나 예측해야 할 정보를 simulator 내부에서 바로 꺼내 observation으로 주지 않는다.**

---

# 5. 두 종류의 identity

AASSR에서는 identity를 하나로 통일하지 않는다.

## Concrete semantic identity

사용처:

- ASEQ
- episode-local exact repetition
- concrete cycle detection
- 실제 environment action execution

```text
route-12 != route-31
```

실제 서로 다른 대상을 구분해야 하기 때문이다.

## Relational transfer identity

사용처:

- Policy
- Prophecy
- Critic
- Skill
- Relational DQN baseline
- DreamerV3 relational adapter

```text
route-12 -> catalog-like role
route-31 -> catalog-like role

=> same relational structure
```

---

# 6. 왜 둘 중 하나만 쓰면 안 되는가?

## Concrete only

```text
identifier rename
-> state identity 전부 변경
-> transfer 약화
```

## Relational only

```text
같은 역할의 서로 다른 concrete entity
-> 같은 대상으로 오인
-> 실제 실행 / self-loop 판정 오류
```

그래서 AASSR은

```text
학습/transfer: relational
실행/정확한 반복 판정: concrete
```

를 분리한다.

---

# 7. Relational state v3의 구조

current v3는 기존 relational v2 descriptor 뒤에 **latest public HTTP status channel**을 추가한다.

현재 코드 기준:

```text
v2 relational descriptor : 35 dimensions
latest status channel     :  8 dimensions
------------------------------------------
v3 descriptor             : 43 dimensions
```

status channel은 다음 public status vocabulary의 one-hot/probability representation이다.

```text
200 / 302 / 400 / 401 / 403 / 404 / 409 / 429
```

---

# 8. 왜 latest HTTP status가 필요했는가?

이전 relational state에서는 전체 semantic structure는 비슷하게 표현하면서도 최근 response의 `403/404/429` 같은 public signal을 잃을 수 있었다.

2026-08-11 Imagination diagnostic에서는 semantic prediction metric이 높게 보여도 실제 override가 이러한 오류 status로 이어지는 문제가 관찰됐다.

즉:

```text
구조적으로 비슷함
!=
decision-critical public outcome까지 같음
```

이었다.

v3는 latest status를 명시적으로 보존해 이 blind spot을 줄인다.

---

# 9. Status는 hidden 위험 신호가 아니다

중요한 방법론 경계다.

AASSR이 보는 것은 실제 response로 공개된 HTTP-like status다.

```text
latest observed 403
```

을 쓰는 것은 허용된다.

반면 simulator 내부의

```text
lockout까지 정확히 1회 남음
hidden audit pressure = 0.93
```

같은 값은 learner에게 직접 주지 않는다.

따라서 status-aware representation은 hidden safety oracle을 추가하는 것이 아니다.

---

# 10. Status vector를 어떻게 얻는가?

current implementation은 public status의 명시적 metadata/fact/vector channel에서 latest status를 복원한다.

우선순위에 따라 이미 relational prediction이 가진 status probabilities를 사용할 수도 있고, 실제 public `last_status` fact 또는 raw public observation channel에서 읽을 수도 있다.

어느 경로든 hidden audit/session state를 읽지 않는 것이 contract다.

---

# 11. Predicted relational state decode

Prophecy는 relational descriptor 자체를 예측한다.

v3 decode는:

```text
predicted base relational semantics
+
predicted legal action mask
+
predicted terminal class
+
predicted status probabilities
```

를 다시 planner가 사용할 `StateSnapshot` 형태로 복원한다.

예측된 latest status는 predicted fact/metadata에도 일관되게 반영된다.

---

# 12. Semantic score v3

World-model calibration에서는 단순 vector distance 하나만 보지 않는다.

current v3 semantic score는 개념적으로 다음 네 종류의 correctness를 함께 본다.

```text
base relational semantics
legal action mask
latest HTTP status
terminal class
```

현재 코드의 가중 구조는:

```text
base semantic quality : 0.35
legal-mask quality    : 0.25
status match          : 0.30
terminal match        : 0.10
```

이다.

이 수치는 reward가 아니라 **Prophecy prediction validation metric**이다.

---

# 13. 왜 status 비중이 꽤 큰가?

과거 diagnostic에서 전체 semantic similarity가 높아도 status error가 실제 decision quality를 망칠 수 있다는 evidence가 나왔기 때문이다.

따라서 calibration metric이 단순 "대부분 비슷하다"만 보지 않고 decision-critical public response를 명시적으로 반영한다.

단, status match를 agent task reward에 더하는 것은 아니다.

---

# 14. 누가 v3 representation을 쓰는가?

current contract 설치 후 핵심 transfer consumer가 v3로 rebound된다.

대표적으로:

- Policy state encoding
- Prophecy relational codec/model
- semantic calibration/evaluator
- Critic/support 관련 relational state key
- DreamerV3 relational adapter

따라서 baseline과 AASSR 비교에서 relational representation 계약을 최대한 일관되게 유지한다.

---

# 15. Raw DQN과 Relational DQN 비교가 중요한 이유

AASSR Full이 raw DQN보다 좋아도 그 차이가 전부 Imagination 때문이라고 할 수 없다.

Representation 자체의 효과가 있을 수 있기 때문이다.

그래서:

```text
dqn_raw
   |
   | state/action representation만 relational로 변경
   v
dqn_relational
```

을 독립 control로 둔다.

이 비교는 AASSR 연구에서 매우 중요한 ablation이다.

---

# 16. State와 Knowledge의 경계

현재 public state에는 이미 실제 response에서 관측한 많은 사실이 포함된다.

KnowledgeStore는 그와 별도로 provenance와 causal timing을 가진 explicit episode context를 관리한다.

```text
State
= 현재 공개 상황 representation

Knowledge
= 어떤 response에서 언제 알게 되었는지까지 관리하는 explicit context
```

같은 사실을 무분별하게 두 경로에서 중복 주입하지 않도록 current Prophecy는 context path를 보수적으로 다룬다.

---

# 17. State와 ASEQ의 경계

Policy/Prophecy는 relational state를 쓰지만 ASEQ는 exact repetition을 판정해야 한다.

따라서 ASEQ까지 같은 relational identity로 뭉치면:

```text
서로 다른 route지만 같은 역할
-> 같은 S라고 오인
-> 정상 행동을 self-loop로 막음
```

이 생길 수 있다.

그래서 concrete semantic state와 relational state를 동시에 유지한다.

---

# 18. 실패 모드

## 18.1 Identifier memorization

Concrete ID에 의존해 unseen rename transfer 실패.

대응: relational role representation.

## 18.2 Over-abstraction

서로 다른 실제 대상을 너무 강하게 같은 state로 압축.

대응: concrete semantic identity를 실행/ASEQ에 별도 유지.

## 18.3 Decision-critical channel loss

전체 구조는 유지하지만 latest status 같은 중요한 public signal을 버림.

대응: relational state v3.

## 18.4 Hidden-state leakage

simulator 내부 정답/압력을 representation에 포함해 benchmark shortcut 발생.

대응: response-causal public observation contract.

## 18.5 Representation drift

Policy, Prophecy, Critic, baseline이 서로 다른 relational definition을 쓰면 비교가 깨진다.

대응: current contract 설치와 manifest/CI validation.

---

# 19. 연구 가설

```text
H1. relational representation이 raw representation보다 unseen transfer에 유리한가?
H2. concrete/relational identity 분리가 self-loop 정확도와 transfer를 동시에 지키는가?
H3. latest public status를 추가하면 Prophecy/calibration의 decision-critical 오류가 줄어드는가?
H4. hidden simulator state 없이도 충분한 문제 구조를 표현할 수 있는가?
H5. 같은 v3 contract를 Policy/Prophecy/Critic/baseline에 적용하면 비교가 더 공정해지는가?
```

---

# 20. 관련 코드

```text
src/aassr_v2/current_relational_state_v3.py
  - latest_status_vector
  - relational_state_descriptor_v3
  - relational_state_vector_v3
  - decode_relational_state_v3
  - semantic_prediction_score_v3
  - install_status_aware_relational_contract

src/aassr_v2/current_manifest.py
  - active observation / policy-state contract
```

---

다음으로 읽기:

- **[Research Architecture](Research-Architecture)**
- **[ASEQ](ASEQ)**
- **[Policy](Policy)**
- **[Prophecy](Prophecy)**
- **[Calibration](Calibration)**
