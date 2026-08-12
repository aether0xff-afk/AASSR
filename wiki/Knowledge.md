# Knowledge

Knowledge는 AASSR에서 **현재 episode 동안 실제 response를 통해 이미 알아낸 사실을 명시적으로 보존하는 context**다.

중요한 것은 자료구조가 `dict`인가 아닌가가 아니다.

연구적으로 더 중요한 질문은 다음이다.

> **어떤 정보를 언제 알게 되었고, 그 정보를 어느 prediction과 decision에 사용할 수 있는가?**

> [!IMPORTANT]
> 현재 manifest 계약: `episode-local-response-knowledge-context-v1`  
> 기본 저장 구조: `src/aassr_v2/knowledge.py`  
> current Prophecy 경계: `KnowledgeBoundProphecy` in `current_generation.py`

---

# 1. 연구 질문

희소 보상 장기 문제에서는 행동 자체뿐 아니라 행동 중 얻은 정보가 이후 행동을 가능하게 한다.

예:

```text
route를 요청함
-> response에서 새로운 사실 발견
-> 다음 행동 후보가 열림
-> 그 뒤에야 목표 경로 진행 가능
```

그래서 AASSR은 단순 state vector 외에도 **명시적인 episode-local Knowledge provenance**를 관리한다.

---

# 2. Knowledge와 State는 같은가?

완전히 같지는 않다.

State는 현재 agent가 의사결정에 사용하는 public observation representation이다.

Knowledge는 response에서 획득한 명시적 사실과 그 출처를 보존한다.

개념적으로:

```text
State
= 지금 관측되는 공개 상황

Knowledge
= 지금까지 실제 response를 통해 획득한 명시적 사실의 context
```

current relational state가 이미 많은 public response fact를 포함하기 때문에 최근 Prophecy repair에서는 concrete Knowledge를 무조건 재주입하지 않는다. 그러나 Skill이나 explicit context path처럼 **언제 정보를 알았는지**가 필요한 경로에서는 Knowledge boundary가 여전히 중요하다.

---

# 3. KnowledgeEntry

기본 entry는 다음 의미를 가진다.

```text
key
value
source_trace_id
confidence
enabled_action_signatures
```

즉 단순히

```text
"token": true
```

만 저장하는 것이 아니라 **어느 real trace에서 나온 정보인지 provenance**를 함께 유지할 수 있다.

이는 debugging과 causality audit에 중요하다.

---

# 4. Episode-local 원칙

current default에서는 Knowledge를 episode 간 영구 정답 메모리처럼 사용하는 것이 핵심이 아니다.

목표는 현재 episode에서 실제로 발견한 response information을 이후 decision에 쓰는 것이다.

```text
Episode start
    |
    v
Knowledge initially limited
    |
real actions / responses
    |
Knowledge accumulates
    |
Episode ends
```

환경 seed가 달라졌는데 이전 episode의 concrete 정답 identifier를 그대로 들고 가는 구조는 transfer 연구를 왜곡할 수 있다.

---

# 5. 가장 중요한 규칙: anti-hindsight boundary

다음 transition을 생각하자.

```text
S_t --A_t--> S_{t+1}
```

`S_{t+1}`의 response에서 새로운 token `K_new`를 발견했다고 하자.

잘못된 학습/검증:

```text
K_new를 얻음
-> 과거로 돌아가
-> A_t 실행 전 Prophecy input에 K_new 사용
```

그러면 모델은 실제 행동 시점에는 알 수 없었던 미래 정보를 사용한다.

이것이 hindsight leak이다.

---

# 6. 올바른 시간 순서

```text
K_t
 |
 v
predict(S_t, A_t, K_t)
 |
 v
execute A_t
 |
 v
observe real response
 |
 v
K_{t+1} = K_t + new response knowledge
```

즉 transition `t`를 예측할 때 사용할 수 있는 것은 **행동 전 Knowledge**뿐이다.

이 경계가 지켜져야 Prophecy 성능이 실제 online decision에서 가능한 수준을 반영한다.

---

# 7. Holdout validation에서는 왜 context-free path가 필요한가?

World model calibration을 할 때 현재 live episode의 Knowledge를 과거 holdout transition에 무분별하게 넣으면 또 다른 leakage가 생길 수 있다.

그래서 current architecture는 대략 다음 두 경로를 구분한다.

```text
context-free predict
-> frozen holdout validation

predict_with_context
-> 현재 real decision / planning에서
   명시적으로 전달된 episode Knowledge만 사용
```

이 경계 덕분에 model validation과 online context usage를 분리할 수 있다.

---

# 8. Knowledge가 action surface를 바꿀 수 있는 이유

어떤 response fact는 새로운 행동을 실제로 가능하게 할 수 있다.

예:

```text
새 route 발견
-> 그 route를 대상으로 한 request action 등장
```

Knowledge entry는 enabled action signature를 연결할 수 있다.

Context-aware prediction에서는 현재 실제 action surface에 존재하는 enabled action을 predicted next-state action map에 합칠 수 있다.

단, 존재하지 않는 concrete action을 임의로 창조하는 것이 아니라 **현재 action surface와 실제 획득 Knowledge의 교집합**을 사용한다.

---

# 9. Knowledge와 ASEQ의 차이

둘 다 경험을 다루지만 목적이 다르다.

```text
Knowledge
= 무엇을 알아냈는가?

ASEQ
= 어떤 실제 (S,A,S') transition을 경험했는가?
```

예를 들어:

```text
"이 route가 존재한다"
```

는 Knowledge일 수 있다.

반면:

```text
같은 semantic S에서 A를 했더니 다시 S였다
```

는 ASEQ self-loop evidence다.

둘을 하나의 memory 개념으로 뭉치면 각 신호의 역할이 불명확해진다.

---

# 10. Knowledge와 Replay의 차이

Replay buffer는 학습을 위해 과거 transition을 저장한다.

KnowledgeStore는 현재 decision context에 사용할 explicit fact를 저장한다.

```text
Replay
= 학습 데이터

Knowledge
= 현재 episode의 명시적 알고 있는 사실
```

Replay transition이 존재한다고 해서 agent가 현재 episode에서 그 concrete 사실을 직접 알고 있는 것으로 취급하면 안 된다.

---

# 11. Branch-local Knowledge

일반적인 Imagination 설계에서는 imagined branch마다 Knowledge가 달라질 수 있다.

```text
root
 |-- branch A -> fact X를 알게 됨
 `-- branch B -> fact Y를 알게 됨
```

따라서 KnowledgeStore는 independent clone을 만들 수 있는 구조를 가진다.

다만 current relational Prophecy에서는 public response fact가 relational state 안에 이미 표현되는 부분이 크므로 concrete Knowledge re-injection은 보수적으로 다룬다.

핵심은 **상상 branch의 정보를 실제 episode Knowledge에 바로 써버리지 않는 것**이다.

---

# 12. Imagined Knowledge는 real Knowledge가 아니다

아주 중요한 구분이다.

```text
Prophecy가 "아마 token을 얻을 것"이라고 예측함
```

과

```text
실제 response에서 token을 얻음
```

은 같은 사실이 아니다.

AASSR의 기본 원칙:

> **상상은 계획에 사용하지만, 실제 학습과 persistent factual knowledge의 근거는 real transition이다.**

따라서 imagined fact를 실제로 관측한 사실처럼 persistent store에 확정해서는 안 된다.

---

# 13. 왜 "딕셔너리를 쓰는 이유"보다 이 문제가 중요한가?

초기 연구 노트에서는 Knowledge를 설명하며 Python dictionary의 장점을 중심으로 생각할 수 있었다.

예:

- key-value lookup이 빠름
- 구현이 간단함
- flexible함

이것들은 구현 선택의 이유일 수는 있지만 AASSR의 연구 기여와 직접 연결되지는 않는다.

연구적으로 중요한 질문은 대신 다음이다.

```text
무엇을 저장하는가?
언제 저장하는가?
어떤 출처를 가지는가?
얼마나 믿는가?
어느 decision부터 사용할 수 있는가?
imagined fact와 real fact를 어떻게 분리하는가?
```

따라서 위키에서는 자료구조보다 **causal knowledge contract**를 중심으로 설명한다.

---

# 14. Knowledge confidence

Entry 자체에도 confidence를 가질 수 있다.

이 값은 Prophecy model reliability와 같은 개념은 아니다.

```text
KnowledgeEntry confidence
= 해당 explicit fact 자체의 신뢰도

Prophecy confidence
= model prediction의 reliability
```

두 종류를 같은 scalar 의미로 해석하면 안 된다.

---

# 15. 실패 모드

## 15.1 Hindsight leakage

행동 후 얻은 정보를 행동 전 prediction에 사용.

결과: offline accuracy는 높아지지만 실제 online agent에서는 재현 불가.

## 15.2 Cross-episode concrete leakage

이전 seed의 concrete identifier를 새 episode에 정답처럼 유지.

결과: transfer benchmark 오염.

## 15.3 Imagined fact promotion

world model prediction을 실제 관측 사실처럼 저장.

결과: model hallucination이 factual memory로 굳어짐.

## 15.4 State/Knowledge double counting

이미 relational public state에 반영된 사실을 별도 context에서 과도하게 재주입해 같은 정보를 중복 강조.

대응: current repaired relational Prophecy의 보수적 context contract.

---

# 16. 연구 가설

```text
H1. explicit episode Knowledge가 long-horizon decision에 도움이 되는가?
H2. provenance를 유지하면 leakage / failure audit이 쉬워지는가?
H3. anti-hindsight boundary를 지켜도 Prophecy가 usable prediction을 학습하는가?
H4. cross-episode concrete Knowledge를 제한하면 unseen transfer가 더 공정해지는가?
H5. relational state와 Knowledge context의 중복을 최소화하면서 필요한 정보는 보존할 수 있는가?
```

---

# 17. 관련 코드

```text
src/aassr_v2/knowledge.py
  - KnowledgeEntry
  - KnowledgeDelta
  - KnowledgeStore

src/aassr_v2/current_generation.py
  - KnowledgeBoundProphecy

src/aassr_v2/current_agent.py
  - current episode Knowledge binding
```

---

다음으로 읽기:

- **[Policy](Policy)**
- **[Prophecy](Prophecy)**
- **[ASEQ](ASEQ)**
- **[Design Rationale](Design-Rationale)**
