# Knowledge

Knowledge는 AASSR에서 **현재 episode 동안 실제 response를 통해 이미 알아낸 사실을 명시적으로 보존하는 [memory/context](MDP-and-POMDP)** 다.

중요한 것은 자료구조가 `dict`인가 아닌가가 아니다.

연구적으로 더 중요한 질문은 다음이다.

> **어떤 정보를 언제 알게 되었고, 그 정보를 어느 prediction과 decision에 사용할 수 있는가?**

즉 Knowledge의 핵심은 **저장 방식보다 [causal information boundary](Causality-Leakage-and-Evaluation)** 다.

> [!IMPORTANT]
> 현재 manifest 계약: `episode-local-response-knowledge-context-v1`  
> 기본 저장 구조: `src/aassr_v2/knowledge.py`  
> current Prophecy 경계: `KnowledgeBoundProphecy` in `current_generation.py`

---

# 0. 먼저 알아두면 좋은 개념

- [MDP and POMDP](MDP-and-POMDP) — state와 observation, history/memory가 필요한 이유
- [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation) — hindsight leakage, privileged information, time order
- [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries) — replay와 current explicit knowledge의 차이
- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — Knowledge confidence와 model reliability의 차이
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — concrete identity를 episode 간 정답처럼 들고 가면 안 되는 이유
- [Model-Based RL & World Models](Model-Based-RL-and-World-Models) — imagined fact와 real fact를 구분해야 하는 이유

---

# 1. 연구 질문

[희소 보상](Sparse-Reward-and-Credit-Assignment) 장기 문제에서는 행동 자체뿐 아니라 행동 중 얻은 정보가 이후 행동을 가능하게 한다.

예:

```text
route를 요청함
→ response에서 새로운 사실 발견
→ 다음 행동 후보가 열림
→ 그 뒤에야 목표 경로 진행 가능
```

그래서 AASSR은 단순 state vector 외에도 **명시적인 episode-local Knowledge provenance**를 관리한다.

이것은 [POMDP](MDP-and-POMDP)에서 history를 보존하려는 일반적인 문제와 연결되지만, AASSR Knowledge는 neural hidden state가 아니라 **명시적 fact + provenance**를 저장한다.

---

# 2. Knowledge와 State는 같은가?

완전히 같지는 않다.

[State / observation](MDP-and-POMDP)는 현재 agent가 의사결정에 사용하는 공개 상황의 representation이다.

Knowledge는 response에서 획득한 명시적 사실과 그 출처를 보존한다.

```text
State
= 지금 decision에 사용되는 public situation representation

Knowledge
= 지금까지 real response를 통해 이미 획득한 explicit facts의 context
```

Current [Relational State v3](State-Representation)가 이미 많은 public response fact를 포함하기 때문에 최근 Prophecy repair에서는 concrete Knowledge를 무조건 다시 주입하지 않는다.

그러나 Skill이나 explicit context path처럼 **언제 정보를 알았는지**가 필요한 경로에서는 Knowledge boundary가 여전히 중요하다.

---

# 3. Knowledge와 GRU hidden state는 같은가?

아니다.

```text
KnowledgeStore
→ explicit fact
→ provenance 추적 가능
→ 사람이 inspect 가능

GRU hidden state
→ learned latent vector
→ 과거 sequence 정보를 압축
→ 각 차원의 의미가 직접 명시되지 않음
```

AASSR의 [Critic](Critic)은 [GRU](GRU-and-Sequence-Models) latent memory를 사용할 수 있지만 Knowledge는 별도의 symbolic/explicit context다.

두 메모리를 하나의 "기억"으로 뭉치면 어떤 정보가 어디서 왔는지 분석하기 어렵다.

---

# 4. KnowledgeEntry

기본 entry는 다음 의미를 가진다.

```text
key
value
source_trace_id
confidence
enabled_action_signatures
```

즉 단순히:

```text
"token": true
```

만 저장하는 것이 아니라 **어느 real trace에서 나온 정보인지 provenance**를 함께 유지할 수 있다.

이는 debugging뿐 아니라 [causality audit](Causality-Leakage-and-Evaluation)에 중요하다.

---

# 5. Provenance란?

Provenance는 정보의 출처를 의미한다.

예:

```text
fact: route X exists
source_trace_id: transition-00427
```

라고 기록하면 나중에 다음을 확인할 수 있다.

```text
이 사실은 실제 response에서 얻었나?
상상 branch에서 생겼나?
어느 episode에서 생겼나?
행동 전에 이미 알고 있었나?
```

즉 provenance는 **Knowledge의 causal history를 검증할 수 있게 만드는 metadata**다.

---

# 6. Episode-local 원칙

Current default에서는 Knowledge를 episode 간 영구 정답 메모리처럼 사용하는 것이 핵심이 아니다.

목표는 현재 episode에서 실제로 발견한 response information을 이후 decision에 쓰는 것이다.

```text
Episode start
    ↓
Knowledge initially limited
    ↓
real actions / responses
    ↓
Knowledge accumulates
    ↓
Episode ends
```

환경 seed가 달라졌는데 이전 episode의 concrete 정답 identifier를 그대로 들고 가는 구조는 [unseen transfer](Relational-Representation-and-Generalization)를 오염시킬 수 있다.

---

# 7. 왜 cross-episode concrete memory가 위험한가?

Training episode에서:

```text
route-12 = 중요한 route
```

를 알았다고 하자.

Evaluation seed에서 같은 역할이:

```text
route-31
```

로 permutation되었다면 이전 concrete fact를 그대로 유지하는 것은 도움이 되지 않을 뿐 아니라 benchmark shortcut이 될 수 있다.

AASSR은 reusable knowledge를 **구조적 학습 parameter/relational pattern**으로 transfer하고, current episode의 concrete known facts는 episode-local로 다루는 방향을 택한다.

---

# 8. 가장 중요한 규칙: anti-hindsight boundary

다음 transition을 생각하자.

```text
S_t --A_t--> S_{t+1}
```

`S_{t+1}`의 response에서 새로운 token `K_new`를 발견했다고 하자.

잘못된 경로:

```text
K_new를 얻음
→ 과거로 돌아감
→ A_t 실행 전 Prophecy input에 K_new 사용
```

그러면 model은 실제 행동 시점에는 알 수 없었던 미래 정보를 사용한다.

이것이 [hindsight leakage](Causality-Leakage-and-Evaluation)다.

---

# 9. 올바른 시간 순서

```text
K_t
 ↓
predict(S_t, A_t, K_t)
 ↓
execute A_t
 ↓
observe real response
 ↓
K_{t+1} = K_t + new response knowledge
```

Transition `t`를 예측할 때 사용할 수 있는 것은 **행동 전 Knowledge**뿐이다.

이 경계가 지켜져야 Prophecy 성능이 실제 online agent에서도 재현 가능하다.

---

# 10. 왜 이것이 'causal'한가?

시간 순서를 보면:

```text
과거 observation
→ current information
→ action
→ future observation
```

이다.

Future observation은 current action의 원인보다 뒤에 있다.

Future information을 current decision feature로 사용하면 prediction relation이 실제 online causal graph와 달라진다.

AASSR의 [response-causal observation contract](State-Representation)도 같은 원칙을 따른다.

---

# 11. Holdout validation에서는 왜 context-free path가 필요한가?

World-model [Calibration](Calibration)을 할 때 현재 live episode의 Knowledge를 과거 holdout transition에 무분별하게 넣으면 leakage가 생길 수 있다.

그래서 architecture는 대략 두 경로를 구분한다.

```text
context-free predict
→ frozen holdout validation

predict_with_context
→ 현재 real decision/planning에서
   명시적으로 전달된 current episode Knowledge만 사용
```

이 경계 덕분에 model validation과 online context usage를 분리할 수 있다.

---

# 12. Knowledge가 action surface를 바꿀 수 있는 이유

어떤 response fact는 새로운 행동을 실제로 가능하게 할 수 있다.

예:

```text
새 route 발견
→ 그 route를 대상으로 한 request action 등장
```

Knowledge entry는 enabled action signature를 연결할 수 있다.

Context-aware prediction에서는 현재 실제 action surface에 존재하는 enabled action을 predicted next-state action map에 합칠 수 있다.

단:

> 존재하지 않는 concrete action을 world model이 마음대로 창조하는 것이 아니라 **현재 action surface와 실제 획득 Knowledge의 인과적으로 유효한 교집합**을 사용해야 한다.

---

# 13. Knowledge와 Legal Action Mask

[Prophecy](Prophecy)는 다음 state의 legal action surface도 예측한다.

Knowledge는 실제로 이미 발견된 public action-enabling fact를 보존할 수 있다.

두 정보원이 중복될 수 있기 때문에:

```text
State / model prediction
+
Knowledge context
```

를 결합할 때 같은 fact를 두 번 과도하게 강조하지 않는 것이 중요하다.

이것이 current relational Prophecy에서 concrete Knowledge reinjection을 보수적으로 다루는 이유 중 하나다.

---

# 14. Knowledge와 ASEQ의 차이

둘 다 경험을 다루지만 목적이 다르다.

```text
Knowledge
= 무엇을 알아냈는가?

ASEQ
= 어떤 실제 (S,A,S') transition을 경험했는가?
```

예:

```text
"이 route가 존재한다"
```

는 Knowledge일 수 있다.

반면:

```text
같은 semantic S에서 A를 했더니 다시 S였다
```

는 [ASEQ](ASEQ) self-loop evidence다.

---

# 15. Knowledge와 Replay의 차이

[Replay buffer](Replay-Buffer-and-Episode-Boundaries)는 learner training을 위해 과거 transition을 저장한다.

KnowledgeStore는 **현재 decision context에서 실제로 알고 있다고 주장할 수 있는 explicit fact**를 저장한다.

```text
Replay
= 학습 데이터

Knowledge
= current episode에서 알고 있는 사실
```

Replay에 `route-12`가 나온 적 있다고 해서 새 episode agent가 `route-12`의 존재를 직접 알고 있다고 처리하면 안 된다.

---

# 16. Knowledge와 Learned Parameter의 차이

과거 경험은 neural parameter나 relational value를 통해 일반화될 수 있다.

```text
과거 episode 경험
→ DQN/Prophecy/Critic parameter 업데이트
→ structural pattern 학습
```

이것과:

```text
현재 episode에서 concrete route X를 실제로 알고 있음
```

은 다른 claim이다.

전자는 **statistical learning**, 후자는 **explicit factual context**다.

---

# 17. Branch-local Knowledge

일반적인 Imagination 설계에서는 imagined branch마다 Knowledge가 달라질 수 있다.

```text
root
 |-- branch A → fact X를 알게 됨
 `-- branch B → fact Y를 알게 됨
```

따라서 `KnowledgeStore.clone()`처럼 branch-local copy를 만들 수 있는 구조가 유용하다.

그러나 imagined branch의 fact는 **그 branch 안의 counterfactual context**이지 real episode factual truth가 아니다.

---

# 18. Imagined Knowledge는 Real Knowledge가 아니다

아주 중요한 구분이다.

```text
Prophecy가 "아마 token을 얻을 것"이라고 예측
```

과:

```text
실제 response에서 token을 얻음
```

은 같은 fact가 아니다.

AASSR 원칙:

> **상상은 [planning](Counterfactual-Planning-and-Search)에 사용하지만, persistent factual knowledge의 근거는 real transition이다.**

Imagined fact를 real Knowledge에 확정 저장하면 [world-model hallucination](Model-Based-RL-and-World-Models)이 factual memory로 굳을 수 있다.

---

# 19. 왜 "dictionary를 쓰는 이유"보다 이 문제가 중요한가?

초기 설명에서는 다음을 강조할 수 있다.

```text
dict lookup이 빠름
key-value가 편함
구현이 간단함
```

이것은 engineering 선택일 뿐 연구 핵심은 아니다.

더 중요한 질문은:

```text
무엇을 저장하는가?
언제 저장하는가?
어떤 provenance를 가지는가?
얼마나 믿는가?
어느 decision부터 사용할 수 있는가?
real fact와 imagined fact를 어떻게 분리하는가?
episode를 넘겨도 되는가?
```

다.

---

# 20. Knowledge confidence

`KnowledgeEntry` 자체에도 confidence가 있을 수 있다.

이 값은 [Prophecy reliability](Stochasticity-Uncertainty-and-Probability)와 같은 개념이 아니다.

```text
KnowledgeEntry confidence
= explicit fact 자체의 신뢰도

Prophecy reliability
= world-model transition prediction의 empirical 신뢰도

Critic support
= value estimate 주변의 real training evidence
```

세 값을 같은 scalar 의미로 섞으면 안 된다.

---

# 21. Knowledge와 Information Value

[Policy](Policy)의 information residual은 **어떤 행동이 미래 decision에 유용한 정보를 제공할 가치**를 나타낸다.

Knowledge는 그 행동 후 실제로 얻은 **명시적 사실**이다.

```text
Information value
= 행동 전: 이 행동이 정보를 줄 가치가 있는가?

Knowledge
= 행동 후: 실제로 무엇을 알게 되었는가?
```

이 구분은 [Information Theory & Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation)에서 더 자세히 다룬다.

---

# 22. Failure mode: Hindsight leakage

행동 후 얻은 정보를 행동 전 prediction에 사용.

결과:

```text
offline prediction metric ↑
real online reproducibility ↓
```

대응:

- trace provenance
- pre-action Knowledge snapshot
- context-free holdout path

---

# 23. Failure mode: Cross-episode concrete leakage

이전 seed의 concrete identifier를 새 episode에서 정답처럼 유지.

결과:

- unseen transfer benchmark 오염
- relational generalization 효과 과대평가

대응:

- episode-local Knowledge
- structural learning과 concrete factual memory 분리

---

# 24. Failure mode: Imagined fact promotion

World model prediction을 실제 관측 fact처럼 persistent store에 저장.

결과:

```text
model hallucination
→ Knowledge
→ future decision
→ 더 많은 hallucination
```

대응:

- branch-local counterfactual context
- real observation에서만 factual commit

---

# 25. Failure mode: State/Knowledge double counting

이미 [Relational State v3](State-Representation)에 반영된 fact를 Knowledge context에서 과도하게 다시 주입하면 같은 정보가 중복 강조될 수 있다.

대응:

- state contract와 context contract 분리
- current repaired Prophecy의 보수적 Knowledge reinjection

---

# 26. Failure mode: Stale Knowledge

환경이 변했는데 이전 fact를 영구적으로 참이라고 유지하면 잘못된 행동을 가능하게 할 수 있다.

Knowledge가 mutable/invalidatable한 이유와 연결된다.

```text
added
changed
removed
```

같은 delta semantics가 필요할 수 있다.

---

# 27. Failure mode: Provenance loss

Fact만 남고 출처가 사라지면:

```text
이 fact가 real인가?
imagined인가?
어느 transition에서 나왔나?
```

를 검증하기 어려워진다.

연구 재현성과 leakage audit 관점에서 provenance는 단순 debugging metadata 이상이다.

---

# 28. Knowledge를 평가하는 방법

Task success만 보면 Knowledge가 어떻게 작동했는지 알기 어렵다.

Diagnostic metric 예:

- knowledge entries created
- provenance-complete fraction
- cross-episode carryover count
- enabled-action additions
- stale/removed entries
- hindsight-leak regression test
- context-free vs context-aware Prophecy difference
- Knowledge OFF/ON success difference

이 중 일부는 correctness metric이고 일부는 activity metric이므로 [proxy metric](Ablation-Benchmarking-and-Reproducibility)을 구분한다.

---

# 29. 연구 가설

```text
H1. explicit episode Knowledge가 long-horizon decision에 도움이 되는가?
H2. provenance를 유지하면 leakage/failure audit이 쉬워지는가?
H3. anti-hindsight boundary를 지켜도 Prophecy가 usable prediction을 학습하는가?
H4. cross-episode concrete Knowledge를 제한하면 unseen transfer가 더 공정해지는가?
H5. relational state와 Knowledge의 중복을 최소화하면서 필요한 history information은 보존되는가?
H6. imagined fact를 persistent Knowledge에서 분리하면 model self-amplification이 줄어드는가?
```

---

# 30. 관련 코드

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

# 31. 한 문장 요약

> **Knowledge는 '무엇을 기억하느냐'보다 '그 사실을 언제, 어떤 real response에서 알았으며 어느 decision부터 사용할 수 있느냐'를 명시하는 episode-local causal memory 계층이다.**

---

다음으로 읽기:

- **[State Representation](State-Representation)**
- **[Policy](Policy)**
- **[Prophecy](Prophecy)**
- **[ASEQ](ASEQ)**
- **[Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation)**
- **[Concept Index](Concept-Index)**
