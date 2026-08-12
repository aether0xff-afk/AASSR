# Design Rationale

이 페이지는 AASSR의 각 설계를 **"무엇을 썼는가"가 아니라 "왜 그렇게 설계했는가"** 중심으로 정리한다.

초기 연구 노트의 `딕셔너리를 쓰는 이유`, `n과 k의 당위성` 같은 질문을 current-generation 기준으로 더 연구적인 질문으로 바꾼 페이지라고 보면 된다.

---

# 1. 왜 sparse reward를 그대로 유지하는가?

가장 쉬운 방법은 사람이 중간 reward를 만드는 것이다.

```text
route 발견   +0.1
login 성공   +0.2
object 발견  +0.2
proof        +1.0
```

그러나 그러면 agent가 스스로 장기 구조를 배운 것인지, 사람이 제공한 subgoal을 따라간 것인지 분리하기 어렵다.

따라서 current benchmark는 외부 task reward를 좁게 유지한다.

```text
success       +1
true failure  -1
otherwise      0
```

AASSR의 내부 신호는 reward shaping과 분리한다.

---

# 2. 왜 concrete state와 relational state를 둘 다 쓰는가?

둘 중 하나만 고르면 각각 다른 오류가 생긴다.

## Concrete만 쓰면

```text
route-12 != route-31
```

새 seed에서 이름만 바뀌어도 새로운 문제처럼 보인다.

## Relational만 쓰면

같은 역할을 가진 서로 다른 concrete object를 같은 대상으로 취급할 수 있다.

그래서:

```text
Concrete semantic identity
-> ASEQ / exact repetition / real execution

Relational transfer identity
-> Policy / Prophecy / Critic / Skill
```

로 역할을 분리한다.

---

# 3. 왜 ASEQ는 모든 반복을 막지 않는가?

반복 행동 자체는 나쁜 것이 아니다.

예:

```text
S0 -> request -> S1
S1 -> request -> S2
```

같은 verb가 반복돼도 실제로 진행하고 있다.

막아야 하는 것은 실제로 관측된

```text
S -> A -> S
```

의 진전 없는 반복이다.

그래서 current ASEQ guard는 intentionally narrow하다.

---

# 4. 왜 information value를 reward에 합치지 않는가?

정보를 얻는 행동은 장기적으로 유용할 수 있다.

하지만 이를 external reward로 만들면 연구자가 사실상 subgoal reward를 설계하는 셈이 될 수 있다.

그래서 Policy는:

```text
Q_task
+
separate information residual
```

로 유지한다.

이렇게 하면 external objective와 internal exploration signal을 구분해서 감사할 수 있다.

---

# 5. 왜 Prophecy는 deterministic하지 않은가?

Partial observability 때문에 같은 public `(S,A)`에서도 여러 실제 미래가 나올 수 있다.

단일 평균 회귀:

```text
future A
future B
   ↓ average
nonexistent C
```

를 만들 수 있다.

그래서 current Prophecy는 conditional mixture를 사용해 여러 outcome mode와 probability mass를 보존한다.

---

# 6. 왜 HTTP status를 별도 categorical target으로 보는가?

`403`, `404`, `429`는 public하고 decision-critical할 수 있다.

숫자 자체를 연속량으로 보면

```text
403과 404는 숫자상 매우 가까움
```

이지만 의미를 그렇게 해석할 근거는 없다.

그래서 서로 배타적인 categorical class로 학습한다.

또 사람이 `403=나쁨` 같은 value rule을 직접 주입하지 않고 class imbalance만 일반적으로 보정한다.

---

# 7. 왜 probability와 reliability를 분리하는가?

```text
outcome probability
= 그 환경 결과가 일어날 확률

reliability
= 모델이 그 예측을 얼마나 잘 알고 있는가
```

예를 들어 모델이

```text
success 90%
```

라고 말해도 해당 region을 한 번도 제대로 학습하지 않았다면 그 90% 자체가 신뢰할 수 없을 수 있다.

따라서 mixture mass와 calibration confidence를 다른 의미로 유지한다.

---

# 8. 왜 confidence를 value bonus로 쓰지 않는가?

높은 confidence는 "좋은 미래"가 아니라 "예측을 더 믿을 수 있음"을 뜻한다.

잘못된 설계:

```text
V = Critic + confidence bonus
```

현재 설계:

```text
confidence 충분?
  yes -> Critic value 비교 가능
  no  -> fail closed
```

그래서 confidence가 Critic branch ranking에 다시 새어 들어가지 않도록 current encoder에서도 이를 중립화한다.

---

# 9. 왜 chance node는 평균이고 decision node는 max인가?

Agent는 자신의 다음 행동은 선택할 수 있지만 환경 outcome은 선택할 수 없다.

따라서:

```math
V_{chance}=\sum_i p_iV_i
```

```math
V_{decision}=\max_aV(a)
```

를 구분한다.

환경 stochasticity에도 `max`를 쓰면 agent가 실제로 통제할 수 없는 좋은 outcome만 고르는 낙관적 planner가 된다.

---

# 10. 왜 단순히 Imagination depth를 크게 하지 않는가?

초기 질문을 `k를 몇으로 할 것인가?`라고 둘 수 있지만 current system에서는 trade-off가 더 중요하다.

```text
depth 증가
-> 더 긴 미래를 봄
-> 하지만 model error 누적
-> compute 증가
-> branch explosion
```

그래서 depth 하나의 정답보다:

- calibration
- root preservation
- branch validity
- Critic horizon
- batching

을 함께 본다.

---

# 11. 왜 branch count `n`을 concrete action 수로 보면 안 되는가?

현재 action surface에는 이름만 다른 structural alias가 많을 수 있다.

```text
172 concrete actions
-> 약 17 relational structures
```

172개를 모두 독립적인 world-model 계산으로 보면 비용이 불필요하게 커진다.

그래서 계산 identity와 실행 identity를 분리한다.

```text
planning compute
= structural dedup

real execution
= concrete action
```

이것이 초기 `n` 질문의 current-generation 답에 가깝다.

---

# 12. 왜 root를 보존하는가?

깊은 branch가 불확실하다고 root action까지 삭제하면 planner가 실제 legal action을 평가 후보에서 잃을 수 있다.

그래서:

```text
deep rollout 실패
-> root 삭제 X
-> shallower / already-computed value fallback O
```

를 사용한다.

---

# 13. 왜 Critic은 actual sparse return을 학습하는가?

Planner 전용 handcrafted score를 Critic target으로 쓰면 최종 objective와 planning value가 달라질 수 있다.

그래서 current Critic은 실제 external outcome을 기반으로 한다.

```text
success +1
failure -1
truncation 0
```

이렇게 하면 Policy와 planning이 궁극적으로 같은 task objective를 향한다.

---

# 14. 왜 Critic을 모든 trajectory suffix에서 학습하는가?

Imagination은 episode 중간 어느 decision point에서도 시작할 수 있다.

하지만 GRU Critic을 episode 시작점에서만 학습하면 inference 시 필요한 과거 hidden memory가 없을 수 있다.

그래서:

```text
S0 S1 S2 S3
```

에서

```text
S0...
S1...
S2...
S3...
```

모든 decision suffix를 zero-memory root로 학습한다.

---

# 15. 왜 global critic-ready만으로 부족한가?

Critic이 training을 완료했다는 것과 현재 state/action region을 경험했다는 것은 다르다.

```text
global ready = yes
local support = no
```

일 수 있다.

2k diagnostic은 실제로 이 문제를 드러냈다.

그래서 current-generation은 local real-training support gate를 둔다.

---

# 16. 왜 support도 value bonus가 아닌가?

Support가 높다는 뜻은 그 행동이 좋다는 게 아니라 **Critic value를 믿을 실증적 근거가 더 많다**는 뜻이다.

따라서:

```text
V + support bonus
```

가 아니라

```text
support threshold 통과?
-> value 비교 허용 여부
```

로 사용한다.

---

# 17. 왜 Training Imagination intervention을 끄는가?

핵심 실험이 같은 checkpoint에서 planner 효과만 보는 것이기 때문이다.

```text
training with intervention
```

을 허용하면 Full과 OFF가 서로 다른 state-distribution에서 학습하게 된다.

현재:

```text
one training run
-> freeze
-> OFF eval
-> ON eval
```

로 비교한다.

---

# 18. 왜 imagined experience를 real truth로 학습하지 않는가?

World model이 틀릴 수 있기 때문이다.

```text
model error
-> imagined transition
-> model/policy가 그것을 real truth처럼 학습
-> error amplification
```

을 피하기 위해 **상상은 planning에 쓰고 factual learning은 real transition에 근거한다**는 원칙을 유지한다.

---

# 19. 왜 Knowledge는 자료구조보다 causal boundary가 중요한가?

Python `dict`는 구현 편의일 뿐 연구 질문의 핵심은 아니다.

더 중요한 것은:

```text
이 사실을 언제 알았는가?
어느 real response에서 나왔는가?
현재 decision 시점에 사용 가능한가?
imagined fact인가 real fact인가?
```

이다.

그래서 Knowledge 문서는 lookup 성능보다 anti-hindsight / provenance contract를 중심으로 다룬다.

---

# 20. 왜 Skill은 concrete macro가 아니라 relational template인가?

Raw 성공 sequence를 저장하면 seed가 바뀌었을 때 target ID가 달라져 재사용하기 어렵다.

그래서 성공 ASeq를 relational action template로 저장하고 현재 action surface에 다시 bind한다.

```text
structure transfer
+
concrete execution
```

원칙이다.

---

# 21. 왜 baseline을 여러 개 둬야 하는가?

`AASSR > DQN` 한 줄로는 무엇이 효과를 만들었는지 알 수 없다.

그래서:

```text
dqn_raw
-> dqn_relational
-> AASSR no-Imagination
-> AASSR Full
```

과 DreamerV3를 둔다.

각 차이는 다른 질문을 검증한다.

```text
raw -> relational
= representation

relational -> AASSR no-img
= non-Imagination stack

no-img -> Full
= planner marginal effect
```

---

# 22. 설계 원칙 요약

AASSR current-generation에서 반복해서 나타나는 공통 원칙은 다음이다.

```text
1. 정답을 직접 넣지 않는다.
2. 공개 관측과 hidden simulator state를 분리한다.
3. 실제 경험과 imagined experience를 분리한다.
4. concrete execution과 relational transfer identity를 분리한다.
5. probability, reliability, value, support를 서로 다른 의미로 유지한다.
6. 실패하면 근거 없는 override보다 Policy fallback을 택한다.
7. 각 효과를 control / ablation으로 분리한다.
```

이 분리들이 current AASSR의 기술 구조를 복잡하게 만들지만, 동시에 **각 성능 향상이 어디서 왔는지 연구적으로 추적 가능하게 하는 핵심**이다.

---

다음으로 읽기:

- **[Research Architecture](Research-Architecture)**
- **[Policy](Policy)**
- **[Calibration](Calibration)**
- **[Critic](Critic)**
- **[Knowledge](Knowledge)**
- **[Skills](Skills)**
