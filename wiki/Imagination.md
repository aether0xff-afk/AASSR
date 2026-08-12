# Imagination

Imagination은 AASSR의 **counterfactual planner**다.

핵심 질문은 다음과 같다.

> **실제 행동을 하기 전에 Prophecy가 예측한 여러 미래를 몇 단계 전개해 보면, 현재 Policy가 고른 행동보다 더 나은 첫 행동을 선택할 수 있는가?**

AASSR에서 Imagination은 학습 데이터를 만들어내는 장치가 아니라 **실행 전 계획 장치**다.

> [!IMPORTANT]
> 현재 핵심 구현: `src/aassr_v2/current_planner.py`  
> 신뢰도 gate: `src/aassr_v2/current_confidence_gate.py`  
> Critic support: `src/aassr_v2/current_critic_support.py`

---

# 1. 기본 아이디어

현재 상태에서 가능한 행동이 세 개 있다고 하자.

```text
A
B
C
```

Model-free Policy는 각각의 현재 Q값을 비교할 수 있다.

```text
Q(S,A)
Q(S,B)
Q(S,C)
```

Imagination은 한 단계 더 나아간다.

```text
A를 하면?
  -> 가능한 미래들
      -> 그 미래에서 다음 행동은?
          -> 그 뒤의 미래는?

B를 하면?
  -> ...

C를 하면?
  -> ...
```

그리고 **최종적으로 지금 실제로 실행할 첫 행동 하나만 선택**한다.

---

# 2. 왜 단순 `n x k` 나무로 설명하면 부족한가?

초기 AASSR 노트에서는

```text
n개의 행동 후보
x
k depth
```

형태의 상상나무로 설명했다.

직관적으로는 맞지만 current-generation planner에서는 더 중요한 구조가 있다.

1. 환경의 stochastic outcome과 agent의 decision을 구분해야 한다.
2. 각 outcome에는 probability mass가 있다.
3. model reliability가 낮으면 branch를 믿지 않아야 한다.
4. Critic이 OOD라면 override를 막아야 한다.
5. concrete alias가 많아도 같은 structural root 계산을 중복하면 안 된다.
6. 깊은 branch가 prune되어도 실제 root action은 보존해야 한다.

따라서 현재 Imagination의 핵심은 단순 branch count보다 **확률적 planning semantics**에 있다.

---

# 3. Chance node와 Decision node

이 구분이 current planner에서 가장 중요하다.

## Chance node

환경 outcome은 agent가 선택할 수 없다.

예:

```text
행동 A
  |-- 0.7 --> 정상 진행
  |-- 0.2 --> 403
  `-- 0.1 --> 429
```

이 경우 좋은 결과만 고르면 안 된다.

가치는 expectation으로 계산해야 한다.

```math
V_{chance} = \sum_i p_i V_i
```

---

## Decision node

예측된 다음 상태에서 다음 행동은 agent가 선택할 수 있다.

```text
predicted S'
   |-- action B
   |-- action C
   `-- action D
```

이 경우에는 가장 좋은 행동을 고를 수 있다.

```math
V_{decision} = \max_a V(S',a)
```

---

# 4. 왜 이 둘을 섞으면 안 되는가?

잘못된 planner:

```text
행동 A의 가능한 outcome 중
가장 좋은 outcome만 선택
```

이렇게 하면 agent가 실제로 통제할 수 없는 환경 randomness를 마치 선택할 수 있는 것처럼 취급한다.

예:

```text
A:
  10% success
  90% failure

B:
  80% moderate success
  20% neutral
```

`max outcome`만 보면 A가 더 좋아 보일 수 있다.

하지만 expected return으로는 B가 더 나을 수 있다.

그래서 AASSR은

```text
chance = expectation
decision = max
```

을 분리한다.

---

# 5. Planning tree

개념적인 전개는 다음과 같다.

```text
real state S0
  |
  +-- root action A
  |      |
  |      +-- p11 --> S11
  |      |            |
  |      |            +-- decision B
  |      |            +-- decision C
  |      |
  |      `-- p12 --> S12
  |
  +-- root action D
  |      |
  |      `-- ...
  |
  `-- root action E
         `-- ...
```

각 root의 최종 값은 하위 chance/decision backup을 통해 계산된다.

---

# 6. Critic은 왜 필요한가?

Planner depth를 무한히 늘릴 수는 없다.

어느 depth에서는 rollout을 멈추고 그 상태 이후의 장기 가치를 추정해야 한다.

```text
S0 -> S1 -> S2 -> S3
                  |
                  v
               Critic
```

현재 Critic은 relational GRU discounted sparse-return model이다.

기본적으로 실제 task return을 학습한다.

```text
success       +1
true failure  -1
truncation     0
```

따라서 Imagination branch를 외부 sparse objective와 같은 기준으로 평가할 수 있다.

---

# 7. Root preservation

깊은 branch가 불확실하거나 pruning되어도 **실제로 실행 가능한 root action 자체가 평가에서 사라지면 안 된다.**

예:

```text
root action A
  -> depth 1 prediction 가능
  -> depth 2에서 일부 branch 불안정
```

잘못된 구현:

```text
깊은 branch 실패
-> A 전체 삭제
```

현재 원칙:

```text
깊은 branch 실패
-> 이미 계산한 root / shallower value로 fallback
-> root action 유지
```

이것이 root preservation이다.

---

# 8. Structural root deduplication

Pentest action surface에서는 concrete action이 매우 많을 수 있다.

예:

```text
GET route-12
GET route-31
GET route-44
...
```

하지만 relational role 관점에서는 같은 구조일 수 있다.

```text
172 concrete roots
       ↓
~17 relational structures
```

같은 structural root를 각각 Prophecy/Critic으로 계산하면 비용이 크게 증가한다.

그래서 current planner는 계산 단계에서 dedup한다.

```text
structural root 1회 계산
      |
      v
concrete aliases에 value fan-out
```

하지만 실제 실행에서는 concrete identity를 유지한다.

```text
compute identity   = relational structure
execution identity = concrete action
```

---

# 9. 왜 concrete action을 끝까지 버리면 안 되는가?

Relational representation은 transfer에는 유리하지만 실제 환경은 concrete action을 요구한다.

예:

```text
"catalog-like route를 요청한다"
```

는 structural decision이고,

```text
GET /r_31
```

은 실제 실행이다.

Planner가 structural value만 계산하더라도 마지막에는 current action surface의 concrete action으로 bind해야 한다.

---

# 10. Prophecy reliability gate

Imagination은 world model을 사용하므로 model error에 민감하다.

그래서 다음 질문을 먼저 본다.

```text
이 predicted future를 믿을 수 있는가?
```

Calibration reliability가 부족하면 aggressive override를 허용하지 않는다.

중요한 점:

```text
confidence는 value bonus가 아니다.
```

예측을 더 신뢰한다고 미래가 더 좋은 것은 아니다.

신뢰도는 **planner input validity**를 판단하는 값이다.

---

# 11. Local Critic support gate

2026-08-11 2k diagnostic에서 중요한 문제가 있었다.

Critic은 전체적으로 학습된 상태였지만 higher-level unseen region에서는 실제 training support가 부족했다.

그런데 Imagination은 그 OOD Critic 값을 믿고 Policy를 여러 번 override했다.

그래서 current-generation은 local support를 확인한다.

```text
현재 imagined state/action이
실제 Critic training data에서 충분히 지원됨?
```

지원 부족:

```text
fail closed
-> override 취소
-> Policy 유지
```

이것은 Imagination의 자유를 무조건 제한하는 장치가 아니라 **OOD value extrapolation을 실제 근거와 분리하기 위한 방법론적 gate**다.

---

# 12. Intervention margin

Imagination alternative가 Policy보다 아주 조금 높다고 매번 행동을 바꾸면 model noise에 민감해질 수 있다.

그래서 실제 override에는 일정 margin이 필요할 수 있다.

개념적으로:

```text
V(imagined best) - V(policy action) > margin
```

일 때만 switch candidate가 된다.

단, margin 자체도 연구 결과를 만들어내는 hidden shaping이 되지 않도록 실험에서 명시적으로 고정하고 보고해야 한다.

---

# 13. Intervention accounting

과거 진단에서는 candidate가 만들어졌지만 후속 gate에서 취소된 경우에도 intervention count가 증가하는 계측 문제가 있었다.

현재는 다음을 구분한다.

```text
plan
switch candidate
suppressed switch
final intervention
changed executed action
```

진짜 intervention은 **모든 gate를 통과하고 실제 실행 action이 Policy 원래 action과 달라진 경우**다.

이 구분이 없으면 "Imagination이 몇 번 행동을 바꿨는가"를 과대평가할 수 있다.

---

# 14. Training에서 왜 Imagination intervention을 끄는가?

현재 AASSR의 주요 비교는 같은 checkpoint에서 Imagination OFF/ON의 marginal effect를 보려는 것이다.

따라서 training 중부터 Imagination이 행동 데이터를 바꾸면 두 조건이 이미 서로 다른 학습 distribution을 갖게 된다.

현재 protocol:

```text
Training
  Imagination intervention OFF
        |
        v
one frozen checkpoint
     /      \
    /        \
OFF eval    ON eval
```

이렇게 하면 평가 차이를 planner 사용 여부에 더 직접적으로 귀속할 수 있다.

---

# 15. 2026-08-11 2k diagnostic이 보여준 것

당시 repaired run에서 Imagination은 더 이상 inert하지 않았다.

```text
Imagination plans         297
switch candidates         218
executed interventions     86
changed actions             86
```

문제는 **행동을 바꿀 수 있게 된 것과 좋은 방향으로 바꾼 것은 다르다**는 점이었다.

많은 intervention이 `403/404/429` 오류로 이어졌고 직접 성공을 만든 intervention은 없었다.

이 결과는 다음 병목을 드러냈다.

```text
과거:
Imagination cannot affect action

수리 후:
Imagination can affect action

새 문제:
Imagination can confidently choose bad actions
```

---

# 16. 그 이후 들어간 수리

현재 current-generation에는 다음이 포함된다.

1. latest public HTTP status를 relational state에 보존
2. status-supervised stochastic Prophecy
3. status-aware calibration
4. local real-training Critic support gate
5. structural root compute deduplication
6. chance expectation / decision max semantics 유지
7. final executed intervention만 counting

중요한 연구 방법론 경계:

> **수리가 코드에 들어갔다는 것과 장기 성능 향상이 실험으로 확인됐다는 것은 다르다.**

새 benchmark에서 재검증해야 한다.

---

# 17. Depth는 깊을수록 좋은가?

아니다.

Depth를 늘리면 장기 결과를 더 많이 볼 수 있지만 동시에 world-model error도 누적된다.

개념적으로:

```text
benefit(depth)
= longer-horizon information

cost(depth)
= model error accumulation
+ compute
+ branching
```

따라서 최적 depth는 단순히 크게 잡는 문제가 아니다.

현재 연구에서는 depth 자체보다

- calibration
- branch validity
- root preservation
- batched execution
- structural dedup

을 함께 봐야 한다.

---

# 18. Branch count는 많을수록 좋은가?

역시 아니다.

모든 concrete action을 그대로 branch하면 계산량이 action surface 크기에 비례해 폭발할 수 있다.

그래서 현재 planner는 relational structural identity를 이용해 계산을 공유한다.

즉 초기의 `n`은 단순 concrete action 수가 아니라 **실질적으로 다른 structural decision 수**와 연결해서 봐야 한다.

---

# 19. Imagination의 실패 모드

## 19.1 Model error exploitation

Planner가 world-model의 오류를 찾아내 실제로는 나쁜 행동을 좋은 행동처럼 평가할 수 있다.

대응:

- calibration
- uncertainty gating
- status-aware prediction

## 19.2 OOD Critic exploitation

Critic이 훈련하지 않은 영역에서 큰 값을 내놓아 override할 수 있다.

대응:

- local Critic support fail-closed

## 19.3 Branch explosion

Concrete alias가 많아 계산량이 폭발한다.

대응:

- structural root dedup
- depth batching

## 19.4 Optimistic stochastic backup

환경 outcome을 decision처럼 `max`하면 위험 outcome을 무시할 수 있다.

대응:

- chance expectation / decision max 분리

## 19.5 Planner가 실제로 행동을 못 바꿈

모든 root value가 비슷하거나 gate가 너무 보수적이면 intervention이 0이 될 수 있다.

대응:

- Critic discrimination audit
- calibration audit
- intervention accounting

---

# 20. Imagination 연구 가설

현재 연구 가설은 단계적으로 보는 것이 정확하다.

```text
H1. Prophecy가 usable future distribution을 만든다.
H2. Planner가 chance/decision semantics를 올바르게 계산한다.
H3. Critic이 branch 간 장기 가치를 구분한다.
H4. Gate가 OOD / unreliable override를 막는다.
H5. Imagination이 실제 Policy action을 유의미하게 변경한다.
H6. 변경된 행동이 오류를 줄이거나 성공을 만든다.
H7. 같은 frozen checkpoint에서 Full이 OFF보다 낫다.
```

`H5`가 성립했다고 `H6`, `H7`이 자동으로 성립하지 않는다.

2026-08-11 diagnostic은 바로 이 차이를 보여줬다.

---

# 21. 어떻게 검증해야 하는가?

Imagination 평가는 성공률 하나만 보면 부족하다.

함께 볼 지표:

- plan count
- switch candidate count
- suppressed switch count
- final intervention count
- changed executed action count
- direct success-producing intervention
- intervention error rate
- root coverage
- local Critic support pass/fail
- Prophecy reliability
- runtime / wall time
- no-Imagination vs Full same-checkpoint success

최종 핵심 비교:

```text
same frozen AASSR checkpoint

OFF
vs
ON
```

이다.

---

다음으로 읽기:

- **[Prophecy](Prophecy)**
- **[Research Architecture](Research-Architecture)**
- **[Experiments](Experiments)**
- **[Current Status](Current-Status)**
