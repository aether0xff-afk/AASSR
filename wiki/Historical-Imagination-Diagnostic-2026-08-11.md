# Historical Imagination Diagnostic — 2026-08-11

> [!WARNING]
> 이 페이지는 **과거 실패 진단(historical diagnostic)** 을 보존한다. 여기의 `4/20 vs 4/20`, `86 interventions`, `58 bad-status interventions`는 **현재 repaired AASSR의 최종 성능 결과가 아니다.**

이 실험의 가치는 성능 숫자 그 자체보다 **Imagination이 왜 잘못된 행동을 자신 있게 선택했는지 root cause를 찾아낸 것**에 있다.

---

# 1. 당시 질문

당시 AASSR의 Imagination은 이전과 달리 실제 Policy 행동을 바꾸기 시작했다.

그래서 질문이 바뀌었다.

```text
이전 질문
Planner가 실제 action에 영향 줄 수 있는가?

          ↓ 해결

새 질문
Planner가 바꾼 action이 실제로 더 좋은가?
```

관련 개념:
- [Imagination](Imagination)
- [Critic](Critic)
- [Calibration](Calibration)
- [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 2. 실험 설정

- 날짜: `2026-08-11`
- research seed: `7`
- real training transitions: `2,048`
- 하나의 AASSR checkpoint만 학습
- 같은 frozen checkpoint에서 planner OFF / ON 비교
- intervention margin: `0.05`

즉 [same-checkpoint comparison](Ablation-Benchmarking-and-Reproducibility)을 사용했다.

```text
one training run
      ↓
frozen AASSR checkpoint
   /                  \
OFF                    ON
Policy-only       Imagination enabled
```

---

# 3. 결과

| Condition | Success | L0 | L1 | L2 | L3 | L4 | True failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-Imagination | **4/20** | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 |
| Full | **4/20** | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 2 |

Planner diagnostics:

```text
Imagination plans         297
switch candidates         218
executed interventions     86
changed actions             86
```

즉 planner가 단순히 계산만 한 것이 아니라 **86번 실제 실행 action을 Policy 선택과 다르게 바꿨다.**

---

# 4. intervention quality

86개 intervention은 모두 L3 `object_choices` 영역에서 발생했다.

그중:

```text
PluginOutcome.error=True : 58 / 86
404                       : 30
403                       : 26
429                       :  2
direct success-producing  :  0
```

따라서 당시에는:

```text
planner active       = yes
planner beneficial   = not shown
planner often wrong  = yes, in this diagnostic
```

이었다.

---

# 5. matched-state audit

단순히 “ON run이 운이 나빴다”는 설명을 줄이기 위해 intervention state를 OFF run의 같은 scenario / semantic state와 맞췄다.

68개 matched intervention state에서:

```text
Full intervention -> error
Policy original    -> no error
= 50 cases

Full intervention -> no error
Policy original    -> error
= 0 cases
```

이 결과는 당시 잘못된 override가 단순 stochastic bad luck만으로 설명되기 어렵다는 evidence였다.

> [!NOTE]
> 이 matched-state audit 역시 최종 causal theorem이 아니라 **root-cause diagnostic evidence**다. episode trajectory coupling과 limited sample size를 고려해야 한다.

---

# 6. Root cause 1 — decision-critical HTTP status 소실

당시 [Relational State](State-Representation) v2는 구조적 similarity에 집중하면서 최근 public response의 `403`, `404`, `429` 같은 status를 명시적으로 보존하지 않았다.

```text
실제 response
403
  ↓
relational abstraction
  ↓
status distinction 약화
```

문제는 `403`과 `200`이 단순한 작은 feature 차이가 아니라 다음 행동에 큰 영향을 주는 **decision-critical public signal**이라는 점이다.

이 사건은 일반적인 [state aliasing](MDP-and-POMDP) 사례로 볼 수 있다.

```text
transfer를 위한 abstraction
        ↓ 너무 강함
서로 다른 future를 가진 상태가
비슷한 representation으로 합쳐짐
```

---

# 7. Root cause 2 — calibration metric blind spot

당시 holdout에서:

```text
probability-weighted semantic quality ≈ 0.916
terminal match                       ≈ 0.991
```

처럼 전체 semantic metric은 높게 보였다.

하지만 실제 planner intervention은 나빴다.

즉:

```text
전체 state similarity 높음
!=
decision-critical channel 정확함
```

이었다.

그래서 [Calibration](Calibration)은 단순 global semantic similarity가 아니라 public HTTP-status 같은 핵심 channel을 명시적으로 평가하는 방향으로 수정됐다.

---

# 8. Root cause 3 — global Critic readiness와 local support 혼동

당시 training success는 낮은 curriculum level에 집중되어 있었다.

그런데 planner는 높은 unseen level에서 Critic value를 이용해 86번 override했다.

```text
Critic has trained somewhere
          ↓
잘못된 추론
          ↓
Critic is trustworthy here
```

둘은 다르다.

```text
Global readiness
= Critic 학습이 시작되고 overall fit이 존재

Local support
= 현재 imagined state/action 주변에 real training evidence 존재
```

이 진단은 현재의 [local Critic support gate](Critic-Support-and-OOD)를 도입하게 된 직접적인 이유 중 하나다.

---

# 9. Root cause 4 — concrete root alias 계산 폭발

L3에서는 대략:

```text
concrete root actions       ~172
relational root structures   ~17
```

이었다.

실행에서는 concrete action을 구분해야 하지만, 관계적으로 같은 구조의 root를 world model과 Critic에 172번 다시 넣을 필요는 없다.

그래서 current planner는:

```text
execution identity = concrete
compute identity   = relational structural slot
```

을 분리하는 [structural root deduplication](Imagination) 방향으로 수리됐다.

---

# 10. 이 diagnostic 이후 들어간 repair

현재 `main`의 `current_manifest.py`는 당시와 다른 contract를 사용한다.

## Relational State v3

```text
response-causal-relational-public-state-v3
+ latest-http-status
```

## Prophecy

```text
relational-conditional-mixture-ensemble-v5-status-balanced
```

## Prophecy status objective

```text
class-balanced-categorical-public-http-status-v2
```

## Calibration

```text
semantic-probability-holdout-calibration-v3-status-aware
```

## Critic support

```text
local-real-training-support-fail-closed-v1
```

## Root handling

```text
root-concrete-execution
+ structural-compute-dedup
```

즉 이 historical diagnostic의 failure mode를 그대로 둔 모델이 current runtime은 아니다.

---

# 11. 그래서 이 결과를 어떻게 인용해야 하나?

## 올바른 표현

> 2026-08-11 diagnostic에서 Imagination은 86회 실제 행동을 변경했지만 성공률 향상은 없었고, 58회의 bad-status intervention이 발생했다. 이 분석은 status-aware representation/calibration, local Critic support, structural root dedup을 도입하는 근거가 되었다.

## 잘못된 표현

> 현재 AASSR Full의 성공률은 4/20이다.

또는:

> 현재 Imagination은 58/86 확률로 잘못된 행동을 한다.

둘 다 잘못이다. 그 숫자는 **특정 과거 checkpoint와 당시 architecture의 diagnostic**이다.

---

# 12. 연구적으로 왜 중요한 negative result인가?

이 실험은 성능 개선을 보여주지 못했지만 다음을 구분하게 만들었다.

```text
Planner가 행동을 바꿀 수 있음
          !=
Planner가 올바르게 바꿈

World model이 평균적으로 잘 맞음
          !=
중요한 decision channel이 맞음

Critic이 학습됨
          !=
현재 imagined state에서 값이 신뢰 가능함

Concrete action이 다름
          !=
계산해야 할 structural root가 다름
```

즉 AASSR의 현재 설계에서 중요한 여러 구분은 이 실패 실험에서 직접 나왔다.

---

## 다음으로 읽기

- [Current Status](Current-Status)
- [Evidence Matrix](Evidence-Matrix)
- [State Representation](State-Representation)
- [Prophecy](Prophecy)
- [Calibration](Calibration)
- [Critic, Support & OOD](Critic-Support-and-OOD)
- [Imagination](Imagination)
