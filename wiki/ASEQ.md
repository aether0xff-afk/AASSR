# ASEQ

[ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)는 AASSR에서 실제 경험을 다루는 핵심 단위다.

```text
(S, A, S')
```

- `S`: 행동 전 상태
- `A`: 실제로 실행한 행동
- `S'`: 행동 후 상태

현재 pentest runtime에서 [ASEQ](ASEQ)는 특히 **진전 없는 [제자리 반복(self-loop)](ASEQ)를 최소한으로 억제하는 memory/[행동(action)](Reinforcement-Learning)-selection component**로 사용된다.

---

## 1. 왜 필요한가?

[DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)이 어떤 행동의 Q값을 높게 평가한 상태에서 그 행동이 실제로 아무것도 바꾸지 못한다면 다음이 발생할 수 있다.

```text
S
|
| A   <- greedy Q 최고
v
S
|
| A
v
S
|
| A
v
...
```

상태가 그대로이므로 [Policy(정책 모델)](Policy) ranking도 그대로 유지되고 같은 행동을 계속 고를 수 있다.

2026-08-07 [전이(transfer)](Relational-Representation-and-Generalization) 진단에서 실제 [DQN](Q-Learning-DQN-and-TD) [체크포인트(checkpoint)](Reproduction)가 L1 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 평가에서 이런 형태로 정체했다. 일부 체크포인트에서는 raw greedy 평가의 마지막 12개 행동이 동일 행동 `12/12` 반복이었다.

따라서 문제는 단순히 “정답 route를 모른다”가 아니라 **이미 방문해서 상태가 변하지 않는 행동을 계속 최고 Q로 선택하는 제자리 반복**였다.

---

## 2. 현재 ASEQ guard의 원칙

현재 설계는 반복을 폭넓게 금지하지 않는다.

억제 대상은 다음 조건에 해당하는 경우다.

```text
S -> A -> S
```

즉 같은 [의미 기반 상태(semantic state)](State-Representation)에서 같은 행동을 했고, 결과도 같은 의미 기반 상태로 돌아온 제자리 반복다.

그리고 이것이 **실제로 반복 관측**되어야 한다.

개념적으로:

```text
1회: S -> A -> S   기록
2회: S -> A -> S   반복 확인
이후: 같은 self-loop 후보를 억제 가능
```

현재 manifest는 이를 `semantic-self-loop-empirical-v3`로 정의한다.

---

## 3. 무엇을 막지 않는가?

### 같은 행동의 정상 반복

```text
S1 -> A -> S2
S2 -> A -> S3
```

상태가 바뀌므로 허용한다.

예를 들어 동일한 종류의 browse 행동을 여러 번 해야 새로운 route/object가 계속 발견되는 환경이라면 이를 막으면 안 된다.

### 같은 `(S,A)`에서 다른 결과가 관측되는 경우

```text
S -> A -> S
S -> A -> S2
```

환경이 stochastic하거나 부분 관측 때문에 결과가 달라질 수 있다는 증거다.

이 경우 `(S,A)`를 무조건 제자리 반복라고 단정해서는 안 된다.

### 모든 행동을 막는 경우

[ASEQ](ASEQ) guard 때문에 가능한 모든 행동이 제거될 경우 원래 행동 freedom을 복원하는 fail-safe가 필요하다.

목표는 [에이전트(agent)](Reinforcement-Learning)의 자유를 없애는 것이 아니라 **이미 경험적으로 무의미하다고 확인한 exact loop만 피하는 것**이다.

---

## 4. 왜 raw vector identity를 쓰지 않는가?

상태에는 request count처럼 매 step 변하는 값이 있을 수 있다.

예를 들어 실제 문제 상황은 똑같은데 단순 카운터만 바뀌면:

```text
raw S1 != raw S2
```

가 된다.

그러면 인간이 보기에는 같은 제자리 반복인데 exact raw vector 비교에서는 매번 다른 상태가 된다.

따라서 [ASEQ](ASEQ)의 `S`는 task-relevant **concrete semantic identity**를 사용한다.

하지만 여기서 또 중요한 점이 있다.

[ASEQ](ASEQ)는 전이용 relational identity와 완전히 같지 않다.

```text
route-A와 route-B가 둘 다 catalog 역할
```

이라고 해도 같은 episode에서 서로 다른 concrete route라면 [ASEQ](ASEQ)에서는 구분해야 한다. 그렇지 않으면 route-A에서 실패한 행동 때문에 route-B까지 막을 수 있다.

정리하면:

| 목적 | identity |
|---|---|
| exact 제자리 반복 detection | concrete semantic |
| [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility) 간 일반화 | relational |

---

## 5. 주요 실험 결과

### 5.1 재학습 없는 root-cause diagnostic

3개 체크포인트 × L1 학습 중 보지 못한 8 난수 시드s에서 비교했다.

| 체크포인트 | raw greedy | exact [ASEQ](ASEQ) guard |
|---|---:|---:|
| L2 first reached | 0/8, stalled 8/8 | 2/8, stalled 0/8 |
| L2 pre-demotion | 0/8, stalled 8/8 | 7/8, stalled 0/8 |
| post-demotion retrained | 0/8, stalled 8/8 | 5/8, stalled 0/8 |

핵심 결과:

```text
24 raw episodes: stalled 24/24
24 ASEQ episodes: stalled 0/24
```

즉 관측된 무한 제자리 반복는 exact [ASEQ](ASEQ) 수준의 최소 제약으로 제거할 수 있었다.

하지만 성공률은 더 강한 `greedy_no_repeat`보다 낮았다. 이것은 [ASEQ](ASEQ)가 일부 불필요한 탐색을 허용하기 때문이다.

이 결과는 오히려 설계 의도와 맞는다.

> [ASEQ](ASEQ)는 성공 행동을 대신 골라주는 oracle이 아니라, 이미 확인된 무진전 loop를 제거하는 장치다.

---

### 5.2 consistent exact-ASEQ retraining

동일한 exact [ASEQ](ASEQ) 규칙을 학습과 평가에 모두 사용한 6,000-[상태 전이(transition)](MDP-and-POMDP) focused experiment에서는:

#### 학습 중 성공

| training mode | episodes | successes | L0 | L1 | L2 |
|---|---:|---:|---:|---:|---:|
| legacy filter | 94 | 29 | 15 | 14 | 0 |
| exact [ASEQ](ASEQ) | 109 | **50** | **30** | **19** | **1** |

#### 최종 unseen + ASEQ guard ON

| trained with | L0 | L1 | L2 |
|---|---:|---:|---:|
| legacy filter | 1/8 | 1/8 | 0/8 |
| exact [ASEQ](ASEQ) | **8/8** | **7/8** | **1/8** |

모든 exact-[ASEQ](ASEQ) evaluation에서 stalled는 0이었다.

다만 이 실험은 research 난수 시드 1개, evaluation 난수 시드 8개, L0~L2 focused 조건이었다. 따라서 최종 일반화 성능으로 과장해서는 안 된다.

---

## 6. ASEQ가 해결한 것과 해결하지 못한 것

### 해결한 것

- 동일한 무진전 행동의 무한 반복
- train/eval repetition-control mismatch의 일부
- [Policy](Policy) 내부에 남아 있던 해결 능력이 제자리 반복 때문에 가려지는 문제

### 해결하지 못한 것

- 어떤 행동이 성공으로 더 빨리 이어지는지 우선순위화
- request budget을 낭비하는 넓은 탐색
- L2 이상 복잡도에서의 장기 dependency reasoning
- [Prophecy(미래 예측 모델)](Prophecy) 정확도
- [Imagination(가상 미래 탐색)](Imagination) [실제 행동 개입(intervention)](Imagination)의 신뢰성

즉 병목은 다음처럼 이동했다.

```text
과거
self-loop 때문에 아무것도 못함
        |
        v
ASEQ 적용
        |
        v
self-loop 제거
        |
        v
이제 어떤 유효 행동을 먼저 고를 것인가?
```

---

## 7. ASEQ와 Skill의 차이

둘 다 ASeq 경험을 사용하지만 역할이 다르다.

### ASEQ guard

```text
이 transition은 반복해도 아무것도 변하지 않았다
-> 다시 고집하지 말자
```

### Skill

```text
이 행동 sequence는 비슷한 goal에서 반복적으로 성공했다
-> 하나의 재사용 가능한 template 후보로 묶자
```

[ASEQ](ASEQ) guard는 **negative repetition memory**에 가깝고, [Skill(성공 절차 재사용)](Skills)은 **positive reusable sequence abstr행동**에 가깝다.

---

## 8. 한 줄 요약

> **[ASEQ](ASEQ)는 반복 행동을 금지하는 장치가 아니라, 실제 경험으로 확인된 `(S,A,S')` 중 `S -> A -> S` 형태의 진전 없는 제자리 반복만 최소한으로 억제하는 경험 메모리다.**

다음: **[Experiments](Experiments)**
