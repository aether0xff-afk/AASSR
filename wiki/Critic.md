# Critic

Critic은 AASSR의 Imagination에서 **예측된 미래가 실제 sparse task objective 관점에서 얼마나 가치 있는지** 평가한다.

현재 Critic은 relational GRU 기반이며, 실제 episode에서 얻은 sparse return을 학습한다.

> [!IMPORTANT]
> 현재 manifest 계약: `relational-gru-discounted-sparse-return+zero-memory-decision-suffixes+batched-train-v3`  
> 핵심 구현: `src/aassr_v2/current_return_critic.py`  
> OOD support: `src/aassr_v2/current_critic_support.py`

---

# 1. 연구 질문

> **실제 성공과 실패가 드문 환경에서 학습한 return model이 Imagination branch의 장기 가치를 구분할 수 있는가? 그리고 현재 state/action에서 그 값을 믿을 실제 근거가 있는지 구분할 수 있는가?**

Prophecy는 "무슨 일이 일어날까?"를 예측한다.

Critic은 그 다음 질문을 담당한다.

```text
그 미래는 최종 목표 관점에서 좋은가?
```

---

# 2. Reward와 Critic target

Critic은 사람이 만든 중간 점수를 학습하지 않는다.

기본 external outcome:

```text
success       +1
true failure  -1
truncation     0
ordinary       0
```

따라서 Critic은 **실제 sparse task return**을 미래 상태/행동 구조에 퍼뜨리는 역할을 한다.

---

# 3. Discounted return

어떤 decision point에서 terminal까지 `n` transitions가 남았다고 하자.

현재 suffix root target은 개념적으로 다음과 같다.

```math
G_s = R_{final}\,\gamma^{n-1}
```

따라서 더 빨리 성공으로 연결되는 상태는 같은 최종 `+1`이라도 더 높은 discounted target을 가질 수 있다.

반대로 실제 failure `-1` 역시 거리에 따라 discount된다.

---

# 4. 왜 GRU인가?

단일 `(state, action)`만으로는 최근 transition 흐름이 가진 장기 구조를 충분히 표현하지 못할 수 있다.

GRU Critic은 sequence를 입력받아 transition trajectory의 문맥을 압축한다.

개념적으로:

```text
(S0,A0,S1)
(S1,A1,S2)
(S2,A2,S3)
       |
       v
      GRU
       |
       v
predicted sparse return
```

다만 Imagination과 training의 recurrent-memory 계약을 맞추는 것이 중요하다.

---

# 5. Zero-memory planning 문제

Imagination은 반드시 episode 시작점에서만 호출되는 것이 아니다.

실제 episode 중간의 현재 state에서 갑자기 planning을 시작할 수 있다.

```text
real trajectory
S0 -> S1 -> S2 -> S3
           ^
           여기서 Imagination 시작 가능
```

이때 planner는 과거 전체 trajectory에서 만들어진 hidden GRU state를 갖고 있지 않을 수 있다.

Critic을 오직 `S0`에서 시작하는 full trajectory만 학습하면 training과 inference contract가 어긋난다.

---

# 6. Decision suffix training

이 문제를 해결하기 위해 현재 Critic은 한 real episode에서 여러 suffix를 만든다.

```text
S0 -> S1 -> S2 -> S3 -> terminal

suffix 0: S0 -> S1 -> S2 -> S3
suffix 1: S1 -> S2 -> S3
suffix 2: S2 -> S3
suffix 3: S3
```

각 suffix는 **zero recurrent memory**에서 시작하는 독립 training example이 된다.

따라서 현재 real decision point에서 planner가 zero-memory Critic 평가를 시작하는 상황과 더 잘 맞는다.

---

# 7. Prefix마다 같은 root-return target

한 suffix 내부의 각 prefix를 학습할 때 현재 구현은 그 suffix의 root return을 target으로 사용한다.

개념적으로:

```text
suffix: S1 -> S2 -> S3 -> success
root target = gamma^2

prefix [S1]             -> gamma^2
prefix [S1,S2]          -> gamma^2
prefix [S1,S2,S3]       -> gamma^2
```

이 설계는 Critic이 "현재 planning root에서 시작했을 때의 return"을 sequence가 길어지면서 추정하도록 맞춘다.

---

# 8. Critic input과 relational identity

Critic은 concrete 이름을 직접 암기하는 대신 relational transition features를 사용한다.

목표:

```text
training seed의 route-12에서 배운 가치 구조
        ↓
unseen seed의 같은 역할 route-31로 transfer
```

ASEQ의 concrete semantic identity와 목적이 다르다.

---

# 9. Confidence는 Critic value feature가 아니다

현재 confidence gate 수리에서 중요한 원칙이다.

Prophecy confidence가 Critic input에 들어가면 Critic이 다음 shortcut을 배울 수 있다.

```text
confidence 높음 -> value 높음
```

하지만 "예측을 잘 믿을 수 있음"과 "그 미래가 좋은 미래임"은 다른 개념이다.

그래서 현재 구조는 기존 tensor shape는 유지하되 confidence feature slot을 상수로 중립화한다.

```text
Critic ranking = sparse-return value
confidence     = reliability gate
```

---

# 10. Global `critic_ready`의 한계

Critic이 어느 정도 training을 했다는 사실만으로 모든 state에서 값을 믿을 수는 없다.

```text
critic_ready = True
```

는

```text
current state/action is supported
```

와 같지 않다.

이 차이가 실제 2k Imagination diagnostic에서 중요한 failure mode로 드러났다.

---

# 11. Local Critic support

현재는 real Critic training transitions를 relational action별로 저장하고, 현재 state와 가까운 실제 training region이 존재하는지 계산한다.

질문은 다음 하나다.

> **이 state/action에서 나온 Critic value가 실제 training experience의 근처에 있는가?**

지원이 부족하면 override를 막는다.

```text
Policy action support 충분?
Candidate action support 충분?
        |
        +-- 둘 다 yes -> value comparison 가능
        `-- 하나라도 no -> fail closed, Policy 유지
```

---

# 12. Support distance

Support distance는 단순 raw vector Euclidean distance가 아니라 **public relational structural region**의 차이를 본다.

비교 요소의 예:

- workflow progress
- known route/profile/object counts
- observed role distributions
- object-related public facts
- latest observed HTTP status

중요한 점은 이 distance가 Critic predicted value를 포함하지 않는다는 것이다.

즉 "비슷한 value를 냈으니 지원된다"는 순환 논리를 피한다.

---

# 13. Support confidence

현재 구현의 support confidence는 가까운 실제 training sample과 sample 수를 함께 반영하는 형태다.

개념적으로:

```math
support \approx e^{-4 d_{nearest}}
\times \frac{N}{N+4}
```

따라서:

- 아주 가까운 sample이 있어도 한두 개뿐이면 confidence가 제한되고,
- sample이 많아도 모두 멀리 있으면 confidence가 낮다.

기본 threshold는 현재 코드에서 보수적으로 사용된다.

---

# 14. 왜 support는 reward가 아닌가?

Local support는 행동을 "좋다/나쁘다"고 평가하지 않는다.

```text
support 높음
!=
좋은 행동
```

뜻은 오직:

```text
이 Critic value estimate를 비교에 사용할 실증적 근거가 충분한가?
```

이다.

따라서 support 역시 value bonus로 더하지 않는다.

---

# 15. Planner에서의 역할

Imagination root evaluation이 다음처럼 나왔다고 하자.

```text
Policy A : V=0.1
Candidate B : V=0.6
```

값만 보면 B로 바꾸고 싶다.

하지만:

```text
support(A)=0.8
support(B)=0.2
```

라면 B의 `0.6`은 OOD extrapolation일 수 있다.

현재 구조는 이 경우 B override를 취소한다.

---

# 16. 2k diagnostic에서 왜 중요했는가?

과거 repaired Imagination은 higher-level unseen state에서 실제 training support가 부족한데도 Critic이 branch 값을 갈라냈고, 그 값으로 86번 행동을 바꿨다.

행동을 바꿀 수 있다는 것 자체는 해결됐지만 많은 변경이 오류로 이어졌다.

이 결과는 다음 구분이 필요함을 보여줬다.

```text
Critic has learned something
!=
Critic is trustworthy here
```

Local support gate는 바로 이 문제를 겨냥한다.

---

# 17. Critic training 안정성

현재 구현은 regression loss로 Smooth L1을 사용하고 gradient clipping을 적용한다.

또 episode suffix들을 replay에 저장하고 batch training을 수행한다.

CUDA current path에서는 여러 branch를 batch scoring하여 Imagination의 많은 Critic 호출을 accelerator-friendly하게 처리한다.

이 최적화는 target semantics를 바꾸는 것이 아니라 실행 비용을 줄이는 목적이다.

---

# 18. 실패 모드

## 18.1 Sparse target starvation

성공/실패 trajectory가 너무 적으면 Critic이 거의 `0` 근처만 학습할 수 있다.

결과: branch 간 value discrimination이 약해지고 Imagination intervention이 0이 될 수 있다.

## 18.2 OOD extrapolation

training frontier 밖에서 큰/작은 value를 근거 없이 출력.

대응: local Critic support fail-closed.

## 18.3 Recurrent contract mismatch

full episode hidden memory로만 학습하고 zero-memory current decision에서 평가하면 mismatch.

대응: every-decision suffix training.

## 18.4 Confidence leakage

Prophecy reliability가 Critic value와 섞임.

대응: confidence-independent Critic input.

---

# 19. 연구 가설

```text
H1. real sparse return만으로 Critic이 branch value를 분리할 수 있는가?
H2. decision suffix training이 arbitrary planning root 평가를 개선하는가?
H3. relational input이 unseen seed transfer에 도움이 되는가?
H4. local support gate가 OOD intervention error를 줄이는가?
H5. gate가 너무 보수적이어서 유효한 intervention까지 모두 막지는 않는가?
```

Critic의 최종 목적은 높은 prediction metric 자체가 아니라 **실제 Imagination decision quality를 높이는 것**이다.

---

# 20. 관련 코드

```text
src/aassr_v2/current_return_critic.py
  - ReturnAwareHardwareRelationalGRUBranchCritic

src/aassr_v2/current_critic_support.py
  - local support replay
  - semantic support distance
  - fail-closed override gate

src/aassr_v2/current_confidence_gate.py
  - confidence-independent Critic encoder
```

---

다음으로 읽기:

- **[Calibration](Calibration)**
- **[Imagination](Imagination)**
- **[Prophecy](Prophecy)**
- **[Experiments](Experiments)**
