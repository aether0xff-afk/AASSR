# Critic, Support and Out-of-Distribution

이 페이지는 **Critic**, **function approximation**, **in-distribution / out-of-distribution(OOD)**, **support**, **interpolation / extrapolation**, **fail-closed gating**을 설명한다.

AASSR에서 "Critic이 학습되었다"와 "지금 이 state/action에서 Critic 값을 믿을 수 있다"를 왜 분리하는지 이해하는 핵심 페이지다.

---

# 1. Critic이란?

넓은 강화학습 용어에서 Critic은 state/action/trajectory의 미래 return을 평가하는 value estimator다.

Actor-Critic에서는:

```text
Actor
→ 행동 선택

Critic
→ 그 행동/상태의 value 평가
```

AASSR에서는 [Policy](Policy)가 기본 action을 만들고, 별도의 [Critic](Critic)이 Imagination branch의 sparse-return value를 평가한다.

---

# 2. Function approximation

Critic이 모든 가능한 state/action을 table로 기억하는 대신 neural network를 쓸 수 있다.

```math
\hat V_\theta(x)
```

또는 trajectory input:

```math
\hat V_\theta(x_{0:t})
```

AASSR current Critic은 relational transition sequence를 처리하는 GRU 계열이다.

Function approximation의 장점은 비슷한 input 사이에 generalization할 수 있다는 것이다.

하지만 training data 밖에서도 항상 어떤 숫자를 출력한다는 위험이 있다.

---

# 3. Training distribution

Critic은 실제 training samples가 분포하는 영역에서 학습된다.

```text
많이 본 region
██████████
██████████
```

이 영역 근처에서는 interpolation이 가능하다.

---

# 4. Interpolation

Training sample 사이의 비슷한 region에서 값을 추정하는 것이다.

```text
train A ---- query ---- train B
```

일반적으로 neural network가 가장 믿을 만한 상황은 충분한 representative data가 있는 region이다.

---

# 5. Extrapolation

Training data support 바깥의 input에 대해 값을 추정하는 것이다.

```text
train region █████
                  query X
```

Neural network는 `X`에서도 숫자를 출력한다.

하지만 그 숫자가 실제 return과 맞다는 보장은 없다.

이것이 OOD value extrapolation 문제다.

---

# 6. OOD: Out-of-Distribution

**Out-of-Distribution**은 현재 input이 training distribution과 의미 있게 다른 경우를 말한다.

AASSR 예:

```text
training에서 Level 0/1 state를 주로 봄
평가에서 더 높은 workflow region 등장
```

또는:

```text
새 relational combination
새 status/action pattern
```

이런 영역에서 Critic은 근거 없는 high value를 낼 수 있다.

---

# 7. 왜 Planner가 OOD 문제를 악화시킬 수 있는가?

Planner는 많은 counterfactual state를 생성하고 그중 높은 value를 찾는다.

```text
Predicted state 1 → value 0.1
Predicted state 2 → value 0.2
Predicted state 3 → value 2.7  ← OOD artifact
```

Optimization은 우연히 큰 오류를 적극적으로 선택할 수 있다.

즉 단순 evaluation보다 **search + function approximation** 조합이 extrapolation error를 더 강하게 exploit할 수 있다.

관련 페이지:

- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)

---

# 8. Global readiness

Critic이 충분한 gradient update를 했거나 training sample 수가 일정 기준을 넘으면:

```text
critic_ready = true
```

같은 global flag를 만들 수 있다.

하지만 이것은:

> Critic이 전체적으로 어느 정도 학습되었다.

는 뜻이지:

> 현재 query state/action이 training support 안에 있다.

는 뜻은 아니다.

---

# 9. Local support

Local support는 현재 query 주변에 **실제 training evidence가 얼마나 있는가**를 묻는다.

```text
Query state/action
      ↓
비슷한 real Critic training samples가 존재?
      ↓
충분하면 supported
```

AASSR current-generation은 Imagination override 전에 Policy root와 candidate root의 local support를 확인한다.

---

# 10. Support는 value가 아니다

다시 중요한 구분:

```text
support 높음
!=
좋은 action
```

Support가 높다는 것은:

> 이 Critic prediction이 training data와 가까운 영역에서 나온다.

라는 evidence다.

실패 state도 training sample이 많으면 support는 높을 수 있다.

---

# 11. Nearest-neighbor intuition

가장 단순한 local support 방법은 현재 input과 가까운 training sample을 찾는 것이다.

```math
d_{nearest}=\min_i d(x,x_i)
```

거리가 작으면 더 familiar한 region이라고 볼 수 있다.

하지만 거리 metric 자체가 의미 있어야 한다.

Raw high-dimensional Euclidean distance가 항상 좋은 것은 아니다.

AASSR current support는 relational/public structural features를 중심으로 semantic distance를 구성한다.

---

# 12. Sample count

가까운 sample 하나만 있다고 충분한 것은 아닐 수 있다.

```text
Case A:
아주 가까운 sample 1개

Case B:
가까운 sample 50개
```

B가 더 강한 empirical support를 제공할 수 있다.

AASSR support confidence는 nearest distance와 sample count를 함께 반영하는 형태다.

---

# 13. Density estimation과의 관계

Local support는 넓게 보면 training density/support estimation 문제와 연결된다.

가능한 방법:

- k-nearest neighbors
- kernel density estimation
- learned density model
- distance in latent space
- ensemble uncertainty

AASSR current 구현은 복잡한 density model보다 auditable한 real-training neighborhood evidence를 사용하는 방향이다.

---

# 14. Support와 Epistemic uncertainty

Training support가 부족하면 epistemic uncertainty가 높을 가능성이 있다.

```text
많이 본 region
→ epistemic uncertainty 낮을 가능성

거의 안 본 region
→ epistemic uncertainty 높을 가능성
```

하지만 support와 epistemic uncertainty가 수학적으로 동일한 것은 아니다.

Support는 **경험 데이터의 존재 여부에 기반한 operational proxy/gate**다.

---

# 15. World-model OOD와 Critic OOD는 다르다

두 model이 따로 틀릴 수 있다.

```text
Prophecy OOD
→ future state prediction이 틀림

Critic OOD
→ state는 맞게 예측했지만 value가 틀림
```

따라서 AASSR은:

```text
Prophecy Calibration
+
Critic local support
```

를 별도로 둔다.

---

# 16. Calibration과 Support 비교

| 질문 | 담당 |
|---|---|
| 이 transition prediction을 믿을 수 있나? | Calibration |
| 이 value prediction을 믿을 real data가 있나? | Critic Support |
| 이 outcome이 일어날 확률은? | Prophecy probability |
| 그 outcome의 task return은? | Critic value |

이 네 값은 다른 의미다.

---

# 17. Fail-closed

Evidence가 충분하지 않을 때 공격적으로 새 action을 실행하지 않고 기본 Policy를 유지하는 방식을 **fail-closed**라고 표현한다.

```text
candidate value 높음
BUT support 부족
→ override 금지
→ Policy 유지
```

장점:

- 근거 없는 extrapolation으로 행동을 바꾸는 위험 감소

단점:

- 너무 보수적이면 실제로 좋은 novel action도 못 선택함

따라서 threshold 자체가 hyperparameter/ablation 대상이다.

---

# 18. Fail-open과 비교

Fail-open:

```text
근거 부족
→ 그래도 model output 사용
```

Fail-closed:

```text
근거 부족
→ 안전한 baseline/fallback 유지
```

AASSR current Imagination은 planning override에 대해 후자에 가깝다.

---

# 19. Conservative RL과의 개념적 연결

Offline RL/Conservative RL에서도 dataset support 밖의 action value overestimation이 큰 문제다.

그래서 data support 밖 action을 보수적으로 다루는 다양한 방법이 연구되어 왔다.

AASSR current local support gate가 특정 conservative offline RL 알고리즘과 동일하다는 뜻은 아니지만, **OOD action-value extrapolation을 경계한다는 문제의식**은 연결된다.

---

# 20. Critic value clipping만으로 충분한가?

Value를 `[-1,1]`로 clamp해도 OOD ranking 오류는 남을 수 있다.

예:

```text
Policy real value ≈ 0.1
OOD candidate가 근거 없이 0.9
```

둘 다 범위 안이다.

따라서 output range만 제한하는 것과 empirical support를 확인하는 것은 다르다.

---

# 21. Uncertainty penalty만으로 충분한가?

Planner value에:

```math
V'=V-\lambda U
```

를 넣을 수도 있다.

하지만 uncertainty estimator 자체가 불안정할 수 있고, task value와 reliability 의미가 섞인다.

AASSR current design은:

```text
reliability/support gate
→ 통과한 branch들끼리 Critic value 비교
```

하는 분리를 선호한다.

---

# 22. 2k diagnostic과 연결

AASSR의 과거 repaired Imagination diagnostic에서:

```text
Critic은 global ready
Imagination은 행동을 실제로 변경
하지만 higher-level region의 real Critic support 부족
```

이라는 문제가 관찰됐다.

이후 local support gate가 들어간 이유다.

핵심 교훈:

```text
"모델이 학습됨"
!=
"이 query에서 모델을 믿어도 됨"
```

---

# 23. Support의 trade-off

Threshold가 너무 낮으면:

```text
OOD bad override 통과
```

Threshold가 너무 높으면:

```text
useful novel override도 차단
```

따라서 최종 평가에서는:

- gate pass rate
- suppressed intervention count
- intervention error rate
- success-producing intervention

을 함께 봐야 한다.

---

# 24. Support와 generalization의 긴장

Generalization은 training sample과 완전히 같은 input이 아니어도 learned structure를 적용하는 능력이다.

Support gate가 너무 strict하면 generalization 자체를 막을 수 있다.

따라서 좋은 support metric은:

```text
exact memorization만 허용 X
구조적으로 관련된 region은 허용 O
근거 없는 extrapolation은 차단 O
```

을 목표로 해야 한다.

AASSR에서 relational distance를 사용하는 이유도 여기에 있다.

---

# 25. 다음으로 읽기

- [Critic](Critic)
- [GRU and Sequence Models](GRU-and-Sequence-Models)
- [Calibration](Calibration)
- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [Imagination](Imagination)

관련 색인: **[Concept Index](Concept-Index)**