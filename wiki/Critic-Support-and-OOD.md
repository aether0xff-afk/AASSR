# Critic, Support and Out-of-Distribution

이 페이지는 **[Critic(미래 가치 평가기)](Critic)**, **function approximation**, **in-distribution / out-of-distribution([학습 분포 밖(OOD)](Critic-Support-and-OOD))**, **[데이터 근거(support)](Critic-Support-and-OOD)**, **interpolation / extrapolation**, **[근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) [조건부 통과 판단(gating)](Terminology-Guide)**을 설명한다.

AASSR에서 "[Critic](Critic)이 학습되었다"와 "지금 이 [상태(state)](State-Representation)/[행동(action)](Reinforcement-Learning)에서 [Critic](Critic) 값을 믿을 수 있다"를 왜 분리하는지 이해하는 핵심 페이지다.

---

# 1. Critic이란?

넓은 강화학습 용어에서 [Critic](Critic)은 상태/행동/trajectory의 미래 [누적 보상(return)](Value-Functions-and-Bellman-Equation)을 평가하는 [가치(value)](Value-Functions-and-Bellman-Equation) [값을 추정하는 모델(estimator)](Terminology-Guide)다.

Actor-[Critic](Critic)에서는:

```text
Actor
→ 행동 선택

Critic
→ 그 행동/상태의 value 평가
```

AASSR에서는 [Policy](Policy)가 기본 행동을 만들고, 별도의 [Critic](Critic)이 [Imagination(가상 미래 탐색)](Imagination) [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)의 sparse-누적 보상 가치를 평가한다.

---

# 2. Function approximation

[Critic](Critic)이 모든 가능한 상태/행동을 table로 기억하는 대신 neural [신경망(network)](Neural-Networks-and-Optimization)를 쓸 수 있다.

```math
\hat V_\theta(x)
```

또는 trajectory [입력(input)](Terminology-Guide):

```math
\hat V_\theta(x_{0:t})
```

AASSR [현재(current)](Current-Status) [Critic](Critic)은 [관계 기반(relational)](Relational-Representation-and-Generalization) [상태 전이(transition)](MDP-and-POMDP) sequence를 처리하는 [GRU(게이트 순환 유닛)](GRU-and-Sequence-Models) 계열이다.

Function approximation의 장점은 비슷한 입력 사이에 [일반화(generalization)](Relational-Representation-and-Generalization)할 수 있다는 것이다.

하지만 [학습 데이터(training data)](Terminology-Guide) 밖에서도 항상 어떤 숫자를 출력한다는 위험이 있다.

---

# 3. Training distribution

[Critic](Critic)은 실제 [학습(training)](Terminology-Guide) samples가 분포하는 영역에서 학습된다.

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

일반적으로 neural 신경망가 가장 믿을 만한 상황은 충분한 representative data가 있는 region이다.

---

# 5. Extrapolation

Training data 데이터 근거 바깥의 입력에 대해 값을 추정하는 것이다.

```text
train region █████
                  query X
```

Neural 신경망는 `X`에서도 숫자를 출력한다.

하지만 그 숫자가 실제 누적 보상과 맞다는 보장은 없다.

이것이 [OOD](Critic-Support-and-OOD) 가치 extrapolation 문제다.

---

# 6. OOD: Out-of-Distribution

**Out-of-Distribution**은 현재 입력이 학습 [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)과 의미 있게 다른 경우를 말한다.

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

이런 영역에서 [Critic](Critic)은 근거 없는 high 가치를 낼 수 있다.

---

# 7. 왜 Planner가 OOD 문제를 악화시킬 수 있는가?

Planner는 많은 counterfactual 상태를 생성하고 그중 높은 가치를 찾는다.

```text
Predicted state 1 → value 0.1
Predicted state 2 → value 0.2
Predicted state 3 → value 2.7  ← OOD artifact
```

Optimization은 우연히 큰 오류를 적극적으로 선택할 수 있다.

즉 단순 [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility)보다 **search + function approximation** 조합이 extrapolation error를 더 강하게 exploit할 수 있다.

관련 페이지:

- [Model-Based RL and World Models](Model-Based-RL-and-World-Models)
- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)

---

# 8. Global readiness

[Critic](Critic)이 충분한 gradient update를 했거나 학습 sample 수가 일정 기준을 넘으면:

```text
critic_ready = true
```

같은 global flag를 만들 수 있다.

하지만 이것은:

> [Critic](Critic)이 전체적으로 어느 정도 학습되었다.

는 뜻이지:

> 현재 query 상태/행동이 학습 데이터 근거 안에 있다.

는 뜻은 아니다.

---

# 9. Local support

[국소 데이터 근거(Local support)](Critic-Support-and-OOD)는 현재 query 주변에 **실제 학습 [증거(evidence)](Evidence-Matrix)가 얼마나 있는가**를 묻는다.

```text
Query state/action
      ↓
비슷한 real Critic training samples가 존재?
      ↓
충분하면 supported
```

AASSR [현재 세대(current-generation)](Current-Status)은 [Imagination](Imagination) [기본 행동 덮어쓰기(override)](Imagination) 전에 [Policy(정책 모델)](Policy) [탐색의 첫 행동(root)](Imagination)와 [선택 후보(candidate)](Terminology-Guide) 탐색의 첫 행동의 [국소 데이터 근거(local support)](Critic-Support-and-OOD)를 확인한다.

---

# 10. Support는 value가 아니다

다시 중요한 구분:

```text
support 높음
!=
좋은 action
```

Support가 높다는 것은:

> 이 [Critic](Critic) [예측(prediction)](Terminology-Guide)이 학습 데이터와 가까운 영역에서 나온다.

라는 증거다.

실패 상태도 학습 sample이 많으면 데이터 근거는 높을 수 있다.

---

# 11. Nearest-neighbor intuition

가장 단순한 국소 데이터 근거 방법은 현재 입력과 가까운 학습 sample을 찾는 것이다.

```math
d_{nearest}=\min_i d(x,x_i)
```

거리가 작으면 더 familiar한 region이라고 볼 수 있다.

하지만 거리 [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility) 자체가 의미 있어야 한다.

Raw high-dimensional Euclidean distance가 항상 좋은 것은 아니다.

AASSR 현재 데이터 근거는 관계 기반/[공개된(public)](State-Representation) [구조 기반(structural)](Relational-Representation-and-Generalization) [학습에 사용하는 특징(features)](Terminology-Guide)를 중심으로 [의미 기준(semantic)](State-Representation) distance를 구성한다.

---

# 12. Sample count

가까운 sample 하나만 있다고 충분한 것은 아닐 수 있다.

```text
Case A:
아주 가까운 sample 1개

Case B:
가까운 sample 50개
```

B가 더 강한 empirical 데이터 근거를 제공할 수 있다.

AASSR 데이터 근거 [예측 신뢰 정도(confidence)](Calibration)는 nearest distance와 sample count를 함께 반영하는 형태다.

---

# 13. Density estimation과의 관계

국소 데이터 근거는 넓게 보면 학습 density/데이터 근거 estimation 문제와 연결된다.

가능한 방법:

- k-nearest neighbors
- kernel density estimation
- learned density [학습 모델(model)](Terminology-Guide)
- distance in latent space
- ensemble uncertainty

AASSR 현재 구현은 복잡한 density 학습 모델보다 auditable한 real-training neighborhood 증거를 사용하는 방향이다.

---

# 14. Support와 Epistemic uncertainty

Training 데이터 근거가 부족하면 [지식 부족에서 오는 불확실성(epistemic uncertainty)](Stochasticity-Uncertainty-and-Probability)가 높을 가능성이 있다.

```text
많이 본 region
→ epistemic uncertainty 낮을 가능성

거의 안 본 region
→ epistemic uncertainty 높을 가능성
```

하지만 데이터 근거와 지식 부족에서 오는 불확실성가 수학적으로 동일한 것은 아니다.

Support는 **경험 데이터의 존재 여부에 기반한 operational [대리 지표(proxy)](Ablation-Benchmarking-and-Reproducibility)/[판정 관문(gate)](Terminology-Guide)**다.

---

# 15. World-model OOD와 Critic OOD는 다르다

두 학습 모델이 따로 틀릴 수 있다.

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
| 이 상태 전이 예측을 믿을 수 있나? | [Calibration(예측 신뢰도 보정)](Calibration) |
| 이 가치 예측을 믿을 [실제 데이터(real data)](Causality-Leakage-and-Evaluation)가 있나? | [가치 평가 데이터 근거(Critic Support)](Critic-Support-and-OOD) |
| 이 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)이 일어날 확률은? | [Prophecy(미래 예측 모델)](Prophecy) [확률(probability)](Stochasticity-Uncertainty-and-Probability) |
| 그 환경 결과의 task 누적 보상은? | [Critic](Critic) 가치 |

이 네 값은 다른 의미다.

---

# 17. Fail-closed

Evidence가 충분하지 않을 때 공격적으로 새 행동을 실행하지 않고 기본 [Policy](Policy)를 유지하는 방식을 **근거가 부족하면 보수적으로 거부하는**라고 표현한다.

```text
candidate value 높음
BUT support 부족
→ override 금지
→ Policy 유지
```

장점:

- 근거 없는 extrapolation으로 행동을 바꾸는 위험 감소

단점:

- 너무 보수적이면 실제로 좋은 novel 행동도 못 선택함

따라서 [판정 기준값(threshold)](Terminology-Guide) 자체가 hyperparameter/[구성요소 제거 비교(ablation)](Ablation-Benchmarking-and-Reproducibility) 대상이다.

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

AASSR 현재 [Imagination](Imagination)은 [계획(planning)](Counterfactual-Planning-and-Search) 기본 행동 덮어쓰기에 대해 후자에 가깝다.

---

# 19. Conservative RL과의 개념적 연결

Offline RL/Conservative RL에서도 dataset 데이터 근거 밖의 행동 가치 overestimation이 큰 문제다.

그래서 data 데이터 근거 밖 행동을 보수적으로 다루는 다양한 방법이 연구되어 왔다.

AASSR 현재 국소 데이터 근거 판정 관문가 특정 conservative offline RL 알고리즘과 동일하다는 뜻은 아니지만, **[OOD](Critic-Support-and-OOD) 행동-value extrapolation을 경계한다는 문제의식**은 연결된다.

---

# 20. Critic value clipping만으로 충분한가?

Value를 `[-1,1]`로 clamp해도 [OOD](Critic-Support-and-OOD) [후보 순위(ranking)](Policy) 오류는 남을 수 있다.

예:

```text
Policy real value ≈ 0.1
OOD candidate가 근거 없이 0.9
```

둘 다 범위 안이다.

따라서 [출력(output)](Terminology-Guide) range만 제한하는 것과 empirical 데이터 근거를 확인하는 것은 다르다.

---

# 21. Uncertainty penalty만으로 충분한가?

Planner 가치에:

```math
V'=V-\lambda U
```

를 넣을 수도 있다.

하지만 uncertainty 값을 추정하는 모델 자체가 불안정할 수 있고, task 가치와 [신뢰도(reliability)](Calibration) 의미가 섞인다.

AASSR 현재 design은:

```text
reliability/support gate
→ 통과한 branch들끼리 Critic value 비교
```

하는 분리를 선호한다.

---

# 22. 2k diagnostic과 연결

AASSR의 과거 repaired [Imagination](Imagination) [진단 실험(diagnostic)](Evidence-Matrix)에서:

```text
Critic은 global ready
Imagination은 행동을 실제로 변경
하지만 higher-level region의 real Critic support 부족
```

이라는 문제가 관찰됐다.

이후 국소 데이터 근거 판정 관문가 들어간 이유다.

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

- 판정 관문 pass rate
- suppressed [실제 행동 개입(intervention)](Imagination) count
- 실제 행동 개입 error rate
- success-producing 실제 행동 개입

을 함께 봐야 한다.

---

# 24. Support와 generalization의 긴장

[일반화(Generalization)](Relational-Representation-and-Generalization)은 학습 sample과 완전히 같은 입력이 아니어도 learned structure를 적용하는 능력이다.

Support 판정 관문가 너무 strict하면 일반화 자체를 막을 수 있다.

따라서 좋은 데이터 근거 평가지표은:

```text
exact memorization만 허용 X
구조적으로 관련된 region은 허용 O
근거 없는 extrapolation은 차단 O
```

을 목표로 해야 한다.

AASSR에서 관계 기반 distance를 사용하는 이유도 여기에 있다.

---

# 25. 다음으로 읽기

- [Critic](Critic)
- [GRU and Sequence Models](GRU-and-Sequence-Models)
- [Calibration](Calibration)
- [Relational Representation and Generalization](Relational-Representation-and-Generalization)
- [Imagination](Imagination)

관련 색인: **[Concept Index](Concept-Index)**