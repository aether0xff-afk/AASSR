# Calibration — 예측 신뢰도 보정

[Calibration(예측 신뢰도 보정)](Calibration)은 AASSR에서 **[Prophecy](Prophecy)가 낸 미래 예측을 실제 [planning](Counterfactual-Planning-and-Search)에 얼마나 믿고 써도 되는지** 측정하는 계층이다.

핵심은 [outcome probability와 prediction reliability](Stochasticity-Uncertainty-and-Probability)를 분리하는 것이다.

```text
outcome probability
= 환경에서 그 결과가 나올 확률

prediction reliability
= world model의 그 예측을 믿을 수 있는 정도
```

> [!IMPORTANT]
> 현재 manifest 계약: `semantic-probability-holdout-calibration-v3-status-aware`  
> 관련 구현: `src/aassr_v2/current_semantic_calibration.py`, `current_confidence_gate.py`

---

# 0. 먼저 알아두면 좋은 개념

- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability) — [확률(probability)](Stochasticity-Uncertainty-and-Probability), [신뢰도(reliability)](Calibration), [지식 부족에서 오는 불확실성(epistemic uncertainty)](Stochasticity-Uncertainty-and-Probability), [가치(value)](Value-Functions-and-Bellman-Equation)의 차이
- [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration) — [검증용 분리 데이터(holdout)](Calibration) calibration, ensemble, probability-weighted correctness
- [Model-Based RL & World Models](Model-Based-RL-and-World-Models) — [학습 모델(model)](Terminology-Guide) error와 [모델 오류 악용(model exploitation)](Model-Based-RL-and-World-Models)
- [Critic, Support & OOD](Critic-Support-and-OOD) — [상태 전이(transition)](MDP-and-POMDP) 신뢰도와 가치 [데이터 근거(support)](Critic-Support-and-OOD)가 왜 다른가?
- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance) — [상태 코드(status)](Terminology-Guide) accuracy, [드문(rare)](Loss-Functions-and-Class-Imbalance) class, calibration [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)
- [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation) — [학습을 멈춘(frozen)](Ablation-Benchmarking-and-Reproducibility) 검증용 분리 데이터과 같은-[체크포인트(checkpoint)](Reproduction) 비교

---

# 1. 연구 질문

> **[world model](Model-Based-RL-and-World-Models)의 평균 성능이 높아 보여도 실제 행동 결정에 중요한 오류가 숨어 있을 수 있는데, 어떤 예측을 [계획기(planner)](Counterfactual-Planning-and-Search)가 믿어도 되는지 [실제 환경에서 관측된(real)](Research-Jargon-Guide) 검증용 분리 데이터 경험으로 판단할 수 있는가?**

AASSR의 [Imagination](Imagination)은 학습 모델 error에 직접 노출된다.

따라서:

```text
예측을 낼 수 있는가?
!=
그 예측을 지금 믿어도 되는가?
```

이다.

이 구분은 [epistemic uncertainty](Stochasticity-Uncertainty-and-Probability)의 operational handling과 연결된다.

---

# 2. 왜 단순 confidence 하나로 부족한가?

[Prophecy(미래 예측 모델)](Prophecy)가 다음 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)을 낸다고 하자.

```text
200 : probability 0.7
403 : probability 0.2
429 : probability 0.1
```

이 숫자는 **[환경(environment)](Reinforcement-Learning) [결과 확률(outcome probability)](Stochasticity-Uncertainty-and-Probability) mass**다.

하지만 학습 모델이 이 [상태(state)](State-Representation)/[행동(action)](Reinforcement-Learning) region을 거의 학습하지 않았다면 이 [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability) 자체를 신뢰하기 어렵다.

예:

```text
outcome distribution:
200 0.7 / 403 0.2 / 429 0.1

prediction reliability:
0.2
```

가 동시에 가능하다.

즉:

```text
softmax/mixture probability
!=
model reliability
```

이다.

---

# 3. Holdout calibration

현재 calibration은 [real replay](Replay-Buffer-and-Episode-Boundaries)에서 분리된 검증용 분리 데이터 상태 전이을 사용한다.

개념적으로:

```text
real holdout transitions
(S,A,S')
      |
      v
Prophecy predicts distribution over S'
      |
      v
semantic correctness 측정
      |
      v
state/action region reliability
```

같은 [관계 기반(relational)](Relational-Representation-and-Generalization) 행동 key에 해당하는 충분한 검증용 분리 데이터 sample이 없으면 calibration은 보수적으로 낮게 유지된다.

즉:

```text
데이터 부족
!=
문제 없음
```

이다.

이것이 [fail-closed](Critic-Support-and-OOD) 원칙이다.

---

# 4. 왜 training data가 아니라 holdout인가?

Model이 직접 학습한 sample에서는 지나치게 좋은 correctness를 보일 수 있다.

```text
train data에서 정확
→ unseen query에서도 정확?
```

는 자동으로 성립하지 않는다.

그래서 학습에 직접 사용하지 않은 실제 상태 전이으로 신뢰도를 추정한다.

이는 일반적인 [train/validation/test 분리](Neural-Networks-and-Optimization)와 같은 문제의식이다.

---

# 5. Probability-weighted semantic score

Stochastic [Prophecy](Prophecy)는 여러 환경 결과을 낸다.

실제 다음 상태와의 correctness를 계산할 때 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)를 동일 가중치로 보면 학습 모델이 assign한 분포을 무시하게 된다.

개념적으로:

```math
C(S,A,S')
=
\sum_i p_i\;score(\hat S_i',S')
```

여기서:

- `p_i`: [predicted outcome probability](Stochasticity-Uncertainty-and-Probability)
- `score`: predicted [의미 기반 상태(semantic state)](State-Representation)와 actual next 상태의 일치도

이 방식은:

```text
1% branch만 actual과 정확히 일치
99% branch는 틀림
```

인 학습 모델을 지나치게 높게 평가하는 것을 줄인다.

관련 일반 개념: [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration)

---

# 6. Semantic correctness란?

AASSR 계획기에서 중요한 next-state correctness는 단순 numeric vector distance 하나가 아니다.

Decision-critical channel에는 다음이 포함된다.

- [relational state semantics](Relational-Representation-and-Generalization)
- [legal action mask](Prophecy)
- latest [공개된(public)](State-Representation) HTTP 상태 코드
- [terminal class / episode semantics](Replay-Buffer-and-Episode-Boundaries)

즉:

```text
전체 숫자가 비슷한가?
```

보다:

```text
실제 다음 decision을 바꿀 중요한 구조를 맞혔는가?
```

가 중요하다.

---

# 7. Frozen holdout

Evaluation에서 calibration reference가 계속 바뀌면 같은 체크포인트 비교가 흔들릴 수 있다.

현재 구현은 검증용 분리 데이터을 freeze할 수 있다.

```text
training / validation data 준비
        ↓
calibration holdout freeze
        ↓
frozen reliability 기준으로 OFF/ON evaluation
```

이는 [same-checkpoint comparison](Ablation-Benchmarking-and-Reproducibility)에서 calibration 기준까지 confound가 되는 것을 줄인다.

---

# 8. 왜 status-aware calibration이 필요한가?

과거 2k [진단 실험(diagnostic)](Evidence-Matrix)에서는 전체 [의미 기준(semantic)](State-Representation) [예측(prediction)](Terminology-Guide) quality가 그럴듯해도 [Imagination(가상 미래 탐색)](Imagination) [실제 행동 개입(intervention)](Imagination)이 `403/404/429`로 이어지는 문제가 있었다.

즉:

```text
전체 semantic similarity 높음
!=
decision-critical outcome을 충분히 잘 예측함
```

이다.

이후 [Relational State v3](State-Representation)는 latest 공개된 HTTP 상태 코드를 명시적으로 보존하고, [Prophecy](Prophecy)와 calibration도 이를 중요한 target/평가지표으로 다룬다.

---

# 9. Rare status와 class imbalance

[Critic(미래 가치 평가기)](Critic)al 상태 코드가 data에서 드물면 majority 상태 코드만 잘 맞혀도 전체 accuracy가 높아질 수 있다.

```text
200: 90%
429: 1%
```

이때 항상 200이라고 예측하면 naive accuracy는 높다.

그래서:

- [class-balanced training](Loss-Functions-and-Class-Imbalance)
- per-class accuracy/recall
- [상태 코드까지 고려하는(status-aware)](Calibration) 의미 기준 신뢰도

가 중요하다.

---

# 10. Reliability는 value가 아니다

[현재 세대(current-generation)](Current-Status)에서 매우 중요한 원칙이다.

잘못된 형태:

```text
Critic value 0.4
confidence   0.9
→ 0.4 + 0.9 bonus
```

이렇게 하면 high-confidence 결과 경로가 task [누적 보상(return)](Value-Functions-and-Bellman-Equation)과 무관하게 좋은 행동처럼 보일 수 있다.

올바른 의미 분리:

```text
reliability 충분?
  yes → Critic value 비교 허용
  no  → branch / override 제한
```

즉 신뢰도는 **eligibility [판정 관문(gate)](Terminology-Guide)**이지 [value function](Value-Functions-and-Bellman-Equation)이 아니다.

---

# 11. 왜 Critic에서도 confidence를 제거하는가?

[Critic](Critic) [입력(input)](Terminology-Guide)에 [Prophecy](Prophecy) [예측 신뢰 정도(confidence)](Calibration)가 직접 들어가면 neural [신경망(network)](Neural-Networks-and-Optimization)가 예측 신뢰 정도를 누적 보상 signal처럼 사용할 수 있다.

[현재(current)](Current-Status) 예측 신뢰 정도 판정 관문는 기존 입력 shape를 유지하면서 해당 scalar slot을 상수로 중립화한다.

```text
network shape 유지
+
confidence feature → constant
```

따라서 결과 경로 [후보 순위(ranking)](Policy)은 sparse-누적 보상 [Critic](Critic) 가치로 이루어지고 예측 신뢰 정도는 신뢰도 판정 관문에만 쓰인다.

이것은:

```text
reliability
!=
value
```

를 코드 수준에서도 강제하는 설계다.

---

# 12. Global coverage gate

현재 상태의 legal 행동 surface 전체에서 [Prophecy](Prophecy)가 충분한 [예측 신뢰도(prediction reliability)](Calibration)를 갖는지 먼저 본다.

```text
coverage < threshold
→ Imagination eligible = false
→ Policy action 유지
```

이것은 계획기가 거의 모르는 상태에서 무리하게 전체 행동을 재평가하는 것을 막는다.

[Exploration](Exploration-and-Exploitation) 때문에 새로운 상태에 들어가는 것과, **근거 없는 학습 모델 [기본 행동 덮어쓰기(override)](Imagination)를 허용하는 것**은 다른 문제다.

---

# 13. Per-root reliability gate

Global [데이터가 어느 영역까지 포함하는지(coverage)](Critic-Support-and-OOD)가 충분해도 특정 [탐색의 첫 행동(root)](Imagination) 행동 예측은 unreliable할 수 있다.

그래서 각 탐색의 첫 행동에도 신뢰도를 확인한다.

```text
reliable roots
→ Critic-only ranking 후보

unreliable roots
→ final override 후보에서 제외
```

---

# 14. 왜 Policy branch reliability도 필요한가?

대안 [선택 후보(candidate)](Terminology-Guide)만 reliable하고 [Policy(정책 모델)](Policy) [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)의 예측이 unreliable하면:

```math
advantage=V_{alt}-V_{policy}
```

를 apples-to-apples하게 해석하기 어렵다.

그래서 현재 판정 관문는 [Policy](Policy) 탐색의 첫 행동가 평가되지 않았거나 예측 신뢰도가 낮으면 기본 행동 덮어쓰기를 [fail-closed](Critic-Support-and-OOD) 한다.

---

# 15. Calibration과 local Critic support의 차이

둘 다 "믿을 수 있는가?"를 묻지만 대상이 다르다.

| 질문 | 담당 계층 |
|---|---|
| 이 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) 환경 결과이 일어날 확률은? | [Prophecy](Prophecy) 결과 확률 |
| 이 상태 전이 예측을 믿을 수 있나? | [Calibration](Calibration) |
| predicted future의 sparse 누적 보상은? | [Critic](Critic) |
| 그 [Critic](Critic) 가치를 뒷받침하는 실제 [학습(training)](Terminology-Guide) [증거(evidence)](Evidence-Matrix)가 있나? | Local [가치 평가 데이터 근거(Critic support)](Critic-Support-and-OOD) |

두 판정 관문 중 하나만 통과하면 충분하지 않다.

```text
Prophecy 정확
+
Critic OOD
→ 미래는 맞게 예측했지만 가치 판단이 틀릴 수 있음
```

반대도 가능하다.

관련 페이지: [Critic, Support & OOD](Critic-Support-and-OOD)

---

# 16. Calibration sample 부족

동일 관계 기반 행동 region의 검증용 분리 데이터 sample이 최소 수에 못 미치면 신뢰도를 낮게 두는 보수적 경로를 가진다.

작은 상태 전이 budget에서는 이 때문에 [Imagination](Imagination)이 거의 개입하지 않을 수 있다.

이것은 성능 면에서는 답답할 수 있지만 방법론적으로:

```text
근거 없음
→ 모른다고 말함
```

이라는 명확한 [실패(failure)](Replay-Buffer-and-Episode-Boundaries) [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)다.

반대로 근거가 없는데 높은 예측 신뢰 정도를 주는 것은 [OOD overconfidence](Critic-Support-and-OOD)와 같은 종류의 위험을 만든다.

---

# 17. Cache와 refresh

[Calibration](Calibration)을 매 decision마다 검증용 분리 데이터 전체에 대해 다시 계산하면 비싸다.

현재 구현은:

- 관계 기반 행동 key
- 검증용 분리 데이터 sample count 구간
- [Prophecy](Prophecy) gradient revision 구간

등을 포함한 cache key를 사용해 신뢰도 계산을 재사용한다.

일정 stride를 넘으면 갱신한다.

이것은 **semantics-preserving performance optimization**이며, calibration 정의를 바꾸는 research trick은 아니다.

[GPU batching과 performance optimization](Neural-Networks-and-Optimization)과 같은 engineering 계층과 연구 의미를 구분해야 한다.

---

# 18. Calibration이 직접 해결하지 않는 것

[Calibration](Calibration)은 만능 안전장치가 아니다.

다음을 직접 해결하지 않는다.

- [Policy](Policy) 자체의 [학습 분포 밖(OOD)](Critic-Support-and-OOD) 후보 순위
- [Critic](Critic) 자체의 [OOD](Critic-Support-and-OOD) 가치
- 상태 [표현(representation)](Relational-Representation-and-Generalization)에서 이미 지운 정보
- 아주 희귀한 환경 결과의 data shortage
- 긴 rollout의 [compounding model error](Model-Based-RL-and-World-Models)
- 잘못된 [보상(reward)](Sparse-Reward-and-Credit-Assignment)/[학습 목표(objective)](Terminology-Guide)

그래서 AASSR은 calibration을 다른 구조와 조합한다.

---

# 19. Failure modes

## 19.1 평균 metric blind spot

전체 의미 기준 [평가 점수(score)](Terminology-Guide)는 높지만 중요한 상태 코드/행동 channel이 틀림.

**대응:** 상태 코드까지 고려하는 평가지표 + downstream 실제 행동 개입 audit.

## 19.2 Sparse holdout

특정 행동 region의 검증용 분리 데이터이 부족.

**대응:** [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD) + 실제 상태 전이 데이터 포함 범위 확대.

## 19.3 Confidence as value leakage

Reliability가 가치에 직접 섞임.

**대응:** confidence-independent [Critic](Critic) encoding + gate-only 사용.

## 19.4 Stale reliability

Model이 크게 update됐는데 cache가 너무 오래 유지됨.

**대응:** 학습 모델 revision 기반 refresh.

## 19.5 Over-conservative gate

모든 novel 결과 경로를 막아 [Imagination](Imagination)이 inert해짐.

**대응:** 판정 관문 pass rate와 bad-실제 행동 개입 rate를 함께 보고 [판정 기준값(threshold)](Terminology-Guide)를 [ablation](Ablation-Benchmarking-and-Reproducibility)으로 검증.

---

# 20. Calibration 평가에서 봐야 할 metric

- 검증용 분리 데이터 의미 기준 평가 점수
- 상태 코드까지 고려하는 correctness
- 신뢰도 데이터 포함 범위
- insufficient-evidence rate
- reliable-root fr행동
- calibration refresh/cache diagnostics
- low-reliability suppressed 실제 행동 개입 count
- suppression 후 bad 실제 행동 개입 rate
- suppression 때문에 놓친 successful 선택 후보 여부

마지막 두 개가 특히 중요하다.

```text
개입을 적게 함
!=
좋은 calibration
```

이다.

목표는 **틀린 기본 행동 덮어쓰기를 줄이면서 좋은 기본 행동 덮어쓰기를 남기는 것**이다.

---

# 21. 연구 가설

```text
H1. holdout semantic reliability가 actual prediction correctness와 연결되는가?
H2. status-aware calibration이 decision-critical error를 더 잘 잡는가?
H3. low-reliability roots를 제거하면 intervention error가 감소하는가?
H4. reliability-only 설계가 confidence-value leakage보다 안정적인가?
H5. gate가 너무 보수적이어서 useful intervention까지 막지는 않는가?
H6. 같은 frozen checkpoint에서 calibrated Full이 no-Imagination보다 실제로 좋아지는가?
```

H1~H4는 mechanism-level 질문이고 H6은 downstream task-level 질문이다.

이 계층 구분은 [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)에서 더 자세히 다룬다.

---

# 22. 관련 코드

```text
src/aassr_v2/current_semantic_calibration.py
  - probability_weighted_semantic_score
  - SemanticCalibratedProphecy

src/aassr_v2/current_confidence_gate.py
  - reliability-only decision gate
  - confidence-independent Critic encoding
```

---

# 23. 한 문장 요약

> **[Calibration](Calibration)은 미래가 좋은지를 평가하는 계층이 아니라, [Prophecy](Prophecy)가 말한 미래 자체를 실제 행동 결정에 사용할 만한 empirical 증거가 있는지 판단하는 신뢰도 판정 관문다.**

---

다음으로 읽기:

- **[Prophecy](Prophecy)**
- **[Critic](Critic)**
- **[Imagination](Imagination)**
- **[Critic, Support & OOD](Critic-Support-and-OOD)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
