# Model-Based RL and World Models

**Model-Based [강화학습(Reinforcement Learning)](Reinforcement-Learning)**은 환경의 [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)를 명시적으로 모델링하고, 그 모델을 이용해 행동 전에 미래를 예측하거나 계획하는 강화학습 계열이다.

AASSR에서는 [Prophecy](Prophecy)가 [학습된(learned)](Neural-Networks-and-Optimization) [세계 모델(world model)](Model-Based-RL-and-World-Models)의 역할을 하고, [Imagination](Imagination)이 그 모델을 이용해 [반사실적 계획(counterfactual planning)](Counterfactual-Planning-and-Search)을 수행한다.

---

# 1. Model-Free와 Model-Based

## Model-Free

환경 [상태 전이(transition)](MDP-and-POMDP) [학습 모델(model)](Terminology-Guide)을 명시적으로 사용하지 않고 [가치(value)](Value-Functions-and-Bellman-Equation)/[정책(policy)](Policy)를 직접 학습한다.

```text
State
 ↓
Q / Policy
 ↓
Action
```

대표 예:

- [Q-러닝(Q-learning)](Q-Learning-DQN-and-TD)
- [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)

AASSR의 기본 [Policy](Policy)는 [환경 예측 모델 없이 직접 학습하는(model-free)](Reinforcement-Learning) [DQN](Q-Learning-DQN-and-TD) 계열이다.

## Model-Based

환경이 어떻게 변하는지 모델을 가지고 미래를 계산한다.

```text
State + Action
      ↓
World Model
      ↓
Predicted Next State
      ↓
Planning
      ↓
Action
```

AASSR의 [Prophecy(미래 예측 모델)](Prophecy) + [Imagination(가상 미래 탐색)](Imagination)이 이 방향이다.

---

# 2. Environment model

가장 기본적인 환경의 상태 변화 규칙 학습 모델은:

```math
\hat P(s'\mid s,a)
```

를 근사한다.

[보상(Reward)](Sparse-Reward-and-Credit-Assignment) 학습 모델도 함께 학습할 수 있다.

```math
\hat R(s,a,s')
```

하지만 AASSR [현재(current)](Current-Status) [Prophecy](Prophecy)의 핵심은 **[공개된(public)](State-Representation) [다음 상태(next-state)](MDP-and-POMDP) [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)**과 [현재 허용된(legal)](Terminology-Guide) [행동(action)](Reinforcement-Learning)/[상태 코드(status)](Terminology-Guide)/[에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) [구조(structure)](Research-Architecture)를 예측하는 데 있다.

외부 [연구 과제(task)](Sparse-Reward-Problem) [보상(reward)](Sparse-Reward-and-Credit-Assignment) 자체를 임의의 학습된 [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment) [학습 신호(signal)](Information-Theory-and-Intrinsic-Motivation)로 바꾸지 않는다.

---

# 3. World Model이란?

넓은 의미에서 세계 모델은 **환경이 어떻게 변하는지에 대한 내부 예측 모델**이다.

가능한 구성:

- 다음 상태 [예측 모델(predictor)](Terminology-Guide)
- [직접 관측되지 않는 잠재 표현(latent)](GRU-and-Sequence-Models) 환경의 상태 변화 규칙 학습 모델
- [관측(observation)](MDP-and-POMDP) 예측 모델
- 보상 예측 모델
- 에피소드 종료 예측 모델

AASSR [Prophecy](Prophecy)는 [관계 기반(relational)](Relational-Representation-and-Generalization) [공개 관측 상태(public state)](State-Representation) 공간에서 다음을 다룬다.

```text
next relational descriptor
latest public HTTP status
legal action mask
terminal class
outcome probability
```

즉 단순한 [수치 벡터(vector)](Neural-Networks-and-Optimization) [회귀 검증(regression)](Ablation-Benchmarking-and-Reproducibility) 하나보다 넓은 상태 전이 학습 모델이다.

---

# 4. 왜 World Model이 도움이 될 수 있나?

Model-free [에이전트(agent)](Reinforcement-Learning)는 실제 경험을 통해 가치를 학습한다.

[세계(World)](Model-Based-RL-and-World-Models) 학습 모델이 있으면 **실제로 행동하지 않고도 가능한 미래를 계산**할 수 있다.

```text
현재 S
 ├→ A를 하면 S_A'
 ├→ B를 하면 S_B'
 └→ C를 하면 S_C'
```

그리고 각 미래에서 다시 행동을 전개할 수 있다.

이것이 lookahead [계획(planning)](Counterfactual-Planning-and-Search)의 핵심이다.

희소 보상에서는 즉시 보상가 모두 0이어도 여러 단계 뒤의 구조 차이를 비교할 수 있다는 기대가 있다.

---

# 5. Planning

세계 학습 모델을 학습하는 것만으로 행동이 자동으로 좋아지지는 않는다.

모델 [출력(output)](Terminology-Guide)을 어떻게 의사결정에 사용할지 계획 algorithm이 필요하다.

```text
World Model
= 미래를 예측

Planner
= 그 예측들을 비교해 현재 행동 결정
```

AASSR에서는:

- [Prophecy](Prophecy) = 학습 모델
- [Imagination](Imagination) = [계획기(planner)](Counterfactual-Planning-and-Search)
- [Critic(미래 가치 평가기)](Critic) = long-horizon evaluator

로 역할을 분리한다.

---

# 6. One-step model과 Multi-step planning

세계 학습 모델은 보통 한 상태 전이을 예측하도록 학습할 수 있다.

```text
(S_t,A_t) → S_{t+1}
```

[계획기(Planner)](Counterfactual-Planning-and-Search)는 이를 반복 호출한다.

```text
S0
 ↓ A0
Ŝ1
 ↓ A1
Ŝ2
 ↓ A2
Ŝ3
```

여기서 `Ŝ`는 [예측된(predicted)](Terminology-Guide) [상태(state)](State-Representation)다.

깊어질수록 실제 상태가 아니라 **예측한 상태 위에서 다시 예측**한다는 점이 중요하다.

---

# 7. Compounding model error

한 단계 [예측(prediction)](Terminology-Guide) [오차(error)](Loss-Functions-and-Class-Imbalance)가 작아도 여러 단계 [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search)하면 누적될 수 있다.

```text
True:      S0 → S1 → S2 → S3
Predicted: S0 → Ŝ1 → Ŝ2 → Ŝ3
               ↑ 작은 오류
                    ↑ 더 커짐
                         ↑ 더 커질 수 있음
```

그래서 계획 [탐색 깊이(depth)](Counterfactual-Planning-and-Search)를 무조건 키우는 것이 좋은 것은 아니다.

AASSR의 [Imagination](Imagination) 탐색 깊이 역시 **longer [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search) vs accumulated 학습 모델 오차**의 [한쪽을 얻으면 다른 쪽을 잃는 상충 관계(trade-off)](Terminology-Guide)를 가진다.

관련 페이지:

- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)
- [Imagination](Imagination)

---

# 8. Model bias

Learned 세계 모델은 실제 환경과 완전히 같지 않다.

그 차이 때문에 계획기가 실제 환경에서는 잘못된 행동을 고를 수 있다.

이를 넓게 **학습 모델 [편향(bias)](Ablation-Benchmarking-and-Reproducibility)**라고 부를 수 있다.

```text
True dynamics P
!=
Learned dynamics P_hat
```

AASSR에서 [Calibration(예측 신뢰도 보정)](Calibration)과 [국소 데이터 근거(local support)](Critic-Support-and-OOD) [판정 관문(gate)](Terminology-Guide)가 필요한 이유 중 하나다.

---

# 9. Model exploitation

계획기는 단순히 학습 모델을 사용하는 것을 넘어 **학습 모델의 오류를 적극적으로 찾아 이용**할 수 있다.

예:

```text
실제로는 나쁜 action
하지만 model이 매우 좋은 outcome을 예측
      ↓
Planner가 그 action을 반복 선택
```

이 현상을 [모델 오류 악용(model exploitation)](Model-Based-RL-and-World-Models)이라고 볼 수 있다.

Optimization이 강할수록 학습 모델의 작은 오류를 찾아낼 수 있다.

AASSR의 [문제를 수정한 뒤의(repaired)](Development-History) [Imagination](Imagination) [진단 실험(diagnostic)](Evidence-Matrix)에서 "계획기가 행동을 바꾸기는 하지만 나쁜 상태 코드로 이어지는" 현상은 이 문제와 밀접하다.

관련 페이지:

- [Calibration](Calibration)
- [Critic, Support and OOD](Critic-Support-and-OOD)

---

# 10. Deterministic model

가장 단순한 세계 모델:

```math
\hat s'=f_\theta(s,a)
```

하나의 [다음(next)](Terminology-Guide) 상태만 출력한다.

환경이 거의 [같은 입력이면 항상 같은 결과인 결정론적(deterministic)](Stochasticity-Uncertainty-and-Probability)이고 상태가 충분히 Markov하면 잘 작동할 수 있다.

하지만 부분 관측이나 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) 환경의 상태 변화 규칙가 있으면 문제가 생길 수 있다.

---

# 11. Stochastic model

여러 가능한 다음 상태의 확률 분포를 모델링한다.

```math
p_\theta(s'\mid s,a)
```

AASSR [Prophecy](Prophecy)는 이 방향이다.

같은 공개된 `(S,A)`에서 여러 [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability)이 가능하므로 단일 평균 상태보다 [여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration) 예측이 필요하다.

관련 페이지:

- [MDP and POMDP](MDP-and-POMDP)
- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

---

# 12. Mean prediction의 위험

두 실제 환경 결과이:

```text
A = [1,0]
B = [0,1]
```

이라고 하자.

MSE 회귀가 평균을 선호하면:

```text
C = [0.5,0.5]
```

를 낼 수 있다.

그런데 `C`가 실제로 존재하지 않는 상태일 수 있다.

특히 [범주형(categorical)](Loss-Functions-and-Class-Imbalance)/[구조 기반(structural)](Relational-Representation-and-Generalization) 상태에서는 이런 평균 상태가 계획기를 심각하게 오도할 수 있다.

AASSR이 [조건부(conditional)](Stochasticity-Uncertainty-and-Probability) [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration)를 사용하는 이유다.

관련 페이지:

- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)

---

# 13. Latent World Model

많은 modern [모델 기반 강화학습(model-based RL)](Model-Based-RL-and-World-Models)은 [가공하지 않은 원본(raw)](State-Representation) 관측을 그대로 예측하지 않고 잠재 표현 [표현(representation)](Relational-Representation-and-Generalization) `z`에서 환경의 상태 변화 규칙를 학습한다.

```text
Observation
 ↓ encoder
Latent z_t
 ↓ dynamics
Latent z_{t+1}
```

장점:

- 고차원 원본 관측 압축
- task-relevant [학습에 사용하는 특징(feature)](Terminology-Guide)에 집중 가능

AASSR 현재 [Prophecy](Prophecy)는 일반적인 pixel 잠재 표현 세계 모델과 달리 **명시적 관계 기반 공개된 [상태를 요약한 표현(descriptor)](State-Representation)**를 중심으로 한다.

Dreamer 계열과 비교할 때 이 차이가 중요하다.

---

# 14. Dreamer와 개념적 비교

Dreamer류 알고리즘은 학습된 잠재 표현 세계 모델 안에서 [모델이 상상한(imagined)](Research-Jargon-Guide) trajectories를 만들고 actor/critic을 학습하는 대표적인 모델 기반 강화학습 계열이다.

AASSR은 "세계 모델 + imagination"이라는 큰 틀에서는 유사한 문제를 다루지만 현재 연구 계약은 다르다.

AASSR의 중요한 차이 중 하나:

```text
Imagined experience
→ planner 계산에 사용
→ current main comparison에서는 real Policy를 직접 재학습시키지 않음
```

또 AASSR은 [명시적인(explicit)](Causality-Leakage-and-Evaluation) 관계 기반 상태, [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ), [Knowledge(에피소드 지식)](Knowledge), [환경의 확률 분기(chance)](Chance-and-Decision-Nodes)/[의사결정(decision)](Chance-and-Decision-Nodes) [의미 규칙(semantics)](State-Representation), 국소 데이터 근거 판정 관문 등을 별도로 둔다.

그래서 [DreamerV3(외부 세계 모델 강화학습 비교군)](Experiments)를 별도 [환경 모델을 사용하는(model-based)](Model-Based-RL-and-World-Models) [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility)으로 비교하는 것이 의미가 있다.

---

# 15. Real-data grounding

세계 학습 모델은 실제 상태 전이으로 학습한다.

```text
real state
real action
real next state
```

AASSR 원칙:

> 상상은 계획에 사용하지만 사실 학습의 기준은 [실제 환경에서 관측된(real)](Research-Jargon-Guide) 상태 전이이다.

Imagined 상태를 다시 세계 모델의 정답으로 학습시키면:

```text
model error
→ imagined data
→ model이 자기 오류를 학습
→ error amplification
```

이 생길 수 있다.

---

# 16. Model uncertainty

세계 학습 모델이 출력 [확률(probability)](Stochasticity-Uncertainty-and-Probability)를 낸다고 해서 모델 자체가 그 예측을 잘 알고 있다는 뜻은 아니다.

예:

```text
model output:
success 0.9
failure 0.1
```

이 상태/행동을 [학습(training)](Terminology-Guide)에서 거의 본 적 없다면 `0.9` 자체를 신뢰하기 어려울 수 있다.

그래서 AASSR은:

```text
outcome probability
!=
prediction reliability
```

를 분리한다.

관련 페이지:

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)
- [Calibration](Calibration)

---

# 17. Calibration

[Calibration](Calibration)은 학습 모델이 **언제 맞고 언제 틀리는지에 대한 [신뢰도(reliability)](Calibration)**를 실제 [검증용 분리 데이터(holdout)](Calibration)으로 점검한다.

AASSR에서는 [예측 신뢰 정도(confidence)](Calibration)를 좋은 미래에 대한 [추가 점수(bonus)](Information-Theory-and-Intrinsic-Motivation)로 쓰지 않는다.

```text
reliability 충분
→ prediction을 planner 비교에 사용 가능

reliability 부족
→ fail closed
```

관련 페이지:

- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)
- [Calibration](Calibration)

---

# 18. Legal action prediction

다음 상태를 맞혀도 그 상태에서 가능한 행동을 틀리면 계획기는 존재하지 않는 행동을 만들 수 있다.

```text
Predicted S'
+
Predicted legal actions
```

AASSR [Prophecy](Prophecy)는 [가능 행동 마스크(legal action mask)](Prophecy)를 [의사결정에 중요한(decision-critical)](Calibration) [대상 또는 학습 목표값(target)](Terminology-Guide)으로 포함한다.

이것은 generic numeric 다음 상태 MSE만 보는 세계 모델과 다른 중요한 설계점이다.

---

# 19. Terminal prediction

계획기는 예측된 [미래(future)](Counterfactual-Planning-and-Search)가:

```text
active
success
true failure
truncation
```

중 무엇인지 알아야 한다.

[에피소드 종료(Terminal)](Replay-Buffer-and-Episode-Boundaries) 의미 규칙를 틀리면 가치 [미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)이 크게 왜곡될 수 있다.

AASSR [Prophecy](Prophecy)는 에피소드 종료 [범주(class)](Loss-Functions-and-Class-Imbalance)를 별도로 예측한다.

---

# 20. Status prediction

AASSR [환경(environment)](Reinforcement-Learning)의 공개된 HTTP-like 상태 코드는 의사결정에 중요한할 수 있다.

예:

```text
200
403
404
429
```

과거 표현에서는 전체 [의미 기준(semantic)](State-Representation) [유사도(similarity)](Critic-Support-and-OOD)가 높아도 상태 코드를 놓쳐 계획기가 실제로 bad 행동을 고를 수 있었다.

그래서 현재 [관계 기반(Relational)](Relational-Representation-and-Generalization) [상태(State)](State-Representation) v3와 [Prophecy](Prophecy)는 [가장 최근의(latest)](Current-Status) 공개된 상태 코드를 명시적으로 보존/예측한다.

관련 페이지:

- [State Representation](State-Representation)
- [Prophecy](Prophecy)

---

# 21. Model-Based RL의 장점

- 실제 행동 전에 미래를 비교 가능
- delayed consequence를 명시적으로 계산 가능
- 학습된 학습 모델을 여러 후보 행동에 재사용 가능
- 계획 탐색 깊이를 조절 가능

---

# 22. Model-Based RL의 위험

- 학습 모델 오차
- compounding 오차
- [학습 분포 밖(OOD)](Critic-Support-and-OOD) 가상 미래 전개
- 모델 오류 악용
- expensive [탐색 트리(tree)](Counterfactual-Planning-and-Search) [탐색(search)](Counterfactual-Planning-and-Search)
- [불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability) misinterpretation

AASSR [현재 세대(current-generation)](Current-Status)의 복잡한 gates는 대부분 이런 위험을 **의미별로 분리**하려는 설계다.

---

# 23. AASSR의 Model-Based stack

```text
Relational public state
        ↓
Prophecy
(stochastic world model)
        ↓
Calibration
(model reliability)
        ↓
Imagination
(counterfactual planning)
        ↓
Critic
(sparse return value)
        ↓
Local support
(value evidence)
        ↓
Override gate
```

여기서:

```text
probability
reliability
value
support
```

는 모두 다르다.

---

# 24. 다음으로 읽기

- [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)
- [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration)
- [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)
- [Prophecy](Prophecy)
- [Imagination](Imagination)

관련 색인: **[Concept Index](Concept-Index)**