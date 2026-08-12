# Glossary

AASSR 위키에서 반복해서 등장하는 용어를 **짧은 정의 + 관련 개념 링크** 형태로 정리한다.

> [!TIP]
> 여기서는 빠른 뜻만 확인한다. 개념을 처음부터 공부하려면 [Concept Index](Concept-Index), AASSR에서 실제로 어떻게 쓰는지 보려면 각 Core Mechanism 페이지로 이동한다.

---

## A

### [Ablation](Ablation-Benchmarking-and-Reproducibility)
전체 시스템에서 특정 구성요소만 제거하거나 바꿔 그 요소의 causal contribution을 분리하려는 실험.

### Action (`A`)
Agent가 environment에 실제로 실행하거나 planner 안에서 counterfactual candidate로 평가하는 행동. AASSR에서는 concrete execution identity와 [relational action representation](Relational-Representation-and-Generalization)을 구분한다.

### Action surface
현재 observation에서 agent가 실제로 선택할 수 있는 legal concrete action들의 집합. [State Representation](State-Representation)과 [Prophecy](Prophecy)의 legal-action mask와 연결된다.

### Advantage
한 행동의 value가 기준 행동보다 얼마나 더 좋은지를 나타내는 차이. AASSR의 planner override에서는 Policy root 대비 fixed margin을 넘는지 확인한다. 관련: [Value Functions & Bellman](Value-Functions-and-Bellman-Equation).

### Agent
Environment와 상호작용하며 action을 선택하는 주체. AASSR에서는 Policy 하나가 아니라 State/ASEQ/Knowledge/Prophecy/Calibration/Critic/Imagination 등을 합친 전체 decision system이 agent다. 관련: [Reinforcement Learning](Reinforcement-Learning).

### ASEQ
AASSR에서 실제로 관측한 transition `(S, A, S')`. 특히 semantic `S → A → S` self-loop evidence를 관리하는 데 사용한다. 자세히: [ASEQ](ASEQ).

---

## B

### [Baseline](Ablation-Benchmarking-and-Reproducibility)
새 방법과 같은 조건에서 비교하기 위한 기준 모델 또는 규칙. AASSR의 예: `dqn_raw`, `dqn_relational`, `dreamerv3_relational`.

### [Bellman Equation](Value-Functions-and-Bellman-Equation)
현재 value와 다음 state/action의 value를 연결하는 강화학습의 핵심 재귀식.

### Bootstrap
현재 target을 계산할 때 다음 state의 추정 value를 사용하는 것. [TD Learning](Q-Learning-DQN-and-TD)과 연결된다.

### Bootstrap boundary
다음 state value를 TD target에 이어 붙이면 안 되는 경계. reward `0`과 bootstrap continuation은 같은 개념이 아니다. 자세히: [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries).

---

## C

### [Calibration](Calibration)
Model confidence/reliability가 실제 holdout prediction quality와 맞는지 평가하고 decision gate에 사용할 수 있게 만드는 과정. 일반 개념: [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration).

### Chance node
Action을 이미 고른 뒤 environment의 stochastic outcome이 갈리는 노드. Agent가 outcome을 고를 수 없으므로 probability-weighted expectation을 사용한다. 자세히: [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes).

### Chance backup

```math
V_{chance}=\sum_i p_iV_i
```

환경 outcome별 value를 outcome probability로 평균하는 backup.

### Concrete semantic identity
한 episode 안에서 서로 다른 실제 entity와 exact repetition을 구분하기 위한 task-relevant identity. [ASEQ](ASEQ), actual execution에 사용된다. [Relational identity](Relational-Representation-and-Generalization)와 목적이 다르다.

### Confidence
문맥에 따라 모호하므로 AASSR에서는 가능하면 **prediction reliability**라는 더 구체적인 말을 사용한다. Confidence를 task value와 섞지 않는다.

### [Counterfactual Planning](Counterfactual-Planning-and-Search)
실제로 행동하기 전에 “이 행동을 했다면?”이라는 여러 future를 model로 계산하는 planning.

### Credit Assignment
최종 reward의 책임/공로를 과거 state와 action에 어떻게 전달할지의 문제. [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment).

### Critic
Imagined future의 discounted external sparse return을 추정하는 value model. Current AASSR은 relational GRU sparse-return Critic을 사용한다. 자세히: [Critic](Critic).

### Critic readiness
Critic이 overall training을 충분히 받아 사용할 최소 조건을 만족하는지 나타내는 global condition.

### Critic local support
현재 state/action의 value estimate 주변에 실제 Critic training data가 충분히 존재하는지 나타내는 local evidence. 자세히: [Critic, Support & OOD](Critic-Support-and-OOD).

### Curriculum
학습 exposure를 쉬운 문제에서 어려운 문제로 조절하는 방법. 정답 action을 직접 주는 guided trajectory와는 다르다. 자세히: [Curriculum Learning](Curriculum-Learning).

---

## D

### Decision node
Predicted state에서 agent가 다음 action을 고르는 노드. Optimal continuation을 가정하면 action value 중 `max`를 사용한다. 자세히: [Chance Nodes & Decision Nodes](Chance-and-Decision-Nodes).

### Decision backup

```math
V_{decision}=\max_a V(S',a)
```

Agent가 실제로 선택 가능한 future action 중 최대 value를 취하는 backup.

### Dense Reward
목표까지 가는 중간 과정에서도 reward가 자주 제공되는 환경. [Sparse Reward](Sparse-Reward-and-Credit-Assignment)의 반대편 개념.

### [DQN](Q-Learning-DQN-and-TD)
Deep Q-Network. Q-learning의 action-value function을 neural network로 근사하는 value-based model-free RL 방법.

### Distribution Shift
Training data와 evaluation/imagined data의 분포가 달라지는 현상. [OOD](Critic-Support-and-OOD)와 연결된다.

### DreamerV3
Learned world model과 latent imagination을 사용하는 외부 model-based RL baseline. AASSR에서는 current relational interface에 pinned official upstream을 adapter로 연결하는 비교를 목표로 한다. 관련: [Experiments](Experiments).

---

## E

### Ensemble
여러 learned model의 prediction을 함께 사용해 model uncertainty 또는 robustness evidence를 얻는 구조. 환경 outcome 자체의 multimodality를 나타내는 **mixture**와 다르다. [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration).

### Episode
Agent가 reset부터 success/failure/truncation 등 boundary까지 경험하는 trajectory 구간. [Reinforcement Learning](Reinforcement-Learning).

### Epistemic Uncertainty
데이터 부족이나 model knowledge 부족에서 오는 uncertainty. 더 많은 적절한 data로 줄어들 가능성이 있다. [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability).

### Evidence Matrix
각 [Research Question](Research-Questions)을 가설, 변수, metric, 현재 evidence, claim boundary로 연결한 페이지. [Evidence Matrix](Evidence-Matrix).

### Expected Return
미래 reward의 discounted cumulative sum의 기대값. [Value Functions & Bellman](Value-Functions-and-Bellman-Equation).

### Exploration
아직 충분히 모르는 state/action을 시도해 새로운 evidence를 얻는 행동. [Exploration & Exploitation](Exploration-and-Exploitation).

### Exploitation
현재까지 학습한 value 기준으로 가장 좋아 보이는 행동을 선택하는 것. [Exploration & Exploitation](Exploration-and-Exploitation).

### External Reward
Environment가 task objective로 제공하는 reward. AASSR pentest 계열 current contract는 success `+1`, true failure `-1`, 그 외 대부분 `0`이다.

---

## F

### Fail closed
필요한 신뢰 조건이 충족되지 않으면 공격적으로 planner override하지 않고 기존 Policy로 fallback하는 설계. Current local Critic support gate가 이 원칙을 사용한다.

### Final Blind
Protocol과 hyperparameter를 고정한 뒤에만 사용하는 미소비 final evaluation seed set. [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility).

---

## G

### Generalization
Training에서 직접 보지 않은 새로운 state/task에서 학습한 구조가 유효하게 작동하는 성질. [Relational Representation & Generalization](Relational-Representation-and-Generalization).

### GRU
Gated Recurrent Unit. Recurrent hidden state를 gate로 업데이트하는 sequence model. Current Critic과 연결된다. [GRU & Sequence Models](GRU-and-Sequence-Models).

### Guided Trajectory
사람 또는 oracle이 성공 action path를 학습 과정에 직접 제공하는 방식. AASSR primary sparse-reward claim에서는 사용하지 않는다.

---

## H

### Hidden Leakage
Agent가 실제 interaction 시점에 알 수 없어야 하는 simulator truth, future outcome, 정답 identity 등이 input에 들어가는 문제. [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation).

### Holdout
Model parameter update와 분리해 calibration/validation을 위해 보관한 real transition set.

### Horizon
Planner가 몇 단계 future까지 전개하는지의 깊이. 깊을수록 long-term consequence를 보지만 model error와 compute가 증가한다. [Counterfactual Planning & Search](Counterfactual-Planning-and-Search).

---

## I

### Imagination
[Prophecy](Prophecy)의 stochastic future를 여러 단계 이어 붙여 실제 행동 전에 root actions를 비교하는 AASSR planner. 현재 primary protocol에서는 factual persistent learning이 아니라 planning에 사용한다. [Imagination](Imagination).

### Imagined Transition
World model이 만들어낸 counterfactual transition. **Real transition과 같은 factual evidence로 자동 취급하지 않는다.**

### Information Gain
어떤 observation/action이 agent uncertainty를 얼마나 줄였는지 나타내는 일반 개념. [Information Theory & Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation).

### Information-value Residual
AASSR Policy에서 task Q-value와 분리해 관리하는 internal information-oriented action value term. External task reward와 동일하지 않다. [Policy](Policy).

### Intervention
Imagination이 최종적으로 Policy가 고른 concrete root action을 다른 action으로 바꾼 실제 개입.

### Intervention Margin
Planner candidate가 Policy root보다 넘어야 하는 최소 fixed value advantage.

### Intrinsic Motivation
Environment external reward와 별개로 novelty, curiosity, information 등을 이용해 exploration을 유도하는 내부 신호 계열. [Information Theory & Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation).

---

## K

### Knowledge
현재 episode에서 real response를 통해 이미 획득한 사실과 causal context/provenance를 보존하는 memory. Future response를 과거 prediction에 넣지 않는다. [Knowledge](Knowledge).

---

## L

### Legal-action Mask
현재 또는 predicted state에서 어떤 structural action slot이 legal한지 표시하는 categorical/binary mask. Prophecy가 next mask를 예측한다.

### Local Support
현재 prediction/value가 training evidence 주변에서 나오는지 보는 local data-support 개념. AASSR에서는 특히 [Critic Support](Critic-Support-and-OOD)를 의미한다.

### Loss
Neural network parameter를 업데이트하기 위해 최소화하는 training objective. Environment reward와는 다르다. [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance).

---

## M

### MDP
Markov Decision Process. State, action, transition, reward를 이용해 sequential decision problem을 표현하는 기본 framework. [MDP and POMDP](MDP-and-POMDP).

### Mixture Model
하나의 input에서도 여러 실제 outcome mode가 가능할 때 각 mode와 probability mass를 유지하는 model. [Mixture, Ensemble & Calibration](Mixture-Ensemble-and-Calibration).

### Model-based RL
Environment transition/reward dynamics model을 학습하거나 이용해 planning하는 강화학습 계열. [Model-Based RL & World Models](Model-Based-RL-and-World-Models).

### Model-free RL
Explicit future transition model을 사용하지 않고 Policy/Value를 직접 학습하는 강화학습 계열. AASSR의 DQN Policy는 이 기반을 가진다. [Reinforcement Learning](Reinforcement-Learning).

### Model Exploitation
Planner가 world model의 prediction error를 실제로 좋은 future인 것처럼 찾아내 최적화하는 실패 모드. [Model-Based RL & World Models](Model-Based-RL-and-World-Models).

### Multimodal Distribution
한 input에서 서로 떨어진 여러 outcome mode가 가능한 확률 분포. 평균 하나로 회귀하면 실제로 존재하지 않는 mean state가 생길 수 있다.

---

## N

### Novelty
지금까지 덜 본 state/action을 더 새로운 것으로 보는 exploration signal. Novelty가 곧 task utility는 아니다. [Exploration & Exploitation](Exploration-and-Exploitation).

---

## O

### Observation (`O`)
Environment의 hidden true state 중 agent가 실제로 접근할 수 있는 관측 정보. **State/Representation과 같은 말이 아니다.** [MDP and POMDP](MDP-and-POMDP), [State Representation](State-Representation).

### Outcome Probability
특정 environment outcome이 실제로 발생할 probability mass. Prediction reliability와 다르다. [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability).

### OOD (Out-of-Distribution)
Training distribution과 충분히 다른 state/action 영역. Neural model이 이곳에서도 숫자를 출력한다고 그 estimate가 trustworthy한 것은 아니다. [Critic, Support & OOD](Critic-Support-and-OOD).

---

## P

### Partial Observability
Agent가 environment hidden state 전체를 직접 보지 못하는 상황. [POMDP](MDP-and-POMDP)와 연결된다.

### Planner Advantage
Imagination이 계산한 root value가 Policy root value보다 얼마나 높은지의 차이. Fixed intervention margin과 비교한다.

### Policy
현재 observation/representation에서 어떤 action을 선택할지 결정하는 규칙 또는 model. AASSR current 기본 Policy는 relational DQN + information residual이다. [Policy](Policy).

### Policy Override
Planner의 candidate가 reliability/support/margin 조건을 통과해 Policy의 원래 concrete action을 다른 root action으로 바꾸는 것.

### POMDP
Partially Observable Markov Decision Process. Agent가 hidden state 대신 observation을 받고 history/belief가 중요해지는 sequential decision framework. [MDP and POMDP](MDP-and-POMDP).

### Prediction Reliability
World-model prediction을 얼마나 믿을 수 있는지에 대한 calibration/gating 값. Outcome probability, value, support와 다르다. [Calibration](Calibration).

### Prophecy
AASSR의 stochastic relational world model. Current contract는 conditional-mixture ensemble v5 status-balanced 계열이다. [Prophecy](Prophecy).

---

## Q

### Q-learning
Action-value `Q(s,a)`를 Bellman target으로 학습하는 off-policy value-based RL 방법. [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD).

### Q-value
현재 state에서 action을 선택했을 때 기대되는 장기 discounted return의 추정값. Immediate reward와 다르다. [Value Functions & Bellman](Value-Functions-and-Bellman-Equation).

---

## R

### Raw DQN
Current raw representation을 사용하는 corrected DQN baseline. Relational representation 효과를 분리하기 위한 기준 조건. [Experiments](Experiments).

### Real Transition
Environment에 실제 concrete action을 실행해 얻은 `(state, action, next observation/state, reward, boundary)` evidence. Imagined transition과 구분한다.

### Relational DQN
AASSR과 같은 relational representation을 사용하지만 AASSR planner/world-model stack을 사용하지 않는 model-free control baseline.

### Relational Identity
Concrete ID보다 역할과 관계 구조를 동일성의 기준으로 보는 representation identity. Rename transfer를 목표로 한다. [Relational Representation & Generalization](Relational-Representation-and-Generalization).

### Relational State v3
Current AASSR transfer representation. Base relational descriptor에 latest public HTTP status categorical channel을 보존한다. [State Representation](State-Representation).

### Reliability
이 위키에서는 가능하면 “무엇의 reliability인지”를 명시한다. 특히 Prophecy prediction reliability를 의미하며 value/support와 구분한다.

### Replay Buffer
과거 real transition을 저장하고 off-policy learner가 재사용하는 memory. [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries).

### Representation
Observation을 learner가 사용할 feature/vector/structure로 변환한 것. **Environment state나 raw observation과 동일하지 않다.** [State Representation](State-Representation).

### Response-causal
Agent가 실제 response 시점까지 관측할 수 있었던 정보만 learner input/Knowledge에 허용하는 원칙. [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation).

### Return
시점 `t` 이후의 discounted cumulative reward.

```math
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
```

[Value Functions & Bellman](Value-Functions-and-Bellman-Equation).

### Reward
Environment가 한 transition에서 제공하는 task signal. Return, Q-value, loss, information residual과 다르다.

### Root Action
현재 real decision에서 바로 실행 가능한 첫 concrete action. Deep imagined continuation과 구분한다.

### Root Preservation
Deep planning branch가 prune/reject되더라도 실제 legal root action 자체의 shallower evaluation을 잃지 않는 planner invariant. [Imagination](Imagination).

---

## S

### Same-checkpoint Evaluation
하나의 frozen AASSR checkpoint를 Planner OFF/ON 두 mode에서 재사용해 Imagination marginal effect를 분리하는 평가. [Experiments](Experiments).

### Scenario Seed
Route/object/session 등 environment instance structure를 결정하는 seed.

### Self-loop
Semantic state가 바뀌지 않는 transition:

```text
S → A → S
```

AASSR은 관측 evidence에 기반해 이런 exact 반복을 억제한다. [ASEQ](ASEQ).

### Skill
반복 성공한 real ASeq를 relational template로 승격해 새 scenario의 concrete action에 다시 bind하는 고수준 재사용 메커니즘. [Skills](Skills).

### Sparse Reward
대부분 transition reward가 `0`이고 성공/실패 같은 드문 event에서만 강한 signal이 나오는 설정. [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment).

### Stall
의미 있는 진행 없이 same/near-same state-action behavior가 반복되는 정체 상태. True failure나 truncation과 동일하지 않다.

### State
문맥에 따라 의미가 달라지므로 위키에서는 가능한 한 더 구체적으로 쓴다.

- **Environment hidden state**: simulator의 실제 내부 상태
- **Semantic state**: ASEQ가 task-relevant exact repetition을 비교하는 상태 identity
- **Relational representation/state**: transfer learner 입력 feature

따라서 `state = observation = representation`으로 쓰지 않는다. [State Representation](State-Representation).

### State Aliasing
실제로 future/action requirement가 다른 두 상황이 같은 representation으로 합쳐지는 문제. [MDP and POMDP](MDP-and-POMDP), [Relational Representation & Generalization](Relational-Representation-and-Generalization).

### Stochastic Future
동일한 public state/action에서도 hidden condition 또는 environment randomness 때문에 여러 outcome이 가능한 미래. [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability).

### Structural Root Deduplication
Concrete ID는 다르지만 같은 relational legal slot/structure를 가진 root의 world-model/Critic 계산을 한 번만 수행하고 결과를 fan-out하는 최적화. [Imagination](Imagination).

### Support
Estimate의 training-data 근거를 뜻한다. “value가 좋다” 또는 “prediction이 정확하다”와 다르다. [Critic, Support & OOD](Critic-Support-and-OOD).

---

## T

### TD (Temporal Difference) Learning
실제 return이 끝날 때까지 기다리지 않고 다음 value estimate를 이용해 현재 value를 업데이트하는 학습 방식. [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD).

### Terminal
Episode가 끝나는 outcome. AASSR에서는 success/failure/truncation을 같은 의미로 뭉개지 않는다.

### True Failure
실제 lockout처럼 task가 실패로 종료된 terminal outcome. Current external reward contract에서 `-1`.

### Truncation
Rate-limit/reset/transition cap 등으로 episode를 멈추지만 task의 실제 failure로 정의하지 않는 boundary. Reward semantics와 TD bootstrap boundary를 별도로 봐야 한다. [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries).

---

## U

### Uncertainty
Future/prediction에 대한 불확실성. Environment randomness 계열과 model knowledge 부족 계열을 구분할 수 있다. [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability).

### Unseen Evaluation
Training에 사용하지 않은 scenario seed/task instance에서의 평가.

---

## V

### Value Function
State 또는 state-action의 expected future return을 추정하는 함수. `V(s)`, `Q(s,a)` 등이 있다. [Value Functions & Bellman](Value-Functions-and-Bellman-Equation).

---

## W

### World Model
Environment transition/outcome을 내부적으로 예측하는 learned model. AASSR의 current world model이 [Prophecy](Prophecy)다. [Model-Based RL & World Models](Model-Based-RL-and-World-Models).

---

# 반드시 구분해야 하는 핵심 묶음

## State / Observation / Representation

```text
Hidden environment state
        ↓ observation process
Public observation
        ↓ encoding
Learner representation
```

세 층은 다르다. [State Representation](State-Representation)

## Reward / Return / Q-value / Loss

```text
Reward
= 한 transition의 task signal

Return
= 미래 reward 누적

Q-value
= return의 기대 추정

Loss
= neural parameter update objective
```

## Outcome Probability / Reliability / Value / Support

```text
Outcome probability
= 환경 outcome이 일어날 확률

Prediction reliability
= world-model prediction을 믿을 수 있는가

Critic value
= future task return이 얼마나 좋은가

Local support
= 그 value estimate에 real training evidence가 있는가
```

## Mixture / Ensemble

```text
Mixture
= 환경에서 가능한 여러 outcome mode

Ensemble
= 여러 learned model의 prediction/evidence
```

## Terminal / Failure / Truncation / Bootstrap Boundary

서로 같은 말이 아니다. [Replay Buffer & Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

## Real / Imagined Evidence

```text
real transition
= 실제 environment interaction

imagined transition
= learned model의 counterfactual prediction
```

Current primary protocol에서 imagined transition은 factual replay truth로 자동 승격되지 않는다.

---

## 연구 문서 바로가기

- [Research Questions](Research-Questions)
- [Evidence Matrix](Evidence-Matrix)
- [Experiments](Experiments)
- [Current Status](Current-Status)
- [Concept Index](Concept-Index)
