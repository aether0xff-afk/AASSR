# Concept Index

이 페이지는 AASSR 위키의 **개념 사전이자 지식 지도**다.

AASSR을 읽다가 모르는 용어가 나오면 해당 단어를 눌러 더 기초적인 개념으로 내려갈 수 있도록 구성한다. 반대로 강화학습의 기초 개념을 읽다가 **그 개념이 AASSR에서 실제로 어디에 쓰이는지** 다시 핵심 메커니즘으로 올라갈 수도 있다.

> [!TIP]
> 이 위키의 링크 규칙은 가능한 한 **첫 등장하는 중요한 전문용어에 내부 링크를 건다**는 것이다. 같은 문단에서 지나치게 반복해서 링크하지는 않지만, 독자가 어느 페이지에서 시작해도 관련 개념을 타고 이동할 수 있게 한다.

> [!NOTE]
> 짧은 정의가 필요하면 [Glossary](Glossary), 개념을 처음부터 이해하려면 이 페이지에서 연결되는 foundation 문서, AASSR의 실제 구현을 보고 싶으면 각 Core Mechanism 페이지로 이동한다.

---

# 1. 전체 지식 그래프

```mermaid
flowchart TD
    RL[Reinforcement Learning] --> MDP[MDP / POMDP]
    RL --> SR[Sparse Reward / Credit Assignment]
    RL --> EE[Exploration / Exploitation]
    RL --> NN[Neural Networks / Optimization]

    SR --> CURR[Curriculum Learning]
    EE --> INFO[Information Theory / Intrinsic Motivation]

    MDP --> MF[Model-Free RL]
    MDP --> MB[Model-Based RL]

    MF --> Q[Value Functions / Bellman]
    Q --> DQN[Q-Learning / DQN / TD]
    DQN --> RP[Replay / Episode Boundaries]

    NN --> LOSS[Loss / Class Imbalance]
    NN --> GRU[GRU / Sequence Models]

    MB --> WM[World Model]
    WM --> U[Stochasticity / Uncertainty]
    U --> MIX[Mixture / Ensemble / Calibration]
    WM --> PLAN[Counterfactual Planning]

    PLAN --> CHANCE[Chance vs Decision]
    PLAN --> CRITIC[Critic / OOD Support]

    MDP --> REP[Relational Representation]
    REP --> GEN[Generalization / Transfer]
    GEN --> HRL[Hierarchical RL / Skills]

    RL --> CAUSAL[Causality / Leakage / Evaluation]
    CAUSAL --> ABL[Ablation / Benchmarking / Reproducibility]

    SR --> AASSR[AASSR]
    DQN --> AASSR
    INFO --> AASSR
    CURR --> AASSR
    WM --> AASSR
    PLAN --> AASSR
    REP --> AASSR
    CRITIC --> AASSR
    HRL --> AASSR
```

---

# 2. 강화학습을 처음부터

## [Reinforcement Learning](Reinforcement-Learning)

가장 먼저 읽을 페이지다.

다음 개념을 한꺼번에 연결한다.

- [에이전트(agent)](Reinforcement-Learning) / [환경(environment)](Reinforcement-Learning)
- [상태(state)](State-Representation) / [관측(observation)](MDP-and-POMDP)
- [행동(action)](Reinforcement-Learning)
- [보상(reward)](Sparse-Reward-and-Credit-Assignment) / [누적 보상(return)](Value-Functions-and-Bellman-Equation)
- [미래 보상의 할인율(discount)](Value-Functions-and-Bellman-Equation) [실험에서 바꾸어 보는 요인(factor)](Ablation-Benchmarking-and-Reproducibility)
- [정책(policy)](Policy)
- [가치(value)](Value-Functions-and-Bellman-Equation) [함수(function)](Terminology-Guide)
- [경험 경로(trajectory)](Reinforcement-Learning) / [한 번의 문제 풀이 구간(episode)](Terminology-Guide)
- [환경 예측 모델 없이 직접 학습하는(model-free)](Reinforcement-Learning) / [환경 모델을 사용하는(model-based)](Model-Based-RL-and-World-Models)
- on-policy / off-policy

AASSR의 `Policy`, `Prophecy`, `Critic`, `Imagination`이 일반 강화학습에서 어디에 위치하는지도 여기서 연결한다.

---

## [MDP and POMDP](MDP-and-POMDP)

- Markov property
- MDP의 `(S, A, P, R, γ)`
- [숨은 환경 상태(hidden state)](MDP-and-POMDP)와 관측
- POMDP
- [관측을 바탕으로 추정한 상태 믿음(belief)](MDP-and-POMDP) 상태
- 상태 aliasing
- [기억(memory)](GRU-and-Sequence-Models)

AASSR의 partial 관측 [명세(contract)](Current-Status)와 [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) [Prophecy(미래 예측 모델)](Prophecy)를 이해하는 기반이다.

---

## [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment)

- dense / [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)
- delayed 보상
- long [미래를 내다보는 범위(horizon)](Counterfactual-Planning-and-Search)
- [시간 순서를 고려하는(temporal)](GRU-and-Sequence-Models) [보상 책임 배분(credit assignment)](Sparse-Reward-and-Credit-Assignment)
- 보상 [인위적인 형태 조정(shaping)](Sparse-Reward-and-Credit-Assignment)
- potential-based 형태 조정
- intrinsic 보상
- Monte Carlo vs TD

AASSR 자체의 구체적인 문제 설정은 **[Sparse Reward Problem](Sparse-Reward-Problem)** 에서 이어진다.

---

## [Exploration and Exploitation](Exploration-and-Exploitation)

- [활용(exploitation)](Exploration-and-Exploitation) / [탐색(exploration)](Exploration-and-Exploitation)
- epsilon-greedy
- epsilon decay
- [무작위(random)](Ablation-Benchmarking-and-Reproducibility) 탐색의 한계
- [새로움(novelty)](Information-Theory-and-Intrinsic-Motivation)
- [새 정보를 찾아보려는 호기심 기반 탐색(curiosity)](Information-Theory-and-Intrinsic-Motivation)
- [정보(information)](Information-Theory-and-Intrinsic-Motivation) [증가량(gain)](Ablation-Benchmarking-and-Reproducibility)
- risky 탐색

AASSR [Policy(정책 모델)](Policy)의 [정보 가치 잔차(information residual)](Policy)과 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)를 이해하는 데 중요하다.

---

## [Information Theory and Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation)

- self-information
- [확률 분포의 불확실성을 나타내는 엔트로피(entropy)](Information-Theory-and-Intrinsic-Motivation)
- [조건부(conditional)](Stochasticity-Uncertainty-and-Probability) 엔트로피
- mutual 정보
- KL divergence
- 정보 증가량
- 호기심 기반 탐색
- noisy-TV problem
- intrinsic vs extrinsic [학습 목표(objective)](Terminology-Guide)

AASSR의 [정보 가치 잔차(information-value residual)](Policy)을 일반 이론과 구분하면서 이해하는 페이지다.

---

## [Curriculum Learning](Curriculum-Learning)

- [고정된(fixed)](Ablation-Benchmarking-and-Reproducibility) / adaptive [난이도 조절 학습(curriculum)](Curriculum-Learning)
- [다음 난이도로 승급(promotion)](Curriculum-Learning) / demotion
- easy-to-hard [학습(learning)](Reinforcement-Learning)
- [전이(transfer)](Relational-Representation-and-Generalization) bottleneck
- catastrophic forgetting
- 난이도 조절 학습과 보상 형태 조정의 차이
- [정답 경로로 유도된(guided)](Causality-Leakage-and-Evaluation) 경험 경로와의 차이

AASSR에서 최초 성공 [스스로 새로운 성공 경로를 발견하는 것(discovery)](Research-Questions)와 [여러 기본 행동을 묶는 상위 수준(higher-level)](Hierarchical-RL-and-Skills) 전이가 왜 별개의 문제인지 연결한다.

---

# 3. 가치 기반 강화학습

## [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation)

- 누적 보상 `G_t`
- 할인율 실험 요인 `γ`
- 상태 가치 `V(s)`
- 행동 가치 `Q(s,a)`
- [다른 선택보다 나은 정도(advantage)](Value-Functions-and-Bellman-Equation)
- Bellman [확률 기댓값(expectation)](Chance-and-Decision-Nodes) equation
- Bellman optimality equation
- [미래 가치를 앞 단계로 되돌려 계산하는 과정(backup)](Value-Functions-and-Bellman-Equation)
- [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries)ping

AASSR의 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) [Policy](Policy)와 sparse-누적 보상 [Critic(미래 가치 평가기)](Critic)의 수학적 기반이다.

---

## [Q-Learning, DQN and Temporal Difference](Q-Learning-DQN-and-TD)

- [Q-러닝(Q-learning)](Q-Learning-DQN-and-TD) [학습 갱신(update)](Neural-Networks-and-Optimization)
- TD [오차(error)](Loss-Functions-and-Class-Imbalance)
- [대상 또는 학습 목표값(target)](Terminology-Guide) [신경망(network)](Neural-Networks-and-Optimization)
- [경험(experience)](Replay-Buffer-and-Episode-Boundaries) [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)
- [DQN](Q-Learning-DQN-and-TD)
- epsilon-greedy
- [에피소드 종료(terminal)](Replay-Buffer-and-Episode-Boundaries) [가능/불가능을 표시하는 마스크(mask)](Terminology-Guide)
- overestimation
- [데이터 분포 변화(distribution shift)](Critic-Support-and-OOD)

`dqn_raw`, `dqn_relational`, `CurrentRelationalPolicy`를 읽기 전에 보면 좋다.

---

## [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries)

- 경험 경험 재사용
- 에피소드 종료
- [외부 제한 종료(truncation)](Replay-Buffer-and-Episode-Boundaries)
- [진전 없이 반복하다 멈춘(stalled)](ASEQ) [환경 초기화(reset)](Replay-Buffer-and-Episode-Boundaries)
- rate-limit 환경 초기화
- [상태 전이(transition)](MDP-and-POMDP) cap
- 보상 [경계(boundary)](Replay-Buffer-and-Episode-Boundaries) vs 다음 상태 가치 이어받기 경계
- 경험 재사용 vs [Knowledge(에피소드 지식)](Knowledge)
- [실제 환경에서 관측된(real)](Research-Jargon-Guide) 상태 전이 vs [모델이 상상한(imagined)](Research-Jargon-Guide) 상태 전이

AASSR에서 **보상가 0이어도 TD 다음 상태 가치 이어받기은 끊어야 할 수 있다**는 수정의 이론적 배경이다.

---

# 4. 신경망과 최적화

## [Neural Networks and Optimization](Neural-Networks-and-Optimization)

- 함수 [근사(approximation)](Value-Functions-and-Bellman-Equation)
- linear [처리 계층(layer)](Research-Architecture) / activation
- forward / backward
- [기울기(gradient)](Neural-Networks-and-Optimization) / backpropagation
- [신경망 파라미터를 갱신하는 최적화 알고리즘(optimizer)](Neural-Networks-and-Optimization)
- 학습 [비율(rate)](Terminology-Guide)
- minibatch
- overfitting / underfitting
- [수치 범위를 맞추는 정규화(normalization)](Neural-Networks-and-Optimization)
- one-hot [학습용 수치 표현으로 바꾸는 인코딩(encoding)](State-Representation)
- GPU [묶음 처리(batching)](Reproduction) / synchronization

AASSR 코드에서 신경망 구현을 읽기 위한 기초다.

---

## [Loss Functions and Class Imbalance](Loss-Functions-and-Class-Imbalance)

- MSE / MAE
- Huber / Smooth L1
- softmax / cross 엔트로피
- BCE
- [범주형(categorical)](Loss-Functions-and-Class-Imbalance) vs multi-label
- [범주(class)](Loss-Functions-and-Class-Imbalance) [데이터 수의 불균형(imbalance)](Loss-Functions-and-Class-Imbalance)
- 범주 weighting / oversampling
- NLL / [여러 결과의 혼합 분포(mixture)](Mixture-Ensemble-and-Calibration) likelihood
- multi-task [학습 손실(loss)](Loss-Functions-and-Class-Imbalance)
- [학습(training)](Terminology-Guide) 학습 손실 vs [검증(validation)](Ablation-Benchmarking-and-Reproducibility) [평가지표(metric)](Ablation-Benchmarking-and-Reproducibility)

특히 [상태 코드까지 고려하는(status-aware)](Calibration) [Prophecy](Prophecy)에서 **[드문(rare)](Loss-Functions-and-Class-Imbalance) 범주를 더 잘 학습하는 것과 보상 형태 조정은 다른 문제**라는 점을 설명한다.

---

# 5. Model-Based RL과 World Model

## [Model-Based RL and World Models](Model-Based-RL-and-World-Models)

- 환경 예측 모델 없는 vs 모델 기반
- 상태 전이 [학습 모델(model)](Terminology-Guide)
- 보상 학습 모델
- [세계 모델(world model)](Model-Based-RL-and-World-Models)
- [학습된(learned)](Neural-Networks-and-Optimization) [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)
- one-step vs multi-step [예측(prediction)](Terminology-Guide)
- compounding 오차
- 학습 모델 [편향(bias)](Ablation-Benchmarking-and-Reproducibility)
- [모델 오류 악용(model exploitation)](Model-Based-RL-and-World-Models)
- [직접 관측되지 않는 잠재 표현(latent)](GRU-and-Sequence-Models) 세계 모델
- Dreamer 계열과의 개념적 비교

AASSR의 [Prophecy](Prophecy)와 [Imagination(가상 미래 탐색)](Imagination)을 일반 [모델 기반 강화학습(model-based RL)](Model-Based-RL-and-World-Models) 안에서 위치시킨다.

---

## [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability)

- [확률(probability)](Stochasticity-Uncertainty-and-Probability)
- 무작위 [변수(variable)](Terminology-Guide)
- [확률을 고려해 기대되는(expected)](Chance-and-Decision-Nodes) 가치
- [분산(variance)](Stochasticity-Uncertainty-and-Probability)
- stochasticity
- aleatoric [불확실성(uncertainty)](Stochasticity-Uncertainty-and-Probability)
- [지식 부족에서 오는 불확실성(epistemic uncertainty)](Stochasticity-Uncertainty-and-Probability)
- [신뢰도(reliability)](Calibration)
- 가치
- [데이터 근거(support)](Critic-Support-and-OOD)
- risk

AASSR에서 반드시 구분해야 하는 관계:

```text
outcome probability
!= prediction reliability
!= Critic value
!= local support
```

---

## [Mixture Models, Ensembles and Calibration](Mixture-Ensemble-and-Calibration)

- [여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration) [확률 또는 데이터 분포(distribution)](Stochasticity-Uncertainty-and-Probability)
- 혼합 분포 학습 모델 / [구성요소(component)](Research-Architecture) / [가중치(weight)](Neural-Networks-and-Optimization)
- 혼합 분포 [여러 결과가 하나로 뭉개지는 붕괴(collapse)](Mixture-Ensemble-and-Calibration)
- [여러 모델을 함께 쓰는 앙상블(ensemble)](Mixture-Ensemble-and-Calibration) / disagreement
- aleatoric vs epistemic 관점
- [검증용 분리 데이터(holdout)](Calibration) [예측 신뢰도 보정(calibration)](Calibration)
- [확률로 가중한(probability-weighted)](Chance-and-Decision-Nodes) [의미 기준(semantic)](State-Representation) [평가 점수(score)](Terminology-Guide)
- [학습을 멈춘(frozen)](Ablation-Benchmarking-and-Reproducibility) 검증용 분리 데이터
- 신뢰도 diagram / ECE 개념

AASSR 확률적 [Prophecy](Prophecy)와 [Calibration(예측 신뢰도 보정)](Calibration)의 직접 배경이다.

---

# 6. Planning과 Imagination

## [Counterfactual Planning and Search](Counterfactual-Planning-and-Search)

- [계획(planning)](Counterfactual-Planning-and-Search) vs 학습
- [실제로 하지 않은 경우를 가정하는 반사실적(counterfactual)](Counterfactual-Planning-and-Search)
- [가상 미래 전개(rollout)](Counterfactual-Planning-and-Search)
- lookahead
- 미래 탐색 범위
- [탐색(search)](Counterfactual-Planning-and-Search) [탐색 트리(tree)](Counterfactual-Planning-and-Search)
- [여러 미래로 갈라지는 분기(branching)](Chance-and-Decision-Nodes) 실험 요인
- [유망 후보만 남기는 빔 탐색(beam)](Counterfactual-Planning-and-Search) 탐색
- [유망하지 않은 탐색 가지를 제거하는 가지치기(pruning)](Counterfactual-Planning-and-Search)
- [탐색의 첫 행동(root)](Imagination) [의미 보존(preservation)](Ablation-Benchmarking-and-Reproducibility)
- [구조 기반(structural)](Relational-Representation-and-Generalization) deduplication
- MPC / receding 미래 탐색 범위과의 개념적 연결
- [실제 행동 개입(intervention)](Imagination) [최소 차이 기준(margin)](Imagination)

AASSR의 [Imagination](Imagination)을 단순 `n × k`보다 깊게 이해하는 페이지다.

---

## [Chance Nodes and Decision Nodes](Chance-and-Decision-Nodes)

```math
V_{chance}=\sum_i p_iV_i
```

```math
V_{decision}=\max_aV(a)
```

- 왜 환경 [무작위성(randomness)](Stochasticity-Uncertainty-and-Probability)에는 확률 기댓값을 쓰는가?
- 왜 에이전트 choice에는 max를 쓸 수 있는가?
- optimistic 확률적 가치 되돌림 계산은 왜 틀리는가?
- 확률와 신뢰도는 왜 다른가?

AASSR [Imagination](Imagination)의 핵심 수학적 [의미 규칙(semantics)](State-Representation)다.

---

# 7. 표현, 일반화, OOD

## [Relational Representation and Generalization](Relational-Representation-and-Generalization)

- [표현(representation)](Relational-Representation-and-Generalization)
- [이름이나 사례를 그대로 외우는 암기(memorization)](Relational-Representation-and-Generalization) vs [일반화(generalization)](Relational-Representation-and-Generalization)
- [이름 순서를 바꾸는 순열(permutation)](Relational-Representation-and-Generalization)
- [이름 등이 바뀌어도 결과가 유지되는 불변성(invariance)](Relational-Representation-and-Generalization) / equivariance
- [관계 기반(relational)](Relational-Representation-and-Generalization) inductive 편향
- abstr행동
- 상태 aliasing
- [실제 개체를 구분하는(concrete)](State-Representation) vs 관계 기반 [식별 방식(identity)](State-Representation)
- 전이 학습
- 구조 기반 탐색의 첫 행동 [중복 계산 제거(dedup)](Reproduction)
- 표현 [정보 누출(leakage)](Causality-Leakage-and-Evaluation)

AASSR `Relational State v3`, 행동 [핵심(key)](Terminology-Guide), [Skill(성공 절차 재사용)](Skills) 전이의 기반이다.

---

## [Critic, Support and OOD](Critic-Support-and-OOD)

- 함수 근사
- interpolation / [학습 범위 밖으로 값을 추정하는 외삽(extrapolation)](Critic-Support-and-OOD)
- in-distribution / [학습 분포 밖(OOD)](Critic-Support-and-OOD)
- [전체 범위(global)](Terminology-Guide) readiness vs [국소 데이터 근거(local support)](Critic-Support-and-OOD)
- nearest-neighbor [증거(evidence)](Evidence-Matrix)
- density/데이터 근거 intuition
- fail-open / [근거가 부족하면 보수적으로 거부하는(fail-closed)](Critic-Support-and-OOD)
- conservative RL과의 문제의식 연결

AASSR에서 [Critic](Critic)이 숫자를 출력한다는 사실만으로 [기본 행동 덮어쓰기(override)](Imagination)를 허용하지 않는 이유를 설명한다.

---

# 8. Sequence와 계층적 행동

## [GRU and Sequence Models](GRU-and-Sequence-Models)

- RNN
- 숨은 환경 상태
- [GRU(게이트 순환 유닛)](GRU-and-Sequence-Models) 학습 갱신/환경 초기화 [판정 관문(gate)](Terminology-Guide)
- [순서열(sequence)](GRU-and-Sequence-Models) 인코딩
- recurrent-state [서로 맞지 않는 불일치(mismatch)](Causality-Leakage-and-Evaluation)
- [과거 기억을 0으로 초기화한(zero-memory)](GRU-and-Sequence-Models) [학습된 모델로 값을 계산하는 추론(inference)](Neural-Networks-and-Optimization)
- [의사결정(decision)](Chance-and-Decision-Nodes) [후속 구간(suffix)](GRU-and-Sequence-Models) 학습
- 순서열 묶음 처리

AASSR [Critic](Critic)의 [과거 정보를 이어가는 순환형(recurrent)](GRU-and-Sequence-Models) 명세를 이해하는 페이지다.

---

## [Hierarchical Reinforcement Learning and Skills](Hierarchical-RL-and-Skills)

- 시간 순서 기반 abstr행동
- [여러 행동을 묶은 상위 행동(macro)](Hierarchical-RL-and-Skills) 행동
- [여러 기본 행동을 묶은 상위 행동 단위(option)](Hierarchical-RL-and-Skills) [문제 표현 틀(framework)](Terminology-Guide)
- [더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills) 행동
- [재사용 가능한 기술(skill)](Skills) 발견
- 난이도 승급
- 관계 기반 [Skill](Skills)
- initiation / availability
- 확률적 기술 가상 미래 전개

AASSR의 [Skill](Skills)이 사람이 넣은 정답 행동 묶음와 어떻게 다른지 설명한다.

---

# 9. 연구 방법론

## [Causality, Leakage and Fair Evaluation](Causality-Leakage-and-Evaluation)

- causality
- [데이터(data)](Terminology-Guide) 정보 누출
- 대상/목표값 정보 누출
- [결과를 본 뒤 얻은 사후 정보(hindsight)](Causality-Leakage-and-Evaluation) 정보 누출
- privileged 정보
- [공개된(public)](State-Representation) 관측 명세
- cross-episode 정보 누출
- 가상 [실제로 관측한 사실(fact)](Causality-Leakage-and-Evaluation) vs 실제 실제 관측 사실
- train/[검사 또는 테스트(test)](Ablation-Benchmarking-and-Reproducibility) contamination
- [같은 체크포인트(same-checkpoint)](Experiments) [비교(comparison)](Ablation-Benchmarking-and-Reproducibility)
- [정답을 알고 있는 기준(Oracle)](Ablation-Benchmarking-and-Reproducibility) / 정답 경로 유도 경험 경로

AASSR의 anti-hindsight, hidden-state 금지, 같은 체크포인트 [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)과 연결된다.

---

## [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility)

- [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)
- [비교 기준(baseline)](Ablation-Benchmarking-and-Reproducibility) / [효과를 비교하기 위한 대조 조건(control)](Ablation-Benchmarking-and-Reproducibility)
- [구성요소 제거 비교(ablation)](Ablation-Benchmarking-and-Reproducibility)
- confounder
- independent / dependent 변수
- [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)
- 학습 [실험에 허용된 전이 수 한도(budget)](Ablation-Benchmarking-and-Reproducibility)
- hyperparameter tuning
- 평가지표 / [대리 지표(proxy)](Ablation-Benchmarking-and-Reproducibility) 평가지표
- mean / standard deviation / [예측 신뢰 정도(confidence)](Calibration) interval
- paired 비교
- [진단 실험(diagnostic)](Evidence-Matrix) vs [최종(final)](Ablation-Benchmarking-and-Reproducibility) 표준 비교 실험
- reproducibility
- artifact [정보의 출처 기록(provenance)](Knowledge)
- [최종 기준(source of truth)](Current-Status)

AASSR의 핵심 비교:

```text
dqn_raw
→ dqn_relational
→ aassr_current_no_imagination
→ aassr_current_full
```

가 왜 필요한지 연구 방법론 수준에서 설명한다.

---

# 10. AASSR 자체 문서

## 연구

- [Sparse Reward Problem](Sparse-Reward-Problem)
- [Research Questions](Research-Questions)
- [Research Architecture](Research-Architecture)
- [Design Rationale](Design-Rationale)

## 메커니즘

- [State Representation](State-Representation)
- [ASEQ](ASEQ)
- [Policy](Policy)
- [Knowledge](Knowledge)
- [Prophecy](Prophecy)
- [Calibration](Calibration)
- [Critic](Critic)
- [Imagination](Imagination)
- [Skills](Skills)
- [Core Architecture](Core-Architecture)

## 검증

- [Experiments](Experiments)
- [Current Status](Current-Status)
- [Reproduction](Reproduction)

## 역사 / 용어

- [Development History](Development-History)
- [Glossary](Glossary)

---

# 11. 단어를 발견했을 때 어디로 가야 하나?

| 단어 | 먼저 읽을 페이지 |
|---|---|
| 강화학습, 에이전트, 환경 | [Reinforcement Learning](Reinforcement-Learning) |
| 상태, 관측, Markov, POMDP | [MDP and POMDP](MDP-and-POMDP) |
| 희소 보상, delayed 보상, 보상 책임 배분 | [Sparse Reward and Credit Assignment](Sparse-Reward-and-Credit-Assignment) |
| 탐색, epsilon, 호기심 기반 탐색 | [Exploration and Exploitation](Exploration-and-Exploitation) |
| 엔트로피, mutual 정보, 정보 증가량 | [Information Theory and Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation) |
| 난이도 조절 학습, 난이도 승급, demotion | [Curriculum Learning](Curriculum-Learning) |
| 가치, 누적 보상, Bellman, 다음 상태 가치 이어받기ping | [Value Functions and Bellman Equation](Value-Functions-and-Bellman-Equation) |
| Q-러닝, [DQN](Q-Learning-DQN-and-TD), TD, 대상/목표값 신경망 | [Q-Learning, DQN and TD](Q-Learning-DQN-and-TD) |
| 경험 재사용, 에피소드 종료, 외부 제한 종료 | [Replay Buffer and Episode Boundaries](Replay-Buffer-and-Episode-Boundaries) |
| [신경망 기반(neural)](Neural-Networks-and-Optimization) 신경망, 기울기, 최적화 알고리즘, GPU [여러 입력 묶음(batch)](Reproduction) | [Neural Networks and Optimization](Neural-Networks-and-Optimization) |
| MSE, cross 엔트로피, Smooth L1, 범주 불균형 | [Loss Functions and Class Imbalance](Loss-Functions-and-Class-Imbalance) |
| 세계 모델, 학습 모델 편향, 모델 오류 악용 | [Model-Based RL and World Models](Model-Based-RL-and-World-Models) |
| 확률, 불확실성, aleatoric, epistemic | [Stochasticity, Uncertainty and Probability](Stochasticity-Uncertainty-and-Probability) |
| 혼합 분포, 앙상블, 예측 신뢰도 보정 | [Mixture, Ensemble and Calibration](Mixture-Ensemble-and-Calibration) |
| 가상 미래 전개, lookahead, 빔 탐색, 가지치기 | [Counterfactual Planning and Search](Counterfactual-Planning-and-Search) |
| [환경 결과 노드(chance node)](Chance-and-Decision-Nodes), [행동 선택 노드(decision node)](Chance-and-Decision-Nodes), 확률 기댓값 | [Chance and Decision Nodes](Chance-and-Decision-Nodes) |
| 관계 기반, 순열, 불변성, 전이 | [Relational Representation and Generalization](Relational-Representation-and-Generalization) |
| [OOD](Critic-Support-and-OOD), 외삽, 데이터 근거, 근거가 부족하면 보수적으로 거부하는 | [Critic, Support and OOD](Critic-Support-and-OOD) |
| [GRU](GRU-and-Sequence-Models), 순환형, 숨은 환경 상태 | [GRU and Sequence Models](GRU-and-Sequence-Models) |
| 기술, 행동 묶음, 상위 행동 단위, 시간 순서 기반 abstr행동 | [Hierarchical RL and Skills](Hierarchical-RL-and-Skills) |
| 정보 누출, 사후 정보, privileged 정보 | [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation) |
| 구성요소 제거 비교, 비교 기준, 난수 시드, 예측 신뢰 정도 interval | [Ablation, Benchmarking and Reproducibility](Ablation-Benchmarking-and-Reproducibility) |

---

# 12. 추천 순서

**완전 처음부터:**  
[Reinforcement Learning](Reinforcement-Learning) → [MDP and POMDP](MDP-and-POMDP) → [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment) → [AASSR in 5 Minutes](AASSR-in-5-Minutes)

**[Policy](Policy)를 이해하고 싶다면:**  
[Value & Bellman](Value-Functions-and-Bellman-Equation) → [DQN & TD](Q-Learning-DQN-and-TD) → [Exploration](Exploration-and-Exploitation) → [Information Theory](Information-Theory-and-Intrinsic-Motivation) → [Policy](Policy)

**[Prophecy](Prophecy)를 이해하고 싶다면:**  
[MDP/POMDP](MDP-and-POMDP) → [World Models](Model-Based-RL-and-World-Models) → [Uncertainty](Stochasticity-Uncertainty-and-Probability) → [Mixture/Ensemble](Mixture-Ensemble-and-Calibration) → [Loss/Class Imbalance](Loss-Functions-and-Class-Imbalance) → [Prophecy](Prophecy)

**[Imagination](Imagination)을 이해하고 싶다면:**  
[World Models](Model-Based-RL-and-World-Models) → [Planning](Counterfactual-Planning-and-Search) → [Chance/Decision](Chance-and-Decision-Nodes) → [Critic/OOD](Critic-Support-and-OOD) → [Imagination](Imagination)

**실험을 검증하고 싶다면:**  
[Causality & Leakage](Causality-Leakage-and-Evaluation) → [Ablation & Benchmarking](Ablation-Benchmarking-and-Reproducibility) → [Experiments](Experiments) → [Current Status](Current-Status)

---

다음으로 읽기:

- **[Home](Home)**
- **[Reinforcement Learning](Reinforcement-Learning)**
- **[Research Architecture](Research-Architecture)**
- **[Glossary](Glossary)**