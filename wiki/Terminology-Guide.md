# 한국어 중심 AASSR 용어 안내서

이 페이지는 영어 전문용어를 이미 안다고 가정하지 않는다.

형식은 다음과 같다.

```text
한국어로 먼저 이해
→ 영어 원어 확인
→ 아주 쉬운 뜻
→ AASSR에서의 정확한 역할
→ 더 깊은 문서
```

영어 단어를 그대로 외우기보다 **“왜 이 단어가 필요한가?”**를 먼저 이해하는 것이 목적이다.

---

# 1. 강화학습의 기본 단어

## 에이전트 — agent

**쉬운 뜻:** 문제를 풀기 위해 행동하는 주체.

게임에서는 플레이어, 로봇에서는 제어 프로그램, AASSR에서는 다음 행동을 결정하는 전체 학습 시스템을 뜻한다.

관련: [강화학습](Reinforcement-Learning)

## 환경 — environment

**쉬운 뜻:** 에이전트가 행동을 실행하는 바깥 세계.

에이전트가 행동하면 환경이 응답하고 상황이 바뀐다.

관련: [강화학습](Reinforcement-Learning)

## 상태 — state

**쉬운 뜻:** 현재 상황이 어떤지를 나타내는 정보.

하지만 실제 환경 전체 상태, 에이전트가 볼 수 있는 정보, 신경망에 넣는 숫자 표현은 서로 다르다.

관련: [상태 표현](State-Representation), [MDP와 POMDP](MDP-and-POMDP)

## 관측 — observation

**쉬운 뜻:** 에이전트가 실제로 볼 수 있는 정보.

환경 내부의 모든 진실을 뜻하지 않는다.

관련: [MDP와 POMDP](MDP-and-POMDP)

## 행동 — action

**쉬운 뜻:** 에이전트가 실제로 선택하는 한 번의 행동.

관련: [강화학습](Reinforcement-Learning)

## 상태 전이 — transition

**영어:** [상태 전이(transition)](MDP-and-POMDP)

**쉬운 뜻:** 한 상태에서 행동을 한 뒤 다음 상태로 바뀌는 과정.

보통 다음처럼 쓴다.

```text
S --A--> S'
```

관련: [MDP와 POMDP](MDP-and-POMDP)

## 보상 — reward

**쉬운 뜻:** 환경이 행동 결과에 대해 주는 점수.

AASSR의 핵심 희소 보상 실험에서는 성공 `+1`, 실제 실패 `-1`, 그 외 대부분 `0`을 사용한다.

관련: [희소 보상](Sparse-Reward-and-Credit-Assignment)

## 누적 보상 — return

**쉬운 뜻:** 앞으로 받을 여러 보상을 시간 순서까지 고려해 합친 장기 점수.

즉시 보상과 다르다.

관련: [가치 함수와 Bellman 식](Value-Functions-and-Bellman-Equation)

## 정책 — policy

**쉬운 뜻:** 현재 상황에서 어떤 행동을 고를지 정하는 규칙 또는 학습 모델.

AASSR에는 기본 행동을 고르는 [Policy](Policy) 모듈이 있다.

관련: [Policy](Policy)

## 가치 — value

**쉬운 뜻:** 어떤 상태나 행동이 장기적으로 얼마나 좋은지 나타내는 값.

보상과 같은 말이 아니다.

관련: [가치 함수](Value-Functions-and-Bellman-Equation)

## Q값 — Q-value

**쉬운 뜻:** 현재 상태에서 특정 행동을 선택했을 때 앞으로 얼마나 좋은 결과가 기대되는지를 나타내는 값.

관련: [Q-learning·DQN·TD](Q-Learning-DQN-and-TD)

---

# 2. 희소 보상과 탐색

## 희소 보상 — sparse reward

**쉬운 뜻:** 성공 전까지 중간 점수를 거의 주지 않는 문제.

예:

```text
중간 행동 0
중간 행동 0
중간 행동 0
최종 성공 +1
```

관련: [희소 보상 문제](Sparse-Reward-Problem)

## 보상 책임 배분 — credit assignment

**쉬운 뜻:** 마지막에 받은 성공 점수를 과거의 어떤 행동들 덕분이라고 봐야 하는지 정하는 문제.

관련: [희소 보상과 보상 책임 배분](Sparse-Reward-and-Credit-Assignment)

## 탐색 — exploration

**쉬운 뜻:** 아직 잘 모르는 행동도 시도해 보는 것.

관련: [탐색과 활용](Exploration-and-Exploitation)

## 활용 — exploitation

**쉬운 뜻:** 지금까지 배운 것 중 가장 좋아 보이는 행동을 사용하는 것.

관련: [탐색과 활용](Exploration-and-Exploitation)

## 내재 동기 — intrinsic motivation

**쉬운 뜻:** 환경의 실제 목표 보상과 별개로, 새 정보나 새로운 상태를 찾도록 내부 신호를 주는 방법 계열.

AASSR에서는 외부 task [보상(reward)](Sparse-Reward-and-Credit-Assignment)와 내부 정보 관련 신호를 구분한다.

관련: [정보 이론과 내재 동기](Information-Theory-and-Intrinsic-Motivation)

## 새로운 정도 — novelty

**영어:** novelty

**쉬운 뜻:** 지금까지 덜 보았던 상태나 행동이 얼마나 새로운지.

새롭다고 해서 반드시 최종 목표에 좋은 것은 아니다.

---

# 3. 부분 관측과 정보 누출

## 부분 관측 — partial observability

**쉬운 뜻:** 환경의 모든 정보를 에이전트가 직접 볼 수 없는 상황.

관련: [MDP와 POMDP](MDP-and-POMDP)

## MDP — Markov Decision Process

**한국어:** 마르코프 의사결정 과정.

**쉬운 뜻:** 상태, 행동, 상태 변화, 보상으로 순차적 의사결정 문제를 표현하는 기본 틀.

관련: [MDP와 POMDP](MDP-and-POMDP)

## POMDP — Partially Observable Markov Decision Process

**한국어:** 부분 관측 마르코프 의사결정 과정.

**쉬운 뜻:** 환경의 실제 상태 일부만 관측할 수 있는 MDP.

관련: [MDP와 POMDP](MDP-and-POMDP)

## 숨은 정보 누출 — hidden leakage

**쉬운 뜻:** 에이전트가 실제 상황에서는 알 수 없어야 할 정답·미래 결과·환경 내부 정보가 학습 입력에 섞이는 문제.

이런 누출이 있으면 성능이 높아도 공정한 실험이 아니다.

관련: [인과성·정보 누출·공정 평가](Causality-Leakage-and-Evaluation)

## 사후 정보 누출 — hindsight leakage

**쉬운 뜻:** 행동 후에 알게 된 정보를 마치 행동 전에 이미 알고 있었던 것처럼 과거 판단에 사용하는 오류.

관련: [인과성·정보 누출·공정 평가](Causality-Leakage-and-Evaluation)

---

# 4. 표현과 일반화

## 표현 — representation

**쉬운 뜻:** 실제 관측을 학습 모델이 사용하기 좋은 형태로 바꾼 것.

관련: [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)

## 관계 기반 표현 — relational representation

**쉬운 뜻:** 객체의 이름 자체보다 객체 사이의 역할과 관계를 중심으로 나타내는 표현.

예:

```text
route-17
```

보다

```text
목록을 보여주는 경로
```

처럼 역할을 중심으로 보는 것이다.

관련: [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)

## 의미 기반 — semantic

**영어:** [의미 기준(semantic)](State-Representation)

**쉬운 뜻:** 단순한 숫자나 문자열이 아니라 문제 해결에서 실제로 같은 의미인지 기준으로 보는 것.

AASSR의 [ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ) 상태 비교에서 중요하다.

관련: [상태 표현](State-Representation), [ASEQ](ASEQ)

## 실제 개체 구분 — concrete identity

**영어:** [실제 개체 구분(concrete identity)](State-Representation)

**쉬운 뜻:** 역할이 같아도 실제로 서로 다른 객체를 구분하기 위한 식별 방식.

관계 기반 표현은 일반화에 좋지만 실제 실행에서는 어떤 구체적인 객체에 행동할지 구분해야 한다.

관련: [상태 표현](State-Representation)

## 일반화 — generalization

**쉬운 뜻:** 훈련에서 보지 않은 새 문제에서도 배운 원리가 작동하는 능력.

관련: [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)

## 전이 — transfer

**쉬운 뜻:** 한 문제에서 배운 지식이나 구조를 다른 문제에 재사용하는 것.

관련: [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)

## 학습 중 보지 못한 — unseen

**쉬운 뜻:** 훈련 데이터에 직접 포함되지 않았던 새 상황.

`unseen performance`는 학습하지 않은 새 문제에서의 성능을 뜻한다.

---

# 5. ASEQ와 반복 억제

## ASEQ

**뜻:** 실제로 관측된 `(현재 상태 S, 행동 A, 다음 상태 S')` 기록.

관련: [ASEQ](ASEQ)

## 제자리 반복 — self-loop

**영어:** [제자리 반복(self-loop)](ASEQ)

**쉬운 뜻:** 행동했는데 의미 있는 상태 변화 없이 다시 같은 상태로 돌아오는 반복.

```text
S → A → S
```

관련: [ASEQ](ASEQ)

## 행동 후보 억제 — suppression

**영어:** suppression

**쉬운 뜻:** 어떤 행동을 영구 삭제하는 것이 아니라, 현재 선택 후보에서 우선적으로 제외하거나 약하게 만드는 것.

[ASEQ](ASEQ)는 실제로 반복 확인된 무진전 행동만 제한하려고 한다.

---

# 6. DQN과 학습

## DQN — Deep Q-Network

**한국어:** 딥 Q-네트워크.

**쉬운 뜻:** Q값을 표 대신 신경망으로 예측하는 Q-learning 계열 방법.

관련: [Q-learning·DQN·TD](Q-Learning-DQN-and-TD)

## TD 학습 — Temporal-Difference learning

**쉬운 뜻:** 지금 받은 보상과 다음 상태의 추정 가치를 이용해 현재 가치 예측을 조금씩 수정하는 방법.

관련: [Q-learning·DQN·TD](Q-Learning-DQN-and-TD)

## Bellman 식 — Bellman equation

**쉬운 뜻:** 현재 가치와 다음 상태의 미래 가치를 연결하는 강화학습의 핵심 관계식.

관련: [가치 함수와 Bellman 식](Value-Functions-and-Bellman-Equation)

## 다음 상태 가치 이어받기 — bootstrap

**영어:** [다음 상태 가치 이어받기(bootstrap)](Replay-Buffer-and-Episode-Boundaries)

**쉬운 뜻:** 지금 정답 전체를 알지 못하므로 다음 상태의 추정 가치를 현재 학습 목표 계산에 사용하는 것.

관련: [Replay Buffer와 에피소드 경계](Replay-Buffer-and-Episode-Boundaries)

## bootstrap 경계

**쉬운 뜻:** 다음 상태가 실제 같은 에피소드의 연속이 아니므로 미래 값을 이어 붙이면 안 되는 지점.

예를 들어 요청 제한으로 강제 reset된 뒤의 새 상태 가치를 이전 에피소드에 연결하면 잘못된 학습이 될 수 있다.

관련: [Replay Buffer와 에피소드 경계](Replay-Buffer-and-Episode-Boundaries)

## 경험 저장소 — replay buffer

**쉬운 뜻:** 실제로 경험한 상태 전이를 저장했다가 학습할 때 다시 꺼내 쓰는 메모리.

관련: [Replay Buffer와 에피소드 경계](Replay-Buffer-and-Episode-Boundaries)

## 학습 손실 — loss

**영어:** [학습 손실(loss)](Loss-Functions-and-Class-Imbalance)

**쉬운 뜻:** 신경망 예측이 학습 목표와 얼마나 다른지 나타내는 최적화용 숫자.

환경의 보상과 같은 것이 아니다.

관련: [손실 함수와 데이터 불균형](Loss-Functions-and-Class-Imbalance)

---

# 7. AASSR의 Policy와 정보 가치

## Policy — 정책 모델

**쉬운 뜻:** 현재 상태만 보고 기본 행동을 선택하는 모델.

관련: [Policy](Policy)

## 정보 가치 잔차 — information-value residual

**쉬운 뜻:** 어떤 행동이 정보를 얻는 데 얼마나 도움이 되는지를 task Q값과 분리해 보는 내부 항목.

외부 보상에 중간 점수를 직접 추가하는 것과 다르다.

관련: [Policy](Policy)

## 행동 순위 — ranking

**영어:** [후보 순위(ranking)](Policy)

**쉬운 뜻:** 여러 행동 후보를 좋은 순서대로 정렬하는 것.

---

# 8. Knowledge와 기억

## Knowledge — 에피소드 내부 지식

**쉬운 뜻:** 이번 문제를 푸는 동안 실제 응답에서 이미 알아낸 사실을 보존하는 메모리.

관련: [Knowledge](Knowledge)

## 에피소드 — episode

**쉬운 뜻:** 환경이 초기화된 순간부터 성공·실패·강제 종료 등으로 한 번의 문제 풀이가 끝날 때까지의 구간.

관련: [강화학습](Reinforcement-Learning)

## 출처 정보 — provenance

**영어:** [정보의 출처 기록(provenance)](Knowledge)

**쉬운 뜻:** 어떤 지식을 언제, 어디서, 어떤 실제 응답을 통해 알게 되었는지 기록하는 정보.

[Knowledge(에피소드 지식)](Knowledge)에서 시간 순서를 지키는 데 중요하다.

---

# 9. 세계 모델과 Prophecy

## 세계 모델 — world model

**쉬운 뜻:** “이 상태에서 이 행동을 하면 다음에 어떤 일이 생길까?”를 예측하는 모델.

관련: [모델 기반 강화학습과 세계 모델](Model-Based-RL-and-World-Models)

## 모델 기반 강화학습 — model-based RL

**쉬운 뜻:** 환경의 다음 상태나 결과를 예측하는 모델을 이용해 행동을 고르는 강화학습 계열.

관련: [모델 기반 강화학습과 세계 모델](Model-Based-RL-and-World-Models)

## 환경 모델 없이 직접 가치 학습 — model-free RL

**쉬운 뜻:** 미래 환경 자체를 명시적으로 예측하기보다 정책이나 가치 함수를 직접 학습하는 강화학습 계열.

[DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)은 대표적인 예다.

관련: [강화학습](Reinforcement-Learning)

## Prophecy — 미래 예측 모델

**쉬운 뜻:** AASSR에서 다음 가능한 상태와 결과 확률을 예측하는 세계 모델 모듈.

관련: [Prophecy](Prophecy)

## 확률적 — stochastic

**쉬운 뜻:** 같은 상태와 행동에서도 결과가 항상 하나로 고정되지 않고 확률적으로 달라질 수 있다는 뜻.

관련: [확률·불확실성](Stochasticity-Uncertainty-and-Probability)

## 여러 결과의 혼합 — mixture

**영어:** mixture

**쉬운 뜻:** 같은 입력에서 여러 실제 결과 후보가 존재할 때 각각을 따로 유지하는 표현.

예:

```text
70% 정상
20% 접근 거부
10% 요청 제한
```

관련: [Mixture·Ensemble·Calibration](Mixture-Ensemble-and-Calibration)

## 여러 모델 묶음 — ensemble

**영어:** ensemble

**쉬운 뜻:** 하나의 모델만 믿지 않고 여러 학습 모델의 예측을 함께 보는 구조.

`mixture`와 다르다.

관련: [Mixture·Ensemble·Calibration](Mixture-Ensemble-and-Calibration)

## 다중 모드 분포 — multimodal distribution

**쉬운 뜻:** 확률 분포가 한 군데에만 몰리지 않고 서로 다른 여러 결과 영역에 나뉘어 있는 상태.

관련: [확률·불확실성](Stochasticity-Uncertainty-and-Probability)

## 결과 확률 — outcome probability

**쉬운 뜻:** 특정 환경 결과가 실제로 발생할 확률.

예측 신뢰도와 다르다.

---

# 10. 불확실성과 Calibration

## 불확실성 — uncertainty

**쉬운 뜻:** 모델이나 환경 결과를 얼마나 확실하게 알 수 없는지.

관련: [확률·불확실성](Stochasticity-Uncertainty-and-Probability)

## 지식 부족에서 오는 불확실성 — epistemic uncertainty

**쉬운 뜻:** 학습 데이터가 부족하거나 모델이 아직 충분히 배우지 못해서 생기는 불확실성.

더 적절한 데이터를 모으면 줄어들 수 있다.

관련: [확률·불확실성](Stochasticity-Uncertainty-and-Probability)

## Calibration — 예측 신뢰도 보정

**쉬운 뜻:** 모델이 “80% 확신”이라고 할 때 실제로도 비슷한 경우 약 80% 맞는지 확인하고 조정하는 과정.

관련: [Calibration](Calibration)

## 검증용 분리 데이터 — holdout

**쉬운 뜻:** 학습에 직접 사용하지 않고 모델 신뢰도나 성능을 확인하기 위해 따로 떼어 둔 실제 데이터.

관련: [Calibration](Calibration)

## 예측 신뢰도 — prediction reliability

**쉬운 뜻:** 모델의 예측이 이 상황에서 얼마나 믿을 만한지.

결과 자체의 발생 확률과 다르다.

---

# 11. Imagination과 계획

## Imagination — 가상 미래 탐색

**쉬운 뜻:** 실제 행동하기 전에 세계 모델을 이용해 여러 행동의 미래를 미리 펼쳐 보는 AASSR 모듈.

관련: [Imagination](Imagination)

## 계획 — planning

**쉬운 뜻:** 현재 행동 하나만 고르는 것이 아니라 미래 결과까지 고려해 행동 순서를 비교하는 과정.

관련: [반사실적 계획과 탐색](Counterfactual-Planning-and-Search)

## 반사실적 계획 — counterfactual planning

**쉬운 뜻:** “실제로 아직 하지 않았지만 만약 이 행동을 했다면?”이라는 가정의 미래를 비교하는 계획 방식.

관련: [반사실적 계획과 탐색](Counterfactual-Planning-and-Search)

## 탐색 깊이 — horizon

**영어:** horizon

**쉬운 뜻:** 미래를 몇 단계까지 상상할지 나타내는 범위.

깊을수록 장기 결과를 볼 수 있지만 예측 오차와 계산량도 커진다.

## 환경 결과 노드 — chance node

**쉬운 뜻:** 행동 후 환경이 어떤 결과를 낼지 갈리는 지점.

에이전트가 결과를 선택할 수 없으므로 확률 평균을 사용한다.

관련: [Chance와 Decision 노드](Chance-and-Decision-Nodes)

## 행동 선택 노드 — decision node

**쉬운 뜻:** 미래 상태에서 에이전트가 다음 행동을 고르는 지점.

가능한 행동 중 가장 좋은 것을 선택할 수 있다.

관련: [Chance와 Decision 노드](Chance-and-Decision-Nodes)

## 뿌리 행동 — root action

**쉬운 뜻:** 여러 단계 미래를 상상한 뒤 결국 지금 현실에서 처음 실행할 행동.

[Imagination(가상 미래 탐색)](Imagination)이 아무리 많은 미래를 계산해도 실제로는 [탐색의 첫 행동(root)](Imagination) [행동(action)](Reinforcement-Learning) 하나만 실행한다.

## 개입 — intervention

**쉬운 뜻:** [Imagination](Imagination)이 기본 [Policy(정책 모델)](Policy)가 고른 행동을 실제로 다른 행동으로 바꾼 경우.

후보를 계산한 것만으로는 [실제 행동 개입(intervention)](Imagination)이라고 하지 않는다.

## 개입 최소 차이 — intervention margin

**쉬운 뜻:** [Imagination](Imagination)이 기본 [Policy](Policy)를 바꾸려면 최소한 이 정도는 더 좋아야 한다고 정한 차이.

작은 예측 잡음 때문에 행동이 자주 뒤집히는 것을 줄인다.

---

# 12. Critic과 OOD

## Critic — 미래 가치 평가기

**쉬운 뜻:** 상상된 미래 상태가 최종 목표 관점에서 얼마나 좋은지 평가하는 모델.

관련: [Critic](Critic)

## GRU — Gated Recurrent Unit

**한국어:** 게이트 순환 유닛.

**쉬운 뜻:** 과거 순서 정보를 내부 상태에 보존하면서 입력을 처리하는 순환 신경망 구조.

관련: [GRU와 순차 모델](GRU-and-Sequence-Models)

## 학습 분포 밖 — OOD, Out-of-Distribution

**쉬운 뜻:** 모델이 학습할 때 충분히 보지 못한 종류의 상태나 행동 영역.

신경망은 [학습 분포 밖(OOD)](Critic-Support-and-OOD)에서도 숫자를 내지만 그 숫자가 믿을 만하다는 보장은 없다.

관련: [Critic Support와 OOD](Critic-Support-and-OOD)

## 국소 데이터 근거 — local support

**쉬운 뜻:** 지금 평가하려는 상태·행동 주변에 실제 학습 데이터가 충분히 있는지 나타내는 근거.

관련: [Critic Support와 OOD](Critic-Support-and-OOD)

## Critic 준비 상태 — Critic readiness

**쉬운 뜻:** [Critic(미래 가치 평가기)](Critic)이 전체적으로 최소한 사용할 만큼 학습됐는지 보는 조건.

전체적으로 준비됐다는 것과 특정 [OOD](Critic-Support-and-OOD) 상태에서 믿을 만하다는 것은 다르다.

## 보수적 실패 — fail closed

**쉬운 뜻:** 신뢰할 근거가 부족할 때 공격적으로 새 결정을 하지 않고 더 안전한 기본 경로로 돌아가는 설계.

AASSR의 [국소 데이터 근거(local support)](Critic-Support-and-OOD) [판정 관문(gate)](Terminology-Guide)는 근거가 부족하면 [Policy](Policy)를 유지한다.

---

# 13. Skill과 재사용

## Skill — 재사용 가능한 성공 절차

**쉬운 뜻:** 반복해서 성공한 행동 구조를 새 문제에 다시 적용하기 위한 패턴.

관련: [Skills](Skills)

## 계층형 강화학습 — hierarchical RL

**쉬운 뜻:** 아주 작은 행동만 매번 고르는 대신 여러 행동을 묶은 상위 수준의 절차나 기술도 다루는 강화학습.

관련: [계층형 강화학습과 Skill](Hierarchical-RL-and-Skills)

## 재연결 — rebind

**영어:** rebind

**쉬운 뜻:** 과거 성공 구조의 역할을 새 문제의 실제 객체에 다시 연결하는 것.

---

# 14. 실험 설계 용어

## 비교 기준 — baseline

**쉬운 뜻:** 새 방법이 실제로 좋은지 비교하기 위한 기준 모델.

관련: [Ablation·Benchmark·재현성](Ablation-Benchmarking-and-Reproducibility)

## 구성요소 제거 비교 — ablation

**쉬운 뜻:** 전체 시스템에서 특정 부품만 끄거나 바꿔 그 부품의 효과를 따로 확인하는 실험.

관련: [Ablation·Benchmark·재현성](Ablation-Benchmarking-and-Reproducibility)

## 표준 비교 실험 — benchmark

**쉬운 뜻:** 같은 조건에서 여러 방법을 일정한 규칙으로 비교하는 시험.

관련: [Ablation·Benchmark·재현성](Ablation-Benchmarking-and-Reproducibility)

## 평가지표 — metric

**쉬운 뜻:** 성공률, 실패율, 예측 오차처럼 실험 결과를 숫자로 나타내는 기준.

## 연구 seed — research seed

**쉬운 뜻:** 모델 초기화와 난수 효과를 여러 번 반복해 보기 위해 사용하는 난수 시작값.

한 [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility) 결과만으로 일반적인 결론을 내리지 않기 위해 여러 난수 시드를 쓴다.

## 최종 비공개 평가 — final blind

**쉬운 뜻:** 모델과 실험 규칙을 완전히 고정한 뒤 처음 사용하는 최종 평가 세트.

결과를 본 뒤 다시 튜닝하는 일을 막는다.

## 재현성 — reproducibility

**쉬운 뜻:** 다른 사람이 같은 코드·설정·난수 시드로 비슷한 실험 결과를 다시 만들 수 있는 성질.

관련: [재현 방법](Reproduction)

## 체크포인트 — checkpoint

**쉬운 뜻:** 특정 시점의 학습된 모델 파라미터를 저장한 파일.

## 동결 — frozen

**쉬운 뜻:** 평가하는 동안 모델 파라미터나 학습 상태를 더 이상 바꾸지 않는 것.

## 같은 체크포인트 비교 — same-checkpoint comparison

**쉬운 뜻:** 두 조건이 서로 다른 학습 결과를 쓰지 않고 동일한 학습 모델을 공유한 채 한 기능만 켜고 끄는 비교.

AASSR에서는 [Imagination](Imagination) 자체의 추가 효과를 보려 할 때 중요하다.

---

# 15. 연구 증거 용어

## 증거 — evidence

**쉬운 뜻:** 어떤 주장을 뒷받침하는 실험 결과·테스트·진단 기록.

AASSR에서는 증거 수준을 구분한다.

```text
구현됨
< 회귀 테스트
< 메커니즘 진단
< 작은 실제 실험
< 여러 seed 비교
< 최종 비공개 평가
```

관련: [Evidence Matrix](Evidence-Matrix)

## 메커니즘 증거 — mechanism evidence

**쉬운 뜻:** 특정 기능이 의도한 현상을 실제로 만들었다는 증거.

예를 들어 [ASEQ](ASEQ)가 제자리 반복을 줄였다는 결과는 [ASEQ](ASEQ)의 메커니즘 증거가 될 수 있다.

## 성능 증거 — performance evidence

**쉬운 뜻:** 최종 성공률 같은 실제 성능이 비교군보다 좋아졌다는 증거.

메커니즘이 작동한다는 것과 성능이 좋아진다는 것은 같은 주장이 아니다.

## 진단 실험 — diagnostic

**쉬운 뜻:** 최종 성능을 발표하기보다 어떤 문제가 왜 생기는지 원인을 찾기 위한 실험.

## 현재 버전 — current

**쉬운 뜻:** 지금 실제 연구에서 활성화된 구조.

## 과거 기록 — historical

**쉬운 뜻:** 이전 버전의 구조·실험·실패 기록.

현재 성능과 섞어 쓰지 않는다.

## 주장 범위 — claim boundary

**쉬운 뜻:** 현재 증거로 어디까지 말할 수 있고 어디부터는 아직 말하면 안 되는지 정한 경계.

관련: [Evidence Matrix](Evidence-Matrix)

---

# 16. 코드·GitHub에서 나오는 단어

## 원본 기준 — source of truth

**쉬운 뜻:** 문서나 코드 설명이 서로 충돌할 때 최종적으로 믿어야 하는 기준 파일.

현재 AASSR 실행 구조의 원본 기준은:

```text
src/aassr_v2/current_manifest.py
```

이다.

## 브랜치 — branch

**쉬운 뜻:** 기존 코드를 바로 덮어쓰지 않고 별도 작업 흐름에서 수정하기 위한 Git의 분기.

## 커밋 — commit

**쉬운 뜻:** 코드나 문서 변경을 하나의 기록 단위로 저장한 것.

## PR — Pull Request

**쉬운 뜻:** 한 브랜치의 변경을 다른 브랜치에 합치기 전에 검토하는 GitHub 단위.

## 머지 — merge

**쉬운 뜻:** 분리된 변경사항을 대상 브랜치에 합치는 것.

## CI — Continuous Integration

**한국어:** 지속적 통합.

**쉬운 뜻:** 코드나 문서를 수정할 때 자동으로 테스트를 실행해 문제가 없는지 확인하는 시스템.

## 회귀 테스트 — regression test

**쉬운 뜻:** 이전에 제대로 되던 기능이 새 수정 때문에 다시 깨지지 않았는지 확인하는 테스트.

## 런타임 — runtime

**쉬운 뜻:** 문서에만 존재하는 설계가 아니라 실제 실행될 때 사용되는 코드 구조.

## manifest

**쉬운 뜻:** 현재 어떤 구성요소가 활성화돼 있는지 목록처럼 기록한 파일.

AASSR의 `current_manifest.py`가 이에 해당한다.

---

# 17. 성능·계산 용어

## 배치 처리 — batching

**쉬운 뜻:** 작은 계산을 하나씩 여러 번 하지 않고 여러 입력을 한꺼번에 묶어 CPU나 GPU에서 계산하는 방법.

## GPU

**쉬운 뜻:** 많은 수치 계산을 병렬로 빠르게 처리하는 하드웨어. 신경망 학습과 추론에 자주 사용한다.

## CPU

**쉬운 뜻:** 일반 프로그램 실행과 다양한 종류의 순차 계산을 담당하는 중앙 처리 장치.

## 동기화 — synchronization

**쉬운 뜻:** GPU 계산 결과를 CPU가 기다리거나 서로의 작업 완료를 맞추는 과정.

너무 자주 발생하면 GPU가 빨라도 전체 프로그램이 느려질 수 있다.

## 처리량 — throughput

**쉬운 뜻:** 일정 시간 동안 얼마나 많은 계산이나 상태 전이를 처리할 수 있는지.

## 실행 시간 — wall-clock time

**쉬운 뜻:** 실제 시계로 측정했을 때 실험이 끝날 때까지 걸린 시간.

---

# 18. 종료 관련 용어

## 종료 상태 — terminal

**쉬운 뜻:** 하나의 에피소드가 끝나는 상태.

## 실제 실패 — true failure

**쉬운 뜻:** 환경의 목표 규칙상 정말로 실패한 경우.

## 외부 제한 종료 — truncation

**쉬운 뜻:** 목표상 성공/실패가 결정된 것이 아니라 시간 제한, 요청 제한, 상태 전이 budget 같은 외부 이유로 에피소드가 끊긴 경우.

## 요청 제한 — rate limit

**쉬운 뜻:** 너무 많은 요청 등으로 환경이 더 이상 요청을 허용하지 않는 상황.

이런 종료를 task [실패(failure)](Replay-Buffer-and-Episode-Boundaries)와 같은 `-1`로 취급하면 학습 의미가 달라질 수 있다.

관련: [Replay Buffer와 에피소드 경계](Replay-Buffer-and-Episode-Boundaries)

---

# 19. AASSR 고유 모듈 빠른 번역표

| 영어 이름 | 이 위키에서 먼저 떠올릴 한국어 | 핵심 질문 |
|---|---|---|
| [State Representation](State-Representation) | 상태를 학습용으로 정리하는 방법 | 지금 상황을 어떤 관점으로 볼까? |
| [ASEQ](ASEQ) | 실제 상태-행동-다음 상태 기록 | 방금 이 행동이 제자리 반복이었나? |
| [Policy](Policy) | 기본 정책 모델 | 지금 당장 무엇을 할까? |
| [Knowledge](Knowledge) | 이번 문제에서 얻은 지식 기억 | 지금까지 실제로 무엇을 알게 됐나? |
| [Prophecy](Prophecy) | 미래 예측 모델 | 이 행동 뒤에 어떤 결과들이 가능한가? |
| [Calibration](Calibration) | 예측 신뢰도 보정 | 그 예측을 얼마나 믿어도 되나? |
| [Imagination](Imagination) | 가상 미래 탐색 | 실제로 하기 전에 여러 미래를 비교할 수 있나? |
| [Critic](Critic) | 미래 가치 평가기 | 그 미래가 최종 목표에 좋은가? |
| [Critic Support](Critic-Support-and-OOD) | 실제 데이터 근거 확인 | 그 가치 평가를 뒷받침할 경험이 있나? |
| [Skills](Skills) | 성공 절차 재사용 | 전에 배운 성공 구조를 새 문제에 쓸 수 있나? |

---

# 20. 비슷해 보여도 다른 것

```text
보상(reward)
!= 누적 보상(return)
!= Q값(Q-value)
```

```text
환경 결과 확률(outcome probability)
!= 예측 신뢰도(prediction reliability)
```

```text
Prophecy의 예측 신뢰도
!= Critic 값의 실제 데이터 근거(local support)
```

```text
환경의 실제 상태
!= 에이전트 관측
!= 신경망 입력 표현
```

```text
여러 환경 결과를 표현하는 mixture
!= 여러 학습 모델을 사용하는 ensemble
```

```text
실제 경험(real transition)
!= 세계 모델이 만든 가상 경험(imagined transition)
```

```text
에피소드 종료(terminal)
!= 실제 실패(true failure)
!= 외부 제한 종료(truncation)
```

---

# 더 찾아보기

- [처음 읽는 사람을 위한 안내서](Beginner-Guide)
- [개념 지도](Concept-Index)
- [짧은 용어 사전](Glossary)
- [현재 연구 상태](Current-Status)
- [연구 질문](Research-Questions)
- [실험](Experiments)
