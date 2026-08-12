# Glossary

AASSR 위키에서 반복해서 등장하는 용어를 짧게 정리한다.

| 용어 | 의미 |
|---|---|
| **State (`S`)** | 에이전트가 현재 상황을 표현한 상태 |
| **Action (`A`)** | 에이전트가 선택해 실제 또는 imagined environment에 적용하는 행동 |
| **Next State (`S'`)** | 행동 이후의 상태 |
| **ASEQ** | `(S, A, S')` 형태의 실제 경험 단위 |
| **Self-loop** | `S -> A -> S`처럼 행동 후 의미 상태가 바뀌지 않는 전이 |
| **Concrete semantic state** | 한 episode 안에서 실제 entity를 구분하는 task-relevant 상태 identity |
| **Relational state** | 구체 ID보다 role/관계를 표현해 seed rename에 강한 상태 |
| **Rename invariance** | route/object 이름이 바뀌어도 구조가 같으면 같은 표현을 만드는 성질 |
| **Policy** | 현재 상태에서 가능한 행동의 가치를 평가하고 후보를 고르는 모델 |
| **DQN** | Q-learning을 신경망으로 근사하는 model-free RL 방법 |
| **Information-value residual** | 즉시 reward 외에 이후 의사결정에 유용한 정보를 여는 행동의 가치를 별도로 학습하는 AASSR Policy 항 |
| **Knowledge** | 현재 episode에서 응답을 통해 얻은 정보 문맥 |
| **Prophecy** | `(state, action)`으로 가능한 다음 상태를 예측하는 AASSR world model |
| **World Model** | 환경 전이를 내부적으로 예측하는 모델 |
| **Stochastic future** | 같은 공개 상태/행동에서도 여러 결과가 확률적으로 가능하다는 표현 |
| **Mixture model** | 여러 가능한 outcome mode를 하나의 평균으로 뭉개지 않고 확률 질량으로 표현하는 모델 |
| **Outcome probability** | 특정 환경 결과가 일어날 확률 질량 |
| **Reliability / Confidence** | 해당 예측을 모델이 얼마나 믿을 수 있는지 나타내는 신뢰도 |
| **Calibration** | 모델 confidence가 실제 예측 정확도와 맞는지 holdout data로 측정/보정하는 과정 |
| **Imagination** | Prophecy를 여러 단계 이어 붙여 실제로 실행하기 전에 여러 미래를 탐색하는 과정 |
| **Chance node** | 어떤 환경 outcome이 나오는지 agent가 고를 수 없는 노드 |
| **Decision node** | 다음 행동을 agent가 선택할 수 있는 노드 |
| **Chance backup** | 여러 environment outcome을 probability-weighted expectation으로 합치는 연산 |
| **Decision backup** | agent가 선택 가능한 future action 중 최대 가치를 취하는 연산 |
| **Critic** | imagined future가 실제 sparse task return 관점에서 얼마나 좋은지 평가하는 모델 |
| **Critic readiness** | Critic이 최소한의 학습 상태에 도달했는지 나타내는 global 조건 |
| **Local Critic support** | 현재 state/action이 실제 Critic training distribution에서 충분히 지원되는지 나타내는 local 조건 |
| **Fail closed** | 신뢰 조건이 충족되지 않으면 공격적으로 행동하지 않고 기존 Policy 쪽으로 안전하게 fallback하는 방식 |
| **Policy override** | Imagination/Critic 판단이 fixed margin을 넘었을 때 Policy가 원래 고른 행동을 다른 행동으로 바꾸는 것 |
| **Intervention** | 실제 실행 행동이 Policy 원래 선택과 달라진 Imagination 개입 |
| **Intervention margin** | Imagination이 Policy를 바꾸기 위해 넘어야 하는 최소 가치 차이 |
| **Root action** | 현재 real decision에서 바로 실행할 수 있는 첫 행동 |
| **Structural root deduplication** | concrete ID만 다르고 relational 구조가 같은 root를 한 번만 계산하는 최적화 |
| **Legal-action mask** | 현재 상태에서 가능한 구조적 행동 슬롯을 표시한 mask |
| **Skill** | 반복적으로 성공한 ASeq를 relational template 형태로 재사용하는 후보 |
| **Sparse Reward** | 대부분의 transition reward가 0이고 성공/실패 끝점에서만 강한 reward가 나오는 설정 |
| **True failure** | 실제 lockout처럼 task가 실패로 종료된 terminal outcome; 현재 reward는 `-1` |
| **Truncation** | rate-limit, transition cap 등으로 episode가 끊겼지만 task failure로 보지 않는 종료; reward는 `0` |
| **Stall** | 의미 있는 진전 없이 반복 행동이나 정체가 이어지는 상태 |
| **Curriculum** | 쉬운 단계부터 어려운 단계로 exposure를 조절하는 학습 구조 |
| **Research seed** | 모델 초기화/실험 반복을 구분하는 최상위 seed |
| **Scenario seed** | route/object/session 등 환경 instance를 결정하는 seed |
| **Unseen evaluation** | 학습에 사용하지 않은 scenario seed에서의 평가 |
| **Same-checkpoint evaluation** | 하나의 frozen checkpoint를 여러 조건에서 동일하게 재사용하는 비교 |
| **Ablation** | 특정 구성요소를 제거하거나 바꿔 효과를 분리하는 실험 |
| **Baseline** | 새 방법과 비교하기 위한 기준 모델/규칙 |
| **Raw DQN** | raw v3 representation을 사용하는 corrected DQN baseline |
| **Relational DQN** | AASSR과 동일한 relational representation을 쓰지만 AASSR 모듈은 없는 DQN baseline |
| **DreamerV3** | learned world model과 latent imagination을 사용하는 외부 model-based RL baseline |
| **Final blind** | 방법론을 고정한 뒤에만 한 번 사용하는 최종 미소비 test seed set |
| **Response-causal** | agent가 실제 응답 시점에 알 수 있는 정보만 observation에 허용하는 원칙 |
| **Hidden leakage** | simulator 정답, 미래 정보, hidden pressure 등 agent가 알 수 없어야 할 정보가 입력에 섞이는 문제 |
| **Holdout** | 학습 업데이트와 분리해 calibration/validation에 사용하는 전이 집합 |
| **Persistent learning state** | replay, neural weights, FeatureMemory, Skill reliability처럼 episode/evaluation 뒤에도 남는 학습 상태 |

---

## 자주 헷갈리는 세 쌍

### ASEQ vs Skill

```text
ASEQ guard
= 실패/무진전 반복을 기억해 피함

Skill
= 반복 성공 sequence를 재사용 가능한 template로 묶음
```

### Outcome probability vs Reliability

```text
Outcome probability
= 실제 환경이 그 결과를 만들 확률

Reliability
= 모델이 자신의 예측을 믿을 수 있는 정도
```

### Critic ready vs Critic locally supported

```text
Critic ready
= Critic이 전반적으로 학습은 됨

Local support
= 지금 이 state/action도 실제 training distribution에 가까움
```

둘을 같은 것으로 보면 unseen 상태에서 과신 intervention이 생길 수 있다.
