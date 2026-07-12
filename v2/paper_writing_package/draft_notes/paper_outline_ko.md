# 논문 초안 구조

## Tentative Title

```text
AASSR: Knowledge-Parameterized Decision Making with Prophecy-Guided Imagination
```

한국어 설명 제목:

```text
지식 기반 행동 파라미터화와 예측 기반 상상 평가를 이용한 폐루프 의사결정 구조
```

## Abstract 초안 재료

본 연구는 관측 지식을 단순히 저장하는 것을 넘어, 이후 행동 생성의
파라미터 후보로 재사용하는 AASSR/APASSR 구조를 제안한다. 제안 구조에서
Knowledge Storage는 KK/KV 형태의 typed parameter pool로 동작하며,
행동 템플릿은 Knowledge Storage의 KV를 KK 슬롯에 바인딩하여 실행 가능한
행동 후보를 만든다. 또한 Prophecy Module은 후보 행동이 만들 수 있는
지식 변화, 오류 가능성, 목표 관련성을 예측하고, Imagination Cycle은 이
예측을 이용해 실행 전 후보 행동을 평가한다. 본 연구는 원래 펜테스팅
자동화에서 출발한 지식-행동 의존성 아이디어를 GridWorld 추상 환경에서
검증하였다. 실험 결과, C3 조건은 v2_complex 환경에서 Q-learning 및
partial-observation DQN baseline보다 높은 성공률과 낮은 반복률을 보였다.
추가 ablation은 현재 구현에서 반복 억제와 오류 회피가 중요한 역할을
한다는 점을 보였다.

## 1. Introduction

### 문제의식

- 많은 의사결정 문제에서는 행동이 단순 primitive action 선택이 아니다.
- 특히 펜테스팅에서는 명령어가 여러 파라미터로 구성된다.
- 예: `nmap {OPTION} {PORT} {TARGET_IP}`
- `TARGET_IP`, `PORT`, `SERVICE`, `PATH` 같은 값은 이전 관측에서 얻어진다.
- 따라서 좋은 에이전트는 관측 지식을 저장할 뿐 아니라, 그 지식을 다음
  행동의 파라미터로 재사용해야 한다.

### 기존 방식의 한계

- Random policy는 희소 보상 환경에서 비효율적이다.
- 단순 policy learning은 행동 후보가 지식에 의해 어떻게 생성되는지
  충분히 반영하지 못한다.
- 일반적인 RL baseline은 충분한 학습 budget에서는 강할 수 있지만,
  지식-행동 파라미터 의존성을 명시적으로 설명하기 어렵다.

### 제안

```text
Action -> Observation -> Knowledge Update -> Parameter Binding -> Next Action
```

핵심 문장:

```text
행동이 지식을 만들고, 지식이 다음 행동을 만든다.
```

### Contribution

1. Knowledge Storage를 관측 기록소가 아니라 행동 파라미터 공급원으로 정의.
2. KK/KV 기반 action template binding 구조 구현.
3. Prophecy Module과 Imagination Cycle을 DMP 폐루프에 결합.
4. GridWorld 추상 환경에서 C0-C4 및 QLEARN/DQN/ORACLE baseline 비교.
5. Table/Transformer Prophecy, reward ON/OFF, rollout depth/branch,
   mechanism/component ablation 수행.

## 2. Background and Motivation

### Original Motivation: Pentesting

원래 문제는 펜테스팅 자동화이다. 펜테스팅 명령은 보통 다음처럼
파라미터화된다.

```text
nmap {WHAT_OPTION} {HOW_OPTION} {PORT} {TARGET_IP}
```

이때 `PORT`, `TARGET_IP`, `SERVICE`, `PATH` 등은 관측에서 얻은 지식이다.
따라서 Knowledge Storage는 단순 로그가 아니라 이후 명령어 생성에 사용될
parameter pool이다.

### GridWorld Abstraction

GridWorld는 펜테스팅을 직접 대체하지 않는다. 대신 다음 구조를 추상화한다.

| Pentesting | GridWorld |
| --- | --- |
| target IP | target cell |
| port/service | key/door/hint/flag cell |
| command template | action template |
| scan result | cell observation |
| discovered value | KV under KK |

예:

```text
MOVE_TOWARD {KK_FRONTIER_CELL}
INSPECT_CELL {KK_UNKNOWN_NEIGHBOR}
USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}
FOLLOW_HINT {KK_HINT_VALUE}
```

## 3. Method

### 3.1 Knowledge Storage

Knowledge Storage는 KK/KV 구조를 사용한다.

- KK: action template이 요구하는 추상 슬롯
- KV: 해당 슬롯에 바인딩되는 구체 값

예:

```text
KK_FRONTIER_CELL = [(3,4), (4,4), (5,2)]
MOVE_TOWARD {KK_FRONTIER_CELL}
-> MOVE_TOWARD (4,4)
```

### 3.2 Action Template Binding

행동 생성은 다음 순서로 이루어진다.

```text
1. 사용 가능한 행동 템플릿 확인
2. 템플릿이 요구하는 KK 슬롯 확인
3. Knowledge Storage에서 KV 후보 검색
4. KV를 슬롯에 바인딩
5. 실행 가능한 ActionCandidate 생성
```

### 3.3 PolicyABC

PolicyABC는 행동 선택을 WHAT/HOW/WHERE로 나눈다.

```text
WHAT = 사용할 행동 템플릿
HOW = 후보 선택 전략
WHERE = 사용할 KK pool
```

### 3.4 Prophecy Module

Prophecy Module은 후보 행동에 대해 다음을 예측한다.

```text
predicted semantic ΔK
predicted error probability
predicted flag/goal relevance
```

중요한 점:

```text
Prophecy Module은 framework 개념이고,
TableProphecyModel, SequenceProphecyModel, TransformerProphecyModel은 구현 변형이다.
```

### 3.5 Imagination Cycle

Imagination Cycle은 실행 전 후보 행동을 평가한다.

현재 구현은 hidden map을 읽거나 실제 미래 행동을 실행하지 않는다. Prophecy
예측을 이용한 depth-limited candidate evaluation이다.

평가 요소:

```text
expected knowledge gain
flag probability
error probability
repeat penalty
policy prior
rollout value
dependency bonus
```

### 3.6 DMP Closed Loop

전체 루프:

```text
Knowledge Storage
-> KK/KV slot binding
-> executable candidate generation
-> PolicyABC selection
-> Prophecy prediction
-> Imagination candidate scoring
-> action execution
-> semantic ΔK
-> reward / prophecy update
-> policy update
```

## 4. Experimental Setup

### Environments

| Environment | Description |
| --- | --- |
| random_key_door | 기본 key-door-hint-flag 구조 |
| v2_complex | 9x6 randomized map, walls, multiple keys/doors/hints |
| locked_bottleneck | mandatory door bottleneck dependency stress test |

### Conditions

| Condition | Meaning |
| --- | --- |
| C0 | RandomScorer |
| C1 | PolicyABC |
| C2 | PolicyABC + Prophecy reward |
| C3 | PolicyABC + Prophecy Module + Imagination Cycle |
| C4 | Optional sequence-based Prophecy variant |
| QLEARN | Tabular Q-learning |
| DQN_PARTIAL | Partial-observation DQN over knowledge/candidate features |
| ORACLE_MDP | Full-map shortest-path upper bound |

### Metrics

- success rate
- steps to flag, successful episodes only in summary table
- repeat rate
- error rate
- semantic ΔK
- prophecy error
- total reward

## 5. Results

### Main Comparison

핵심 결과는 v2_complex 30x10에서 C3가 QLEARN/DQN_PARTIAL보다 성공률이 높고
반복률이 낮았다는 것이다.

논문에 넣을 문장:

```text
In the tested v2_complex setting, C3 achieved higher success and lower repeat
rate than QLEARN and DQN_PARTIAL, suggesting that knowledge-parameterized
candidate generation and Prophecy-guided Imagination can improve exploration
efficiency in structured partial-observation tasks.
```

### Locked Bottleneck

locked_bottleneck에서는 QLEARN/DQN이 C3보다 좋을 수 있다. 이는 실패가
아니라 환경 의존성을 보여주는 결과로 해석한다.

안전한 해석:

```text
APASSR does not dominate all baselines in all environments. Its advantage is
most visible in settings where knowledge reuse and invalid-action reduction are
central to exploration.
```

## 6. Ablation Studies

### A1: Prophecy Implementation

TableProphecyModel이 TransformerProphecyModel보다 안정적이었다. 이는 현재
성능이 복잡한 neural architecture 때문이 아님을 보여준다.

### A2: Prediction-Error Reward

Prediction-error reward는 단순 환경에서는 큰 차이가 없지만 복잡 환경에서는
보조 신호로 도움이 되었다.

### A3: Imagination Depth/Branch

얕은 rollout은 도움이 되었지만 branch 수를 늘리는 것은 일관된 향상을
만들지 않았다.

### A4: Imagination Mechanisms

가장 중요한 발견:

```text
NO_REPEAT_PENALTY가 크게 성능을 떨어뜨렸다.
```

즉 반복 행동 억제는 현재 구현에서 핵심적이다.

### A5: Prophecy Score Components

가장 중요한 발견:

```text
NO_ERROR_AVOIDANCE가 복잡 환경에서 크게 성능을 떨어뜨렸다.
NO_KNOWLEDGE_GAIN은 오히려 좋아졌다.
```

즉 무조건 지식을 많이 얻는 것보다 오류를 피하고 반복을 줄이는 것이 더
중요했다.

## 7. Discussion

### 핵심 해석

```text
APASSR의 핵심은 더 많은 지식을 모으는 것이 아니라,
행동 가능한 지식을 이용해 후보 행동을 줄이고,
반복과 오류를 줄이는 것이다.
```

### C5 Future Direction

현재 결과에 기반한 개선형:

```text
C5 = C3
   + lower or zero knowledge_weight
   + zero policy_prior_weight
   + repeat penalty retained
   + error avoidance retained
```

이것은 본 논문의 main method가 아니라 future work 또는 improved variant로
다룬다.

## 8. Limitations

- GridWorld는 펜테스팅의 직접 대체 실험이 아니다.
- DQN은 큰 학습 budget에서 더 좋아질 수 있다.
- 현재 TransformerProphecyModel은 작은 NumPy attention implementation이며
  대규모 neural model이 아니다.
- 현재 Imagination은 true environment rollout이 아니다.
- C3 성능은 환경 구조에 따라 달라진다.

## 9. Conclusion

마무리 문장 초안:

```text
This study supports the view that knowledge storage should be treated not only
as memory, but also as a source of action parameters. The proposed AASSR loop
connects observation, typed knowledge storage, action-template binding,
prediction, imagination, execution, and knowledge update. Experiments in
controlled GridWorld environments show that this structure can improve
exploration efficiency in tasks with knowledge-action dependencies, especially
by reducing repeated and error-prone actions.
```
