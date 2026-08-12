# Research Questions

AASSR의 출발점은 특정 모듈을 만드는 것이 아니라 **희소 보상 환경에서 에이전트가 어떻게 스스로 장기 행동 구조를 만들어낼 수 있는가**라는 질문이다.

이 페이지는 AASSR의 연구 질문을 가장 상위 수준에서 정리하고, 각 질문이 어떤 설계와 실험으로 이어지는지를 연결한다.

---

## 1. 핵심 연구 질문

> **중간 보상이 거의 없고, 가능한 행동이 많으며, 환경을 완전히 관찰할 수 없는 상황에서 에이전트가 경험의 구조를 학습하고 미래 결과를 예측하여 스스로 최종 목표에 도달하는 행동 과정을 만들 수 있는가?**

초기 연구 노트에서는 이를 더 철학적으로 다음처럼 표현했다.

> **최종 목표만 존재하는 희소 보상 환경에서 에이전트가 인간이 미리 정해준 경로 없이 목표 수행 과정을 스스로 만들어낼 수 있는가?**

두 문장은 같은 문제를 다른 수준에서 표현한다.

- 첫 번째는 실험 가능한 연구 질문이다.
- 두 번째는 AASSR이 지향하는 장기적인 연구 철학이다.

---

# 2. 왜 이 질문이 어려운가?

일반적인 dense-reward 환경에서는 행동 직후 비교적 자주 학습 신호를 얻는다.

```text
행동 -> +0.1
행동 -> +0.2
행동 -> -0.1
행동 -> +0.5
```

그러나 AASSR이 겨냥하는 환경은 다음에 더 가깝다.

```text
A0 -> A1 -> A2 -> A3 -> A4 -> A5 -> success
 0     0     0     0     0       +1
```

또는 실제 실패가 있는 경우:

```text
success       +1
true failure  -1
otherwise      0
```

이때 에이전트는 단순히 직전 reward만 보고는 다음을 판단하기 어렵다.

- 어떤 행동이 몇 단계 뒤 성공에 기여했는가?
- 현재 행동이 진짜 진행인지 self-loop인지?
- 지금 얻은 정보가 나중에 어떤 행동에 필요한가?
- 같은 구조인데 이름만 바뀐 새로운 문제를 같은 문제로 알아볼 수 있는가?
- 실제로 실패하기 전에 위험한 경로를 추정할 수 있는가?

AASSR의 각 구성 요소는 이 하위 문제 중 하나 이상을 겨냥한다.

---

# 3. 연구 질문의 분해

## RQ1. 희소 보상만으로 최초 성공 경험을 만들 수 있는가?

> **guided trajectory, oracle action injection, intermediate shaping reward 없이도 에이전트가 스스로 최초 성공을 발견할 수 있는가?**

이 질문은 AASSR 전체 연구의 출발점이다.

관련 요소:

- curriculum
- exploration
- Policy
- ASEQ

관련 실험:

- autonomous first-proof pilot
- curriculum promotion / demotion
- exact ASEQ retraining

---

## RQ2. 상태를 관계 구조로 표현하면 일반화가 좋아지는가?

> **concrete identifier를 직접 외우는 대신 역할과 관계를 표현하면 seed가 바뀐 unseen 환경으로 더 잘 transfer할 수 있는가?**

예:

```text
Scenario A: route-12 = catalog-like route
Scenario B: route-31 = catalog-like route

concrete identity: route-12 != route-31
relational role  : same structural role
```

관련 요소:

- relational public state v3
- relational action features
- relational DQN
- relational Prophecy
- relational Critic

핵심 비교:

```text
dqn_raw -> dqn_relational
```

이 차이는 representation 효과를 분리하기 위한 control이다.

---

## RQ3. 실제 경험 `(S,A,S')`을 이용해 진전 없는 반복을 줄일 수 있는가?

> **같은 semantic state에서 같은 행동을 반복해 다시 같은 상태로 돌아오는 self-loop만 억제하면 탐색 효율이 개선되는가?**

ASEQ는 다음 실제 transition이다.

```text
(S, A, S')
```

AASSR은 좁게 다음 패턴만 억제한다.

```text
S -> A -> S
```

반면 실제 상태 변화가 있으면 반복을 허용한다.

```text
S -> A -> S'
S' != S
```

관련 페이지: **[ASEQ](ASEQ)**

---

## RQ4. 미래를 예측하는 world model이 행동 선택에 유용한가?

> **현재 public state와 행동으로부터 가능한 다음 상태의 분포를 학습하면 장기 의사결정에 도움이 되는가?**

현재 Prophecy는 단일 next-state를 예측하지 않는다.

```text
P(S' | S, A, K)
```

여러 가능한 결과를 conditional mixture로 표현한다.

예측 대상에는 다음이 포함된다.

- next relational descriptor
- latest public HTTP status
- legal action mask
- active / success / failure / truncation
- outcome probability

관련 페이지: **[Prophecy](Prophecy)**

---

## RQ5. 예측을 여러 단계 연결한 Imagination이 Policy보다 좋은 결정을 만들 수 있는가?

> **실제 행동 전에 여러 counterfactual future를 계산하면 같은 checkpoint의 Policy보다 더 좋은 첫 행동을 선택할 수 있는가?**

이 질문은 반드시 같은 checkpoint로 비교한다.

```text
one AASSR training run
        |
        v
frozen checkpoint
      /    \
     /      \
OFF eval  ON eval
```

따라서

```text
AASSR no-Imagination -> AASSR Full
```

의 차이가 Imagination의 marginal effect다.

관련 페이지: **[Imagination](Imagination)**

---

## RQ6. world model의 confidence만으로 충분한가?

> **예측이 신뢰 가능하더라도 Critic이 현재 state/action region을 실제로 학습한 적 없다면 그 값을 믿어도 되는가?**

최근 실험에서 답은 **아니었다.**

그래서 AASSR은 두 신뢰 개념을 분리한다.

```text
Prophecy reliability
= world model prediction을 믿을 수 있는가?

Critic local support
= 현재 영역의 value estimate를 실제 training data가 지지하는가?
```

둘 중 하나라도 부족하면 Imagination override는 fail-closed 된다.

---

## RQ7. AASSR의 효과는 어떤 구성 요소에서 오는가?

최종 current-generation 비교는 다음 구조를 사용한다.

| Condition | 목적 |
|---|---|
| `dqn_raw` | 단순 model-free baseline |
| `dqn_relational` | representation 효과 |
| `dreamerv3_relational` | 표준 world-model RL baseline |
| `aassr_current_no_imagination` | AASSR stack의 non-Imagination 효과 |
| `aassr_current_full` | Imagination marginal effect |

따라서 단순히 "AASSR이 DQN보다 높다"가 아니라 **어느 층에서 차이가 발생하는가**를 분리해서 본다.

---

# 4. 장기 연구 질문: 창의성

초기 AASSR 노트의 두 번째 큰 질문은 다음이었다.

> **에이전트가 인간이 미리 정해준 정답 경로와 다른 유효한 목표 수행 경로를 만들어낼 수 있는가?**

이 질문은 흥미롭지만 현재 단계에서는 최종 성능 질문과 분리한다.

먼저 확인해야 할 것은:

1. 스스로 성공 가능한가?
2. unseen 환경으로 transfer 가능한가?
3. Imagination이 실제 행동 품질을 높이는가?
4. 그 뒤에 인간이 제공하지 않은 새로운 해결 경로가 반복적으로 나타나는가?

따라서 **창의성은 현재 primary benchmark claim이 아니라 후속 분석 질문**으로 둔다.

---

# 5. 연구 질문과 모듈의 대응

| 연구 질문 | 핵심 모듈 / 설계 |
|---|---|
| 최초 성공을 스스로 찾는가? | Policy, curriculum, ASEQ |
| unseen에서 일반화하는가? | relational representation |
| self-loop를 줄일 수 있는가? | ASEQ |
| 미래를 예측할 수 있는가? | Prophecy |
| 예측을 믿어도 되는가? | Calibration |
| 미래를 비교해 행동을 바꿀 수 있는가? | Imagination |
| 미래 가치를 평가할 수 있는가? | Critic |
| OOD value를 막을 수 있는가? | local Critic support |
| 성공 구조를 재사용할 수 있는가? | Skill |

이 대응 관계가 AASSR 문서 전체의 기본 구조다.

---

# 6. 현재 연구에서 중요한 원칙

AASSR의 성능을 높이기 위해 연구 질문 자체를 흐리는 shortcut은 허용하지 않는다.

현재 원칙:

- intermediate shaping reward 없음
- oracle action injection 없음
- guided success trajectory 없음
- hidden curriculum metadata를 observation으로 주지 않음
- hidden lockout/session pressure를 직접 주지 않음
- evaluation 중 checkpoint 재학습 없음
- no-Imagination / Full은 같은 frozen checkpoint 사용

즉 목표는 **정답을 더 많이 알려줘서 성공시키는 것**이 아니라, 공개 관측과 실제 경험만으로 문제 해결 구조를 학습하는 것이다.

---

다음으로 읽기:

- **[Sparse Reward Problem](Sparse-Reward-Problem)**
- **[Research Architecture](Research-Architecture)**
- **[Core Architecture](Core-Architecture)**
- **[Experiments](Experiments)**
