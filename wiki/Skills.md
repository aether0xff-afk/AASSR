# Skills — 성공 절차 재사용

[Skill(성공 절차 재사용)](Skills)은 AASSR에서 **반복해서 성공한 실제 ASeq 구조를 [관계 기반(relational)](Relational-Representation-and-Generalization) [재사용 가능한 틀(template)](Skills)로 승격해 다시 사용할 수 있게 하는 메커니즘**이다.

사람이 정답 macro를 미리 넣어주는 기능이 아니다.

```text
실제 성공 trajectory
      ↓
반복되는 relational ASeq 발견
      ↓
Skill template 승격
      ↓
새 scenario의 concrete actions에 다시 bind
```

> [!IMPORTANT]
> 현재 manifest 계약: `relational-aseq-template-v1`  
> 재사용 가능한 틀 promotion: `RelationalSkillLibrary` in `current_generation.py`  
> [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) rollout: `src/aassr_v2/current_relational_skill_prophecy.py`

---

# 1. 연구 질문

> **한 번 배운 성공 행동 구조를 [실제 개체를 구분하는(concrete)](State-Representation) identifier가 바뀐 새로운 상황에서도 재사용 가능한 고수준 행동 단위로 만들 수 있는가?**

희소 보상 장기 문제에서 매번 primitive [행동(action)](Reinforcement-Learning)부터 다시 탐색하면 sample efficiency가 낮다.

반복적으로 성공하는 구조가 있다면 다음에는 하나의 고수준 후보처럼 다룰 수 있다.

---

# 2. Skill의 근거는 real successful ASeq다

[Skill](Skills) 후보는 실제 [상태 전이(transition)](MDP-and-POMDP) trace에서 나온다.

```text
A1 -> A2 -> A3 -> goal achieved
```

같은 목표를 해결하는 구조가 반복되면 승격 후보가 된다.

핵심:

```text
human-written correct macro X
real successful experience O
```

즉 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility) 정답 경로를 사람이 [Skill](Skills)로 입력하는 구조가 아니다.

---

# 3. 왜 raw action sequence를 저장하면 안 되는가?

훈련 [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility):

```text
GET route-12
LOGIN profile-4
REQUEST object-7
```

새 난수 시드:

```text
GET route-31
LOGIN profile-9
REQUEST object-2
```

구조는 같아도 실제 개체를 구분하는 ID가 다르다.

Raw signature sequence를 그대로 재생하면 [전이(transfer)](Relational-Representation-and-Generalization)가 실패한다.

그래서 [현재(current)](Current-Status) [Skill](Skills)은 각 primitive를 **관계 기반 행동 재사용 가능한 틀**로 저장한다.

---

# 4. Relational template

실제 성공 trace의 각 행동을 관계 기반 key로 바꾼다.

```text
trace action A0 -> relational template T0
trace action A1 -> relational template T1
trace action A2 -> relational template T2
```

[Skill](Skills)은 다음 구조를 기억한다.

```text
Skill = (T0, T1, T2, ...)
```

새 [상태(state)](State-Representation)에서는 `T_i`와 같은 관계 기반 [역할(role)](Relational-Representation-and-Generalization)을 가진 현재 legal [실제 실행 행동(concrete action)](State-Representation)을 찾아 다시 bind한다.

---

# 5. Concrete rebinding

[Skill](Skills) step `i`에서 현재 행동 surface를 검색한다.

```text
현재 legal actions
   |
   | relational_action_key == T_i ?
   v
matching concrete candidates
```

후보가 없으면 그 [Skill](Skills)은 현재 상태에서 실행 불가능하다.

후보가 있으면 deterministic하게 실제 실행 행동 하나를 resolve한다.

따라서 [Skill](Skills)은

```text
구조는 transfer
실행은 concrete
```

라는 AASSR의 [식별 방식(identity)](State-Representation) 원칙을 그대로 따른다.

---

# 6. Promotion

현재 관계 기반 library는 goal completion에서 최근 ASeq를 관측해 동일 재사용 가능한 틀가 반복 성공했는지 센다.

기본적인 아이디어:

```text
첫 성공
-> candidate

같은 relational template의 반복 성공
-> confidence 증가

promotion threshold 충족
-> Skill 생성
```

현재 기본 promotion은 매우 적은 횟수에서도 가능하도록 설계되어 있지만, 최종 연구에서는 promotion [판정 기준값(threshold)](Terminology-Guide) 자체도 명시적으로 보고해야 한다.

---

# 7. Skill은 action surface에 어떻게 나타나는가?

승격된 [Skill](Skills)은 [Policy(정책 모델)](Policy)/Planner가 평가할 수 있는 고수준 행동처럼 노출될 수 있다.

```text
primitive A
primitive B
primitive C
skill-0001
```

하지만 [Skill](Skills)을 실행하면 내부적으로 현재 상태에 맞는 primitive sequence로 풀린다.

즉 [환경(environment)](Reinforcement-Learning)에 새로운 초능력 행동을 추가하는 것이 아니다.

---

# 8. Skill과 Policy

Primitive 행동의 [환경이 주는 외부(external)](Terminology-Guide) [가치(value)](Value-Functions-and-Bellman-Equation)는 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)이 담당한다.

[Skill](Skills)은 별도 식별 방식를 가지기 때문에 [Policy](Policy)에서 [Skill](Skills) 가치를 별도로 관리할 수 있다.

중요한 점은 [Skill](Skills)이 존재한다고 항상 선택되는 것이 아니라 다른 현재 후보들과 가치 비교를 거친다는 것이다.

---

# 9. Skill과 Prophecy

[Skill](Skills) 하나가 여러 primitive를 포함한다면 [Skill](Skills)의 미래를 예측하려면 [세계 모델(world model)](Model-Based-RL-and-World-Models)을 연속해서 적용해야 한다.

```text
Skill T0,T1,T2
   |
   v
Prophecy(T0)
   |
   v
Prophecy(T1)
   |
   v
Prophecy(T2)
```

초기 구현에서 매 primitive마다 가장 높은 [확률(probability)](Stochasticity-Uncertainty-and-Probability) [환경 결과(outcome)](Stochasticity-Uncertainty-and-Probability) 하나만 선택하면 확률적 future가 collapse된다.

[현재 세대(current-generation)](Current-Status)은 이 문제를 수리해 [Skill](Skills) rollout에서도 여러 환경 결과 [갈라진 결과 경로(branch)](Chance-and-Decision-Nodes)를 작은 beam으로 유지한다.

---

# 10. Skill에서도 probability와 reliability를 분리한다

[Skill](Skills) rollout의 결과 경로는 두 누적량을 따로 가진다.

```text
outcome mass
= 그 stochastic path가 발생할 probability mass

reliability
= 각 primitive prediction을 신뢰할 수 있는 정도의 누적
```

이 둘을 곱해 하나의 의미로 섞어버리면 [계획기(planner)](Counterfactual-Planning-and-Search) semantics가 흐려진다.

현재 확률적 [Skill](Skills) [Prophecy(미래 예측 모델)](Prophecy)는 여러 환경 결과의 mass를 유지하면서 [신뢰도(reliability)](Calibration)도 별도로 누적한다.

---

# 11. Beam을 쓰는 이유

[Skill](Skills) 길이가 `L`이고 각 primitive가 `M`개의 가능한 환경 결과을 낸다면 모든 결과 경로를 보존할 경우 대략 `M^L`로 증가할 수 있다.

그래서 현재 [Skill](Skills) [Prophecy](Prophecy)는 제한된 환경 결과 beam을 유지한다.

```text
각 step에서 stochastic candidates 생성
       ↓
outcome mass / reliability 기준 정렬
       ↓
상위 beam 유지
       ↓
mass renormalization
```

이것은 단순히 가장 좋은 미래 하나를 고르는 것이 아니라 **중요한 확률적 mass를 제한된 계산량에서 유지하려는 근사**다.

---

# 12. Skill 실패가 왜 중요한가?

훈련에서 성공했던 관계 기반 재사용 가능한 틀라도 새 scenario에서는 중간 primitive가 legal하지 않을 수 있다.

```text
T0 resolve 성공
T1 resolve 성공
T2 matching concrete action 없음
```

이 경우 [Skill](Skills)을 억지로 실행하면 안 된다.

현재 path는 unavailable 상태를 표시하거나 [예측 신뢰 정도(confidence)](Calibration)를 낮춰 계획기가 이를 신뢰하지 않도록 한다.

---

# 13. Skill과 창의성

[Skill](Skills)은 "창의성 모듈" 자체는 아니다.

오히려 이미 발견된 성공 구조를 압축하고 재사용하는 기능에 가깝다.

창의성 연구 질문은 별도다.

> 기존 [Skill](Skills)과 [학습(training)](Terminology-Guide) trajectory를 그대로 복제하지 않고도 새로운 유효한 해결 경로가 나오는가?

따라서 [Skill](Skills) 사용 성공과 새로운 해결 경로 생성은 구분해서 분석해야 한다.

---

# 14. Skill과 ASEQ

둘은 밀접하지만 역할이 다르다.

```text
ASEQ
= 실제 한 transition (S,A,S')

Skill
= 반복 성공한 여러 relational ASeq의 template
```

[ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)가 local experience 단위라면 [Skill](Skills)은 성공 경험의 고수준 재사용 단위다.

---

# 15. 실패 모드

## 15.1 Concrete macro memorization

raw ID sequence를 저장하면 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 난수 시드 전이 실패.

대응: 관계 기반 재사용 가능한 틀 + 현재 실제 개체를 구분하는 rebinding.

## 15.2 Premature promotion

우연히 한 번 성공한 sequence를 강한 [Skill](Skills)로 고정하면 잘못된 macro가 강화될 수 있다.

대응: 반복 성공 [증거(evidence)](Evidence-Matrix), 신뢰도/[실패(failure)](Replay-Buffer-and-Episode-Boundaries) accounting.

## 15.3 Stochastic collapse

[Skill](Skills) rollout에서 매 step 가장 높은 환경 결과만 남기면 위험한 확률적 결과 경로를 잃는다.

대응: 확률적 환경 결과 beam.

## 15.4 Unavailable primitive

새 상태에서 재사용 가능한 틀에 맞는 legal 실제 실행 행동이 없음.

대응: unavailable / low-confidence 처리.

## 15.5 Skill domination

한 [Skill](Skills)의 가치가 과도하게 높아져 primitive [탐색(exploration)](Exploration-and-Exploitation)을 막을 수 있다.

대응: primitive와 같은 외부 [학습 목표(objective)](Terminology-Guide) 기준에서 평가하고 별도 [진단 실험(diagnostic)](Evidence-Matrix)을 유지해야 한다.

---

# 16. 연구 가설

```text
H1. repeated successful ASeq를 relational template로 안정적으로 추출할 수 있는가?
H2. concrete ID가 바뀐 unseen seed에 Skill이 다시 bind되는가?
H3. Skill이 primitive-only보다 긴 성공 구조의 재사용 효율을 높이는가?
H4. stochastic Skill Prophecy가 deterministic best-path rollout보다 안전한가?
H5. Skill promotion이 새로운 exploration을 지나치게 줄이지 않는가?
```

---

# 17. 관련 코드

```text
src/aassr_v2/current_generation.py
  - RelationalSkillLibrary
  - relational template extraction
  - concrete primitive resolution

src/aassr_v2/current_relational_skill_prophecy.py
  - RelationalStochasticSkillProphecy
  - multi-outcome beam rollout

src/aassr_v2/skills.py
  - generic Skill data model / library base
```

---

다음으로 읽기:

- **[ASEQ](ASEQ)**
- **[Prophecy](Prophecy)**
- **[Policy](Policy)**
- **[Research Questions](Research-Questions)**
