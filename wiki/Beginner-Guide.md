# 처음 읽는 사람을 위한 AASSR 안내서

이 문서는 **강화학습 전공자가 아닌 아마추어 개발자·학생 연구자·개인 연구자**가 AASSR을 처음 읽을 때 막히지 않도록 만든 입문 안내서다.

이 위키의 목표는 어려운 전문용어를 많이 쓰는 것이 아니다. 오히려 다음 순서로 이해시키는 것이다.

```text
왜 필요한가?
↓
쉬운 말로 무엇인가?
↓
AASSR에서는 무엇을 하는가?
↓
정확한 전문용어는 무엇인가?
↓
수식과 코드에서는 어떻게 표현되는가?
↓
어떤 실험으로 검증하는가?
```

전문용어는 필요하다. 하지만 **전문용어를 알아야만 첫 문장을 이해할 수 있게 쓰면 안 된다.**

---

# 1. 이 연구를 한 문장으로 말하면

AASSR은 다음 질문에서 출발한다.

> **중간 힌트가 거의 없고 마지막 성공 여부만 알려주는 문제에서, 인공지능 에이전트가 스스로 시행착오를 하면서 성공에 필요한 여러 단계의 행동 순서를 만들어낼 수 있을까?**

예를 들어 어떤 시스템에서 최종 목표가 `proof 획득`이라고 하자.

```text
정보 보기       → 0점
경로 확인       → 0점
로그인 시도     → 0점
대상 조사       → 0점
추가 정보 획득  → 0점
최종 proof 획득 → +1점
```

중간에는 거의 아무 점수도 없다.

이런 문제를 **희소 보상([희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment))** 문제라고 부른다.

AASSR은 이런 환경에서 단순히 “직전에 점수가 높았던 행동”만 반복하지 않고,

- 지금까지 실제로 무엇을 관측했는지 기억하고,
- 같은 제자리 행동을 반복하지 않게 하고,
- 비슷한 구조의 문제에서 배운 것을 재사용하고,
- 행동하기 전에 가능한 미래를 예측해 보고,
- 예측을 믿어도 되는지 확인하고,
- 상상한 미래가 실제 목표에 좋은지 평가한 뒤,
- 충분한 근거가 있을 때만 기본 행동을 바꾸는

구조를 연구한다.

---

# 2. 먼저 알아야 할 가장 작은 강화학습 개념

강화학습을 한 번도 배우지 않았어도 다음 다섯 단어만 먼저 알면 된다.

## 에이전트(agent)

문제를 풀기 위해 행동하는 주체다.

게임이라면 플레이어, 로봇이라면 로봇의 제어기, AASSR이라면 웹 환경에서 다음 요청을 결정하는 학습 시스템 전체가 에이전트다.

관련 문서: [강화학습](Reinforcement-Learning)

## 환경(environment)

에이전트가 행동을 실행하는 바깥 세계다.

에이전트가 어떤 요청을 보내면 환경은 응답하고 상태가 달라질 수 있다.

## 상태(state)

“현재 세계가 어떤 상황인가?”를 표현하는 말이다.

하지만 AASSR에서는 이 단어를 조심해서 쓴다. 실제 세계 전체 상태와, 에이전트가 볼 수 있는 관측, 신경망에 넣기 위해 만든 숫자 표현은 서로 다르다.

관련 문서: [상태·관측·표현](State-Representation)

## 행동(action)

에이전트가 실제로 선택하는 한 번의 행동이다.

예를 들면 `browse`, `login`, `request object` 같은 선택이 행동이 될 수 있다.

## 보상(reward)

환경이 행동 결과에 대해 주는 점수다.

AASSR의 핵심 연구 환경에서는 보상을 최대한 단순하게 둔다.

```text
최종 성공      +1
진짜 실패      -1
그 외 대부분    0
```

중간 과정에 “이 행동 잘했음 +0.2” 같은 정답 힌트를 넣지 않으려는 것이다.

관련 문서: [희소 보상과 보상 책임 배분](Sparse-Reward-and-Credit-Assignment)

---

# 3. 왜 일반적인 강화학습만으로 어렵나?

## 문제 1. 중간 행동의 가치가 바로 드러나지 않는다

어떤 행동이 지금 당장 0점을 받아도, 세 단계 뒤 성공에 꼭 필요한 정보였을 수 있다.

```text
정보 획득 0
   ↓
인증 0
   ↓
대상 선택 0
   ↓
성공 +1
```

따라서 마지막 +1을 과거 행동에 어떻게 연결할지가 어렵다.

이 문제를 **보상 책임 배분([보상 책임 배분(credit assignment)](Sparse-Reward-and-Credit-Assignment))** 문제라고 부른다.

## 문제 2. 세계의 모든 정보를 볼 수 없다

에이전트는 환경 내부의 정답을 직접 보면 안 된다.

```text
환경 내부에는 존재하지만 에이전트가 보면 안 되는 것
- 실제 정답 대상
- 미래에 성공할 행동
- 숨겨진 실패 임계치

에이전트가 실제로 볼 수 있는 것
- 방금 받은 응답
- 공개된 상태 코드
- 과거 응답에서 실제로 발견한 정보
```

이처럼 세계 일부만 볼 수 있는 상황을 **부분 관측([부분 관측(partial observability)](MDP-and-POMDP))**이라고 한다.

관련 문서: [MDP와 POMDP](MDP-and-POMDP)

## 문제 3. 행동 후보가 상황마다 달라진다

현재 상태에 따라 가능한 행동이 달라질 수 있다.

예를 들어 아직 어떤 객체를 발견하지 않았다면 그 객체에 대한 요청 행동도 존재하지 않을 수 있다.

그래서 AASSR은 “고정된 몇 개 버튼 중 하나를 누르는 문제”보다 더 복잡한 행동 공간을 다룬다.

---

# 4. AASSR의 전체 구조를 사람의 생각 과정에 비유하면

AASSR의 핵심 모듈을 아주 거칠게 사람의 문제 해결 과정에 비유하면 다음과 같다.

| AASSR 구성요소 | 쉬운 말 | 사람이 문제를 풀 때의 비유 |
|---|---|---|
| [State Representation](State-Representation) | 현재 상황을 정리하는 방식 | “지금까지 상황을 어떤 관점으로 볼까?” |
| [ASEQ](ASEQ) | 실제 행동 결과 기록 | “아까 이 행동 했는데 아무 변화 없었지.” |
| [Policy](Policy) | 기본 행동 선택기 | “지금 당장 가장 좋아 보이는 행동은 이거야.” |
| [Knowledge](Knowledge) | 이번 문제에서 얻은 사실 기억 | “아까 응답에서 토큰을 하나 봤어.” |
| [Prophecy](Prophecy) | 미래 결과 예측기 | “이 행동을 하면 다음에 무슨 일이 생길까?” |
| [Calibration](Calibration) | 예측 신뢰도 검사 | “내 예상이 이 상황에서도 믿을 만한가?” |
| [Imagination](Imagination) | 가상 미래 탐색 | “A를 하면 이렇게, B를 하면 저렇게 될 것 같은데?” |
| [Critic](Critic) | 미래 가치 평가기 | “그 미래가 최종 목표에 실제로 좋은가?” |
| [Critic Support](Critic-Support-and-OOD) | 평가 근거 확인 | “근데 내가 이런 상황을 실제로 충분히 본 적이 있나?” |
| [Skills](Skills) | 성공 절차 재사용 | “전에 비슷한 문제에서 이 순서가 통했지.” |

중요한 점은 이 모듈들이 모두 같은 일을 하는 것이 아니라는 것이다.

AASSR은 일부러 역할을 나눈다.

```text
예측한다
!=
좋은지 평가한다
!=
예측을 믿어도 되는지 판단한다
!=
실제로 충분한 데이터 근거가 있는지 확인한다
```

---

# 5. 상태(state), 관측(observation), 표현(representation)은 왜 따로 말하나?

초보자가 가장 많이 헷갈리는 부분이다.

## 실제 환경 상태

환경 내부에 실제로 존재하는 모든 정보다.

에이전트가 이것을 전부 볼 수 있다고 가정하면 너무 쉬운 문제가 될 수 있다.

## 관측(observation)

에이전트가 실제로 받아볼 수 있는 정보다.

예:

```text
HTTP 상태 코드
응답 본문에서 찾은 토큰
공개된 객체 목록
```

## 표현(representation)

관측을 학습 모델이 사용하기 좋은 형태로 바꾼 것이다.

예를 들어 문자열 이름 그대로를 쓰지 않고 “이 객체가 어떤 역할을 하는지”를 숫자로 표현할 수 있다.

```text
환경의 실제 상태
      ↓ 일부만 보임
관측
      ↓ 학습 가능한 형태로 변환
표현
      ↓
신경망
```

관련 문서: [State Representation](State-Representation)

---

# 6. 관계 기반 표현(relational representation)은 왜 필요한가?

훈련 중에 본 이름과 평가 때 나오는 이름이 다를 수 있다.

훈련:

```text
route-12
profile-4
object-7
```

새 문제:

```text
route-31
profile-9
object-2
```

이름 자체만 외우면 전혀 다른 문제처럼 보인다.

하지만 구조적으로는 같을 수 있다.

```text
목록을 보여주는 경로
→ 인증된 사용자
→ 후보 객체
```

그래서 AASSR은 일부 학습 모듈에서 **이름보다 관계와 역할을 중심으로 표현**한다.

이를 관계 기반 표현([관계 기반 표현(relational representation)](Relational-Representation-and-Generalization))이라고 한다.

관련 문서: [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)

---

# 7. ASEQ는 무엇인가?

[ASEQ(실제 상태-행동-다음 상태 기록)](ASEQ)는 AASSR에서 실제로 관측된 한 번의 상태 전이를 다음 세 요소로 기록한 것이다.

```text
현재 상태 S
행동 A
다음 상태 S'
```

즉:

```text
(S, A, S')
```

이다.

특히 AASSR은 다음 패턴에 관심이 있다.

```text
S → A → S
```

현재 상태에서 어떤 행동을 했는데 실제 상황이 그대로라면 **제자리 반복([제자리 반복(self-loop)](ASEQ))**일 수 있다.

같은 무진전 패턴이 실제 경험으로 반복 확인되면 그 행동을 잠시 억제한다.

하지만 다음처럼 실제로 상황이 변하면 허용한다.

```text
S1 → browse → S2
S2 → browse → S3
```

따라서 [ASEQ](ASEQ)는 “같은 행동 두 번 금지”가 아니다.

정확히는 **실제로 관측된 의미상 동일한 제자리 반복만 줄이려는 장치**다.

관련 문서: [ASEQ](ASEQ)

---

# 8. Policy는 무엇인가?

[Policy(정책 모델)](Policy)는 현재 상태에서 기본 행동을 정하는 정책 모델이다.

AASSR의 현재 [Policy](Policy)는 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD) 계열을 사용한다.

[DQN](Q-Learning-DQN-and-TD)은 아주 간단히 말하면:

> “현재 상태에서 이 행동을 했을 때 장기적으로 얼마나 좋은가?”

를 행동마다 숫자로 추정한다.

예:

```text
browse        0.21
login         0.48
request       0.33
```

가장 큰 숫자만 고르면 `login`을 선택하게 된다.

이 숫자를 Q값([Q값(Q-value)](Value-Functions-and-Bellman-Equation))이라고 한다.

하지만 이 값은 **즉시 보상**과 같은 것이 아니다.

최종 성공까지 이어질 가능성을 장기적으로 반영하려는 값이다.

관련 문서: [Policy](Policy), [Q-learning·DQN·TD](Q-Learning-DQN-and-TD)

---

# 9. Knowledge는 무엇인가?

현재 응답 하나만 봐서는 과거에 발견한 정보가 사라질 수 있다.

예:

```text
1단계: token 발견
2단계: 다른 페이지 이동
3단계: 현재 화면에는 token이 안 보임
```

사람이라면 “아까 token을 봤다”는 사실을 기억한다.

[Knowledge(에피소드 지식)](Knowledge)는 이런 **현재 에피소드에서 실제로 얻은 사실**을 보존한다.

단, 미래에 알게 된 사실을 과거 판단에 몰래 넣으면 안 된다.

그래서 시간 순서를 지킨다.

```text
현재 알고 있는 지식
↓
예측과 행동 결정
↓
실제 행동
↓
새 응답
↓
새 지식 추가
```

관련 문서: [Knowledge](Knowledge), [정보 누출과 공정 평가](Causality-Leakage-and-Evaluation)

---

# 10. Prophecy는 무엇인가?

[Prophecy(미래 예측 모델)](Prophecy)는 **행동 뒤에 어떤 다음 상황들이 생길 수 있는지 예측하는 모델**이다.

쉽게 말하면 AASSR의 미래 예측기다.

예:

```text
행동 A
├─ 70% → 정상 응답
├─ 20% → 접근 거부
└─ 10% → 요청 제한
```

중요한 점은 결과가 하나로 정해져 있지 않을 수 있다는 것이다.

만약 세 결과를 그냥 평균내면 실제로는 존재하지 않는 애매한 미래가 생길 수 있다.

그래서 현재 [Prophecy](Prophecy)는 여러 가능한 결과와 각각의 확률을 따로 표현한다.

이를 **혼합 분포(mixture)**라고 생각하면 된다.

또 여러 학습 모델을 함께 사용해 모델 사이의 차이도 본다.

이를 **앙상블(ensemble)**이라고 한다.

둘은 다르다.

```text
mixture
= 환경 자체에서 여러 결과가 가능함

ensemble
= 여러 학습 모델이 서로 다른 예측을 낼 수 있음
```

관련 문서: [Prophecy](Prophecy), [세계 모델](Model-Based-RL-and-World-Models), [혼합·앙상블·보정](Mixture-Ensemble-and-Calibration)

---

# 11. Calibration은 무엇인가?

예측 모델이 숫자를 낸다고 해서 항상 믿을 수 있는 것은 아니다.

예를 들어 모델이 이렇게 말할 수 있다.

```text
성공 가능성 80%
```

그런데 과거에 비슷한 상황에서 80%라고 말한 예측이 실제로는 절반만 맞았다면 그 모델은 과신하고 있는 것이다.

[Calibration(예측 신뢰도 보정)](Calibration)은 이런 **예측의 신뢰도를 실제 검증 데이터와 비교해 보는 과정**이다.

AASSR에서는 특히 중요한 공개 상태 코드까지 제대로 맞추는지 본다.

관련 문서: [Calibration](Calibration)

---

# 12. Imagination은 무엇인가?

[Imagination(가상 미래 탐색)](Imagination)은 실제 행동하기 전에 [Prophecy](Prophecy)의 예측을 여러 단계 연결해 보는 계획 탐색기다.

```text
현재
├─ 행동 A
│  ├─ 결과 A1
│  │  ├─ 다음 행동 ...
│  │  └─ 다음 행동 ...
│  └─ 결과 A2
└─ 행동 B
   ├─ 결과 B1
   └─ 결과 B2
```

여기서 매우 중요한 구분이 있다.

## 환경 결과는 에이전트가 선택할 수 없다

행동 A 이후 성공 결과가 10%, 실패 결과가 90%라면 10% 성공만 골라서 계산하면 안 된다.

환경 결과는 확률에 따라 평균해야 한다.

이를 **환경 결과 노드([환경 결과 노드(chance node)](Chance-and-Decision-Nodes))**라고 한다.

## 다음 행동은 에이전트가 선택할 수 있다

어떤 미래 상태에서 가능한 행동이 세 개 있다면 가장 좋아 보이는 행동을 선택할 수 있다.

이를 **행동 선택 노드([행동 선택 노드(decision node)](Chance-and-Decision-Nodes))**라고 한다.

관련 문서: [Imagination](Imagination), [환경 결과 노드와 행동 선택 노드](Chance-and-Decision-Nodes)

---

# 13. Critic은 무엇인가?

[Prophecy](Prophecy)는 미래를 예측하지만, 그 미래가 목표에 좋은지는 별도 문제다.

```text
Prophecy
= 무슨 일이 생길까?

Critic
= 그 미래가 최종 성공에 좋은가?
```

[Critic(미래 가치 평가기)](Critic)은 상상된 미래 상태가 장기적으로 얼마나 좋은지 평가한다.

현재 AASSR은 실제 희소 보상 경험으로 [Critic](Critic)을 학습한다.

관련 문서: [Critic](Critic)

---

# 14. Critic Support는 왜 또 필요한가?

신경망은 처음 보는 입력에도 숫자를 낸다.

예:

```text
Critic = 0.95
```

하지만 “0.95라는 숫자를 냈다”와 “그 숫자를 믿을 실제 데이터가 있다”는 다르다.

AASSR은 그래서 별도의 **국소 데이터 근거([국소 데이터 근거(local support)](Critic-Support-and-OOD))**를 확인한다.

```text
Critic이 높은 값을 냄
      ↓
주변에 실제 학습 경험이 충분한가?
      ↓ 아니오
Imagination이 기본 Policy를 덮어쓰지 못하게 막음
```

이것은 과감하게 틀리는 것보다, 근거가 없을 때 기존 정책으로 돌아가는 **보수적 실패(fail closed)** 설계다.

관련 문서: [Critic Support와 OOD](Critic-Support-and-OOD)

---

# 15. Skill은 무엇인가?

[Skill(성공 절차 재사용)](Skills)은 성공했던 행동 순서를 그대로 외우는 기능과는 조금 다르다.

AASSR은 반복 성공한 실제 행동 구조에서 관계 패턴을 뽑아 새 문제의 실제 객체에 다시 연결하려고 한다.

```text
과거 성공
route-A → profile-B → object-C

관계 구조 추출
목록 경로 → 인증 사용자 → 목표 객체

새 문제
route-X → profile-Y → object-Z
```

이를 통해 이름이 달라도 구조가 비슷한 문제에서 성공 절차를 재사용할 수 있는지 연구한다.

관련 문서: [Skills](Skills), [계층형 강화학습과 Skill](Hierarchical-RL-and-Skills)

---

# 16. AASSR에서 가장 헷갈리는 값들

아래는 절대 같은 값이 아니다.

## 보상(reward)

환경이 바로 주는 점수.

## 누적 보상(return)

앞으로 받을 여러 보상을 시간 할인까지 고려해 합친 값.

## Q값(Q-value)

현재 상태에서 특정 행동을 했을 때 기대되는 장기 가치를 추정한 값.

## 정보 가치 잔차(information residual)

정보를 얻는 행동이 미래 선택에 도움을 줄 가능성을 별도로 평가하는 내부 항목.

외부 보상 자체를 바꾸는 것은 아니다.

## 결과 확률(outcome probability)

특정 미래 결과가 발생할 확률.

## 예측 신뢰도(prediction reliability)

그 확률 예측 자체를 얼마나 믿을 수 있는지.

## Critic 값

상상된 미래가 최종 목표에 얼마나 좋은지 추정한 값.

## 국소 데이터 근거(local support)

그 [Critic](Critic) 값을 뒷받침하는 실제 학습 데이터가 주변에 충분히 있는지.

```text
보상
!= 누적 보상
!= Q값
!= 정보 가치
!= 결과 확률
!= 예측 신뢰도
!= Critic 값
!= 데이터 근거
```

---

# 17. 실험에서 자주 나오는 단어

## 비교 기준 모델(baseline)

새 방법이 정말 좋은지 비교하기 위한 기준 모델이다.

예:

```text
Raw DQN
Relational DQN
DreamerV3
AASSR without Imagination
AASSR Full
```

## 구성요소 제거 실험(ablation)

AASSR 전체가 좋아졌다고 해도 어느 부분 때문인지 알 수 없다.

그래서 한 요소만 끄거나 바꿔 비교한다.

예:

```text
AASSR Full
vs
같은 checkpoint + Imagination OFF
```

이 비교는 [Imagination](Imagination)의 추가 효과를 보려는 것이다.

## 학습 중 보지 못한 문제(unseen)

훈련에서 직접 본 문제와 다른 새 문제다.

## 전이(transfer)

한 문제에서 배운 것을 이름이나 세부 구성이 다른 새 문제에 재사용하는 능력이다.

## 최종 비공개 평가(final blind)

모델과 실험 방법을 모두 고정한 뒤 처음 공개하는 평가 세트다.

결과를 보고 다시 튜닝하지 않기 위한 장치다.

관련 문서: [실험 설계·비교·재현성](Ablation-Benchmarking-and-Reproducibility)

---

# 18. 현재 연구가 어디까지 왔는지 읽는 법

AASSR 위키에서는 “코드가 존재한다”와 “성능이 증명됐다”를 구분한다.

증거 수준을 대략 다음처럼 본다.

```text
1. 구조가 구현됨
2. 단위 테스트와 회귀 테스트 통과
3. 특정 메커니즘이 작동하는 진단 실험
4. 작은 규모 실제 환경 실험
5. 여러 seed를 사용한 비교 실험
6. 최종 비공개 평가
```

현재 어디까지 왔는지는 [Current Status](Current-Status)를 본다.

각 연구 질문별로 무엇이 증명됐고 무엇이 남았는지는 [Evidence Matrix](Evidence-Matrix)를 본다.

---

# 19. 코드 문서에서 자주 나오는 단어

## current

현재 실제로 사용되는 버전이라는 뜻이다.

## historical

과거 버전 또는 과거 진단 기록이라는 뜻이다.

## source of truth

여러 문서가 서로 다를 때 최종적으로 무엇을 기준으로 삼을지 정한 원본이다.

AASSR의 현재 실행 구조는 다음 파일이 최종 기준이다.

```text
src/aassr_v2/current_manifest.py
```

## checkpoint

학습된 신경망 파라미터를 저장한 시점이다.

## seed

난수 생성 결과를 재현하기 위해 사용하는 시작 숫자다.

같은 [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)와 같은 코드/환경이면 가능한 한 같은 실험 조건을 다시 만들 수 있다.

## regression test

예전에 제대로 작동하던 기능이 새 수정 때문에 다시 깨지지 않았는지 확인하는 테스트다.

---

# 20. 추천 읽기 순서

## 정말 처음이라면

```text
이 문서
↓
AASSR in 5 Minutes
↓
용어 안내서
↓
State Representation
↓
ASEQ
↓
Policy
↓
Prophecy
↓
Imagination
↓
Current Status
```

## 강화학습은 조금 안다면

```text
Sparse Reward Problem
↓
Research Questions
↓
Research Architecture
↓
State / Policy / Prophecy / Critic / Imagination
↓
Evidence Matrix
↓
Experiments
```

## 실험을 재현하고 싶다면

```text
Current Status
↓
Experiments
↓
Reproduction
```

---

# 21. 영어 용어를 만났을 때의 규칙

이 위키는 앞으로 다음 표기 규칙을 사용한다.

## 처음 등장

```text
한국어 설명(영어 원어)
```

또는 AASSR 고유 모듈이라면:

```text
Prophecy(미래 예측 모델)
```

처럼 쓴다.

## 같은 페이지에서 다시 등장

가능하면 한국어를 우선 사용하고, 고유 모듈 이름처럼 꼭 필요한 경우에만 영어 이름을 유지한다.

## 영어 약어

약어만 던지지 않는다.

나쁜 예:

```text
OOD에서 Critic support가 부족하다.
```

더 좋은 예:

```text
학습 분포 밖(OOD) 상태에서는 Critic의 값을 뒷받침할 실제 데이터 근거(local support)가 부족할 수 있다.
```

## 코드 이름

실제 코드 식별자는 번역하지 않는다.

```text
current_manifest.py
build_current_pentest_aassr_core
```

대신 바로 앞뒤 문장에서 무엇을 뜻하는지 설명한다.

---

# 다음 문서

- **[AASSR 5분 설명](AASSR-in-5-Minutes)** — 전체 동작을 빠르게 훑기
- **[한국어 중심 용어 안내서](Terminology-Guide)** — 영어 전문용어를 쉬운 말로 찾기
- **[개념 지도](Concept-Index)** — 개념 간 연결 보기
- **[현재 연구 상태](Current-Status)** — 무엇이 검증됐고 무엇이 남았는지 보기
- **[연구 질문](Research-Questions)** — 이 연구가 무엇을 증명하려는지 보기
