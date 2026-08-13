# 연구 문서에서 자주 나오는 영어 해설

이 페이지는 AASSR 고유 기술보다 **연구·개발 문서 자체에서 너무 쉽게 영어로 적어 버리는 표현**을 설명한다.

전공자는 `runtime`, `contract`, `diagnostic`, `evidence`, `proxy`, `frozen` 같은 말을 별 설명 없이 사용하기 쉽다. 하지만 이 단어를 모르면 문장의 핵심을 놓치게 된다.

여기서는 영어를 외우게 하기보다 **그 단어가 문장에서 무슨 역할을 하는지** 설명한다.

---

# 1. current — 현재 사용 중

`current`는 **지금 실제 연구에서 기준으로 삼는 것**이라는 뜻이다.

예를 들어:

```text
current architecture
```

는 “현재 구조”라는 뜻이다.

AASSR에는 과거 버전과 현재 버전이 함께 저장되어 있기 때문에 이 구분이 중요하다.

```text
현재 구조
= 지금 실제 실험과 설명의 기준

과거 구조
= 이전 실험을 재현하거나 실패 원인을 기록하기 위해 보존
```

AASSR의 현재 실행 구조는 [현재 연구 상태](Current-Status)에서 확인한다.

---

# 2. current-generation — 현재 세대

단순히 최신 파일이라는 뜻보다 조금 더 강하다.

AASSR에서 **[현재 세대(current-generation)](Current-Status)**는 지금 연구의 기준이 되는 한 묶음의 설계 세대를 뜻한다.

예:

```text
aassr-current-generation-v2
```

이 문자열을 외울 필요는 없다.

“현재 실험이 어떤 세대의 구조를 사용하는지 구분하기 위한 버전 이름”이라고 이해하면 된다.

---

# 3. runtime — 실제 실행 구조

`runtime`은 문서에 적힌 아이디어가 아니라 **프로그램을 실제로 실행했을 때 사용되는 코드 구조**를 뜻한다.

예를 들어 문서에 오래된 [Prophecy(미래 예측 모델)](Prophecy) 설명이 남아 있어도 실제 실행 코드가 다른 모델을 사용한다면 현재 연구를 판단할 때는 실행 구조가 더 중요하다.

AASSR에서는 현재 실행 구조의 최종 기준을 다음 파일에 둔다.

```text
src/aassr_v2/current_manifest.py
```

관련: [현재 연구 상태](Current-Status)

---

# 4. contract — 반드시 지켜야 하는 명세

개발 문서에서 `contract`는 법률 계약이 아니다.

**“이 구성요소는 반드시 이런 의미와 동작을 가져야 한다”는 약속 또는 명세**를 뜻한다.

예를 들어:

```text
보상 contract
성공 = +1
실제 실패 = -1
그 외 = 0
```

이라고 하면 코드를 최적화하더라도 이 의미를 바꾸면 안 된다.

AASSR에서는 실험 공정성을 위해 이런 명세를 많이 둔다.

---

# 5. invariant — 바뀌면 안 되는 성질

`invariant`는 **어떤 수정이나 계산 과정에서도 유지되어야 하는 성질**이다.

예:

```text
최적화 전 행동 선택 = A
최적화 후 행동 선택 = A
```

처럼 성능 최적화를 했더라도 연구 의미가 같아야 한다.

회귀 테스트는 이런 불변 조건이 깨지지 않았는지 확인한다.

---

# 6. component — 구성요소

`component`는 시스템의 한 부품을 뜻한다.

AASSR에서:

```text
Policy
Prophecy
Calibration
Critic
Imagination
```

등은 각각 다른 역할을 가진 구성요소다.

구성요소를 나누는 이유는 “전체 성능이 변했다”는 결과만 보지 않고 **어떤 기능이 무엇을 했는지 따로 검증하기 위해서**다.

---

# 7. active — 현재 실제로 사용 중

코드 파일이 존재한다고 해서 그 기능이 현재 모델에서 쓰인다는 뜻은 아니다.

`active`는 **실제 현재 실행 경로에 연결되어 사용 중**이라는 뜻이다.

반대로 과거 코드가 저장소에 남아 있어도 현재 실행 경로에서 호출되지 않는다면 [현재 활성(active)](Current-Status)가 아니다.

---

# 8. historical — 과거 기록

`historical`은 **과거 버전이나 과거 실험을 기록·재현하기 위해 남긴 것**이라는 뜻이다.

AASSR에서는 과거 실패도 삭제하지 않는다.

왜 실패했는지 알 수 있는 중요한 연구 자료이기 때문이다.

하지만 과거 결과를 현재 모델의 성능으로 인용하면 안 된다.

관련: [개발 역사](Development-History)

---

# 9. diagnostic — 원인을 찾는 진단 실험

`diagnostic`은 최종 성능 경쟁을 위한 실험이 아니다.

**“왜 이런 현상이 생겼지?”를 좁혀 가기 위한 진단 실험**이다.

예:

```text
성공률이 낮다
↓
Prophecy가 틀린가?
Critic이 과신하는가?
데이터가 부족한가?
Imagination이 행동을 너무 자주 바꾸는가?
```

이런 원인들을 분리하기 위해 작은 실험을 수행할 수 있다.

진단 실험에서 좋은 결과가 나왔다고 바로 최종 성능이 증명되는 것은 아니다.

관련: [연구 질문-증거 연결표](Evidence-Matrix)

---

# 10. evidence — 증거

연구 문서에서 `evidence`는 단순한 느낌이나 추측이 아니라 **주장을 뒷받침하는 관측·실험·검증 결과**를 뜻한다.

AASSR 위키에서는 증거 수준을 나눈다.

```text
구현됨
↓
회귀 테스트 통과
↓
메커니즘 진단
↓
작은 실제 실험
↓
여러 난수 시드 비교
↓
최종 비공개 평가
```

따라서 “코드가 있다”는 것도 한 종류의 증거지만 “성능이 더 좋다”는 주장에 충분한 증거는 아니다.

---

# 11. claim — 연구 주장

`claim`은 **논문이나 연구 결과에서 우리가 사실이라고 말하려는 문장**이다.

예:

```text
ASEQ는 제자리 반복을 줄인다.
```

와

```text
ASEQ는 전체 성공률을 항상 높인다.
```

는 서로 다른 주장이다.

첫 번째를 뒷받침하는 증거만 있다고 두 번째까지 말하면 과장이다.

그래서 AASSR의 [연구 질문-증거 연결표](Evidence-Matrix)는 “현재 말할 수 있는 것”과 “아직 말하면 안 되는 것”을 나눈다.

---

# 12. public — 에이전트에게 공개된

`public`은 모든 사람이 공개적으로 볼 수 있다는 뜻이라기보다, AASSR 문맥에서는 **에이전트가 정상적인 상호작용을 통해 볼 수 있는 정보**라는 뜻으로 자주 쓴다.

예:

```text
공개된 응답 상태 코드
공개된 응답 본문
실제로 발견한 객체
```

등이다.

관련: [상태 표현](State-Representation)

---

# 13. hidden — 에이전트에게 숨겨진

`hidden`은 환경 내부에는 존재하지만 에이전트에게 직접 보여주지 않는 정보다.

예:

```text
환경 내부의 실제 정답
숨겨진 실패 카운터
미래에 성공할 행동
```

이런 정보가 학습 입력에 섞이면 정보 누출이 생길 수 있다.

관련: [인과성·정보 누출·공정 평가](Causality-Leakage-and-Evaluation)

---

# 14. real — 실제 환경에서 관측된

AASSR에서 `real`은 매우 중요하다.

**세계 모델이 상상한 것이 아니라 실제 환경과 상호작용해서 얻은 것**이라는 뜻이다.

```text
실제 상태 전이
= 실제 행동을 실행하고 실제 응답을 관측
```

반대로 [Imagination(가상 미래 탐색)](Imagination)이 만든 미래는 가상 정보다.

```text
실제 경험
≠ 가상 미래
```

이 구분은 학습 근거와 계획 결과를 섞지 않기 위해 필요하다.

---

# 15. imagined — 모델이 상상한

`imagined`는 실제 환경에서 일어난 것이 아니라 **세계 모델이 예측해 만든 가상 미래**라는 뜻이다.

가상 미래는 행동 계획에 사용할 수 있다.

하지만 실제로 일어난 사실처럼 [Knowledge(에피소드 지식)](Knowledge)나 실제 학습 증거에 자동으로 넣으면 안 된다.

관련: [Imagination](Imagination)

---

# 16. training — 학습

`training`은 모델의 파라미터를 실제 데이터로 수정하는 과정이다.

```text
입력
↓
예측
↓
정답 또는 학습 목표와 비교
↓
오차 계산
↓
파라미터 수정
```

평가와 구분해야 한다.

---

# 17. validation — 검증

`validation`은 학습 중 또는 실험 설계 중에 **현재 모델이 제대로 배우고 있는지 확인하는 평가**다.

검증 결과를 보고 모델 설정을 바꿀 수 있다면 최종 시험과는 역할이 다르다.

---

# 18. evaluation — 평가

`evaluation`은 학습된 모델의 성능을 재는 과정이다.

중요한 실험에서는 평가 중에 모델이 몰래 더 학습하지 않도록 막아야 한다.

```text
학습 단계 → 파라미터 변경 허용
평가 단계 → 파라미터 변경 금지
```

---

# 19. frozen — 학습을 멈춘

`frozen`은 **평가 중 모델 파라미터나 지속적인 학습 상태를 바꾸지 않는 것**을 뜻한다.

AASSR의 [Imagination](Imagination) 효과를 비교할 때 특히 중요하다.

```text
동일한 학습 체크포인트
├─ Imagination 끔
└─ Imagination 켬
```

두 조건 사이에서 다시 학습하면 [Imagination](Imagination)의 효과인지 학습 차이인지 알 수 없기 때문이다.

---

# 20. protocol — 미리 정한 실험 규칙

`protocol`은 실험을 어떤 규칙으로 수행할지 미리 정한 문서다.

예를 들면:

- 어떤 난수 시드를 쓸지
- 학습 상태 전이 수가 몇 개인지
- 어떤 비교 모델을 쓸지
- 무엇을 성공으로 볼지
- 최종 평가를 언제 열지

등을 정한다.

결과를 본 뒤 규칙을 바꾸면 비교가 불공정해질 수 있다.

---

# 21. proxy — 대신 측정하는 값

`proxy`는 우리가 진짜 알고 싶은 것을 직접 측정하기 어려울 때 대신 보는 지표다.

예를 들어 최종 관심사가 성공률인데:

```text
계획 깊이
예측 오차
개입 횟수
```

같은 값은 원인을 분석하는 데 유용하지만 성공률 그 자체는 아니다.

따라서 **대리 지표가 좋아졌다고 최종 성능도 좋아졌다고 단정하면 안 된다.**

---

# 22. input / output — 입력과 출력

`input`은 모델에 넣는 정보, `output`은 모델이 계산해서 내놓는 결과다.

예:

```text
Prophecy 입력
= 현재 상태 + 행동 + 현재까지 얻은 지식

Prophecy 출력
= 가능한 다음 상태들과 각 결과의 확률
```

---

# 23. model — 학습 모델

`model`은 문맥에 따라 뜻이 다르지만 AASSR 위키에서는 보통 **데이터에서 규칙을 학습해 예측이나 평가를 수행하는 계산 구조**를 뜻한다.

예:

- [Policy(정책 모델)](Policy)의 [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD)
- [Prophecy](Prophecy)의 미래 예측 모델
- [Critic(미래 가치 평가기)](Critic)의 가치 평가 모델

`world model`처럼 특별한 의미를 가질 때는 별도로 설명한다.

---

# 24. predictor — 예측 모델

`predictor`는 어떤 값을 예측하는 모델이다.

예를 들어 다음 상태를 예측하거나 정보 가치를 예측할 수 있다.

어떤 값을 예측하는지 항상 같이 확인해야 한다.

---

# 25. objective — 학습 목표

`objective`는 모델이 학습하면서 줄이거나 늘리려고 하는 수학적 목표다.

예를 들어 상태 코드 예측 모델이라면 실제 상태 코드와 예측 분포 사이의 오차를 줄이는 것이 학습 목표가 될 수 있다.

환경의 최종 보상과 같은 개념은 아니다.

---

# 26. categorical — 여러 범주 중 하나를 예측하는

`categorical`은 연속적인 숫자 하나를 예측하는 대신 **정해진 여러 종류 중 어느 것인지** 예측하는 방식을 말한다.

예를 들어 HTTP와 비슷한 상태 코드가:

```text
200
403
404
429
```

처럼 몇 개 범주로 나뉜다면 범주형 예측으로 다룰 수 있다.

---

# 27. reliability — 신뢰도

`reliability`는 **예측값이 나왔다는 사실과 별개로 그 예측을 얼마나 믿을 수 있는지**를 나타낸다.

AASSR에서:

```text
결과 확률
≠ 예측 신뢰도
```

이다.

자세히: [Calibration](Calibration)

---

# 28. root — 지금 실행할 첫 행동

계획 트리에서 `root`는 가장 처음에 있는 선택 지점을 뜻한다.

AASSR의 [Imagination](Imagination)이 여러 단계 미래를 계산하더라도 현실에서 가장 먼저 필요한 것은:

> **그래서 지금 무슨 행동을 할 것인가?**

이다.

이 첫 행동 후보를 `root action` 또는 `root candidate`라고 부르는 경우가 있다.

관련: [Imagination](Imagination)

---

# 29. override — 기본 선택을 바꾸기

`override`는 **기본 [Policy](Policy)가 고른 행동 대신 다른 행동을 실제로 사용하도록 바꾸는 것**이다.

AASSR에서 [Imagination](Imagination)은 미래를 계산했다는 이유만으로 무조건 행동을 바꾸지 않는다.

신뢰도·가치 차이·실제 데이터 근거 등의 조건을 통과해야 한다.

---

# 30. gate — 조건을 통과해야 하는 판정 관문

`gate`는 어떤 기능을 사용할지 결정하는 조건 검사다.

예:

```text
예측 신뢰도가 충분한가?
↓
Critic의 실제 데이터 근거가 충분한가?
↓
기본 행동보다 충분히 좋은가?
↓
모두 통과해야 행동 변경 허용
```

---

# 31. fallback — 기본 경로로 돌아가기

`fallback`은 새 방법을 믿기 어려울 때 **더 기본적인 안전한 선택 방식으로 돌아가는 것**이다.

현재 AASSR에서는 [Imagination](Imagination)의 근거가 부족하면 [Policy](Policy)의 기본 행동을 유지하는 방식이 대표적이다.

---

# 32. margin — 최소한 필요한 차이

두 행동의 예측 가치가 거의 같다면 작은 계산 오차 때문에 선택이 쉽게 뒤집힐 수 있다.

그래서:

```text
새 행동 가치가 기본 행동보다
최소한 일정 정도 이상 좋아야
행동 변경을 허용
```

하는 최소 차이를 둘 수 있다.

이런 기준을 `margin`이라고 부른다.

---

# 33. action surface — 현재 가능한 행동 집합

환경에서는 항상 모든 행동을 할 수 있는 것이 아닐 수 있다.

현재 상황에서 실제로 선택 가능한 행동들의 집합을 `action surface`라고 부르는 경우가 있다.

예:

```text
아직 객체를 발견하지 않음
→ 그 객체를 대상으로 하는 행동은 아직 없음
```

따라서 훈련과 평가에서 가능한 행동 집합이 다르면 공정성 문제가 생길 수 있다.

---

# 34. legal action — 현재 허용된 행동

`legal`은 법률적 의미가 아니라 **현재 환경 규칙상 실행 가능한 행동**이라는 뜻이다.

[Prophecy](Prophecy)는 가상 미래에서도 어떤 행동이 가능한지 예측해야 한다.

---

# 35. dynamics — 환경의 상태 변화 규칙

강화학습에서 `dynamics`는:

> **현재 상태에서 어떤 행동을 했을 때 다음 상태가 어떻게 변하는가?**

를 나타내는 환경의 변화 규칙이다.

[Prophecy](Prophecy)는 이 변화를 학습하는 세계 모델 역할을 한다.

---

# 36. stochastic — 확률적인

`stochastic`은 같은 상태에서 같은 행동을 해도 결과가 항상 하나로 고정되지 않을 수 있다는 뜻이다.

```text
같은 행동
├─ 결과 A
├─ 결과 B
└─ 결과 C
```

각 결과가 확률을 가진다.

반대 개념은 같은 입력이면 항상 같은 결과가 나오는 결정론적 환경이다.

---

# 37. relational — 관계 기반

`relational`은 객체 이름 자체보다 **객체 사이의 역할과 관계**를 본다는 뜻이다.

```text
route-17이라는 이름
```

보다:

```text
목록을 보여주는 경로라는 역할
```

을 학습하는 쪽에 가깝다.

관련: [관계 기반 표현과 일반화](Relational-Representation-and-Generalization)

---

# 38. multimodal — 서로 다른 여러 결과 형태가 존재하는

`multimodal`은 가능한 결과가 하나의 평균 근처에만 모이지 않고 **서로 다른 여러 종류로 나뉠 수 있다**는 뜻이다.

예:

```text
정상 응답
접근 거부
요청 제한
```

을 하나의 평균값으로 섞으면 실제 의미를 잃을 수 있다.

관련: [혼합·앙상블·보정](Mixture-Ensemble-and-Calibration)

---

# 39. support — 데이터 근거

AASSR에서 `support`는 “지원 기능”이라는 일반 영어 뜻이 아니라 **현재 판단 주변에 실제 학습 데이터가 얼마나 존재하는가**라는 뜻으로 자주 사용한다.

[Critic](Critic)이 높은 값을 예측해도 실제 비슷한 경험이 없다면 데이터 근거가 부족한 것이다.

관련: [가치 평가 데이터 근거와 OOD](Critic-Support-and-OOD)

---

# 40. fail closed — 근거가 부족하면 보수적으로 거부

`fail closed`는 판단 근거가 부족하거나 검증에 실패했을 때 **공격적으로 새 행동을 허용하기보다 기본 행동을 유지하는 설계**다.

AASSR에서는 불확실한 [Imagination](Imagination)이 [Policy](Policy)를 함부로 덮어쓰지 못하게 하는 데 사용한다.

---

# 빠른 구분표

| 영어 | 먼저 떠올릴 한국어 | 핵심 의미 |
|---|---|---|
| [현재(current)](Current-Status) | 현재 | 지금 기준으로 사용 중 |
| [실행 구조(runtime)](Current-Status) | 실제 실행 구조 | 프로그램이 실제로 실행하는 경로 |
| [명세(contract)](Current-Status) | 명세 | 반드시 지켜야 하는 의미·조건 |
| [과거 기록(historical)](Development-History) | 과거 기록 | 재현·분석을 위해 보존한 이전 결과 |
| [진단 실험(diagnostic)](Evidence-Matrix) | 진단 실험 | 원인을 좁히는 실험 |
| [증거(evidence)](Evidence-Matrix) | 증거 | 주장을 뒷받침하는 결과 |
| [연구 주장(claim)](Evidence-Matrix) | 연구 주장 | 현재 증거로 사실이라고 말하려는 문장 |
| [공개된(public)](State-Representation) | 에이전트에게 공개된 | 정상 상호작용으로 볼 수 있음 |
| [숨겨진(hidden)](MDP-and-POMDP) | 에이전트에게 숨겨진 | 환경 내부에는 있지만 직접 볼 수 없음 |
| [실제 환경에서 관측된(real)](Research-Jargon-Guide) | 실제 환경에서 관측된 | 상상이 아니라 실제 경험 |
| [모델이 상상한(imagined)](Research-Jargon-Guide) | 모델이 상상한 | 세계 모델이 만든 가상 미래 |
| [학습(training)](Terminology-Guide) | 학습 | 모델 파라미터를 바꿈 |
| [검증(validation)](Ablation-Benchmarking-and-Reproducibility) | 검증 | 개발 과정에서 상태 확인 |
| [평가(evaluation)](Ablation-Benchmarking-and-Reproducibility) | 평가 | 학습된 모델 성능 측정 |
| [학습을 멈춘(frozen)](Ablation-Benchmarking-and-Reproducibility) | 학습 중지 | 평가 중 모델을 바꾸지 않음 |
| [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility) | 실험 규칙 | 결과 보기 전에 정한 비교 방법 |
| [대리 지표(proxy)](Ablation-Benchmarking-and-Reproducibility) | 대리 지표 | 최종 목표 대신 원인을 보기 위한 값 |
| [탐색의 첫 행동(root)](Imagination) | 지금 실행할 첫 행동 | 계획 결과 중 현실에서 첫 번째로 쓸 행동 |
| [기본 행동 덮어쓰기(override)](Imagination) | 기본 행동 변경 | [Policy](Policy) 선택을 다른 행동으로 바꿈 |
| [기본 경로로 돌아가기(fallback)](Imagination) | 기본 경로 복귀 | 근거가 부족하면 기본 선택 유지 |
| [판정 관문(gate)](Terminology-Guide) | 판정 관문 | 특정 조건을 통과해야 기능 사용 |
| [최소 차이 기준(margin)](Imagination) | 최소 차이 기준 | 행동을 바꾸기 위해 요구하는 최소 우위 |
| [확률적(stochastic)](Stochasticity-Uncertainty-and-Probability) | 확률적 | 같은 입력에도 여러 결과 가능 |
| [관계 기반(relational)](Relational-Representation-and-Generalization) | 관계 기반 | 이름보다 역할과 관계를 봄 |
| [여러 결과 형태를 가진(multimodal)](Mixture-Ensemble-and-Calibration) | 여러 결과 형태 | 하나의 평균으로 설명하기 어려운 여러 결과 |
| [데이터 근거(support)](Critic-Support-and-OOD) | 실제 데이터 근거 | 판단 주변에 실제 경험이 있는 정도 |

---

# 함께 읽으면 좋은 문서

- [처음 읽는 사람을 위한 AASSR 안내서](Beginner-Guide)
- [한국어 중심 AASSR 용어 안내서](Terminology-Guide)
- [개념 지도](Concept-Index)
- [현재 연구 상태](Current-Status)
- [연구 질문-증거 연결표](Evidence-Matrix)
