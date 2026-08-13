# Information Theory and Intrinsic Motivation

AASSR의 [Policy](Policy)는 외부 sparse task [누적 보상(return)](Value-Functions-and-Bellman-Equation)과 별도로 **[정보 가치 잔차(information-value residual)](Policy)**을 유지한다. 이 페이지는 그 배경을 이해하는 데 필요한 정보이론과 [내재 동기(intrinsic motivation)](Information-Theory-and-Intrinsic-Motivation) 개념을 정리한다.

AASSR의 [현재(current)](Current-Status) residual이 아래 모든 이론을 그대로 구현한다는 뜻은 아니다. **관련 개념을 구분하기 위한 배경 문서**다.

---

# 1. 정보란 무엇인가?

직관적으로 어떤 [관측(observation)](MDP-and-POMDP)이 우리가 모르던 것을 줄여주면 정보를 얻었다고 말할 수 있다.

예:

```text
행동 전:
어떤 route가 중요한지 모름

response 관측 후:
새 route의 역할을 알게 됨
```

Task [보상(reward)](Sparse-Reward-and-Credit-Assignment)는 즉시 `0`이어도 이후 [행동(action)](Reinforcement-Learning) choice는 크게 달라질 수 있다.

---

# 2. Self-information

확률 `p(x)`인 사건 `x`의 정보량을:

```math
I(x)=-\log p(x)
```

로 정의할 수 있다.

드문 사건일수록 더 큰 surprise를 준다.

하지만 **surprise가 크다고 task에 유용한 정보라는 뜻은 아니다.**

랜덤 noise도 매우 surprising할 수 있다.

---

# 3. Entropy

확률분포의 평균적인 불확실성을 나타낸다.

```math
H(X)=-\sum_x p(x)\log p(x)
```

Distribution이 한 outcome에 거의 확정되어 있으면 entropy가 낮다.

여러 outcome이 비슷한 probability를 가지면 entropy가 높다.

관련 페이지:

- [Stochasticity, Uncertainty & Probability](Stochasticity-Uncertainty-and-Probability)

---

# 4. Conditional entropy

다른 변수 `Y`를 알고 있을 때 `X`의 남은 uncertainty:

```math
H(X|Y)
```

이다.

[관측(Observation)](MDP-and-POMDP)을 얻은 뒤 [숨겨진(hidden)](MDP-and-POMDP) situation에 대한 uncertainty가 얼마나 줄었는지 생각할 때 연결된다.

---

# 5. Mutual information

두 변수 사이에 공유되는 정보량:

```math
I(X;Y)=H(X)-H(X|Y)
```

이다.

예:

```text
Action outcome Y를 관측하면
hidden/task-relevant state X에 대해 얼마나 더 알게 되는가?
```

를 formalize하는 방향과 연결된다.

---

# 6. KL divergence

두 probability distribution 차이를 측정하는 대표량:

```math
D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}
```

대칭 distance는 아니다.

Bayesian information gain에서는 행동 전 belief와 행동 후 posterior 사이 KL divergence를 사용할 수 있다.

---

# 7. Information gain

행동 전 belief `p(z)`와 관측 후 posterior `p(z|o)`의 변화:

```math
IG=D_{KL}(p(z|o)\|p(z))
```

를 정보 획득으로 볼 수 있다.

```text
행동 A
→ observation
→ 중요한 uncertainty 크게 감소
→ information gain 큼
```

---

# 8. Task information과 Novelty는 다르다

처음 보는 관측이라고 모두 유용한 것은 아니다.

```text
새로운 랜덤 문자열
→ novelty 높음
→ task에는 아무 도움 없음
```

좋은 information signal은 단순 novelty보다 **future decision quality와 연결**되어야 한다.

AASSR가 [정보 가치 잔차(information residual)](Policy)을 이해할 때 이 차이가 중요하다.

---

# 9. Intrinsic motivation

환경의 외부 task 보상와 별개로 [에이전트(agent)](Reinforcement-Learning) 내부에서 [탐색(exploration)](Exploration-and-Exploitation)을 유도하는 signal을 만든다.

대표적인 계열:

- curiosity
- novelty
- count-based 탐색
- information gain
- empowerment

관련 페이지:

- [Exploration & Exploitation](Exploration-and-Exploitation)

---

# 10. Extrinsic vs Intrinsic

```text
Extrinsic reward
= environment가 task 성공/실패에 대해 주는 신호

Intrinsic signal
= agent 내부에서 exploration/knowledge acquisition을 위해 계산한 신호
```

AASSR 현재 design의 중요한 원칙:

```text
External sparse reward
!=
Information residual
```

이다.

---

# 11. Curiosity by prediction error

한 방법은 [세계 모델(world model)](Model-Based-RL-and-World-Models) [예측(prediction)](Terminology-Guide) error가 큰 관측을 흥미롭다고 보는 것이다.

```math
r_{int}\propto\|\hat s'-s'\|
```

처음 보는 [환경의 상태 변화 규칙(dynamics)](Model-Based-RL-and-World-Models)를 탐색하는 데 도움이 될 수 있다.

하지만 [환경(environment)](Reinforcement-Learning) noise가 본질적으로 예측 불가능하면 계속 큰 보상가 생길 수 있다.

---

# 12. Noisy-TV problem

[에이전트(Agent)](Reinforcement-Learning)가 예측하기 어려운 랜덤 noise source만 계속 바라보는 [실패(failure)](Replay-Buffer-and-Episode-Boundaries) mode다.

```text
랜덤 noise
→ prediction error 계속 큼
→ curiosity reward 큼
→ task progress 없이 반복
```

그래서 "예측 불가능함"과 "유용한 정보"는 같지 않다.

---

# 13. Count-based exploration

덜 방문한 [상태(state)](State-Representation)에 bonus를 주는 방법이다.

```math
B(s)\propto\frac1{\sqrt{N(s)}}
```

큰/continuous 상태 space에서는 exact 상태 count가 어렵기 때문에 pseudo-count나 [표현(representation)](Relational-Representation-and-Generalization)-based count를 사용할 수 있다.

AASSR에서는 [관계 기반(relational)](Relational-Representation-and-Generalization) 상태를 쓰므로 structural novelty를 정의할 가능성도 있지만 현재 main external 보상에 count bonus를 넣는 구조는 아니다.

---

# 14. Empowerment

에이전트의 행동이 미래 상태를 얼마나 다양하게 제어할 수 있는지와 관련된 intrinsic [학습 목표(objective)](Terminology-Guide)다.

대략 행동과 future 상태 사이 mutual information을 최대화하는 관점과 연결된다.

AASSR 현재 정보 가치 잔차과 동일한 개념은 아니지만 **정보/통제 가능성에 내부 가치를 줄 수 있다**는 관련 연구 배경이다.

---

# 15. Information value와 Expected task return

어떤 정보 행동은 지금 task 누적 보상 추정치가 낮아도 미래 선택을 개선할 수 있다.

```text
Action A
→ reward 0
→ 새 route 정보 획득
→ 다음 decision space가 바뀜
→ 몇 단계 뒤 success
```

Information [가치(value)](Value-Functions-and-Bellman-Equation)를 완전히 external 누적 보상 안에서만 학습하려면 성공 sample을 통해 긴 [보상 책임 배분(credit assignment)](Sparse-Reward-and-Credit-Assignment)가 필요하다.

AASSR [Policy(정책 모델)](Policy)는 별도 residual로 이 내부 값을 추적한다.

---

# 16. 왜 external reward에 합치지 않는가?

```math
r'=r_{task}+\beta r_{info}
```

처럼 합치면 [학습 주체(learner)](Terminology-Guide) 학습 목표 자체가 바뀐다.

AASSR의 연구 질문은 sparse external 보상를 유지하는 것이므로:

```text
DQN external Q
+
separate information residual
```

로 분리한다.

관련 페이지:

- [Policy](Policy)
- [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment)

---

# 17. Residual이라는 말

기본 [학습 모델(model)](Terminology-Guide) [출력(output)](Terminology-Guide)에 추가되는 보정값을 residual이라고 부를 수 있다.

AASSR 개념식:

```math
score(S,A)=Q_{task}(S,A)+I(S,A)
```

여기서 `I`가 정보 가치 잔차이다.

하지만 `I`는 task 보상 자체가 아니다.

---

# 18. Information residual dominance

Internal signal이 너무 크면 에이전트가 목표를 끝내기보다 계속 정보를 모으는 행동만 선호할 수 있다.

```text
탐색 → 정보
탐색 → 정보
탐색 → 정보
목표 completion 안 함
```

그래서 [진단 실험(diagnostic)](Evidence-Matrix)에서 external Q와 information [구성요소(component)](Research-Architecture)를 따로 보는 것이 중요하다.

---

# 19. Information과 Knowledge

```text
Information value
= 어떤 행동이 future decision에 줄 수 있는 내부 가치

Knowledge
= 실제 response에서 획득해 명시적으로 저장한 사실
```

같은 개념이 아니다.

관련 페이지:

- [Knowledge](Knowledge)

---

# 20. Information과 Uncertainty

Uncertainty가 높은 행동이 항상 informative한 것은 아니다.

```text
높은 uncertainty
→ 관측해도 계속 random
→ information gain은 낮을 수 있음
```

반대로 현재 outcome은 거의 deterministic하지만 중요한 숨겨진 fact를 공개하는 행동은 매우 informative할 수 있다.

---

# 21. Information와 Prophecy

World 학습 모델은 행동 outcome을 예측한다.

Information-seeking [계획기(planner)](Counterfactual-Planning-and-Search)라면 "이 행동이 학습 모델 uncertainty를 얼마나 줄일까?"까지 계획할 수 있다.

AASSR 현재 [Imagination(가상 미래 탐색)](Imagination)의 주요 학습 목표는 external sparse-누적 보상 [계획(planning)](Counterfactual-Planning-and-Search)이며, uncertainty 자체를 positive task 가치로 사용하지 않는다.

---

# 22. Information signal의 공정성

Internal information signal이 [표준 비교 실험(benchmark)](Ablation-Benchmarking-and-Reproducibility)의 숨겨진 correct structure를 직접 참조하면 사실상 shaping/oracle이 된다.

허용 가능한 방향:

```text
실제 observation 변화
실제 action availability 변화
agent가 관측 가능한 novelty
```

주의해야 할 방향:

```text
hidden target까지 거리
정답 workflow stage
```

관련 페이지:

- [Causality, Leakage & Evaluation](Causality-Leakage-and-Evaluation)

---

# 23. AASSR 연구에서 확인해야 할 질문

```text
H1. information residual이 최초 성공 discovery를 돕는가?
H2. residual이 external Q를 압도하지 않는가?
H3. unseen relational state에서도 transfer되는가?
H4. 같은 external reward contract에서 실제 success가 증가하는가?
H5. residual OFF/ON ablation에서 효과가 분리되는가?
```

---

# 24. 다음으로 읽기

- [Policy](Policy)
- [Exploration & Exploitation](Exploration-and-Exploitation)
- [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment)
- [Knowledge](Knowledge)
- [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)

관련 색인: **[Concept Index](Concept-Index)**