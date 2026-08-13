# Policy — 기본 정책 모델

[Policy(정책 모델)](Policy)는 AASSR에서 **현재 상태에서 실제로 실행할 기본 행동을 선택하는 [model-free 강화학습](Reinforcement-Learning) 학습기**다.

[현재 세대(current-generation)](Current-Status)의 핵심은 단순 [DQN](Q-Learning-DQN-and-TD) 하나가 아니라 다음 두 신호를 의도적으로 분리한다는 점이다.

```text
외부 task return을 학습하는 DQN
+
별도의 information-value residual
```

> [!**중요**]
> 현재 manifest 계약: `relational-invariant-dqn+information-residual-v1`  
> 핵심 구현: `src/aassr_v2/current_generation.py`의 `CurrentRelationalPolicy`

---

# 0. 먼저 알아두면 좋은 개념

이 문서에서 처음 보는 단어가 있다면 다음 순서로 읽으면 된다.

- [Reinforcement Learning](Reinforcement-Learning) — [정책(policy)](Policy), [보상(reward)](Sparse-Reward-and-Credit-Assignment), [누적 보상(return)](Value-Functions-and-Bellman-Equation), [환경 예측 모델 없이 직접 학습하는(model-free)](Reinforcement-Learning)
- [Value Functions & Bellman Equation](Value-Functions-and-Bellman-Equation) — `Q(s,a)`, 누적 보상, discounting
- [Q-Learning, DQN & TD](Q-Learning-DQN-and-TD) — [DQN(딥 Q-네트워크)](Q-Learning-DQN-and-TD), TD [대상 또는 학습 목표값(target)](Terminology-Guide), 대상/목표값 [신경망(network)](Neural-Networks-and-Optimization), [저장된 경험의 재사용(replay)](Replay-Buffer-and-Episode-Boundaries)
- [Exploration & Exploitation](Exploration-and-Exploitation) — epsilon-greedy
- [Information Theory & Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation) — [정보(information)](Information-Theory-and-Intrinsic-Motivation) [증가량(gain)](Ablation-Benchmarking-and-Reproducibility), intrinsic [학습 신호(signal)](Information-Theory-and-Intrinsic-Motivation)
- [Relational Representation & Generalization](Relational-Representation-and-Generalization) — [식별자(identifier)](State-Representation) [이름 순서를 바꾸는 순열(permutation)](Relational-Representation-and-Generalization)과 [전이(transfer)](Relational-Representation-and-Generalization)
- [Sparse Reward & Credit Assignment](Sparse-Reward-and-Credit-Assignment) — 왜 중간 보상를 따로 만들지 않는가?

---

# 1. 연구 질문

> **[희소한 외부 reward](Sparse-Reward-and-Credit-Assignment)만 유지하면서도, 이름이 바뀐 [학습 중 보지 못한(unseen)](Relational-Representation-and-Generalization) 환경에서 [관계 구조](Relational-Representation-and-Generalization)를 이용해 현재 행동의 가치를 학습할 수 있는가?**

[Policy](Policy)는 AASSR의 [기본 경로로 돌아가기(fallback)](Imagination)이기도 하다.

[Prophecy](Prophecy)나 [Critic](Critic)이 불확실하면 최종적으로 [Policy](Policy) 행동을 그대로 실행한다.

따라서 AASSR [전체 AASSR 조건(Full)](Experiments)의 성능을 이해하려면 먼저 **[Imagination](Imagination)이 없어도 동작하는 [Policy](Policy)가 무엇을 학습하는지**를 분리해야 한다.

---

# 2. 왜 raw state가 아니라 relational state인가?

훈련 환경에서 중요한 route가 `route-12`였다고 하자.

평가 [난수 시드(seed)](Ablation-Benchmarking-and-Reproducibility)에서는 같은 역할이 `route-31`이라는 새 이름을 가질 수 있다.

Raw [식별 방식(identity)](State-Representation)를 강하게 사용하면:

```text
route-12 != route-31
-> 새로운 문제처럼 보임
```

[Relational representation](Relational-Representation-and-Generalization)에서는:

```text
route-12 -> catalog-like role
route-31 -> catalog-like role

=> 같은 구조
```

로 볼 수 있다.

실제 [현재(current)](Current-Status) [입력(input)](Terminology-Guide) 계약은 [State Representation](State-Representation)에서 더 자세히 설명한다.

이것이 `dqn_raw`와 `dqn_relational`을 따로 두는 이유다.

```text
dqn_raw -> dqn_relational
= representation effect
```

AASSR 전체 효과와 [표현(representation)](Relational-Representation-and-Generalization) 효과를 섞지 않는다. 이런 효과 분리는 [ablation/control](Ablation-Benchmarking-and-Reproducibility)의 핵심이다.

---

# 3. 기본 DQN objective

Primitive [행동(action)](Reinforcement-Learning)의 외부 [연구 과제(task)](Sparse-Reward-Problem) [가치(value)](Value-Functions-and-Bellman-Equation)는 [DQN](Q-Learning-DQN-and-TD)이 담당한다.

개념적으로 일반적인 [TD target](Q-Learning-DQN-and-TD)은 다음 형태다.

```math
y_t = r_t + \gamma (1-d_t) \max_{a'} Q_{\theta^-}(S_{t+1},a')
```

여기서:

- `r_t`: 현재 [상태 전이(transition)](MDP-and-POMDP)의 외부 보상
- `γ`: [discount factor](Value-Functions-and-Bellman-Equation)
- `d_t`: [bootstrap을 끊는 episode boundary](Replay-Buffer-and-Episode-Boundaries)
- `Q_{θ^-}`: 대상/목표값 Q-network

AASSR의 외부 보상 [명세(contract)](Current-Status)는 좁게 유지한다.

```text
success       +1
true failure  -1
otherwise      0
```

즉 route 발견, token 발견, object 발견 같은 중간 이벤트를 외부 보상로 바꾸지 않는다. 이것은 [reward shaping](Sparse-Reward-and-Credit-Assignment)을 의도적으로 피하는 연구 설계다.

---

# 4. Information residual

[희소 reward](Sparse-Reward-and-Credit-Assignment)에서는 어떤 행동이 바로 성공을 만들지는 않더라도 **나중의 의사결정을 가능하게 하는 정보**를 얻을 수 있다.

AASSR은 이 신호를 외부 보상에 합쳐버리지 않고 별도 [기본 값에 더하는 잔차(residual)](Policy)로 유지한다.

개념적으로:

```math
Q_{total}(S,A)
= Q_{task}(S,A) + I(S,A)
```

여기서:

- `Q_task`: 실제 환경 [드문 보상만 있는(sparse)](Sparse-Reward-and-Credit-Assignment) 누적 보상을 학습하는 [DQN](Q-Learning-DQN-and-TD) 값
- `I`: 내부 information-value [추정값(estimate)](Value-Functions-and-Bellman-Equation)

핵심은 다음이다.

```text
I(S,A)는 환경 reward가 아니다.
```

[DQN](Q-Learning-DQN-and-TD) 대상/목표값에 `정보를 얻었으니 +0.2` 같은 [shaping reward](Sparse-Reward-and-Credit-Assignment)를 주는 구조가 아니다.

`information gain`, `intrinsic motivation`, `curiosity`와 어떤 점이 같고 다른지는 [Information Theory & Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation)에서 설명한다.

---

# 5. 왜 둘을 분리하는가?

외부 보상와 정보 가치를 하나로 섞으면 연구 질문이 달라질 수 있다.

예를 들어:

```text
route 발견 +0.2
login      +0.3
proof      +1.0
```

처럼 만들면 [에이전트(agent)](Reinforcement-Learning)가 성공한 이유가

- [희소 보상(sparse reward)](Sparse-Reward-and-Credit-Assignment)를 스스로 연결했기 때문인지,
- 사람이 설계한 중간 보상를 따라갔기 때문인지

분리하기 어렵다.

현재 구조는

```text
External task objective
!=
Internal information signal
```

을 유지해 이 혼동을 줄인다.

이 경계는 [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation) 및 [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility) 관점에서도 중요하다.

---

# 6. Information residual은 어떻게 저장되는가?

Primitive 행동에 대해서는 [relational state key와 relational action key](Relational-Representation-and-Generalization)의 쌍을 사용한다.

개념적으로:

```text
(relational_state_key, relational_action_key)
    -> running information value
```

따라서 [실제 개체를 구분하는(concrete)](State-Representation) ID가 바뀌어도 관계 구조가 같으면 잔차을 재사용할 수 있다.

[Skill](Skills)은 [더 쪼개지 않는 기본 행동 단위(primitive)](Hierarchical-RL-and-Skills)와 다른 식별 방식를 사용하므로 별도의 [재사용 가능한 기술(skill)](Skills) 가치 table을 가진다.

---

# 7. 행동 선택

Evaluation에서 [탐색(exploration)](Exploration-and-Exploitation)을 끄면 [Policy](Policy)는 가능한 행동을 점수화하고 가장 높은 행동을 고른다.

```text
score(A)
= DQN external value
+ information residual
```

[학습(Training)](Reinforcement-Learning)에서는 [epsilon-greedy exploration](Exploration-and-Exploitation)을 사용할 수 있다.

```text
with probability epsilon:
    random legal action
otherwise:
    highest ranked action
```

중요한 것은 [Imagination(가상 미래 탐색)](Imagination)이 이 [Policy](Policy)를 대체하는 별도 학습 actor가 아니라는 점이다.

[Policy](Policy)는 언제나 기본 행동을 제공한다.

---

# 8. Imagination과 Policy의 관계

현재 [의사결정(decision)](Chance-and-Decision-Nodes) flow는 다음처럼 이해하면 된다.

```text
Policy proposes A_policy
        |
        v
Imagination evaluates roots
        |
        v
reliable + supported + enough advantage ?
      /                         \
    yes                          no
     |                            |
A_imagined 실행               A_policy 실행
```

여기서:

- `reliable`: [Prophecy prediction reliability](Stochasticity-Uncertainty-and-Probability)
- `supported`: [Critic local support](Critic-Support-and-OOD)
- `advantage`: [계획기(planner)](Counterfactual-Planning-and-Search) [선택 후보(candidate)](Terminology-Guide)와 [Policy](Policy) [탐색의 첫 행동(root)](Imagination)의 가치 차이

즉 [Imagination](Imagination)이 실패하거나 불확실하더라도 에이전트는 행동할 수 있다.

이것이 [fail-closed](Critic-Support-and-OOD) 설계의 기준점이다.

---

# 9. 왜 Imagination으로 Policy를 학습시키지 않는가?

현재 [실험 규칙(protocol)](Ablation-Benchmarking-and-Reproducibility)에서는 [모델이 상상한(imagined)](Research-Jargon-Guide) [경험(experience)](Replay-Buffer-and-Episode-Boundaries)가 [실제 환경에서 관측된(real)](Research-Jargon-Guide) [Policy](Policy)를 직접 강화하지 않는다.

이유는 두 가지다.

## 9.1 Model error self-amplification 방지

잘못된 [world model](Model-Based-RL-and-World-Models)이 만든 가상 상태 전이을 [Policy](Policy) 학습 데이터로 쓰면:

```text
world-model error
-> imagined experience
-> Policy update
-> 더 많은 잘못된 행동
```

처럼 오류가 자기증폭될 수 있다.

## 9.2 Same-checkpoint 비교 유지

현재 핵심 실험은:

```text
one AASSR training
        |
        v
frozen checkpoint
   /           \
OFF eval     ON eval
```

이다.

학습 [경험 경로(trajectory)](Reinforcement-Learning) 자체를 [Imagination](Imagination)이 바꾸면 OFF/ON의 차이가 계획기 [다른 조건이 같을 때의 추가 기여(marginal)](Ablation-Benchmarking-and-Reproducibility) [효과(effect)](Ablation-Benchmarking-and-Reproducibility)만이 아니게 된다.

이 비교 원칙은 [same-checkpoint evaluation](Ablation-Benchmarking-and-Reproducibility)에서 더 깊게 설명한다.

---

# 10. Policy Memory와 imagined delta

[계획기(Planner)](Counterfactual-Planning-and-Search) 내부에서는 branch-local [Policy](Policy) [기억(memory)](GRU-and-Sequence-Models)를 사용할 수 있다.

이것은 실제 [에피소드가 끝나도 유지되는(persistent)](Knowledge) [Policy](Policy) weights를 업데이트하는 것과 다르다.

```text
branch-local imagined memory
= 해당 상상 경로 안의 temporary preference

persistent DQN / information table
= 실제 학습 상태
```

이 경계를 유지하면 상상 중 계산과 실제 학습 상태를 분리할 수 있다.

`imagined fact/experience`와 `real fact/transition`을 분리하는 일반 원칙은 [Causality, Leakage & Fair Evaluation](Causality-Leakage-and-Evaluation)에서 본다.

---

# 11. Policy의 공정성 control

현재 실험에서 [Policy](Policy) 관련 효과는 다음 순서로 분리한다.

```text
dqn_raw
   |
   | relational representation
   v
dqn_relational
   |
   | Knowledge / ASEQ / information-value 등 AASSR stack
   v
aassr_current_no_imagination
   |
   | counterfactual planner
   v
aassr_current_full
```

따라서 `AASSR Full > raw DQN` 하나만 보고 모든 차이를 [Imagination](Imagination) 효과라고 말하면 안 된다.

각 화살표가 어떤 가설을 검증하는지는 [Ablation, Benchmarking & Reproducibility](Ablation-Benchmarking-and-Reproducibility)에서 정리한다.

---

# 12. 실패 모드

## 12.1 Concrete memorization

ID 자체에 과도하게 의존하면 학습 중 보지 못한 난수 시드 전이가 무너진다.

대응: [relational state/action representation](Relational-Representation-and-Generalization).

## 12.2 Self-loop preference

특정 행동 Q가 조금 높게 고정되면 같은 상태에서 무한 반복할 수 있다.

대응: [정확히 동일한(exact)](ASEQ) [의미 기준(semantic)](State-Representation) [ASEQ](ASEQ) [잘못된 행동을 제한하는 보호 규칙(guard)](ASEQ).

## 12.3 Information residual dominance

내부 정보 학습 신호이 외부 연구 과제 [학습 목표(objective)](Terminology-Guide)를 압도하면 탐색만 하고 목표를 끝내지 못할 수 있다.

그래서 외부 [DQN](Q-Learning-DQN-and-TD)과 잔차을 별도 신호로 관리하고 [진단 실험(diagnostic)](Evidence-Matrix)을 분리해야 한다.

관련 일반 [실패(failure)](Replay-Buffer-and-Episode-Boundaries) [서로 다른 결과 유형(mode)](Mixture-Ensemble-and-Calibration)는 [Intrinsic Motivation](Information-Theory-and-Intrinsic-Motivation)에서 본다.

## 12.4 OOD action ranking

새로운 [관계 기반(relational)](Relational-Representation-and-Generalization) [상태 공간의 영역(region)](Critic-Support-and-OOD)에서 [Policy](Policy) 자체도 틀릴 수 있다.

[Imagination](Imagination)은 이를 고칠 가능성이 있지만, 반대로 [Prophecy(미래 예측 모델)](Prophecy)/[Critic(미래 가치 평가기)](Critic)도 [OOD](Critic-Support-and-OOD)라면 [Policy](Policy)를 유지하는 쪽이 더 안전할 수 있다.

---

# 13. 연구 가설

[Policy](Policy)에 대한 검증은 다음처럼 단계적으로 보는 것이 좋다.

```text
H1. raw DQN보다 relational DQN이 unseen transfer에 유리한가?
H2. information residual이 shaping reward 없이 useful exploration을 만드는가?
H3. ASEQ와 함께 쓸 때 self-loop가 줄어드는가?
H4. AASSR no-Imagination이 relational DQN보다 나은가?
H5. Full에서 override되지 않은 Policy fallback이 안정적으로 작동하는가?
```

이 가설들은 하나의 성공률 숫자로 뭉치지 않고 별도 [ablation](Ablation-Benchmarking-and-Reproducibility)으로 확인하는 것이 좋다.

---

# 14. 관련 코드

```text
src/aassr_v2/current_generation.py
  - RelationalInvariantDQN
  - CurrentRelationalPolicy

src/aassr_v2/current_hardware.py
  - accelerator-aware DQN path

src/aassr_v2/current_confidence_gate.py
  - Policy vs Imagination final selection
```

신경망, [학습 손실(loss)](Loss-Functions-and-Class-Imbalance), GPU [여러 입력 묶음(batch)](Reproduction) 같은 구현 기초는:

- [Neural Networks & Optimization](Neural-Networks-and-Optimization)
- [Loss Functions & Class Imbalance](Loss-Functions-and-Class-Imbalance)

에서 볼 수 있다.

---

다음으로 읽기:

- **[ASEQ](ASEQ)**
- **[Knowledge](Knowledge)**
- **[Imagination](Imagination)**
- **[Experiments](Experiments)**
- **[Concept Index](Concept-Index)**
