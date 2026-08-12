# Policy

Policy는 AASSR에서 **현재 상태에서 실제로 실행할 기본 행동을 선택하는 model-free 학습기**다.

current-generation의 핵심은 단순 DQN 하나가 아니라 다음 두 신호를 의도적으로 분리한다는 점이다.

```text
외부 task return을 학습하는 DQN
+
별도의 information-value residual
```

> [!IMPORTANT]
> 현재 manifest 계약: `relational-invariant-dqn+information-residual-v1`  
> 핵심 구현: `src/aassr_v2/current_generation.py`의 `CurrentRelationalPolicy`

---

# 1. 연구 질문

> **희소한 외부 reward만 유지하면서도, 이름이 바뀐 unseen 환경에서 관계 구조를 이용해 현재 행동의 가치를 학습할 수 있는가?**

Policy는 AASSR의 fallback이기도 하다.

Prophecy나 Critic이 불확실하면 최종적으로 Policy 행동을 그대로 실행한다.

따라서 AASSR Full의 성능을 이해하려면 먼저 **Imagination이 없어도 동작하는 Policy가 무엇을 학습하는지**를 분리해야 한다.

---

# 2. 왜 raw state가 아니라 relational state인가?

훈련 환경에서 중요한 route가 `route-12`였다고 하자.

평가 seed에서는 같은 역할이 `route-31`이라는 새 이름을 가질 수 있다.

Raw identity를 강하게 사용하면:

```text
route-12 != route-31
-> 새로운 문제처럼 보임
```

Relational representation에서는:

```text
route-12 -> catalog-like role
route-31 -> catalog-like role

=> 같은 구조
```

로 볼 수 있다.

이것이 `dqn_raw`와 `dqn_relational`을 따로 두는 이유다.

```text
dqn_raw -> dqn_relational
= representation effect
```

AASSR 전체 효과와 representation 효과를 섞지 않는다.

---

# 3. 기본 DQN objective

Primitive action의 외부 task value는 DQN이 담당한다.

개념적으로 일반적인 TD target은 다음 형태다.

```math
y_t = r_t + \gamma (1-d_t) \max_{a'} Q_{\theta^-}(S_{t+1},a')
```

AASSR의 외부 reward contract는 좁게 유지한다.

```text
success       +1
true failure  -1
otherwise      0
```

즉 route 발견, token 발견, object 발견 같은 중간 이벤트를 외부 reward로 바꾸지 않는다.

---

# 4. Information residual

희소 reward에서는 어떤 행동이 바로 성공을 만들지는 않더라도 **나중의 의사결정을 가능하게 하는 정보**를 얻을 수 있다.

AASSR은 이 신호를 외부 reward에 합쳐버리지 않고 별도 residual로 유지한다.

개념적으로:

```math
Q_{total}(S,A)
= Q_{task}(S,A) + I(S,A)
```

여기서:

- `Q_task`: 실제 환경 sparse return을 학습하는 DQN 값
- `I`: 내부 information-value estimate

핵심은 다음이다.

```text
I(S,A)는 환경 reward가 아니다.
```

DQN target에 `정보를 얻었으니 +0.2` 같은 shaping reward를 주는 구조가 아니다.

---

# 5. 왜 둘을 분리하는가?

외부 reward와 information value를 하나로 섞으면 연구 질문이 달라질 수 있다.

예를 들어:

```text
route 발견 +0.2
login      +0.3
proof      +1.0
```

처럼 만들면 agent가 성공한 이유가

- sparse reward를 스스로 연결했기 때문인지,
- 사람이 설계한 중간 reward를 따라갔기 때문인지

분리하기 어렵다.

현재 구조는

```text
External task objective
!=
Internal information signal
```

을 유지해 이 혼동을 줄인다.

---

# 6. Information residual은 어떻게 저장되는가?

Primitive action에 대해서는 relational state key와 relational action key의 쌍을 사용한다.

개념적으로:

```text
(relational_state_key, relational_action_key)
    -> running information value
```

따라서 concrete ID가 바뀌어도 관계 구조가 같으면 residual을 재사용할 수 있다.

Skill은 primitive와 다른 identity를 사용하므로 별도의 skill value table을 가진다.

---

# 7. 행동 선택

Evaluation에서 exploration을 끄면 Policy는 가능한 행동을 점수화하고 가장 높은 행동을 고른다.

```text
score(A)
= DQN external value
+ information residual
```

Training에서는 epsilon-greedy exploration을 사용할 수 있다.

```text
with probability epsilon:
    random legal action
otherwise:
    highest ranked action
```

중요한 것은 Imagination이 이 Policy를 대체하는 별도 학습 actor가 아니라는 점이다.

Policy는 언제나 기본 행동을 제공한다.

---

# 8. Imagination과 Policy의 관계

현재 decision flow는 다음처럼 이해하면 된다.

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

즉 Imagination이 실패하거나 불확실하더라도 agent는 행동할 수 있다.

이것이 fail-closed 설계의 기준점이다.

---

# 9. 왜 Imagination으로 Policy를 학습시키지 않는가?

current protocol에서는 imagined experience가 real Policy를 직접 강화하지 않는다.

이유는 두 가지다.

## 9.1 Model error self-amplification 방지

잘못된 Prophecy가 만든 imagined transition을 Policy 학습 데이터로 쓰면:

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

Training trajectory 자체를 Imagination이 바꾸면 OFF/ON의 차이가 planner marginal effect만이 아니게 된다.

---

# 10. Policy Memory와 imagined delta

Planner 내부에서는 branch-local Policy memory를 사용할 수 있다.

이것은 실제 persistent Policy weights를 업데이트하는 것과 다르다.

```text
branch-local imagined memory
= 해당 상상 경로 안의 temporary preference

persistent DQN / information table
= 실제 학습 상태
```

이 경계를 유지하면 상상 중 계산과 실제 학습 상태를 분리할 수 있다.

---

# 11. Policy의 공정성 control

현재 실험에서 Policy 관련 효과는 다음 순서로 분리한다.

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

따라서 `AASSR Full > raw DQN` 하나만 보고 모든 차이를 Imagination 효과라고 말하면 안 된다.

---

# 12. 실패 모드

## 12.1 Concrete memorization

ID 자체에 과도하게 의존하면 unseen seed transfer가 무너진다.

대응: relational state/action representation.

## 12.2 Self-loop preference

특정 행동 Q가 조금 높게 고정되면 같은 상태에서 무한 반복할 수 있다.

대응: exact semantic ASEQ guard.

## 12.3 Information residual dominance

내부 information signal이 외부 task objective를 압도하면 탐색만 하고 목표를 끝내지 못할 수 있다.

그래서 외부 DQN과 residual을 별도 신호로 관리하고 diagnostic을 분리해야 한다.

## 12.4 OOD action ranking

새로운 relational region에서 Policy 자체도 틀릴 수 있다.

Imagination은 이를 고칠 가능성이 있지만, 반대로 Prophecy/Critic도 OOD라면 Policy를 유지하는 쪽이 더 안전할 수 있다.

---

# 13. 연구 가설

Policy에 대한 검증은 다음처럼 단계적으로 보는 것이 좋다.

```text
H1. raw DQN보다 relational DQN이 unseen transfer에 유리한가?
H2. information residual이 shaping reward 없이 useful exploration을 만드는가?
H3. ASEQ와 함께 쓸 때 self-loop가 줄어드는가?
H4. AASSR no-Imagination이 relational DQN보다 나은가?
H5. Full에서 override되지 않은 Policy fallback이 안정적으로 작동하는가?
```

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

---

다음으로 읽기:

- **[ASEQ](ASEQ)**
- **[Knowledge](Knowledge)**
- **[Imagination](Imagination)**
- **[Experiments](Experiments)**
