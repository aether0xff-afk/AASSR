# Ablation, Benchmarking and Reproducibility

이 페이지는 AASSR의 성능 수치를 **어떻게 연구 주장으로 바꿀 수 있는가**를 다룬다.

좋은 모델을 만드는 것과, **왜 좋아졌는지 증명하는 것**은 다른 문제다.

핵심 개념은 다음과 같다.

```text
baseline
control
ablation
confounder
seed
training budget
unseen evaluation
same-checkpoint comparison
metric
variance / uncertainty
reproducibility
source of truth
```

---

# 1. Benchmark란?

**Benchmark**는 여러 방법을 같은 조건에서 비교하기 위해 정의한 문제, 데이터, protocol, metric의 묶음이다.

단순히 환경 하나만 있으면 benchmark가 아니다.

좋은 benchmark에는 적어도 다음이 필요하다.

```text
Task definition
Observation contract
Action contract
Reward contract
Training budget
Evaluation split
Metrics
Baselines
Random seeds
Reproduction procedure
```

AASSR current pentest lab도 이 관점에서 봐야 한다.

---

# 2. Baseline

**Baseline**은 새 방법이 실제로 어떤 이득을 만드는지 비교하기 위한 기준 모델이다.

예:

```text
AASSR success = 30%
```

만 보면 좋은지 나쁜지 알기 어렵다.

같은 환경에서:

```text
Random = 0%
Raw DQN = 2%
Relational DQN = 8%
DreamerV3 = 15%
AASSR = 30%
```

처럼 비교해야 의미가 생긴다.

하지만 여기서도 **각 baseline이 정확히 무엇을 통제하는지**가 중요하다.

---

# 3. Control condition

Control은 특정 요인을 바꾸지 않은 비교 조건이다.

예를 들어 relational representation의 효과를 알고 싶다면:

```text
Raw DQN
vs
Relational DQN
```

에서 나머지 training budget, reward, environment, evaluation split은 가능한 한 같아야 한다.

그러면 차이를 representation에 더 직접적으로 귀속할 수 있다.

---

# 4. Ablation study

**Ablation**은 전체 시스템에서 특정 구성 요소를 제거하거나 바꿔 그 요소의 기여를 측정하는 실험이다.

예:

```text
Full AASSR
vs
AASSR without Imagination
```

차이가 크면 Imagination이 성능에 영향을 주었다는 evidence가 된다.

하지만 ablation condition이 다른 곳까지 바뀌면 해석이 깨진다.

---

# 5. AASSR의 핵심 비교 사슬

현재 AASSR 연구에서 중요한 기본 구조:

```text
dqn_raw
   |
   | relational representation effect
   v
dqn_relational
   |
   | AASSR stack beyond representation
   v
aassr_current_no_imagination
   |
   | Imagination marginal effect
   v
aassr_current_full
```

추가로 model-based RL 계열의 강한 비교점으로 DreamerV3 relational adapter를 둔다.

각 화살표가 서로 다른 연구 질문에 답한다.

---

# 6. 왜 `AASSR vs DQN` 하나로는 부족한가?

AASSR Full과 raw DQN 사이에는 여러 차이가 동시에 있다.

예:

- relational state representation
- relational action representation
- ASEQ
- Knowledge
- information residual
- Prophecy
- Calibration
- Critic
- Skill
- Imagination

따라서:

```text
AASSR Full > Raw DQN
```

이어도 **어느 요소 때문에 좋아졌는지 알 수 없다.**

이를 confounding이라고 볼 수 있다.

---

# 7. Confounder

**Confounder**는 결과에 영향을 주지만 실험에서 관심 있는 변수와 함께 변해 원인 해석을 어렵게 하는 요인이다.

예:

```text
Model A: 2k transitions, CPU
Model B: 20k transitions, GPU, larger network
```

B가 더 좋더라도 algorithm 때문인지 data/compute 때문인지 알기 어렵다.

AASSR 비교에서도:

- transition budget
- checkpoint
- observation representation
- evaluation seed
- exploration schedule

을 최대한 분리해야 한다.

---

# 8. Independent variable과 Dependent variable

실험에서 의도적으로 바꾸는 것이 **independent variable**이다.

측정하는 결과가 **dependent variable**이다.

예:

```text
Independent variable:
Imagination OFF vs ON

Dependent variables:
success rate
intervention error rate
wall time
```

Ablation은 independent variable을 가능한 한 하나씩 바꾸는 것이 이상적이다.

---

# 9. Same-checkpoint comparison

AASSR Imagination의 marginal effect를 측정할 때 매우 중요하다.

```text
one AASSR training run
         ↓
frozen checkpoint
      /       \
OFF eval     ON eval
```

이렇게 해야 training trajectory, learned Policy, Prophecy, Critic 등이 같다.

차이는 평가 시 planner를 사용했느냐에 집중된다.

관련 페이지:

- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)
- [Imagination](Imagination)

---

# 10. 잘못된 OFF/ON 비교

```text
Agent A
→ Imagination OFF로 training

Agent B
→ Imagination ON으로 training
```

을 비교하면 두 agent가 경험한 state/action distribution부터 달라진다.

성능 차이가:

- planner 때문인지
- training data distribution 때문인지
- random exploration 차이 때문인지

분리하기 어렵다.

그래서 current main protocol은 training Imagination intervention을 끄고 same-checkpoint evaluation을 사용한다.

---

# 11. Random seed

Neural network training과 environment에는 randomness가 많다.

예:

- parameter initialization
- epsilon-greedy action
- replay sampling
- stochastic environment outcome
- mixture training bootstrap

따라서 seed 하나의 결과는 우연일 수 있다.

여러 research seed를 사용해야 variance를 볼 수 있다.

---

# 12. Seed의 두 역할을 구분

AASSR benchmark에서는 seed라는 단어가 최소 두 의미로 쓰일 수 있다.

```text
Research seed
→ learner/RNG 반복 실험

Environment/scenario seed
→ opaque identifiers, scenario realization 등
```

둘을 문서에서 명확히 구분해야 한다.

---

# 13. Training seed와 Evaluation seed

Generalization을 보려면 평가 환경을 training에서 분리한다.

```text
Training scenarios
≠
Unseen evaluation scenarios
```

특히 concrete identifier permutation이 있는 benchmark에서는 같은 seed를 반복하면 memorization shortcut이 생길 수 있다.

관련 페이지:

- [Relational Representation and Generalization](Relational-Representation-and-Generalization)

---

# 14. Transition budget

RL에서 중요한 compute/data budget 중 하나가 environment transition 수다.

```text
2k transitions
10k transitions
100k transitions
```

모델마다 transition budget이 다르면 sample efficiency 비교가 어렵다.

그래서 AASSR benchmark는 가능한 한 condition별 real transition budget을 명시한다.

---

# 15. Real transition과 Model compute를 분리

Model-based method는 같은 real transition 수에서도 훨씬 많은 내부 compute를 사용할 수 있다.

```text
Real environment interaction = 10k
Imagined model evaluations = millions 가능
```

따라서 두 종류의 efficiency를 구분해야 한다.

```text
Sample efficiency
= 실제 environment interaction 대비 성능

Compute efficiency
= wall time / FLOPs / accelerator cost 대비 성능
```

AASSR은 둘 다 보고하는 것이 좋다.

---

# 16. Wall time

같은 transition budget이라도 runtime이 100배 다르면 실제 사용성에 큰 차이가 있다.

AASSR current-generation에서는 Prophecy/Critic batching, structural root dedup 같은 최적화가 중요한 이유다.

Wall time은 성능 metric은 아니지만 engineering feasibility를 판단하는 중요한 보조 지표다.

---

# 17. Hyperparameter tuning

Hyperparameter 예:

- learning rate
- gamma
- epsilon schedule
- Imagination depth
- branch/beam width
- calibration threshold
- intervention margin
- support threshold

이 값을 evaluation 결과를 계속 보면서 맞추면 test overfitting이 생길 수 있다.

그래서:

```text
Development set
→ tuning

Validation set
→ model selection

Final blind/unseen set
→ 최종 claim
```

같은 분리가 필요하다.

---

# 18. Ablation과 Hyperparameter sweep은 다르다

## Ablation

구성 요소가 필요한지 묻는다.

```text
Calibration ON vs OFF
```

## Hyperparameter sweep

구성 요소 내부 설정의 좋은 범위를 찾는다.

```text
Calibration threshold 0.3 / 0.5 / 0.7
```

두 실험의 목적을 섞지 않는 것이 좋다.

---

# 19. Metric

Metric은 성능을 수치화한 값이다.

AASSR에서 중요한 metric은 하나가 아니다.

## Task-level

- success rate
- true failure rate
- stall rate
- truncation rate
- transitions to success

## World-model

- semantic prediction quality
- status accuracy
- legal-mask accuracy
- terminal accuracy
- probability-weighted quality
- calibration reliability

## Imagination

- plan count
- switch candidate count
- suppressed switch count
- executed intervention count
- changed-action count
- intervention error rate
- direct success-producing intervention

## Compute

- wall time
- model calls
- batch size / batch calls
- planning nodes

---

# 20. Proxy metric의 위험

World-model accuracy가 높다고 agent success가 자동으로 높아지는 것은 아니다.

```text
Prophecy semantic accuracy ↑
```

하지만 decision-critical status를 틀리면 planner가 나쁜 행동을 고를 수 있다.

마찬가지로:

```text
Imagination intervention count ↑
```

도 좋은 일이 아니다.

많이 바꿨지만 더 많이 실패할 수 있다.

따라서 proxy metric과 final task metric을 분리해야 한다.

---

# 21. Aggregate와 Per-cell 결과

난도/seed를 합친 aggregate만 보면 특정 cell의 실패가 가려질 수 있다.

예:

```text
Overall success 30%
```

이어도:

```text
Easy 80%
Medium 10%
Hard 0%
```

일 수 있다.

그래서 AASSR은 difficulty × seed cell과 aggregate를 함께 보는 것이 좋다.

---

# 22. Mean

여러 seed success rate의 평균:

```math
\bar x=\frac1n\sum_i x_i
```

를 쓸 수 있다.

하지만 mean만으로 seed variability를 알 수 없다.

---

# 23. Standard deviation

Sample standard deviation:

```math
s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}
```

seed마다 결과가 얼마나 흔들리는지 보여준다.

RL은 variance가 큰 경우가 많아서 평균과 함께 보는 것이 중요하다.

---

# 24. Standard error와 Confidence Interval

평균 추정의 불확실성을 표현할 수 있다.

단순한 큰-sample 근사에서는:

```math
SE=\frac{s}{\sqrt n}
```

를 쓸 수 있다.

하지만 seed 수가 매우 적으면 normal approximation이 부정확할 수 있다.

성공/실패 비율에는 binomial interval, seed aggregate에는 bootstrap interval 같은 방법도 고려할 수 있다.

핵심은 숫자 하나보다 **불확실성을 함께 보고하는 것**이다.

---

# 25. Statistical significance와 Practical significance

통계적으로 차이가 있어도 실제 효과 크기가 작을 수 있다.

반대로 sample이 적어 p-value는 불확실하지만 효과 크기는 매우 클 수도 있다.

그래서:

```text
effect size
uncertainty
raw counts
```

를 함께 보는 것이 좋다.

AASSR처럼 아직 seed 수가 제한된 연구에서는 특히 raw success counts와 per-seed 결과를 숨기지 않는 것이 중요하다.

---

# 26. Paired comparison

Same scenario/seed에서 OFF와 ON을 비교하면 paired structure를 활용할 수 있다.

```text
Scenario 1: OFF fail / ON success
Scenario 2: OFF success / ON success
Scenario 3: OFF success / ON fail
```

단순 aggregate success rate뿐 아니라 **어떤 episode에서 행동이 실제로 개선/악화되었는지** 볼 수 있다.

Imagination marginal effect 분석에 특히 유용하다.

---

# 27. Regression test와 Performance benchmark

둘은 목적이 다르다.

## Regression test

코드 계약이 깨지지 않았는지 확인.

예:

- chance node가 expectation을 쓰는가?
- confidence가 Critic value에 들어가지 않는가?
- hidden state leakage가 없는가?

## Performance benchmark

실제 agent가 더 잘 푸는지 확인.

예:

- success rate 향상
- intervention error 감소

Regression test 통과는 성능 향상 증명이 아니다.

---

# 28. Diagnostic experiment

작은 2k run은 failure mechanism을 찾는 데 매우 유용하다.

예:

```text
Imagination이 실제로 action을 바꾸는가?
왜 바꾼 action이 403/429로 가는가?
Critic support가 있는가?
```

하지만 작은 diagnostic을 최종 benchmark claim으로 확대해석하면 안 된다.

---

# 29. Mechanism evidence와 Final performance

AASSR 위키에서는 결과를 다음 계층으로 나누는 것이 좋다.

```text
1. Unit / contract validation
2. Mechanism diagnostic
3. Reduced benchmark
4. Multi-seed current benchmark
5. Final blind/unseen benchmark
```

각 단계에서 말할 수 있는 주장이 다르다.

---

# 30. Oracle baseline

Oracle은 hidden correct action/path를 알고 성공 가능한 upper-bound sanity check로 사용할 수 있다.

Oracle이 성공하지 못하면 환경 자체가 잘못되었을 가능성이 있다.

하지만 Oracle은 agent fairness baseline이 아니다.

```text
Oracle
→ benchmark solvability 확인

Learned agent
→ 실제 비교 대상
```

이다.

---

# 31. Random baseline

Random policy가 너무 잘하면 benchmark가 너무 쉽거나 action space가 제대로 어렵지 않을 수 있다.

반대로 Oracle만 성공하고 모든 non-oracle method가 영원히 0이면 benchmark가 너무 어려워 model 차이를 측정하기 어렵다.

좋은 benchmark는 비교 가능한 난도 영역을 가져야 한다.

---

# 32. Heuristic baseline

단순한 human-designed heuristic은 benchmark의 구조적 난도를 보는 데 유용하다.

예:

```text
항상 browse-first
항상 새 action 우선
```

다만 heuristic에 benchmark 정답 구조를 너무 많이 넣으면 사실상 oracle이 된다.

---

# 33. Strong learned baseline

새 RL architecture를 제안한다면 오래된 tabular/Q-learning만 비교하기보다 강한 현대 baseline이 필요하다.

AASSR에서는 official pinned DreamerV3 relational adapter를 model-based comparison으로 둔다.

중요한 점은 baseline을 약하게 만드는 것이 아니라 **가능한 한 공정하게 같은 observation/action/reward contract를 적용하는 것**이다.

---

# 34. Reproducibility

다른 사람이 같은 실험을 다시 실행했을 때 비슷한 결과를 얻을 수 있어야 한다.

필요한 것:

- exact code revision
- environment/version
- dependencies
- command line
- seed
- transition budget
- hardware assumptions
- output artifact schema
- evaluation protocol

AASSR은 이를 [Reproduction](Reproduction)에서 관리한다.

---

# 35. Reproducibility와 Replicability

문헌에 따라 정의가 다르지만 흔히:

```text
Reproducibility
→ 같은 code/data로 결과 재현

Replicability
→ 독립 구현/실험으로 같은 현상 확인
```

처럼 구분하기도 한다.

AASSR 현재 단계에서는 먼저 repository 내부 reproducibility를 강하게 만드는 것이 중요하다.

---

# 36. Source of truth

여러 문서에 current component 정보가 복제되면 쉽게 drift한다.

그래서 AASSR은 active runtime definition의 source of truth를:

```text
src/aassr_v2/current_manifest.py
```

로 둔다.

Wiki는 이를 설명하지만, 코드와 충돌하면 manifest/current entrypoint를 우선 확인해야 한다.

---

# 37. Historical code와 Current code

Repository에 과거 구현이 남아 있는 것은 reproducibility에는 좋다.

하지만 독자가 옛 클래스를 보고 current runtime이라고 착각할 수 있다.

그래서:

```text
historical / reproduction path
!=
active current-generation path
```

를 위키에서 명시한다.

관련 페이지:

- [Development History](Development-History)

---

# 38. Artifact provenance

실험 결과 파일에는 가능하면 다음 metadata가 있어야 한다.

```text
commit SHA
branch
runner
seed
budget
config
model version
condition
start/end time
```

그래야 나중에 오래된 결과와 current 결과가 섞이는 문제를 줄일 수 있다.

---

# 39. Result cherry-picking

여러 seed 중 가장 좋은 하나만 보고하면 실제 평균 성능보다 과장될 수 있다.

```text
seed 7   → 40%
seed 42  → 5%
seed 100 → 0%
```

인데 seed 7만 보고하면 매우 다른 인상을 준다.

따라서 사전에 정한 seed set 전체를 보고하는 것이 중요하다.

---

# 40. Early stopping과 선택 편향

좋은 결과가 나온 시점에만 실험을 멈추면 선택 편향이 생길 수 있다.

Training budget/stop criterion을 사전에 고정하고, 중간 checkpoint 분석과 final checkpoint claim을 분리하는 것이 좋다.

---

# 41. Multiple comparisons

많은 hyperparameter/architecture를 동시에 시험하면 우연히 좋은 결과 하나가 나올 확률이 커진다.

따라서 최종 claim에서는:

- 얼마나 많은 후보를 시험했는지
- validation에서 어떻게 선택했는지
- final test를 몇 번 사용했는지

를 가능한 한 투명하게 관리해야 한다.

---

# 42. AASSR에서 각 비교가 답하는 질문

| Comparison | 연구 질문 |
|---|---|
| Raw DQN vs Relational DQN | relational representation이 transfer에 도움이 되는가? |
| Relational DQN vs AASSR no-Imagination | ASEQ/Knowledge/information/Skill 등 non-planner stack의 추가 효과가 있는가? |
| AASSR no-Imagination vs Full | 같은 checkpoint에서 Imagination의 marginal effect가 있는가? |
| Full vs DreamerV3 relational | AASSR의 명시적 구조가 강한 world-model baseline과 비교해 경쟁력 있는가? |
| ASEQ OFF vs ON | exact self-loop guard가 반복을 줄이는가? |
| Calibration/support OFF vs ON | bad intervention을 실제로 줄이는가? |

---

# 43. 좋은 Ablation 결과 해석 예시

```text
Relational DQN > Raw DQN
→ representation 효과 evidence

AASSR no-img > Relational DQN
→ non-Imagination stack additional evidence

Full ≈ no-img
→ 현재 checkpoint/budget에서 Imagination marginal benefit evidence 없음
```

이 경우 "AASSR 전체가 효과 없음"도 아니고 "Imagination이 효과 있음"도 아니다.

**효과가 어느 층에서 발생했는지 분리해 말해야 한다.**

---

# 44. Negative result도 결과다

예:

```text
Imagination intervention 86회
직접 성공 intervention 0회
```

이라면 중요한 정보다.

이는:

```text
planner가 action을 바꿀 수 있다
```

는 mechanism evidence와:

```text
그 변경이 task success를 개선한다
```

는 performance claim이 다르다는 것을 보여준다.

Negative result를 숨기지 않고 failure mechanism으로 연결하는 것이 위키의 역할이다.

---

# 45. 최종 체크리스트

실험 결과를 위키에 추가하기 전에 확인한다.

```text
[ ] 어떤 research question을 검증하는가?
[ ] control/baseline은 무엇인가?
[ ] 다른 변수는 고정되었는가?
[ ] 같은 observation/reward contract인가?
[ ] real transition budget은 같은가?
[ ] seed를 여러 개 사용했는가?
[ ] unseen evaluation인가?
[ ] raw counts를 보존했는가?
[ ] variance/uncertainty를 보고했는가?
[ ] diagnostic과 final claim을 구분했는가?
[ ] exact commit/config를 기록했는가?
[ ] 결과가 current-generation인지 historical인지 명시했는가?
```

---

# 46. AASSR 연구 문서의 기본 원칙

```text
성공률 숫자 하나
        ↓
좋은 연구 결론 X

연구 질문
+ 공정한 control
+ 같은 budget
+ 여러 seed
+ 명확한 metric
+ failure analysis
+ reproducible artifacts
        ↓
해석 가능한 연구 결론
```

---

# 47. 다음으로 읽기

- [Experiments](Experiments)
- [Current Status](Current-Status)
- [Reproduction](Reproduction)
- [Causality, Leakage and Evaluation](Causality-Leakage-and-Evaluation)
- [Research Questions](Research-Questions)

관련 색인: **[Concept Index](Concept-Index)**