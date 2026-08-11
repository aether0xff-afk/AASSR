# Current Status

> 마지막 정리 기준: **2026-08-11**  
> Current runtime: `aassr-current-generation-v2`  
> Research branch: `agent/imagination-gate-ablation`

이 페이지는 “코드에 구현된 것”, “짧은 regression으로 검증된 것”, “실제 성능 실험으로 확인된 것”을 분리한다.

---

# 1. 한눈에 보기

| 영역 | 구현 | 구조 검증 | 성능 evidence | 현재 판단 |
|---|---|---|---|---|
| Response-causal observation | ✅ | ✅ | ✅ | 🟢 Active |
| Relational representation | ✅ | ✅ | 일부 | 🟢 Active |
| ASEQ self-loop guard | ✅ | ✅ | ✅ | 🟢 Active |
| Relational DQN Policy | ✅ | ✅ | 진행 중 | 🟢 Active |
| Episode-local Knowledge | ✅ | ✅ | 부분적 | 🟢 Active |
| Stochastic Prophecy v3 | ✅ | ✅ | 새 장기 검증 전 | 🟡 Experimental |
| Status-aware calibration | ✅ | ✅ | 새 장기 검증 전 | 🟡 Experimental |
| Sparse-return GRU Critic | ✅ | ✅ | 2k에서 override 가능 확인 | 🟡 Experimental |
| Local Critic support | ✅ | ✅ | 새 장기 검증 전 | 🟡 Experimental |
| Imagination tree | ✅ | ✅ | 행동 변경 확인, 성능 향상 미확인 | 🟡 Experimental |
| Structural root dedup | ✅ | ✅ | 새 benchmark runtime 검증 전 | 🟡 Experimental |
| Skill | ✅ | regression 존재 | 제한적 | 🟡 Experimental |
| DreamerV3 official baseline | adapter ✅ | smoke/contract ✅ | canonical CUDA result 전 | 🟡 Experimental |
| Five-condition final suite | assembler ✅ | contract ✅ | full run 전 | ⚪ Pending |
| Final blinded evaluation | protocol 방향 존재 | — | ❌ | ⚪ Not run |

---

# 2. 지금 확실하게 말할 수 있는 것

## 2.1 Benchmark는 agent 평가용으로 사용 가능

HTTP in-process benchmark는 40 evaluation seeds에서 다음 조건을 통과했다.

- Oracle 100% across tiers
- Random은 사실상 0%
- Response-guided는 Easy 100%, Medium 30%, Hard 20%
- seed별 난도 편차 존재
- target-ID 순서 shortcut 제거
- 실제 network/shell 없음

따라서 benchmark 자체가 너무 쉽거나 구조적으로 불가능한 환경은 아니다.

## 2.2 ASEQ는 관측된 self-loop를 제거한다

재학습 없는 L1 diagnostic에서:

```text
raw greedy stalled        24 / 24
exact ASEQ stalled         0 / 24
```

따라서 현재 ASEQ의 최소 self-loop guard는 명확한 기능 evidence가 있다.

## 2.3 Imagination은 이제 실제 행동을 바꿀 수 있다

2026-08-11 2k same-checkpoint validation:

```text
plans                 297
switch candidates     218
actual interventions   86
changed actions         86
```

즉 과거의 “planner는 돌지만 Policy와 항상 tie라 행동을 못 바꿈” 병목은 적어도 이 run에서는 사라졌다.

---

# 3. 아직 확실하게 말할 수 없는 것

## 3.1 Imagination이 성능을 높이는가?

아직 아니다.

2k validation:

```text
no-Imagination : 4 / 20
Full           : 4 / 20
```

Full에는 true failure가 2회 있었고, intervention 86회 중 58회가 error였다.

따라서 “Imagination is active”는 맞지만 “Imagination improves success”는 아직 증명되지 않았다.

## 3.2 Current AASSR이 DQN보다 최종적으로 좋은가?

아직 current-generation five-condition result가 완성되지 않았다.

과거 세대의 AASSR 결과를 가져와 current-generation의 최종 성능 주장으로 쓰지 않는다.

## 3.3 DreamerV3보다 좋은가?

아직 canonical DreamerV3 CUDA baseline과 동일 protocol aggregate가 완성되지 않았다.

---

# 4. 2k validation에서 발견된 문제

## 4.1 Decision-critical HTTP status가 relational v2에서 사라짐

당시 raw observation에는 `403`, `404`, `429` 같은 public status가 있었지만 relational state가 이를 명시적으로 보존하지 않았다.

결과적으로 “방금 위험 신호가 관측됐다”는 정보가 world model / Critic input에서 약해질 수 있었다.

## 4.2 Semantic calibration이 실제 위험을 충분히 벌점 주지 못함

2k run에서 probability-weighted semantic quality가 약 `0.916`, terminal match가 약 `0.991`이었는데도 intervention은 나빴다.

즉 global semantic similarity는 높지만 decision-critical error channel을 놓치는 metric blind spot이 있었다.

## 4.3 Critic이 training support 밖에서 너무 자신 있게 override

training successes는 L0에 집중되고 curriculum focus도 L1까지만 갔는데 Full은 L3에서 86회 override했다.

이것은 다음 둘을 분리해야 함을 보여줬다.

```text
Critic trained globally
vs
current state/action locally supported
```

## 4.4 Root alias가 너무 많아 계산 낭비

L3에서 대략:

```text
concrete root actions     ~172
relational root structures ~17
```

구조적으로 같은 concrete alias를 매번 world-model/critic에 넣는 것은 계산 낭비다.

---

# 5. 현재 코드에 이미 반영된 repair

`current_manifest.py` 기준 현재 contract는 2k run 당시보다 한 단계 더 나아가 있다.

## 5.1 Relational state v3

```text
relational-public-structural-v3
+ latest-http-status
```

- public response status 보존
- hidden audit pressure는 계속 마스킹
- hidden session countdown도 계속 마스킹

즉 누락된 public signal만 복원하고 hidden leakage를 다시 열지 않는다.

## 5.2 Prophecy v3

```text
relational-stochastic-world-model-v3-status-supervised
```

예측 대상에 public HTTP status를 명시적으로 포함하고 status supervision을 별도 objective로 둔다.

## 5.3 Status-aware calibration

```text
semantic-probability-holdout-calibration-v3-status-aware
```

403/404/429 같은 response-status error가 semantic quality에서 무시되지 않도록 한다.

## 5.4 Local Critic support gate

```text
local-real-training-support-fail-closed-v1
```

현재 state/action이 Critic 실제 training data에서 충분히 지원되지 않으면 override를 허용하지 않는다.

이 support score는 value bonus나 reward가 아니다.

## 5.5 Structural root compute dedup

```text
concrete execution
+
structural compute dedup
```

같은 relational legal slot을 공유하는 concrete alias는 한 번만 imagination 계산하고 결과를 fan-out한다.

---

# 6. 현재 Imagination 계약

현재 planner가 지켜야 하는 핵심 invariant:

### Chance backup

```text
환경 outcome -> probability-weighted expectation
```

### Decision backup

```text
future agent actions -> max
```

### Reliability

```text
reliability는 gate 용도
value bonus로 사용하지 않음
```

### Failure semantics

```text
success       +1
truncation     0
true failure  -1
```

### Root preservation

모든 real root action은 평가 가능해야 한다. deep branch pruning 때문에 root 전체가 사라지면 안 된다.

### Same-checkpoint comparison

training-time Imagination intervention은 끄고, 하나의 frozen AASSR checkpoint를 OFF/ON으로 평가한다.

---

# 7. Current five-condition suite

최종 비교 예정 row:

1. `dqn_raw`
2. `dqn_relational`
3. `dreamerv3_relational`
4. `aassr_current_no_imagination`
5. `aassr_current_full`

research seed 하나에서 학습되는 checkpoint는 4개다.

```text
Raw DQN
Relational DQN
DreamerV3
AASSR
```

AASSR OFF/ON은 같은 checkpoint다.

10k 기준 nominal real training transitions:

```text
4 checkpoints * 10,000
= 40,000 per research seed
```

---

# 8. 다음 validation gate

현재 코드 상태에서 바로 full-scale 성능 주장을 하면 안 된다.

순서는 다음과 같다.

```text
[1] current-generation unit/regression gates
        |
        v
[2] target CUDA hardware check
        |
        v
[3] short real-environment smoke
        |
        v
[4] repaired reduced AASSR validation
        |
        v
[5] official DreamerV3 reduced JAX/CUDA run
        |
        v
[6] reduced five-condition assembly
        |
        v
[7] protocol freeze
        |
        v
[8] full main
        |
        v
[9] final blinded evaluation
```

## 다음 reduced AASSR run에서 특히 볼 것

단순 success만 보면 안 된다.

- L2/L3에서 local Critic support가 실제로 fail-closed하는지
- intervention count가 0으로 다시 붕괴하지 않는지
- intervention error rate가 58/86에서 유의미하게 내려가는지
- status prediction accuracy
- 403/404/429 뒤의 action switching behavior
- structural root dedup으로 Full runtime이 얼마나 줄었는지
- no-Imagination vs Full same-checkpoint success/failure

---

# 9. 현재 연구의 정확한 한 문장

> **AASSR current-generation은 sparse-reward 환경에서 relational DQN, empirical ASEQ, stochastic relational world model, sparse-return Critic, multi-step Imagination을 하나의 closed loop로 통합했고, Imagination이 실제 Policy 행동을 바꿀 수 있음까지 확인했다. 현재는 그 행동 변경을 신뢰할 수 있는 상태/행동 영역에 제한하고 실제 성능 향상으로 연결되는지 검증 중이다.**

이 문장보다 강한 성능 주장은 다음 reduced/final evidence 이후에 업데이트한다.
