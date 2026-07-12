# 핵심 코드 참조 가이드

이 폴더의 `code_reference/`에는 논문 작성에 필요한 핵심 코드 파일이 복사되어
있다. 아래 표는 논문 개념과 코드 파일의 대응 관계이다.

## Code Map

| Paper Concept | Code File | What to Cite/Explain |
| --- | --- | --- |
| KK/KV Knowledge Storage | `code_reference/knowledge.py` | KK enum, KV metadata, KnowledgeStore |
| DMP closed loop | `code_reference/gridworld.py` | GridWorldDMP, candidate generation, execute loop |
| Action Template Binding | `code_reference/gridworld.py` | ActionCandidate, templates, required_kk_slots |
| PolicyABC | `code_reference/policy.py` | WHAT/HOW/WHERE probability tables |
| Prophecy Module | `code_reference/prophecy.py` | ProphecyModule protocol, prediction/update |
| Table Prophecy | `code_reference/prophecy.py` | TableProphecyModel |
| Transformer variant | `code_reference/prophecy.py` | TransformerProphecyModel |
| Imagination Cycle | `code_reference/imagination.py` | ImaginationConfig, ImaginationCycle |
| Reward | `code_reference/reward.py` | semantic gain, error/repeat reward components |
| Experiment Conditions | `code_reference/experiment.py` | C0-C4, ExperimentSpec |
| Ablation Suites | `code_reference/ablation.py` | A1-A5 definitions |
| Metrics | `code_reference/metrics.py` | StepMetric, EpisodeMetric, SummaryMetric |
| Environments | `code_reference/worlds.py` | random_key_door, v2_complex, locked_bottleneck |
| Analysis | `code_reference/analysis.py` | summary table, bootstrap CI, report generation |
| DQN baseline | `code_reference/dqn_baseline.py` | DQN_PARTIAL |
| Q-learning baseline | `code_reference/traditional_baselines.py` | QLEARN |
| Oracle upper bound | `code_reference/mdp_baseline.py` | ORACLE_MDP |

## Important Code Concepts

### 1. KK/KV

Look at:

```text
code_reference/knowledge.py
```

Explain:

```text
KK is an abstract knowledge key used as an action-template slot.
KV is a concrete value stored under a KK and later bound into an action.
```

논문 표현:

```text
Knowledge Storage is a typed parameter pool rather than a passive memory.
```

### 2. ActionCandidate

Look at:

```text
code_reference/gridworld.py
```

Explain:

```text
ActionCandidate contains:
- name
- template
- required_kk_slots
- bindings
- strategy
```

이 구조가 `MOVE_TOWARD {KK_FRONTIER_CELL}` 같은 추상 템플릿을 실제 실행 가능한
행동으로 바꾼다.

### 3. Prophecy Module

Look at:

```text
code_reference/prophecy.py
```

The common interface:

```text
predict(state_signature, candidate)
update(state_signature, candidate, actual_delta, actual_error, actual_flag)
```

Predicted targets:

```text
semantic ΔK
error probability
flag probability
```

주의:

```text
Prophecy Module is the framework concept.
Table/Sequence/Transformer are implementation variants.
```

### 4. Imagination Cycle

Look at:

```text
code_reference/imagination.py
```

Current score terms:

```text
knowledge_weight * expected_kk_gain
+ flag_weight * predicted_flag_prob
- error_weight * predicted_error_prob
- repeat_weight * repeat_penalty
+ policy_prior_weight * policy_prior
+ rollout_discount * rollout_value
```

Important:

```text
The current Imagination Cycle does not read the hidden world map.
It does not execute future actions.
It evaluates candidates using Prophecy predictions.
```

### 5. Experiment Conditions

Look at:

```text
code_reference/experiment.py
```

Conditions:

```text
C0 = RandomScorer
C1 = PolicyABC
C2 = PolicyABC + Prophecy reward
C3 = PolicyABC + Prophecy Module + Imagination Cycle
C4 = optional sequence-based Prophecy variant
```

### 6. Ablation Definitions

Look at:

```text
code_reference/ablation.py
```

Suites:

```text
A1: Table vs Transformer Prophecy
A2: Prophecy reward ON/OFF
A3: Imagination depth/branch
A4: Imagination mechanism terms
A5: Prophecy score components
```

## Pseudocode for Method Section

Use this in the paper:

```text
for each step:
    K <- current Knowledge Storage
    T <- available action templates
    C <- []
    for template in T:
        slots <- required KK slots
        values <- retrieve candidate KV values from K
        C <- C + bind(template, values)

    for candidate in C:
        prediction <- Prophecy.predict(K, candidate)
        imagination_score <- Imagination.score(candidate, prediction)

    action <- PolicyABC.select(C, imagination_score)
    observation <- execute(action)
    delta_K <- extract_KV(observation)
    Knowledge Storage.update(delta_K)
    Prophecy.update(prediction, delta_K)
    PolicyABC.update(reward)
```

## Best Code Snippets to Include in Paper/Appendix

If the paper includes code snippets, use short snippets from:

1. `ActionCandidate` dataclass from `gridworld.py`
2. `ProphecyPrediction` and `ProphecyModule` from `prophecy.py`
3. `ImaginationConfig` from `imagination.py`
4. `ExperimentCondition` and `ExperimentSpec` from `experiment.py`
5. `AblationSuite` definitions from `ablation.py`

Avoid including long full files in the paper body. Use them only as appendix or
repository reference.
