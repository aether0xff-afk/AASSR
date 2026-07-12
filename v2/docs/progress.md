# AASSR/APASSR Progress Report

## Summary

현재 저장소는 초기 README-only 상태에서 APASSR GridWorld 프로토타입으로 확장되었다. 핵심 구현 방향은 다음 문장으로 고정했다.

```text
행동은 지식을 만들고,
지식은 다음 행동의 파라미터가 된다.
```

Knowledge Storage는 단순 관측 기록이 아니라, 행동 템플릿의 `KK` 슬롯에 대입할 `KV` 후보를 공급하는 저장소로 구현되었다.

## Implemented Modules

| Path | Status | Purpose |
| --- | --- | --- |
| `src/aassr/knowledge.py` | Implemented | `KK`, `KV`, `KnowledgeStore`, lifecycle 상태, `KnowledgeDelta` |
| `src/aassr/gridworld.py` | Implemented | GridWorld, action candidates, DMP loop, `StepResult` |
| `src/aassr/dashboard.py` | Implemented | Streamlit-independent dashboard table helpers |
| `src/aassr/reward.py` | Implemented | sparse external reward + knowledge-change intrinsic reward |
| `src/aassr/policy.py` | Implemented | `RandomScorer`, `PolicyABC` scaffold |
| `src/aassr/prophecy.py` | Implemented | `ProphecyModule` interface with `TableProphecyModel` and optional `SequenceProphecyModel` implementations |
| `src/aassr/imagination.py` | Implemented | depth-limited candidate rollout using Prophecy predictions |
| `src/aassr/worlds.py` | Implemented | fixed and randomized GridWorld builders |
| `src/aassr/metrics.py` | Implemented | step, episode, and summary experiment metrics |
| `src/aassr/experiment.py` | Implemented | C0/C1/C2/C3/C4/C5 runner and CSV output |
| `src/aassr/analysis.py` | Implemented | result aggregation, bootstrap CI, report generation |
| `src/aassr/plotting.py` | Implemented | matplotlib figures for result analysis |
| `src/aassr/visualization.py` | Implemented | ASCII visualization and Mermaid loop renderer |
| `app.py` | Implemented | Streamlit dashboard for GridWorld/DMP visualization |
| `tests/` | Implemented | Unit tests for Knowledge, DMP behavior, PolicyABC, Prophecy, Imagination, visualization, dashboard helpers |
| `README.md` | Updated | Design explanation, run commands, prototype description |

## Knowledge Storage Changes

### KK/KV Model

Implemented:

```text
KK = abstract action-template slot
KV = concrete value bound into a KK slot
```

Example:

```text
Template:
MOVE_TOWARD {KK_FRONTIER_CELL}

Knowledge:
KK_FRONTIER_CELL = [(3, 4), (4, 4)]

Executable action:
MOVE_TOWARD (3, 4)
```

### Singleton Current Position

`KK_CURRENT_POS` was changed into a singleton state key.

Before:

```text
KK_CURRENT_POS = [(0, 0), (1, 0), (2, 0), ...]
```

Now:

```text
KK_CURRENT_POS = current position only
KK_VISITED_CELL = historical visited positions
KK_KNOWN_CELL = known/confirmed cells
```

Implemented through:

```python
KnowledgeStore.set_singleton(...)
```

### Knowledge Lifecycle

Known/unknown/frontier lifecycle was tightened.

Rules now implemented:

```text
cell becomes known:
- remove from KK_UNKNOWN_NEIGHBOR
- remove from KK_FRONTIER_CELL
- add to KK_KNOWN_CELL

cell is wall:
- remove from KK_UNKNOWN_NEIGHBOR
- remove from KK_FRONTIER_CELL
- add to KK_WALL_CELL with blocked status

current position changes:
- replace KK_CURRENT_POS singleton
- add to KK_VISITED_CELL
- promote cell to known
```

### KnowledgeDelta

Added explicit `KnowledgeDelta`.

```python
@dataclass(frozen=True)
class KnowledgeDelta:
    added: tuple[tuple[KK, KV], ...]
    updated: tuple[tuple[KK, KV], ...]
    status_changed: tuple[tuple[KK, KV], ...]
    removed: tuple[tuple[KK, Any], ...]
    usage_updated: tuple[tuple[KK, KV], ...]
```

This replaces rough row-count based knowledge-change tracking.

Purpose:

```text
ΔK is now explicit and can be used for reward, policy update, prophecy, and metrics.
```

`KnowledgeDelta` separates semantic changes from usage metadata changes.

Semantic delta:

```text
added
updated
status_changed
removed
```

Usage delta:

```text
usage_updated
```

Usage-only updates include `used_count`, `success_count`, and `last_updated`.
They are visible for debugging, but they do not count as information gain.

Verified repeat behavior:

```text
first execution:
semantic_gain = 5
usage_updated = 1
intrinsic_reward = 0.5

repeat same candidate:
semantic_gain = 0
usage_updated = 6
intrinsic_reward = -0.05
```

Reward and Prophecy use semantic delta only.

## DMP / StepResult

`GridWorldDMP.execute()` now returns `StepResult` instead of a raw dict.

```python
@dataclass(frozen=True)
class StepResult:
    step: int
    action: ActionCandidate
    observation: dict
    delta_k: KnowledgeDelta
    external_reward: float
    intrinsic_reward: float
    total_reward: float
    error: bool
    flag_found: bool
    done: bool
```

Current step loop:

```text
1. generate executable action candidates
2. select candidate
3. execute action
4. receive observation
5. update Knowledge Storage
6. compute ΔK
7. compute reward
8. return StepResult
```

## Action Generation

Implemented action candidates based on template-slot binding.

Current templates:

```text
MOVE_TOWARD {KK_FRONTIER_CELL}
MOVE_TOWARD {KK_HINT_CELL}
MOVE_TOWARD {KK_KEY_CELL}
MOVE_TOWARD {KK_DOOR_CELL}
MOVE_TOWARD {KK_FLAG_CELL}
INSPECT_CELL {KK_UNKNOWN_NEIGHBOR}
INSPECT_CELL {KK_FRONTIER_CELL}
USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}
FOLLOW_HINT {KK_HINT_VALUE}
```

`OPEN_DOOR` was removed from the active action set because it overlapped with:

```text
USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}
```

This better matches the KK/KV parameter-binding design.

## Reward Module

Added `RewardModule`.

Current reward:

```text
external_reward = 1.0 if flag_found else 0.0
intrinsic_reward = knowledge_gain_weight * |ΔK|
                 + error_penalty if error
                 + repeat_penalty if repeated

total_reward = external_reward + intrinsic_reward
```

This is an initial scaffold for sparse reward experiments.

## Policy / Scorer Work

Added:

```text
RandomScorer
PolicyABC
```

Current meaning:

```text
RandomScorer:
- C0-style random selector

PolicyABC:
- C1-style WHAT / HOW / WHERE probability-table selector
```

Implemented `PolicyABC` behavior:

```text
1. decompose candidate into WHAT / HOW / WHERE
2. sample candidate by P(WHAT) * P(HOW) * P(WHERE)
3. update selected axes after StepResult reward
4. use multiplicative update:
   p[selected] *= exp(lr * reward)
5. normalize each axis table
6. apply minimum probability floor to prevent early collapse
```

Verified example:

```text
candidate_axes = (INSPECT_CELL, least_tried, KK_UNKNOWN_NEIGHBOR)
reward = 0.6

before:
WHAT INSPECT_CELL = 0.25
HOW least_tried = 0.1667
WHERE KK_UNKNOWN_NEIGHBOR = 0.125

after:
WHAT INSPECT_CELL = 0.3779
HOW least_tried = 0.2671
WHERE KK_UNKNOWN_NEIGHBOR = 0.2065
```

Important note:

```text
Hand-coded lifecycle and dependency heuristics have been removed from active
experiment conditions.
C0 = RandomScorer.
C1 = PolicyABC.
C2 = PolicyABC + Prophecy.
C3 = PolicyABC + Prophecy + Imagination.
C4 = PolicyABC + optional sequence-based Prophecy implementation + Imagination.
C5 = improved APASSR with C3 loop plus ablation-derived Imagination weights.
```

Recommended experiment naming:

```text
C0: Random
- random candidate selection
- no policy learning
- no prophecy
- no imagination

C1: PolicyABC
- WHAT / HOW / WHERE policy tables
- reward-based policy update
- no prophecy
- no imagination

C2: PolicyABC + Prophecy
- C1 plus Prophecy Module reward
- prediction error can be added to reward
- no imagination

C3: PolicyABC + Prophecy + Imagination
- main paper-aligned framework condition
- C2 plus pre-execution candidate evaluation

C4: PolicyABC + sequence Prophecy implementation + Imagination
- optional Prophecy implementation variant
- not a replacement for C3 or the paper framework

C5: improved APASSR
- C3 loop with TableProphecyModel
- knowledge_weight = 0.0
- repeat penalty and error avoidance retained
- not a replacement for vanilla C3
```

## Prophecy Module

Added the general Prophecy Module interface and concrete implementations:

```text
src/aassr/prophecy.py
```

Implemented:

```text
ProphecyModule
- predict(state_signature, candidate)
- update(state_signature, candidate, actual_delta, actual_error, actual_flag)

ProphecyPrediction
- kk_probs: dict[KK, float]
- error_prob: float
- flag_prob: float

ProphecyUpdate
- prediction_error: float
- loss: float

TableProphecyModel
- predict(state_signature, candidate)
- update(state_signature, candidate, actual_delta, actual_error, actual_flag)
- current lightweight C3 implementation

SequenceProphecyModel
- optional recurrent/sequence-based Prophecy implementation variant
- used by C4 as an ablation, not as a replacement for the framework

gridworld_state_signature(dmp)
- has_key
- door_known
- hint_known
- flag_candidate_known
- frontier_count bucket
- unknown_neighbor_count
```

The table key is:

```text
(state_signature, candidate WHAT, candidate WHERE)
```

This estimates:

```text
P(ΔKK | state, action)
P(error | state, action)
P(flag | state, action)
```

Prophecy learns from semantic ΔK only. Usage-only metadata changes are excluded from `P(ΔKK)`.

## C2 Prophecy Reward Loop

Prophecy is now connected to the DMP step loop through `DMPConfig`.

```python
@dataclass(frozen=True)
class DMPConfig:
    use_prophecy: bool = False
    use_imagination: bool = False
    prophecy_beta: float = 0.3
```

C2 step flow:

```text
1. compute state_signature before execution
2. Prophecy.predict(state_signature, candidate)
3. execute candidate
4. compute semantic KnowledgeDelta, error, flag_found
5. Prophecy.update(state_signature, candidate, delta_k, error, flag_found)
6. add prophecy_beta * prediction_error to total_reward
7. PolicyABC.update(candidate, total_reward)
```

`StepResult` now includes:

```text
prophecy_prediction
prophecy_error
prophecy_loss
predicted_kk_count
predicted_error_prob
predicted_flag_prob
```

Verified C2 behavior:

```text
Prophecy.predict is called before execution.
Prophecy.update receives semantic ΔK after execution.
StepResult records prophecy_error and prophecy_loss.
total_reward includes beta * prophecy_error.
PolicyABC updates with prophecy-adjusted total_reward.
Usage-only delta is not used as prophecy target.
```

## C3 Imagination Cycle

Added depth-limited imagination module:

```text
src/aassr/imagination.py
```

Implemented:

```text
ImaginationConfig
- knowledge_weight
- flag_weight
- error_weight
- repeat_weight
- policy_prior_weight
- rollout_depth
- rollout_branching
- rollout_discount
- dependency_weight

ImaginationScore
- candidate
- score
- expected_kk_gain
- predicted_flag_prob
- predicted_error_prob
- rollout_value
- rollout_depth
- repeat_penalty
- policy_prior

ImaginationTrace
- selected
- scores

ImaginationCycle
- score_candidate(state_signature, candidate, policy=None, dmp=None)
- choose(state_signature, candidates, policy=None, dmp=None)
```

C3 selection flow:

```text
1. generate executable candidates from Knowledge Storage
2. compute current state_signature
3. Prophecy.predict(state_signature, candidate) for each candidate
4. score each candidate using predicted ΔK, error, flag, repeat count, PolicyABC prior, and depth-limited rollout value
5. select highest-scoring candidate
6. execute selected candidate through the normal DMP execute path
7. run Prophecy.update, reward calculation, and PolicyABC.update after execution
```

Current score formula:

```text
score =
  knowledge_weight * expected_kk_gain
  + flag_weight * predicted_flag_prob
  - error_weight * predicted_error_prob
  - repeat_weight * repeat_penalty
  + policy_prior_weight * policy_prior
```

Important boundary:

```text
Imagination does not read the hidden GridWorld map.
Imagination does not inspect actual cell kind.
Imagination does not execute candidates before selection.
```

It only uses:

```text
state_signature
candidate metadata
PolicyABC candidate probability
ProphecyPrediction
KV usage metadata for repeat penalty
```

`StepResult` now includes `imagination_trace`, and `to_dict()` exposes:

```text
imagination_selected_score
imagination_candidate_count
imagination_best_flag_prob
imagination_best_error_prob
```

Verified C3 behavior:

```text
flag_prob increases candidate score.
error_prob decreases candidate score.
expected ΔKK gain increases candidate score.
ImaginationCycle.choose selects the highest score.
Imagination does not mutate DMP step index or position.
C3 StepResult records the selected imagination trace.
```

## ExperimentRunner

Added experiment runner:

```text
src/aassr/experiment.py
src/aassr/metrics.py
```

Implemented conditions:

```text
C0 = RandomScorer
C1 = PolicyABC
C2 = PolicyABC + Prophecy reward
C3 = PolicyABC + Prophecy + Imagination
C4 = PolicyABC + optional sequence-based Prophecy implementation + Imagination
C5 = improved APASSR with ablation-derived Imagination weights
```

Execution model:

```text
for each condition:
  for each seed:
    create persistent scorer / prophecy components
    for each episode:
      create a fresh GridWorld and KnowledgeStore
      reuse the condition components across episodes
      run until flag, no candidates, or step limit
      emit step and episode metrics
```

This keeps environment state episode-local while allowing C1/C2/C3 learning
state to accumulate within a seed. C4 is available as an optional Prophecy
implementation variant, but C3 remains the main paper-aligned framework
condition.

Required commands:

```powershell
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C0 --episodes 100 --seeds 10 --workers 6
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C1 --episodes 100 --seeds 10 --workers 6
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C2 --episodes 100 --seeds 10 --workers 6
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C3 --episodes 100 --seeds 10 --workers 6
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C4 --episodes 100 --seeds 10 --workers 6
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C5 --episodes 100 --seeds 10 --workers 6
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --episodes 100 --seeds 10 --workers 6
```

`--workers` parallelizes independent seeds while preserving sequential learning
within each seed.

CSV outputs:

```text
runs/gridworld/{condition}/gridworld_steps.csv
runs/gridworld/{condition}/gridworld_episodes.csv
runs/gridworld/{condition}/gridworld_summary.csv
runs/gridworld/all/combined_summary.csv
```

The default output directory is condition-safe:

```text
--condition C0 -> runs/gridworld/C0
--condition C1 -> runs/gridworld/C1
--condition C2 -> runs/gridworld/C2
--condition C3 -> runs/gridworld/C3
--condition C4 -> runs/gridworld/C4
--condition C5 -> runs/gridworld/C5
--condition all -> runs/gridworld/all
```

World options:

```text
fixed
random_flag
random_wall_flag
random_key_door
```

Example randomized-map command:

```powershell
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C3 --world random_key_door --episodes 100 --seeds 10
```

Episode metrics:

```text
condition
seed
episode
success
steps_to_flag
total_reward
external_reward
semantic_gain_total
prophecy_error_mean
repeat_count
error_count
knowledge_reuse_count
unique_action_count
```

Repeat metrics use action signatures that exclude `KK_CURRENT_POS`.

```text
MOVE_TOWARD {KK_FLAG_CELL}, current=(1,1), target=(5,2)
MOVE_TOWARD {KK_FLAG_CELL}, current=(2,1), target=(5,2)
```

These are counted as the same repeated target action, because current position
is execution state rather than the stable action target.

Smoke run results:

```text
C0: episodes=4, seeds=2, success_rate=0.500, steps_to_flag_mean=46.250
C1: episodes=4, seeds=2, success_rate=0.250, steps_to_flag_mean=49.250
C2: episodes=4, seeds=2, success_rate=0.000, steps_to_flag_mean=50.000
C3: episodes=4, seeds=2, success_rate=1.000, steps_to_flag_mean=34.500
```

These smoke numbers are only a wiring check, not a paper result.

Latest all-condition randomized smoke:

```text
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 1 --seeds 1 --step-limit 20
-> wrote condition folders and combined_summary.csv
```

## Result Analysis And Plots

Added analysis and plotting modules:

```text
src/aassr/analysis.py
src/aassr/plotting.py
```

Analysis command:

```powershell
$env:PYTHONPATH='src'; python -m aassr.analysis --input runs/gridworld/all --output runs/gridworld/all/analysis
```

Outputs:

```text
runs/gridworld/all/analysis/summary_table.csv
runs/gridworld/all/analysis/condition_stats.csv
runs/gridworld/all/analysis/learning_curve.csv
runs/gridworld/all/analysis/figure_success_rate.png
runs/gridworld/all/analysis/figure_steps_to_flag.png
runs/gridworld/all/analysis/figure_semantic_gain.png
runs/gridworld/all/analysis/figure_repeat_error_rate.png
runs/gridworld/all/analysis/figure_learning_curve.png
runs/gridworld/all/analysis/report.md
```

Summary table fields:

```text
condition
success_rate_mean
success_rate_ci95_low
success_rate_ci95_high
steps_to_flag_mean
steps_to_flag_ci95_low
steps_to_flag_ci95_high
semantic_gain_mean
repeat_rate_mean
error_rate_mean
prophecy_error_mean
total_reward_mean
```

Important analysis rule:

```text
steps_to_flag_mean is computed over successful episodes only.
```

Confidence intervals:

```text
seed-level bootstrap 95% CI
```

Plotting uses matplotlib only. Seaborn is not used.

Full-run protocol:

```powershell
# 1. Smoke
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 20 --seeds 5 --step-limit 80
$env:PYTHONPATH='src'; python -m aassr.analysis --input runs/gridworld/all --output runs/gridworld/all/analysis

# 2. Medium
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 100 --seeds 10 --step-limit 100
$env:PYTHONPATH='src'; python -m aassr.analysis --input runs/gridworld/all --output runs/gridworld/all/analysis

# 3. Paper-candidate
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 200 --seeds 20 --step-limit 120
$env:PYTHONPATH='src'; python -m aassr.analysis --input runs/gridworld/all --output runs/gridworld/all/analysis
```

Analysis smoke verification:

```text
$env:PYTHONPATH='src'; python -m aassr.analysis --input runs/verify_all_random --output runs/verify_all_random/analysis --bootstrap-samples 50 --learning-window 1
-> wrote summary table, condition stats, five figures, and report.md
```

## Streamlit Dashboard

Added dashboard in:

```text
app.py
```

Dashboard sections:

```text
GridWorld
Last Result
KK Slot Binding
Executable Action Candidates
Policy WHAT / HOW / WHERE
Knowledge Storage
DMP Trace
Action Template Library
Core Loop
Paper vs Project
```

Dashboard selector modes:

```text
Random C0
PolicyABC C1
PolicyABC + Prophecy C2
PolicyABC + Prophecy + Imagination C3
```

The dashboard now exposes the original design idea directly:

```text
KK slot -> KV bound -> generated command -> execution -> ΔK
```

Dashboard table helpers were moved into `src/aassr/dashboard.py` so tests can
run without importing Streamlit. `app.py` is now Streamlit runtime glue only.

When C3 is active, the dashboard also shows an Imagination Scores table:

```text
candidate
score
expected_kk_gain
flag_prob
error_prob
repeat_penalty
policy_prior
```

The dashboard now has two top-level tabs:

```text
DMP Runtime
Paper vs Project
```

`Paper vs Project` shows:

```text
Original APASSR / prior setting vs this GridWorld implementation
module-by-module comparison
implementation status
C0/C1/C2/C3/C4/C5 condition mapping
```

Run command:

```powershell
$env:PYTHONPATH='src'; streamlit run app.py
```

The Streamlit server is currently not running.

## Verified Behavior

Manual trace confirmed the expected paper-aligned loop:

```text
INSPECT key
-> ΔK includes KK_KEY_CELL

MOVE_TOWARD key
-> position changes
-> KK_CURRENT_POS singleton updated
-> KK_VISITED_CELL updated
-> key acquired as KK_KEY_OBJECT

INSPECT door
-> ΔK includes KK_DOOR_CELL

USE_OBJECT key ON door
-> door status becomes consumed

INSPECT flag
-> external_reward = 1.0
-> done = True
```

## Test Status

Latest verification:

```text
python -m compileall app.py src tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

Result:

```text
Ran 56 tests

OK
```

Streamlit render test:

```text
streamlit_app_render=ok
dataframes=6
```

## Current Limitations

Still not complete as a full paper reproduction.

Remaining work:

```text
1. Split gridworld.py into paper-aligned modules:
   gridworld_env.py
   action_space.py
   dmp.py
   metrics.py
   experiment.py

2. Expand experiment reporting:
   persist policy probability trajectories
   add configuration snapshots for each full run

3. Run and review full protocol:
   smoke run
   medium run
   paper-candidate run
   inspect analysis/report.md before claiming final results
```

## Current Interpretation

The current code is no longer just a simple GridWorld solver. It now has the minimum paper-aligned skeleton:

```text
Knowledge Storage
-> KK/KV slot binding
-> executable action generation
-> StepResult
-> KnowledgeDelta
-> RewardModule
-> policy/scorer separation
-> TableProphecyModel
-> ImaginationCycle
-> ExperimentRunner
-> CSV metrics
-> Analysis CLI
-> Plot generation
```

However, it is still not a final paper result package until full-scale repeated-seed runs are executed and the generated analysis report is reviewed.

## Baseline Semantics Update

`ORACLE_MDP` names the full-map shortest-path oracle upper bound. It knows the
complete GridWorld map before acting and is therefore a ceiling reference, not a
same-information-condition baseline for the APASSR family. APASSR conditions,
QLEARN, and DQN_PARTIAL should be interpreted separately from this oracle upper
bound.

`DQN_PARTIAL` has been added as the deep reinforcement learning baseline under
the same partial-information condition. It uses only Knowledge Storage masks and
candidate-action features, not the full hidden map, so it can be compared
directly against C0/C1/C2/C3/C4/C5 and QLEARN.

`LOCKED_BOTTLENECK` has been added as a structured dependency stress
environment.
The flag sits behind mandatory door bottlenecks, making key/door KV discovery
and reuse central to success. This environment is intended to test the claim
that knowledge-bound action generation can be especially strong in structured
dependency tasks, even if a neural baseline is competitive on generic maps.

## Framework Positioning

C3 remains the main paper-aligned framework condition:

```text
PolicyABC + Prophecy Module + Imagination Cycle
```

The framework contribution is the closed loop connecting KK/KV Knowledge
Storage, action parameter binding, Prophecy prediction, Imagination candidate
evaluation, execution, and observation-based knowledge update. The specific
Prophecy implementation is an implementation choice. `TableProphecyModel` is
the current lightweight implementation used for C3, while `SequenceProphecyModel`
is an optional C4 implementation variant. The repository should not describe
the whole method as a specific sequence model, and it should not claim
Transformer usage unless a Transformer implementation is actually added.

The GridWorld benchmark is not intended to replace the original nmap-based
pentesting experiment. It is an abstract environment used to test the
knowledge-action dependency mechanism under controlled conditions.

The current Imagination Cycle performs depth-limited candidate rollout using
Prophecy predictions. It should not be described as environment-simulated
rollout because it does not execute future actions or read the hidden map.

## Paper Ablation Plan

The vanilla paper condition remains C3:

```text
C3 = PolicyABC + Prophecy Module + Imagination Cycle
```

The ablation suites vary one factor at a time around that C3 loop:

```text
ablation_1:
  A1_TABLE_C3 vs A1_TRANSFORMER_C3
  Question: does Prophecy implementation complexity affect performance?

ablation_2:
  A2_REWARD_ON vs A2_REWARD_OFF
  Question: does adding prediction-error reward affect learning?

ablation_3:
  A3_D{depth}_B{branch}
  Question: how do Imagination rollout depth and branch count affect performance?
```

Environment sweeps are separate from ablations. Running the same ablation
suites on `random_key_door`, `v2_complex`, and `locked_bottleneck` checks task
robustness, but the environment itself is not the ablated variable.

Implemented command:

```powershell
$env:PYTHONPATH='src'; python -m aassr.ablation --suite all --world all --episodes 30 --seeds 10 --step-limit 120 --workers 6 --output-dir runs\ablation_env_sweep
```
