# AASSR

AASSR/APASSR is a knowledge-driven agent design in which actions create
knowledge, and knowledge creates the next actions.

Current implementation progress is summarized in `docs/progress.md`.

The current paper-facing Korean research summary is available at
`docs/research_summary_ko.md`.

The central idea is:

```text
Action -> Observation -> Knowledge Update -> Parameter Binding -> Next Action
```

Knowledge Storage is therefore not just a memory of past observations. It is a
typed parameter pool that supplies concrete values for action templates.

```text
Knowledge Storage = memory + action parameter supplier
```

## Baseline Semantics

The v2 comparison uses `ORACLE_MDP` for the full-map shortest-path oracle upper
bound. It is a ceiling reference that knows the complete GridWorld map before
acting, so it should not be interpreted as a same-information-condition baseline
against the APASSR conditions.

The same-information-condition baseline set now includes `DQN_PARTIAL`, a small
numpy MLP DQN baseline that receives only the current Knowledge Storage and
candidate-action features. It does not receive the hidden full map.

## Framework Positioning

The proposed framework is a knowledge-parameterized decision-making loop, not a
specific neural architecture. Observations are stored as KK/KV knowledge items,
and those KV values are reused as parameters for future action templates. The
Prophecy Module predicts possible knowledge gain, error likelihood, and
flag/goal relevance for candidate actions. The Imagination Cycle compares
candidate actions using those predictions before execution. The DMP then closes
the loop through observation, knowledge update, action parameter binding, and
execution.

`C3` is the main paper-aligned framework condition:

```text
C3 = PolicyABC + Prophecy Module + Imagination Cycle
```

The current C3 implementation uses `TableProphecyModel` as a lightweight
implementation of the general Prophecy Module. `C4` uses `SequenceProphecyModel`
as an optional sequence-based implementation variant. C4 does not replace the
framework and should not be described as the central contribution.

`C5` is an improved APASSR condition derived from later ablation findings. It
keeps the C3 loop but disables unconditional knowledge-gain scoring inside the
Imagination score. C5 is reported as an improved variant, not as the vanilla
paper framework.

## Scope Notes

The GridWorld benchmark is not intended to replace the original nmap-based
pentesting experiment. It is an abstract environment used to test the
knowledge-action dependency mechanism under controlled conditions.

The current Imagination Cycle performs depth-limited candidate rollout using
Prophecy predictions. It should not be described as environment-simulated
rollout because it does not execute future actions or read the hidden map.

## Core Concepts

### KK and KV

`KK` is a knowledge key: an abstract parameter slot required by an action
template.

`KV` is a knowledge value: a concrete value that can be bound into a `KK` slot.

For example:

```text
Action template:
MOVE_TOWARD {KK_FRONTIER_CELL}

Knowledge Storage:
KK_FRONTIER_CELL = [(3, 4), (4, 4), (5, 2)]

Bound executable action:
MOVE_TOWARD (4, 4)
```

This means the agent does not merely choose primitive directions such as
`up/down/left/right`. It generates executable actions by filling action
templates with values discovered through previous interaction.

### Knowledge Storage

Knowledge Storage stores KV candidates that can be reused in later action
generation.

It serves two roles at the same time:

1. It records what the agent has learned.
2. It supplies parameters for future actions.

The APASSR loop can be summarized as:

```text
Action creates knowledge.
Knowledge creates the next action.
```

## GridWorld Knowledge Schema

GridWorld uses knowledge keys that are directly usable as action parameters.

| KK | Meaning | Used by |
| --- | --- | --- |
| `KK_CURRENT_POS` | Current agent position | `MOVE_TOWARD`, `INSPECT_CELL` |
| `KK_DIRECTION` | Basic interaction directions | `INSPECT_CELL`, initial exploration |
| `KK_SELF` | Agent identity/self reference | Self-state updates |
| `KK_KNOWN_CELL` | Confirmed cells | Path calculation |
| `KK_UNKNOWN_NEIGHBOR` | Adjacent unknown cells | `INSPECT_CELL` |
| `KK_FRONTIER_CELL` | Boundary between known and unknown space | `MOVE_TOWARD`, `INSPECT_CELL` |
| `KK_WALL_CELL` | Blocked cells | Invalid-action pruning |
| `KK_HINT_CELL` | Cells containing hints | `MOVE_TOWARD`, `INSPECT_CELL` |
| `KK_HINT_VALUE` | Hint contents | `FOLLOW_HINT` |
| `KK_KEY_CELL` | Known key locations | `MOVE_TOWARD` |
| `KK_KEY_OBJECT` | Acquired key objects | `USE_OBJECT` |
| `KK_DOOR_CELL` | Known door locations | `MOVE_TOWARD`, `USE_OBJECT` |
| `KK_FLAG_CELL` | Observed or inferred flag location | Final target movement |

Initial knowledge must only include the minimum interface needed to interact
with the environment:

```text
KK_CURRENT_POS = start position
KK_DIRECTION = [UP, DOWN, LEFT, RIGHT]
KK_SELF = agent
```

Initial knowledge must not include the map layout, target location, hint
contents, key location, or door location.

## KV Metadata

Each KV should carry metadata so policies can reason about candidate quality,
reuse, and lifecycle state.

```text
{
  value: concrete value,
  type: value type,
  source: observed | inferred | imagined | prophetic,
  confidence: 0.0-1.0,
  status: active | visited | blocked | consumed | stale,
  used_count: number,
  success_count: number,
  last_updated: step index
}
```

Recommended value types include:

| Type | Example |
| --- | --- |
| `CellCoord` | `(4, 5)` |
| `Direction` | `UP` |
| `ObjectType` | `key` |
| `ObjectInstance` | `key#1` |
| `HintText` | `"flag is east"` |
| `HintTarget` | `(7, 2)` |

The distinction between location and object is important. For example,
`KK_KEY_CELL` represents where a key is located, while `KK_KEY_OBJECT`
represents a key the agent has acquired and can use.

## KV Lifecycle

Knowledge update is not append-only. KV entries must be added, revised, removed,
or marked according to observations.

Typical lifecycle rules:

| Event | Knowledge update |
| --- | --- |
| Unknown neighbor discovered | Add to `KK_UNKNOWN_NEIGHBOR` |
| Cell inspected and confirmed | Move from `KK_UNKNOWN_NEIGHBOR` to `KK_KNOWN_CELL` |
| Wall confirmed | Add to `KK_WALL_CELL`, mark blocked, exclude from movement candidates |
| Reachable boundary found | Add to `KK_FRONTIER_CELL` |
| Frontier visited/exhausted | Mark visited or stale, stop using as active frontier |
| Hint found | Add cell to `KK_HINT_CELL`, add content to `KK_HINT_VALUE` |
| Key found | Add location to `KK_KEY_CELL` |
| Key acquired | Mark key cell consumed, add object to `KK_KEY_OBJECT` |
| Door found | Add to `KK_DOOR_CELL` |
| Door opened | Mark door consumed/opened, exclude from repeated open candidates |
| Flag observed or inferred | Add or update `KK_FLAG_CELL` |

Observed, inferred, imagined, and prophetic KVs must remain distinguishable.
Policies may prefer observed values, but inferred or prophetic values can still
be used when exploration requires them.

## Action Generation

Actions are generated by binding KV candidates into KK slots.

Each action template has this structure:

```text
template: action expression with KK slots
required_kk_slots: KK slots that must have bindable KV candidates
preconditions: conditions that must hold before execution
binding_strategy: how candidates are selected
execution_rule: how the bound action is executed
observation_to_kv_update_rule: how results update Knowledge Storage
```

GridWorld v1 action templates:

| Template | Required slots | Notes |
| --- | --- | --- |
| `MOVE_TOWARD {KK_FRONTIER_CELL}` | `KK_CURRENT_POS`, `KK_FRONTIER_CELL` | High-level path movement toward unexplored boundary |
| `MOVE_TOWARD {KK_HINT_CELL}` | `KK_CURRENT_POS`, `KK_HINT_CELL` | Move toward known hint location |
| `MOVE_TOWARD {KK_KEY_CELL}` | `KK_CURRENT_POS`, `KK_KEY_CELL` | Move toward known key location |
| `MOVE_TOWARD {KK_DOOR_CELL}` | `KK_CURRENT_POS`, `KK_DOOR_CELL` | Move toward known door location |
| `MOVE_TOWARD {KK_FLAG_CELL}` | `KK_CURRENT_POS`, `KK_FLAG_CELL` | Move toward observed or inferred flag |
| `INSPECT_CELL {KK_UNKNOWN_NEIGHBOR}` | `KK_CURRENT_POS`, `KK_UNKNOWN_NEIGHBOR` | Inspect adjacent unknown cell |
| `INSPECT_CELL {KK_FRONTIER_CELL}` | `KK_CURRENT_POS`, `KK_FRONTIER_CELL` | Inspect frontier cell when reachable |
| `USE_OBJECT {KK_KEY_OBJECT} ON {KK_DOOR_CELL}` | `KK_KEY_OBJECT`, `KK_DOOR_CELL` | Binds an acquired object to a target door |
| `FOLLOW_HINT {KK_HINT_VALUE}` | `KK_HINT_VALUE` | Converts hint into target candidates |

For GridWorld v1, `MOVE_TOWARD` is the only high-level path action. `INSPECT`
and `USE_OBJECT` require their local preconditions to be satisfied before they
execute.

## Policy Structure

A policy decision is represented as:

```text
WHAT = action template
HOW = KV selection strategy
WHERE = KK slot or candidate pool
```

Examples:

```text
(WHAT=INSPECT_CELL, HOW=least_tried, WHERE=KK_UNKNOWN_NEIGHBOR)
-> INSPECT_CELL (2, 3)

(WHAT=MOVE_TOWARD, HOW=nearest, WHERE=KK_FRONTIER_CELL)
-> MOVE_TOWARD (5, 6)

(WHAT=USE_OBJECT, HOW=normal, WHERE=KK_DOOR_CELL)
-> USE key#1 ON door at (7, 4)
```

Recommended `HOW` strategies:

| Strategy | Behavior |
| --- | --- |
| `nearest` | Prefer closest reachable KV |
| `random` | Sample from valid candidates |
| `least_tried` | Prefer low `used_count` candidates |
| `high_uncertainty` | Prefer lower-confidence candidates |
| `prophecy_best` | Prefer candidates scored highly by Prophecy |

## Candidate Pruning

To avoid candidate explosion, action generation should apply pruning before
policy scoring:

1. Check template preconditions first.
2. Exclude `blocked`, `consumed`, and stale candidates.
3. Keep only reachable movement targets.
4. Limit each KK slot to top-k candidates.
5. Remove invalid object-target pairs before scoring.

This keeps APASSR knowledge-driven without generating unbounded combinations
such as every object against every target.

## DMP Flow

The Decision-Making Process runs the same loop every step:

```text
1. Read current Knowledge Storage.
2. Read available action templates.
3. Inspect each template's required KK slots.
4. Retrieve KV candidates for those slots.
5. Generate executable candidate actions through binding.
6. Score candidates with Policy, Prophecy, and Imagination.
7. Execute the selected action.
8. Receive observation.
9. Extract new KV entries from the observation.
10. Update Knowledge Storage.
11. Use the new KV entries as parameters for future actions.
```

This gives APASSR its central cycle:

```text
Action -> Observation -> Knowledge Update -> Parameter Binding -> Next Action
```

## Evaluation

Basic success metrics such as task completion and step count are useful but not
enough to prove that Knowledge Storage is active in action generation.

Additional metrics should include:

| Metric | Purpose |
| --- | --- |
| `slot_binding_success_rate` | Measures how often templates can be filled with available KV |
| `valid_action_candidate_ratio` | Measures how many generated candidates are executable |
| `invalid_action_reduction_rate` | Measures whether knowledge reduces repeated invalid actions |
| `knowledge_reuse_count` | Measures how often stored KV is reused in later actions |
| `prophecy_imagination_action_success_rate` | Measures success of predicted or imagined candidates |

## GridWorld Scenario Tests

The GridWorld implementation should validate these scenarios:

1. Initial seed knowledge generates the first inspect or move candidates.
2. Discovering a new frontier creates `MOVE_TOWARD {KK_FRONTIER_CELL}` candidates.
3. Confirmed walls are excluded from later movement candidates.
4. Inspecting a cell moves it from unknown to known state.
5. Finding a hint creates both hint-cell and hint-value KVs.
6. Finding and acquiring a key updates `KK_KEY_CELL` and `KK_KEY_OBJECT`.
7. Opening a door consumes or updates the relevant door/key candidates.
8. Consumed keys and opened doors do not remain active repeated-action targets.
9. Inferred, imagined, and prophetic values remain distinguishable from observed values.

## Prototype Implementation

This repository includes a small Python prototype for the design above.

| Path | Purpose |
| --- | --- |
| `src/aassr/knowledge.py` | KK/KV types, KV metadata, lifecycle state, and `KnowledgeStore` |
| `src/aassr/gridworld.py` | GridWorld environment, action candidates, DMP loop, and observation-to-KV updates |
| `src/aassr/dashboard.py` | Streamlit-independent dashboard table helpers |
| `src/aassr/reward.py` | Sparse external reward and knowledge-change intrinsic signal |
| `src/aassr/policy.py` | C0/C1-ready selectors plus Policy A/B/C probability table scaffold |
| `src/aassr/prophecy.py` | Prophecy Module interface plus table, optional sequence, and optional Transformer implementations |
| `src/aassr/imagination.py` | Depth-limited C3 candidate rollout using Prophecy predictions |
| `src/aassr/worlds.py` | Fixed and randomized GridWorld builders for experiment generality checks |
| `src/aassr/metrics.py` | Step, episode, and summary metric rows for experiments |
| `src/aassr/experiment.py` | C0/C1/C2/C3/C4/C5 GridWorld experiment runner and CSV writer |
| `src/aassr/ablation.py` | Paper-facing ablation suites for Prophecy implementation, Prophecy reward, and Imagination depth/branching |
| `src/aassr/analysis.py` | Experiment result analysis, seed-level bootstrap CI, report generation |
| `src/aassr/plotting.py` | Matplotlib figures for paper-facing result plots |
| `src/aassr/visualization.py` | ASCII GridWorld, Knowledge Storage, action candidate, and Mermaid flow renderers |
| `app.py` | Streamlit dashboard for stepping through the GridWorld DMP |
| `tests/` | Scenario and unit tests for Knowledge, DMP, PolicyABC, Prophecy, Imagination, dashboard helpers, and visualization |

The DMP now returns a `StepResult` for every action. `StepResult` contains the
executed action, observation, `KnowledgeDelta`, external reward, intrinsic
reward, total reward, error flag, flag-found signal, and episode `done` state.
When C2 prophecy is enabled, it also contains prophecy prediction metadata,
prediction error, and prophecy loss. When C3 imagination is enabled, it also
contains the selected imagination score and the full candidate score trace.

`KnowledgeDelta` separates semantic knowledge changes from usage metadata
changes. Reward and Prophecy use semantic changes only; `used_count`,
`success_count`, and `last_updated` updates are tracked as `usage_updated`.

`KK_CURRENT_POS` is a singleton state key. Historical movement is recorded in
`KK_KNOWN_CELL` and `KK_VISITED_CELL`, while frontier and unknown-neighbor KVs
are removed from active candidate pools as soon as a cell becomes known or
blocked.

Use `DMPConfig(use_prophecy=True)` with a Prophecy Module implementation such as
`TableProphecyModel` to run the C2
reward loop:

```text
predict before execution -> execute -> update prophecy -> add beta * prediction_error to reward
```

Use `DMPConfig(use_prophecy=True, use_imagination=True)` with a shared Prophecy
Module and `ImaginationCycle` to run C3:

```text
generate executable candidates -> predict candidate outcomes -> rollout candidate branches -> execute selected candidate
```

Current development conditions are:

| Condition | Meaning |
| --- | --- |
| `C0` | RandomScorer |
| `C1` | PolicyABC |
| `C2` | PolicyABC + Prophecy Module reward |
| `C3` | Main framework: PolicyABC + Prophecy Module + Imagination |
| `C4` | Optional ablation: PolicyABC + sequence-based Prophecy implementation + Imagination |
| `C5` | Improved APASSR: C3 loop with ablation-derived Imagination weights |

Run experiment conditions with:

```powershell
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C0 --episodes 100 --seeds 10
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C1 --episodes 100 --seeds 10
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C2 --episodes 100 --seeds 10
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C3 --episodes 100 --seeds 10
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C4 --episodes 100 --seeds 10
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C5 --episodes 100 --seeds 10
```

By default, each condition writes to a condition-safe folder:

```text
runs/gridworld/{condition}/gridworld_steps.csv
runs/gridworld/{condition}/gridworld_episodes.csv
runs/gridworld/{condition}/gridworld_summary.csv
```

Run all conditions and write a combined summary with:

```powershell
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --episodes 100 --seeds 10 --workers 6
```

This writes:

```text
runs/gridworld/all/C0/gridworld_summary.csv
runs/gridworld/all/C1/gridworld_summary.csv
runs/gridworld/all/C2/gridworld_summary.csv
runs/gridworld/all/C3/gridworld_summary.csv
runs/gridworld/all/C4/gridworld_summary.csv
runs/gridworld/all/C5/gridworld_summary.csv
runs/gridworld/all/combined_summary.csv
```

Use randomized maps with:

```powershell
$env:PYTHONPATH='src'; python -m aassr.experiment --condition C3 --world random_key_door --episodes 100 --seeds 10
```

Available worlds are `fixed`, `random_flag`, `random_wall_flag`,
`random_key_door`, `v2_complex`, and `locked_bottleneck`.

## Paper Ablations

C3 remains the vanilla paper-aligned APASSR condition. Ablations vary one
factor around that loop:

| Suite | Question | Conditions |
| --- | --- | --- |
| `ablation_1` | Does Prophecy implementation complexity matter? | `A1_TABLE_C3` vs `A1_TRANSFORMER_C3` |
| `ablation_2` | Does prediction-error reward help? | `A2_REWARD_ON` vs `A2_REWARD_OFF` |
| `ablation_3` | How do Imagination rollout depth and branch count affect performance? | `A3_D{depth}_B{branch}` |
| `ablation_4` | Which Imagination mechanism matters? | full C3 vs no dependency/repeat/prior/rollout terms |
| `ablation_5` | Which Prophecy score component matters? | full C3 vs no knowledge/flag/error score terms |

Run all ablations on one environment:

```powershell
$env:PYTHONPATH='src'; python -m aassr.ablation --suite all --world v2_complex --episodes 30 --seeds 10 --step-limit 120 --workers 6 --output-dir runs/ablations
```

Long experiment CLIs print seed-level progress and ETA, for example:

```text
[v2_complex/ablation_1_prophecy_model/A1_TABLE_C3] 4/10 ( 40.0%) elapsed=1m12s eta=1m48s
```

Run the same ablations across multiple environments:

```powershell
$env:PYTHONPATH='src'; python -m aassr.ablation --suite all --world all --episodes 30 --seeds 10 --step-limit 120 --workers 6 --output-dir runs/ablation_env_sweep
```

Environment changes are robustness/generalization checks, not ablations. The
ablation variable is the controlled module/configuration change inside the same
world.

Recommended full-run protocol:

```powershell
# Smoke
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 20 --seeds 5 --step-limit 80

# Medium
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 100 --seeds 10 --step-limit 100

# Paper-candidate
$env:PYTHONPATH='src'; python -m aassr.experiment --condition all --world random_key_door --episodes 200 --seeds 20 --step-limit 120
```

Analyze results with:

```powershell
$env:PYTHONPATH='src'; python -m aassr.analysis --input runs/gridworld/all --output runs/gridworld/all/analysis
```

For v2 comparisons with Q-learning, DQN_PARTIAL, and ORACLE_MDP, use:

```powershell
$env:PYTHONPATH='src'; python -m aassr.v2_compare --episodes 30 --seeds 10 --step-limit 120 --world v2_complex --workers 6 --output-dir runs/paper_rollout_v2_complex_30x10
```

`--workers` parallelizes independent seeds. Episodes within each seed remain
sequential so PolicyABC, Prophecy, and DQN learning state still accumulates in
the same way as the non-parallel run.

The analysis command writes:

```text
summary_table.csv
condition_stats.csv
learning_curve.csv
figure_success_rate.png
figure_steps_to_flag.png
figure_semantic_gain.png
figure_repeat_error_rate.png
figure_learning_curve.png
report.md
```

`steps_to_flag_mean` is computed over successful episodes only. Confidence
intervals use seed-level bootstrap 95% CI.

The episode CSV includes:

```text
condition, seed, episode, success, steps_to_flag, total_reward,
external_reward, semantic_gain_total, prophecy_error_mean,
repeat_count, error_count, knowledge_reuse_count, unique_action_count
```

Run the tests with:

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

Render a small demo state with:

```powershell
$env:PYTHONPATH='src'; python -m aassr.visualization
```

Run the Streamlit dashboard with:

```powershell
$env:PYTHONPATH='src'; streamlit run app.py
```

The dashboard shows the observed GridWorld, current Knowledge Storage, generated
action candidates, policy metrics, latest action result, and C3 imagination
score table when available. You can execute a selected candidate directly, run
one policy-selected step, or auto-run several steps.

Map legend:

| Symbol | Meaning |
| --- | --- |
| `A` | Agent |
| `?` | Active frontier or unknown neighbor |
| `.` | Known empty cell |
| `#` | Known wall |
| `H` | Known hint cell |
| `K` | Known key cell |
| `D` | Known door cell |
| `F` | Known or inferred flag cell |

## Research Wording

본 연구에서 Knowledge Storage는 단순한 관측 기록이 아니라, 행동 생성에
필요한 파라미터 후보를 저장하는 구조이다. 에이전트는 행동 결과로부터 새로운
KV를 획득하고, 이 값들은 이후 행동 템플릿의 KK 슬롯에 대입되어 새로운 행동
후보를 생성한다. 따라서 APASSR의 탐색은 "행동 -> 관측 -> 지식 갱신 ->
파라미터 대치 -> 다음 행동 생성"의 순환 구조로 이루어진다.

GridWorld에서 새로 발견한 칸, 미확인 경계, 힌트, 열쇠, 문 위치는 모두 다음
행동의 파라미터로 사용된다. 예를 들어 에이전트가 미확인 경계 칸을 발견하면
해당 위치는 `KK_FRONTIER_CELL`의 KV로 저장되고, 이후
`MOVE_TOWARD {KK_FRONTIER_CELL}` 또는 `INSPECT_CELL {KK_FRONTIER_CELL}`
행동의 대상으로 대치된다.
