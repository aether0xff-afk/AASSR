# AASSR core structure audit v2.2

This audit corrects the earlier Development report.  The immutable v2.1 result
used a fixed terminal GOAL and `TabularProphecy`; it is historical diagnostic
evidence, not evidence for the condition now named `full_aassr`.

| Existing class | Original purpose | v2.2 core connection | Actual call site | Earlier omission | Justified? | Reconnection |
|---|---|---|---|---|---|---|
| `LearningAgent` | Coordinate planner, evaluator, GOAL completion and skill promotion | Not instantiated | Legacy `agent.py:18`; no production caller found | The first core reimplemented its loop | Partly: its environment contract cannot consume `EnvironmentPlugin`, but silently bypassing it was not justified | Its owned modules and lifecycle are injected into `AASSRCore`; `LearningAgent` remains legacy rather than being falsely listed |
| `AutonomousLearningAgent` | Standalone online policy/Prophecy experimental agent | Not part of full factory | `autonomous_experiment.py`, `paper_runner.py`, escape training | Separate research path | Yes; combining its private policy loop would create a second core | Retained as legacy/baseline only |
| `GoalGenerator` | Generate internal desires from observed state gaps or blocked actions | Connected through `ObservableGoalRuntime` | `goal_runtime.py:146`, called from `aassr_core.py:611` path | Omitted; fixed terminal goal only | No | Called after each training transition using only before/after visible snapshots and public action success |
| `GoalSet` | Store/evaluate multiple goals | Injected and checkpointed | `aassr_core.py:505`, `aassr_core.py:611` | Contained only terminal goal | No | Dynamic goals share the same set used by selection and scoring |
| `GoalStateScorer` | Score final and internal GOAL satisfaction deltas | Injected into imagination and information-value estimator | Factory assembly in `aassr_core.py`; `ObservableGoalProgressEstimator` | `internal_goal_weight=0.0` | No | Full factories use configured nonzero internal weight; tabular comparison keeps zero |
| `OnlineGRUProphecy` | Online recurrent next-state model | Directly injected in both full factories | `aassr_core.py:938`; evaluator learn and tree predict paths | Replaced by tabular model | No | Uses the existing `learn`, `predict_step`, loss and recurrent memory |
| `GRUMemory` | Immutable recurrent state for real sequences and imagined branches | Connected | `gru_prophecy.py:62`, `skills.py:219`, `imagination_tree.py` nodes | Branches started from generic zero memory | No | `SkillAwareProphecy.initial_memory()` clones current training memory; predictions return new immutable branch memory |
| `ImaginationTree` | Branching Prophecy rollout with branch-local policy/model memory | Injected | `aassr_core.py:505` | Present, but with fixed-goal tabular model | Insufficient | Full factory uses GRU and dynamic `GoalStateScorer`; no-imagination factory keeps the object disabled for the ablation |
| `AdvancedTransitionEvaluator` | Execute real transitions, separate knowledge/model/holdout gains and learn | Injected | `aassr_core.py:611` | Present | Yes, but previously incompletely fed downstream | Its information features now flow into FeatureMemory and Policy |
| `KnowledgeStore` | Store observed facts with trace provenance | Injected and checkpointed | Evaluator `execute`, core GOAL evaluation | Present | Yes | No GridPush-private knowledge adapter added |
| `OnlineFeatureMemory` | Learn reusable information/action-slot value | Injected and checkpointed | `_observe_feature_memory` and episode finish in `aassr_core.py` | Used only `real_reward` | No | Receives predicted information value, immediate information value and delayed terminal credit |
| `InformationValuePredictor` | Predict delayed realized information value from general features | Injected | `learning.py:63`, evaluator execute/finish | Present but output did not reach feature policy path | No | Predicted value is included in both FeatureMemory and policy reinforcement target |
| `DelayedCreditAssigner` | Discount terminal outcomes back across the episode | Injected | `learning.py:412`, called at `aassr_core.py:738` episode finish | Present | Yes, but FeatureMemory did not receive it | Credit is logged and passed to FeatureMemory and Policy |
| `WeightedPolicy` | Rank actions and hold real/imagined value updates | Injected | `aassr_core.py:505`; `learning.py:412` | Present | Yes | FeatureMemory provides branch-neutral `PolicyMemory` deltas; episode learning reinforces the same policy |
| `SkillLibrary` | Promote repeated goal-achieving primitive sequences | Injected | `aassr_core.py:611` goal completion path | Present, terminal-only GOAL limited observations | Insufficient | Dynamic internal GOAL achievements call the existing promotion method |
| `SkillAwareProphecy` | Roll learned skills as primitive Prophecy sequences | Injected around the selected Prophecy | Factory assembly; `skills.py:202` | Present over tabular model | Insufficient | Wraps OnlineGRUProphecy in both full factories and propagates branch memory |

No solver result, path, optimal action, goal distance, block role, private
plate-door link, private viability or intermediate environment reward is passed
through these reconnections.

