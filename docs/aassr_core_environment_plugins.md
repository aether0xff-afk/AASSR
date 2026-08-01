# AASSRCore and EnvironmentPlugin

`AASSRCore` is the only condition that may be named `full_aassr` in new
GridPush runs.  The historical GridPush-specific causal agent remains available
as `reduced_causal_agent`; its existing artifacts are not rewritten.

## Runtime boundary

An `EnvironmentPlugin` exposes only:

- reset by world seed;
- the current `RawCausalObservation`;
- primitive `ActionSchema` values;
- primitive action execution as an observable before/after transition;
- terminal state and final sparse reward;
- rendering.

The contract has no solver, optimal-action, goal-distance, progress-shaping,
object-role, private-link or heuristic method.  A non-terminal environment
reward is rejected at the contract boundary.

`GridPushEnvironmentPlugin` exposes four parameterless movement schemas.  Push
behavior remains a consequence of `GridPushWorld` physics.  The plugin imports
the runtime physics module, not `grid_push_solver`.

## Core lifecycle

Every primitive transition follows one core-owned path:

```text
plugin raw observation
  -> CoreObservationEncoder
  -> GOAL / Knowledge / OnlineFeatureMemory queries
  -> Policy and ImaginationTree action selection
  -> plugin primitive execution
  -> AdvancedTransitionEvaluator
  -> Prophecy + Replay/Holdout + Knowledge updates
  -> OnlineFeatureMemory update
  -> terminal GOAL observation and SkillLibrary update
  -> DelayedCreditAssigner
  -> InformationValuePredictor and Policy updates
```

The encoder hashes only visible observation tokens.  It always writes
`StateSnapshot.goal_progress = 0.0`; terminal success is observed only through
the terminal reward/fact after the environment has ended.

## Frozen evaluation

Training ends at an episode boundary and `AASSRCore.export_checkpoint()` stores
the complete learned state of KnowledgeStore, feature memory, GOAL definitions,
Policy, Prophecy, Replay/Holdout, information-value components, delayed-credit
configuration, SkillLibrary, RNG and evaluator counters.  Runtime call-audit
telemetry is intentionally external to this learned-state checkpoint.

Evaluation restores a fresh core from the trusted local checkpoint.  It uses
the same inference lifecycle with `learn=False`.  Knowledge, Prophecy,
Replay/Holdout, feature memory, Policy, information value and SkillLibrary are
not updated, and the checkpoint fingerprint must match before and after the
entire evaluation.

## Call audit semantics

The audit separates three quantities:

- `calls`: a real module method was invoked;
- `learning_updates`: persistent trainable state was updated;
- `work_units`: non-trainable work such as imagined nodes, achieved GOAL events
  or assigned credit records was produced.

The Development module-call probe requires every core module to have a positive
call count, every trainable module to have a positive learning-update count, and
GOAL, ImaginationTree and DelayedCreditAssigner to produce positive work.  The
fixture is explicitly engineering evidence and is not a performance benchmark.

## Solver separation

`grid_push_world.py` contains runtime physics.  `grid_push_solver.py` contains
bounded search, certification and procedural acceptance.  Solver references
are frozen before a plugin or core is constructed.  Solver result objects are
discarded and never passed into the plugin, core, checkpoint, KnowledgeStore,
Prophecy or ImaginationTree.  A fresh-process test verifies that importing the
runtime plugin does not load the solver module.
