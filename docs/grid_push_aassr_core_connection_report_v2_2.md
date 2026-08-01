# GridPush AASSR core connection report v2.2

This report is a correction and a Development-only engineering record.  The
final code-matched run is under
`paper_results_v2/development/paper-grid-push-core-development-v2.2/connection-audit-r4-20260801`;
no Locked Confirmation, Pilot or Final was run.  The earlier r1/r2/r3 runs are
also preserved and were not overwritten.

## 1. Structures missing from the preceding implementation

The preceding core omitted `GoalGenerator`, dynamic GOAL creation/selection,
nonzero internal-goal scoring, `OnlineGRUProphecy`, real recurrent sequence
memory and branch-memory cloning.  FeatureMemory received only sparse
environment reward, so nearly all pre-success uses had value zero.

## 2. Why its `full_aassr` name was wrong

That run instantiated `TabularProphecy`, a single fixed terminal goal and an
internal-goal weight of zero.  Owning many module-shaped fields did not prove
the original recurrent and autonomous-GOAL paths were active.  Its immutable
artifact remains unchanged and is treated as a misnamed Development diagnostic.

## 3. Corrected condition names

- `random`
- `contextual_policy`
- `reduced_causal_agent`
- `tabular_fixed_goal_core`
- `full_aassr_no_imagination`
- `full_aassr`

Only the final condition is produced by `build_full_aassr_core()`.

## 4. Actual full factory

The factory in `src/aassr_v2/aassr_core.py` computes the visible state-vector
size, instantiates the existing `OnlineGRUProphecy`, and injects it into the
common component builder with dynamic goals and imagination enabled:

```python
def build_full_aassr_core(*, config=None, seed=0):
    resolved = config or AASSRCoreConfig()
    return _build_gru_core(
        condition_name="full_aassr",
        config=resolved,
        seed=seed,
        use_imagination=True,
    )
```

The comparison factories use the same orchestrator rather than a GridPush
training loop.

## 5. Actual full-condition class list

The run manifest records:

`KnowledgeStore`, `OnlineFeatureMemory`, `GoalSet`, `GoalGenerator`,
`ObservableGoalRuntime`, `WeightedPolicy`, `OnlineGRUProphecy`,
`ImaginationTree`, `AdvancedTransitionEvaluator`, `ReplayBuffer`,
`PredictionValidator`, `InformationValuePredictor`, `DelayedCreditAssigner`,
`SkillLibrary`, `SkillAwareProphecy`, `GridPushEnvironmentPlugin`.

## 6. GoalGenerator trace

The connection fixture called GoalGenerator 6 times.  The first evidence
observation SHA-256 was
`2a832583e46299e1b5824746cb2621c3ec43ba4dd67a93119077ed1491c3244b`.
It was captured only after a public successful transition; no solver object was
present.

## 7. Internal GOAL example

One generated record was:

- ID: `internal:e0:s0:fact:0:2a832583e4`
- kind: `fact_present`
- target: `spatial:cell:1,1=floor`
- source: `state_gap`
- selected: true
- achieved: true
- discarded: false

This target was derived from the observed before/after state gap.  No block,
pit, plate, door or direction objective was pre-seeded.

## 8. GRU training trace

For trace `aseq-000001`, action `MOVE_EAST|_|_|_` produced loss
`0.07523649440314292`.  The real hidden fingerprint changed from
`8b5a42b74752538af074a4da16334de41a4e4d17d00676f12b307b35cd1a821d`
to `0a3e98b371c08eb45593d16fc805018693e7ceed40a4f2625b1da891b91a92f6`.
Across the fixture: predict 24, learn 5, hidden update 5, sequence reset 6.

## 9. Imagined branch hidden-state evidence

The fixture created 12 non-root branch memories.  Six planner calls had the
same real hidden fingerprint before and after planning.  Frozen evaluation also
left the complete serialized checkpoint unchanged.

## 10. Information value to FeatureMemory and Policy

The trace contains values after the predictor had learned.  For
`aseq-000003`: predicted information value `1.245`, immediate information value
`1.0`, delayed terminal credit `1.0`, FeatureMemory value `3.245`, and policy
update value `3.245`.  The environment still emitted only terminal reward 1;
these are core learning values, not GridPush reward shaping.

## 11. Full module counts

Connection-fixture evidence counts were: GoalGenerator 6, internal GOAL 2,
GRU predict 24, GRU learn 5, hidden update 5, reset 6, branch clone 12,
real-hidden unchanged checks 6, Knowledge updates 6, FeatureMemory updates 18,
information-predictor updates 6, delayed credits 6, policy updates 6, skill
observations 6, imagined nodes 12.  Every required count was greater than zero.

## 12. Small Development result

The run used one research seed, three certified procedural worlds, eight
training episodes per learned condition and three frozen episodes per
condition.  It wrote 58 episode rows.  All six frozen success rates were 0.0
and all learned-condition training tails were 0.0.  This does not support a
performance advantage; it only shows the connected paths executed.

All three worlds were solver-certified before agent construction.  Each was
solvable and adequate, with no private-observation leaks.  Every checkpoint was
unchanged during frozen evaluation, evaluation learning calls were zero, the
environment reward was strict sparse, and the gzip trace replayed completely.

## 13. Remaining omissions and limits

- This run is one-seed Development evidence, not statistical evidence.
- Dynamic goals are learned from observed state gaps and failures, but their
  usefulness on longer worlds is not established by this zero-success run.
- The GRU predicts the existing `StateSnapshot` vector; the separate v2.0/v2.1
  multi-head prophecy interface is not substituted into this legacy full-core
  path.
- The no-imagination and full conditions share component classes by design;
  the former disables planner execution through the injected runtime flag.
- No Confirmation, Pilot, Final, human or real Minecraft run was attempted.

## 14. Test result

The final complete repository test result was `162 passed, 3 skipped in 53.33s`.
The skipped tests are optional-dependency cases.  Test success must be read
together with the immutable run manifest; no performance claim is inferred
from it.
