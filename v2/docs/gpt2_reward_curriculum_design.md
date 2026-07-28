# GPT-2: Actionable Reward and Autonomous Academy Design

This branch was created from the updated `main` branch. Only the design document from `GPT-1` is carried forward here; the `GPT-1` implementation files are not copied.

## Status

- Base branch: updated `main`
- Branch: `GPT-2`
- Carried forward: reward/curriculum design only
- Not carried forward: `aassr.actionable`, `aassr.curriculum`, `aassr.gpt1_experiment`, GPT-1 tests, or GPT-1 CI
- Implementation status: pending against the updated v2 codebase

## Design goals

1. Minimize hand-authored task knowledge.
2. Reward consequences that are useful across tasks rather than specific KK names.
3. Teach transition, binding, lifecycle, and recovery fundamentals without teaching a fixed solution order.
4. Preserve creativity by supporting model-only transfer: academy Prophecy experience is retained while PolicyABC is reset before target evaluation.
5. Measure successful-trajectory diversity and novelty against academy trajectories.

## Planned actionable reward

The new reward condition should begin from the strongest current APASSR condition in the updated `main` branch and disable prediction-error curiosity during the first reward-alignment experiment.

The intrinsic signal should be based on:

```text
unique executable actions unlocked
+ completed knowledge lifecycle transitions
+ small capped semantic information signal
- typed execution errors
- repeated action with no semantic change
- A-B-A-B movement cycles
```

No key/door/flag-specific reward weights should be used. Candidate unlocks must be deduplicated at the executable-action level, ignoring HOW labels and current-position bindings.

## Planned autonomous academy

The academy scheduler should choose among procedural training bands rather than store fixed solution trajectories.

| Band | Environment family | Intended capacity |
| --- | --- | --- |
| foundation | simple goal environments | observation and movement |
| control | obstacle environments | error discovery and recovery |
| composition | knowledge-dependent environments | knowledge-action composition |
| adversarial | complex dependency environments | transfer and long dependency |

The teacher should use recent learning progress, learnable-zone proximity, and exploration. It must not store or provide action sequences.

### Transfer modes

- `model_only`: retain learned Prophecy/transition experience, reset PolicyABC before evaluation. This is the primary creativity-preserving condition.
- `full_prior`: retain both model experience and PolicyABC. This is an ablation for measuring policy-prior interference.

## Creativity guardrails

Academy reports should include:

- successful trajectory count
- unique successful trajectory count
- successful trajectory diversity
- novel strategy rate relative to successful academy sequences
- normalized trajectory entropy
- policy override rate when target evidence conflicts with academy prior

The academy should only be accepted when target success improves without a material collapse in trajectory diversity, novel-strategy rate, or policy override behavior.

## Required implementation order

```text
1. Inspect the updated main branch and freeze the new baseline.
2. Rebuild the reward condition against the current v2 interfaces.
3. Run reward-component ablations.
4. Add the procedural academy and learning-progress teacher.
5. Compare model-only transfer with full-prior transfer.
6. Validate creativity guardrails.
7. Add KK-update embedding only after reward and academy alignment are confirmed.
```

## Required comparison

Run the updated baseline and the new conditions on identical worlds and seeds:

```text
current best APASSR condition from main
new actionable-reward condition
academy model-only transfer
academy full-prior transfer
```

The next phase, KK-update embedding, must not begin until the reward and academy objectives align with actual success and do not suppress strategy diversity.
