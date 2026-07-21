# APASSR_FULL Design Note

`APASSR_FULL` is the paper-aligned implementation path added beside the
legacy C0-C5 experiments. It keeps C3/C5 reproducibility intact and separates
the new architecture behind an explicit condition.

## Legacy Versus Full

`C3` and `C5` use Prophecy-guided candidate scoring with lightweight dependency
lookahead. They score current candidates and estimate whether one candidate's
predicted delta-KK can help another current candidate.

`APASSR_FULL` uses predicted-state multi-step imagination with virtual Knowledge
Store transitions and future action regeneration. Each rollout step:

1. predicts delta-KK/error/flag from the current imagined state and action,
2. applies predicted semantic KK additions to a cloned Knowledge Store,
3. builds a new imagined state signature,
4. regenerates future candidates from the virtual knowledge state,
5. accumulates discounted rollout value.

The rollout is a belief/knowledge-state rollout. It does not read hidden
GridWorld cells, hidden object positions, or execute future environment actions.

## Core Classes

- `CandidateGenerator`: shared candidate binding rules for real DMP state and
  imagined state.
- `GridKnowledgeState`: position/bounds/open-door state view used by candidate
  generation.
- `PredictedKnowledgeDelta`: Prophecy-derived virtual semantic delta.
- `ImaginedState`: virtual Knowledge Store plus position belief and rollout
  metadata.
- `ImaginedStep`: one predicted transition.
- `ImaginedTrajectory`: multi-step rollout result.
- `PredictedStateImaginationCycle`: full APASSR imagination engine.

## Policy A/B/C

Legacy C0-C5 candidate generation keeps the old single-HOW behavior for
reproducibility. `APASSR_FULL` enables independent HOW expansion, so the same
WHAT/WHERE binding can produce several HOW candidates such as `least_tried`,
`high_uncertainty`, and `random`. Candidate probability remains:

```text
P(candidate) = P_A(WHAT) * P_B(HOW) * P_C(WHERE)
```

## Prophecy Features

The table key now includes state signature, WHAT, HOW, WHERE, generalized
binding signature, and recent transition summary. The richer state signature is
derived from the Knowledge Store and includes key/door/hint/flag counts,
frontier/unknown buckets, visited ratio bucket, position region, last action,
last semantic delta, last error, and recent transition summary.

## Reward Semantics

The default C2/C3/C5 reward behavior remains curiosity-style:

```text
total_reward = external + intrinsic + beta * prediction_error
```

This is documented as a surprise/curiosity bonus, not an accuracy reward.
`DMPConfig.prediction_error_mode` can also select `accuracy` or `disabled`.

## Diagnostics

`APASSR_FULL` now records structure-first diagnostics for virtual transitions,
future candidate regeneration, newly unlocked actions, setup-action selection,
Prophecy KK alignment, imagined-next-action match, and placeholder/belief KV
usage. Metric definitions are in `docs/apassr_full_diagnostic_metrics.md`; the
first 30x10 results are in `docs/apassr_full_diagnostic_results.md`.

## Calibrated Variant

`APASSR_FULL_CAL` is a separate calibrated condition for comparing against the
original full rollout. It adds candidate signature deduplication,
raw-versus-unique future expansion diagnostics, confidence-discounted future
rollout value, and placeholder grounding discount. It does not change the
environment, reward function, Prophecy update rule, rollout depth, or hidden-map
boundary. Details are in `docs/apassr_full_calibrated_design.md`.
