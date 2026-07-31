# AASSR paper protocol implementation status

This document records the implementation evidence for
`paper_autonomy_creativity_experiment_protocol.md`. It is an engineering
completion record, not a claim that the large RQ1/RQ2 Final experiments or
human recruitment have been completed.

## P0 — reliability and leakage control

- Paper configs declare protocol/stage, disjoint research and
  train/seen/unseen world seeds, phase learning permissions, real-transition
  budgets, and frozen statistical settings.
- `OpaqueDependencyWorld` keeps branch viability private. Correct and
  incorrect actions are observationally indistinguishable until terminal
  reward.
- Frozen evaluation uses no exploration and does not advance policy RNG,
  decision counters, Prophecy, holdout, replay, or skill state. Checkpoint
  hashes include policy, Prophecy, holdout, counters, RNG, and effect motifs.
- Manifests contain commit/config hashes, seeds, phases, timestamps,
  software/library versions, hardware, execution settings, failure/exclusion
  lists, and optional human/protocol-lock metadata.

Evidence: `tests/test_paper_protocol.py`,
`tests/test_paper_suite_integration.py`, and the manifest validators.

## P1 — RQ1 and ablations

- Random, Contextual Policy, tabular Q-learning, DQN, Prophecy without
  Imagination, Full AASSR, and privileged Oracle run through one actual
  transition protocol. Oracle is excluded from inference.
- The ablation config includes the seven required component conditions and a
  declarative 4 × 3 × 3 depth/branching/aggregation matrix.
- Analysis first aggregates episodes inside each research seed, then reports
  paired differences, bootstrap intervals, paired permutation tests, Holm
  correction, learning AUC, first-success transitions, final-tail success,
  effect size, runtime, prediction, holdout, and imagination cost.

Pilot evidence:

- `paper_results/paper-autonomy-pilot-v1`: 14,700 episode rows.
- `paper_results/paper-ablation-pilot-v1`: 18,060 episode rows.

Both pass `validate_paper_artifacts.py`.

## P2 — structural transfer

- Checkpoints are split into policy, Prophecy, holdout, empirical effect
  representation, counters, and RNG.
- Effect profiles contain execution/error rates, state/fact changes, unlock,
  risk/goal changes, uncertainty, and information gain without action names.
- Every `0/1/4/16/64` adaptation branch uses the same branch seed and starting
  checkpoint. Learning occurs only in adaptation; the following evaluation is
  hash-frozen.
- Outputs include adaptation curves/AUC, 50%/80% thresholds, sample savings,
  transfer gain, and unseen prediction calibration error.

Pilot evidence: `paper_results/paper-transfer-pilot-v1`, 13,820 episode rows,
100 transfer branch groups, exactly one origin fingerprint per group.

## P3 — creativity environment and strategy analysis

- `MultiSolutionDependencyWorld` has five private causal solution families,
  within-family variations, resource/risk/length trade-offs, and an emergent
  combination.
- Human, baseline, and AASSR paths use the same `StrategyRecord` JSONL schema,
  including the full normalized trace and causal effect graph.
- Novelty reports graph-edit approximation, motif Jaccard, prerequisite-edge,
  solution-family, and effect-sequence distances separately.
- The Pilot threshold is computed from deduplicated baseline causal graphs.
  `configs/frozen_creativity_rules_v1.json` freezes threshold `0.06` and the
  utility/reuse criteria before any creativity Final execution.

Pilot evidence: `paper_results/paper-creativity-pilot-v1`, 805 episode rows
and 700 strategies across all five solution families.

## P4 — anonymous human tooling

- The localhost-only standard-library UI issues anonymous IDs, exposes only
  primitive action descriptions, stores full traces in SQLite, resumes after
  browser/server interruption, assigns blind randomized ratings, prevents
  duplicate/self ratings, and exports JSONL/CSV plus metadata.
- Ratings use 1–5 novelty, utility, coherence, and surprise fields.
- Final merging requires a matching approval ID and dataset version plus at
  least two raters. Agreement and human/automatic concordance conflicts are
  reported separately.
- The software does not collect direct identifiers and does not replace
  institutional approval or participant consent.

Human recruitment, consent, institutional approval, and an approved Final
human dataset remain outside repository implementation scope.

## P5 — isolated safe application

- The Compose network is internal, publishes no host ports, runs as
  `65534:65534`, drops all capabilities, uses a read-only root filesystem, and
  enables `no-new-privileges`.
- World seed changes the internal service, port, distractors, route paths,
  token, and configuration evidence. Three safe solution families reach only
  a local synthetic FLAG.
- Runtime verification checks all routes, invalid isolation assumptions,
  external egress blocking, root-write blocking, health, user, capabilities,
  ports, and network mode, then removes containers/networks/volumes.

Pilot evidence: 25 rows. Optional Final evidence: 600 rows across all three
families. Both artifact sets validate.

## Final protocol lock

`configs/paper_acceptance_gates_v1.json` pins the P0–P3 Pilot manifest hashes.
All Final configs have at least 30 non-Pilot research seeds, disjoint world
seeds, explicit statistics, and valid gate references. The large autonomy,
ablation, transfer, and creativity Final experiments are deliberately not run
as part of implementation validation.

## Verification commands

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/validate_paper_artifacts.py --results <result-directory>
python scripts/run_safe_application.py `
  --config configs/paper_safe_application_pilot_v1.json smoke
```

The completion audit passed 90 tests, including the opt-in Docker runtime test.
