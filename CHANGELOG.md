# Changelog

All notable research/software milestones are recorded here. Architecture-generation names such as `AASSR v2` are tracked separately from package versions; see [`docs/VERSIONING.md`](docs/VERSIONING.md).

## [0.4.0] - 2026-08-08

### Added

- observation-derived semantic ASEQ `(S,A,S')` handling,
- repeated empirical `S -> A -> S` self-loop suppression with all-guarded fallback,
- response-causal pentest observation contract v3,
- predeclared 2x2 learning-mechanism development experiment,
- dedicated curriculum-validation seed pool and fixed diagnostic matrix,
- final-evaluation seed blinding guardrails,
- reproducible per-seed experiment artifacts and release evidence documentation.

### Changed

- removed hidden workflow-depth normalization and other privileged stage metadata from the policy observation,
- made `goal_progress` terminal-only in the audited pentest contract,
- removed duplicate own/target role re-randomization in the audited wrapper,
- separated historical train-only repetition filtering from corrected TD episode-boundary handling as explicit experimental factors,
- hardened no-checkpoint diagnostic output and methodology source guardrails.

### Development evidence

Run `31240514649`, launched from commit `83a6f23698ad23987b3c878925bc94fa88ae4038`, completed all 12 predeclared 10k-transition cells for research seeds 7, 42, and 100.

Across both learning backgrounds, ASEQ changed fixed-diagnostic success from `23/432 (5.3%)` to `76/432 (17.6%)` and stalled episodes from `328/432 (75.9%)` to `21/432 (4.9%)`.

This release does not claim high-level transfer is solved; L4 and above remained at zero success in the development diagnostic. See [`docs/releases/v0.4.0.md`](docs/releases/v0.4.0.md).

## [0.3.0]

Previous package baseline. Historical details are intentionally not reconstructed retroactively without a frozen release record.
