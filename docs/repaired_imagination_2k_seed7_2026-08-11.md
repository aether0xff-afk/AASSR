# Repaired Imagination 2k validation — seed 7

Date: 2026-08-11
Branch: `agent/imagination-gate-ablation`
Validation: one 2,048-real-transition training checkpoint, then no-Imagination vs Full from the same frozen checkpoint.

## Result

| Condition | Success | L0 | L1 | L2 | L3 | L4 | True failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-Imagination | 4/20 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0 |
| Full Imagination | 4/20 | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 | 2 |

Full Imagination diagnostics:

- Imagination plans: 297
- switch candidates: 218
- executed interventions / changed actions: 86
- all 86 interventions occurred at L3 `object_choices`
- 58/86 interventions produced `PluginOutcome.error=True`
- errors: 404 × 30, 403 × 26, 429 × 2
- direct success-producing interventions: 0

The previous repaired bottleneck was `0` interventions because future values collapsed into ties. That bottleneck is resolved: the Critic now separates alternatives enough to override Policy. The new bottleneck is **confident wrong intervention**.

## Matched-state audit

68 intervention states had an exact same-scenario + same-semantic-state counterpart in the no-Imagination trace.

- Full intervention -> error, original Policy action -> no error: 50
- Full intervention -> no error, original Policy action -> error: 0

This shows the intervention problem is not explained by stochastic luck alone.

## Root cause 1 — public HTTP status dropped by relational v2

The audited raw observation contains the latest HTTP response status as a public 8-way channel:

`200 / 302 / 400 / 401 / 403 / 404 / 409 / 429`

The repaired relational v2 state preserved public control flags, request usage, workflow progress, resource counts, semantic roles, object information, and the action surface, but dropped this latest response channel.

As a consequence, repeated `403`/`404` outcomes can remain relationally close to safe transitions even though they are directly observed warning signals. Hidden audit pressure and hidden session countdown must remain masked; latest HTTP status is public and should be preserved.

## Root cause 2 — semantic calibration underweights decision-critical response errors

Probability-weighted semantic score was about 0.916 and terminal match about 0.991, yet intervention quality was poor. The metric can remain high because the missing response-risk channel is not explicitly scored.

The next contract must make a wrong `403/404/429` prediction materially reduce semantic quality/reliability.

## Root cause 3 — global Critic readiness permits OOD overrides

Training successes were all L0 and the training curriculum only reached focus L1. Nevertheless the same Critic executed 86 overrides in unseen L3.

`critic_ready=True` currently means the Critic has trained globally; it does not establish that the current public relational state/action is supported by the Critic training distribution.

The next gate must add a **local Critic support** requirement. Support remains a reliability gate only and must never become a value bonus or reward.

## Scaling result

L3 evaluation exposed roughly 172 concrete root actions but only about 17 structural relational roots. Full Imagination was dramatically slower than no-Imagination (L3 approximately 42.8× in this run).

Future planning is already relational below the root, but depth-1 still computes duplicate aliases. The next planner must evaluate one representative per relational root structure and fan the value back to its concrete aliases. Concrete identity is bound only when the final real action is executed.

## Required repair before another long run

1. Relational public state v3: preserve latest public HTTP status, keep hidden audit/session values masked.
2. Status-aware Prophecy/semantic calibration: explicit status correctness in training diagnostics and semantic score.
3. Local Critic support gate: fail closed on state/action regions unsupported by real Critic training data.
4. Structural root deduplication: one imagined computation per relational root, then fan out to concrete aliases.

No further 2k run should be started until these changes pass short unit/current-generation gates and the 192-transition real-environment smoke.