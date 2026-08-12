# AASSR Core / Plugin Boundary

The current runtime is assembled from two different responsibilities. They are
kept separate so an environment-specific trick cannot silently become part of the
AASSR algorithm.

## AASSR Core

The core is the reusable learning/reasoning system:

- **ASEQ**: suppress only exact semantic self-loops.
- **Policy**: choose real actions from learned value and information residuals.
- **Prophecy role**: learn a stochastic model of possible next states.
- **Knowledge**: retain facts observed in the current episode.
- **Skills**: promote reusable relational ASeq templates.
- **Imagination**: build multi-step possible futures and back up chance outcomes.
- **Critic**: estimate signed sparse return for imagined branches.
- **Reliability gates**: fail closed when model confidence or real Critic support is insufficient.

The executable source of truth is `current_core_manifest.py`. It intentionally
contains no HTTP, route, profile, CSRF, or pentest semantics.

## Runtime Plugin

A plugin binds that core to one concrete task family. The current plugin is
`plugins/current_pentest.py`.

The pentest plugin owns:

- response-causal public observation rules;
- relational state/action encoding for route/profile/object roles;
- public HTTP-status channels and their categorical supervision;
- the concrete status-aware conditional-mixture Prophecy head;
- pentest terminal and sparse-reward semantics;
- hidden audit/session-pressure masking.

These are not claims about the universal AASSR algorithm. A future GridPush,
robotics, browser, or other plugin should provide its own observation/action
binding while reusing the same core responsibilities.

## Assembly

`current_entrypoint.build_current_pentest_aassr_core()` is now an assembler:

```text
AASSR Core
  Policy / Prophecy role / Knowledge / Skills / ASEQ / Imagination / Critic
                              +
Pentest Plugin
  observation / codec / HTTP status head / environment outcome semantics
                              |
                              v
                    Current Pentest Runtime
```

The assembled agent exposes both sides separately:

- `agent.aassr_core`
- `agent.runtime_plugin`
- `agent.current_core_components`
- `agent.current_plugin_components`

`agent.diagnostics()` reports `aassr_core` and `runtime_plugin` as separate blocks.

## Performance boundary

Performance patches are orthogonal to both the scientific core and plugin
semantics. They may change only execution mechanics. The scaling contract keeps
seeds, replay rows, update cadence, batch size, losses, exploration, curriculum,
and action semantics fixed.

Current performance v2 adds:

1. ensemble-dimension batched GEMM for Prophecy **inference only** while retaining
   the original independent models and optimizers;
2. once-per-update padded Critic tensor construction while retaining the original
   GRUCell recurrence and loss.

Any optimization that fails the equivalence regressions is not eligible for the
10k scaling run.
