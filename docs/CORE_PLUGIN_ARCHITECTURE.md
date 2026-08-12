# AASSR Core / Plugin Boundary

The current runtime separates four responsibilities so an environment-specific
trick or a runtime optimization cannot silently become part of the AASSR
algorithm.

## Four layers

| Layer | Owns | Must not own |
|---|---|---|
| **Core** | learning/reasoning roles and generic stateful primitives | HTTP/pentest rules or hardware tricks |
| **Plugin** | one task family's observation/action/outcome semantics | universal AASSR claims |
| **Assembly** | composing one Core with one Plugin | hidden algorithm changes |
| **Performance** | execution mechanics only | reward, curriculum, loss, action semantics |

`current_architecture_layers.py` is the executable module-ownership registry.
Some historically named files remain in their old physical paths for checkpoint
and import compatibility, but their ownership is explicit now. A large physical
file move is intentionally not mixed into the pre-10k scientific scaling patch.

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

`agent.aassr_core` is a live view. If episode state such as the Knowledge store is
replaced, the view follows the current object instead of retaining a stale
construction-time reference.

`agent.diagnostics()` reports `aassr_core` and `runtime_plugin` as separate blocks.

## Performance boundary

Performance patches are orthogonal to both the scientific core and plugin
semantics. They may change only execution mechanics. The scaling contract keeps
seeds, replay rows, update cadence, batch size, losses, exploration, curriculum,
and action semantics fixed.

Current performance v2 adds:

1. ensemble-dimension batched GEMM for Prophecy **inference only**, while retaining
   the original independent models and optimizers;
2. revision-aware cached ensemble parameter packs, rebuilt after model parameters
   change and reused across repeated Imagination prediction calls;
3. once-per-update padded Critic tensor construction while retaining the original
   GRUCell recurrence and loss.

The Prophecy training ensemble remains independently optimized. This patch does
not merge Adam states or change bootstrap sampling/order merely to raise GPU
utilization.

## Validation before 10k

First run the small raw-network CUDA benchmark:

```powershell
python scripts\benchmark_current_performance_v2.py --device cuda:0
```

It compares the current three-model sequential MLP path against cached and
fresh-pack fused inference over several batch sizes and hard-fails if numerical
error exceeds `1e-5`.

Then run the end-to-end scientific-contract profiler:

```powershell
python scripts\profile_current_runtime_performance.py `
  --device cuda:0 `
  --transitions 512 `
  --seed 7 `
  --output runs\current_runtime_performance_profile_v2.json
```

The 10k run is eligible only when replay/episode contracts remain exact,
parameter drift remains within the existing `1e-5` bound, and the optimized path
actually improves wall time on the target GPU.
