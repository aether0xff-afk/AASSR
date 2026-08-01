# Minecraft backend comparison for Protocol v2

Status: 2026-08-01. This comparison is a design note, not evidence that a real
Minecraft runtime was executed. Protocol v2 currently depends only on the
backend-neutral `MinecraftAdapter` and the deterministic
`MockMinecraftAdapter`.

## Scope boundary

The implemented interface accepts high-level skills and returns only structured,
observable deltas. RGB perception, keyboard/mouse control, a live server, and
backend-specific installation are outside this implementation. Semantic-skill
and opaque-skill results are stored and interpreted separately.

## Official-source comparison

| Backend | Officially documented characteristics | Fit for a later phase |
|---|---|---|
| MineStudio | The [official repository](https://github.com/CraftJarvis/MineStudio) describes an agent-development toolkit, and the [official PyPI package](https://pypi.org/project/minestudio/) publishes its supported Python requirement and release metadata. | Most plausible first candidate for a new integration spike. Assets, runtime compatibility, licenses, and reproducibility must be certified before adoption. |
| MineRL | The [official repository](https://github.com/minerllabs/minerl) exposes Gym environments and documents the Java/runtime requirements; its [official installation tutorial](https://minerl.readthedocs.io/en/latest/tutorials/index.html) describes platform setup. | Useful for pixel/action research, but it does not directly implement this repository's high-level skill contract. A separately tested adapter would be required. |
| Project Malmo / MalmoEnv | The [official repository](https://github.com/microsoft/malmo) documents the Minecraft mod, Java client, mission interface, MalmoEnv, display requirements, and networking constraints. The [Microsoft Research project page](https://www.microsoft.com/en-us/research/project/project-malmo/) gives project context. | Its mission-oriented structure may suit controlled tasks, but runtime, display, port, version, and maintenance risks require an isolated feasibility study. |

## Decision

No real backend is selected or installed in Protocol v2.0. A future phase may
implement one adapter only after the mock suite's engineering and adequacy gates
pass and after dependency/version, asset provenance, deterministic reset,
observation privacy, and seed replay are independently certified. Results from
different backends must never be pooled without a protocol-version change.
