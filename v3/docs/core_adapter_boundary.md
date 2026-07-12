# v3 Core and Web Adapter Boundary

v3 should be read as:

```text
v2 final APASSR core + local web pentesting adapter
```

It is not a new framework. The paper-aligned core remains:

```text
observation
-> KK/KV Knowledge Store update
-> action template parameter binding
-> PolicyABC selection
-> Prophecy prediction
-> Imagination candidate evaluation
-> tool execution through an adapter
-> new observation
```

## Core Modules

- `knowledge.py`: typed KK/KV store and seed interface.
- `policy.py`: WHAT/HOW/WHERE PolicyABC.
- `prophecy.py`: Prophecy Module implementation. The current implementation is
  `TableProphecyModel`.
- `imagination.py`: one-step candidate evaluation using Prophecy predictions.
- `dmp.py`: the closed-loop decision-making process.

## Current Web Adapter

- `actions.py`: maps web concepts to action templates and KK slots.
- `tools.py`: executes primitive fixed wrappers such as GET, POST, HEAD,
  OPTIONS, shallow nmap, and passive fingerprinting.
- `parser.py`: converts HTTP and tool output into new KV candidates.
- `reward.py`: optional local-lab observers such as Juice Shop solved
  challenge feedback.

The adapter may know how to run primitive tools, parse observations, and expose
domain-specific KK slots. It should not encode a human-written challenge stage
list or canonical solution order.

## v3.5 Plugin Boundary

v3 now exposes a plugin interface and registry in `plugins.py`:

- `seed(base_url)`: creates the initial KK/KV interface knowledge.
- `candidates(store)`: binds KK/KV values into domain actions.
- `parse(result)`: converts tool output back into KV observations.
- `reward_observer(name, base_url)`: optionally attaches local-lab feedback.
- `PluginMetadata`: describes the plugin, supported reward observers, adapter
  modules, and safety notes.
- `register_plugin(plugin)`: registers a new target plugin.
- `python -m apassr_tool.plugins`: prints the available plugin manifest.

The default `web` plugin wraps the existing web pentesting adapter. The core
DMP receives a plugin object, but PolicyABC, Prophecy, Imagination, and reward
updates remain domain-independent.

The `json-api-actions` plugin extends the generic web surface with JSON
POST/PUT/PATCH/DELETE actions built from observed endpoints, fields, and values.
`input-mutation-actions` adds generic query/form/JSON mutation candidates, and
`file-surface-actions` adds file/static asset probes anchored to observed paths.
The `juice-shop-full` plugin is the broad training adapter for Juice Shop. It
combines executable web, JSON API, mutation, file surface, recon, and reward
observer capabilities, while recording browser DOM and storage-state plugins as
planned dependencies. It does not seed a challenge order or known answers.

This is intentionally minimal. A plugin is not allowed to provide a human
challenge stage list or a canonical solution path. It provides the environment
interface: action schemas, primitive tool execution surface, observation
parsing, and optional reward feedback.

## Adding a Plugin

1. Add a plugin class that implements `TargetPlugin`.
2. Add or reuse action templates and parser rules.
3. Register the plugin with `register_plugin(...)`.
4. Add tests that prove DMP can instantiate and use the plugin.

Minimal shape:

```python
from apassr_tool.plugins import PluginMetadata, register_plugin


class MyPlugin:
    name = "my-plugin"
    metadata = PluginMetadata(
        name="my-plugin",
        description="My controlled local target.",
        domain="local_lab",
        reward_observers=("none",),
    )

    def seed(self, base_url):
        ...

    def candidates(self, store):
        ...

    def parse(self, result):
        ...

    def reward_observer(self, name, base_url):
        return None


register_plugin(MyPlugin)
```

## v4 Direction

v4 should formalize this boundary as a plugin system. A plugin should provide:

- KK slots and action templates for one domain.
- fixed tool wrappers or simulated executors.
- parsers that extract KV observations.
- optional reward observers for controlled local environments.

The APASSR core should remain stable across plugins.
