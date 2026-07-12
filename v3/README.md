# AASSR v3 Local Web Pentesting Adapter

This prototype keeps the v2 APASSR core and attaches a local web pentesting
adapter. The framework is still the same closed loop:

```text
KK/KV Knowledge Store -> action parameter binding -> PolicyABC
-> Prophecy Module -> Imagination Cycle -> DMP execution
-> observation -> knowledge update
```

The web layer is not a replacement for the v2 framework. It is the first
domain adapter that makes the same core useful against a controlled local web
target. v4 should turn this adapter boundary into a cleaner plugin interface.
The current v3 code already exposes a minimal `web` plugin boundary so new
targets can be added without changing PolicyABC, Prophecy, Imagination, or DMP.

The goal is not to attack external systems. The agent can only interact with explicitly allowed local targets, and tool execution is restricted to predefined command templates.

## What v3 Demonstrates

```text
tool output -> KK/KV knowledge update -> action template binding -> next tool command
```

For larger labs such as Juice Shop, APASSR should not receive a human-authored
task stage list. The learnable version treats the whole local target as one
environment. A reward observer may read environment feedback, such as Juice
Shop's solved challenge scoreboard, only after actions execute. That feedback
updates PolicyABC but is not used to seed KK/KV values or hand-pick the next
task.

PolicyABC and the Prophecy Module both learn from execution data. PolicyABC
updates the WHAT/HOW/WHERE probability tables from reward. The current
`TableProphecyModel` stores experience statistics for executed candidates,
templates, policy axes, endpoints, and parameters. The Imagination Cycle uses
those Prophecy predictions to score future candidates before execution. With no
prior experience, Imagination is neutral; it only becomes stronger from
observed reward, knowledge gain, solved challenge deltas, and errors.

For optional experiments where the goal is to discover non-standard paths rather than
quickly imitate known writeups, v3 also has novelty objectives. These objectives
do not encode canonical Juice Shop solutions. They reward candidates, action
chains, and response transitions that are rare in the agent's own experience.
This deliberately allows inefficient exploration when it creates a different
route through the target.

The decision layer is PolicyABC over web-pentesting action axes:

```text
WHAT  = HTTP_GET | HTTP_METADATA | QUERY_PROBE | FORM_POST | AUTHENTICATED_GET | PORT_SCAN | WEB_FINGERPRINT
HOW   = NORMAL | PARAMETERIZED | HEADER_ONLY | METHOD_DISCOVERY | PROBE_VALUE | AUTH_ATTEMPT | AUTHENTICATED | SHALLOW_SCAN | PASSIVE_FINGERPRINT
WHERE = KK_PATH | KK_ENDPOINT | KK_PARAM_NAME | KK_USERNAME | KK_AUTH_PATH | KK_BASE_URL | KK_HOST
```

The tool layer is only an execution adapter. PolicyABC selects a parameterized
action candidate; the adapter converts that candidate into a fixed, allowlisted
tool template such as `curl GET`, `curl POST`, or `nmap`.

Tool wrappers should remain primitive. They provide basic actions such as
`GET`, `HEAD`, `OPTIONS`, query probes, `POST`, and shallow `nmap`; they should
not contain the strategy for solving the lab. Strategy belongs in PolicyABC,
Prophecy, Imagination, and the DMP loop.

## Core and Adapter Boundary

Core modules inherited from the v2 APASSR design:

- `knowledge.py`: KK/KV Knowledge Store and seed interface.
- `policy.py`: PolicyABC over WHAT/HOW/WHERE.
- `prophecy.py`: Prophecy Module implementation, currently `TableProphecyModel`.
- `imagination.py`: one-step candidate evaluation using Prophecy predictions.
- `dmp.py`: closed-loop decision-making process.

Current web pentesting adapter modules:

- `plugins.py`: target plugin interface and default `web` plugin.
- `actions.py`: web action templates and KK slot binding.
- `tools.py`: primitive fixed tool wrappers.
- `parser.py`: observations extracted from HTTP/tool output.
- `reward.py`: optional local-lab reward observers.

Optional exploration module:

- `novelty.py`: diversity pressure for unusual traces. It is not the main
  APASSR framework and should be reported as an optional objective/ablation.

## Plugin System

v3 has a small but real plugin system. The APASSR core does not know whether
the target is a sandbag page, Juice Shop, DVWA, or a future local pentesting
lab. A plugin provides the target interface:

```text
plugin.seed(base_url)
-> initial KK/KV interface knowledge

plugin.candidates(store)
-> bind current KV values into action candidates

plugin.parse(tool_result)
-> extract new KV values from observations

plugin.reward_observer(name, base_url)
-> optional local environment feedback after execution
```

Inspect installed plugins:

```powershell
cd X:\Dev\AASSR\v3
$env:PYTHONPATH='src'
python -m apassr_tool.plugins
python -m apassr_tool.plugins --json
```

Current plugins:

```text
web
- local web pentesting adapter
- fixed HTTP/nmap/whatweb action surface
- supports reward observers: none, juice-shop

web-recon-actions
- shallow nmap, passive fingerprint, OPTIONS/header observation
- no exploit modules

json-api-actions
- generic JSON API action adapter
- creates JSON POST/PUT/PATCH/DELETE candidates from observed endpoints, fields, and values
- does not inject endpoint-specific payloads

input-mutation-actions
- generic query/form/JSON mutation candidates
- uses observed endpoints and parameter names
- uses small generic value mutations such as numeric boundaries, booleans, null-like values, and string edge values

file-surface-actions
- file/static asset probing based on observed paths
- backup suffix and directory file probes anchored to observed file-like paths

juice-shop
- Juice Shop target adapter built on the generic web surface
- uses challenge API only as post-action reward/progress feedback
- does not seed challenge order, credentials, flags, or writeup-derived actions

juice-shop-full
- composite Juice Shop training adapter
- combines web, JSON API, mutation, file surface, recon, and reward observer capabilities
- includes planned browser/storage plugin dependencies in metadata

browser-dom-actions
- registered dependency plugin for future Playwright-backed navigation/click/fill
- not executable yet

storage-state-observer
- registered dependency plugin for future localStorage/sessionStorage/JWT-like observation
- cookie observation already exists through the web tool session
```

`juice-shop` and `juice-shop-full` are intentionally target adapters, not
solution packs. The broad training plugin is `juice-shop-full`:

```text
available in executable plugins now:
- SPA asset discovery through observed links/scripts
- REST/API endpoint discovery through parsing
- GET/HEAD/OPTIONS
- query probes
- form POST and combo POST
- JSON POST/PUT/PATCH/DELETE through `json-api-actions`
- generic input mutation through `input-mutation-actions`
- file/static asset probing through `file-surface-actions`
- cookie session reuse
- Juice Shop scoreboard reward observer

planned plugin capabilities:
- browser DOM click/fill/navigation
- localStorage/sessionStorage observation
- file upload and binary metadata inspection
```

Run directly with the JSON API capability:

```powershell
python -m apassr_tool.experiment --plugin json-api-actions --base-url http://127.0.0.1:3000 --condition APASSR --episodes 5 --step-limit 80 --reward-observer juice-shop --no-curl
```

For broader Juice Shop learning, prefer the composite plugin:

```powershell
python -m apassr_tool.juice_train --plugin juice-shop-full --base-url http://127.0.0.1:3000 --episodes 50 --step-limit 100 --objective balanced --no-curl --output-dir runs\juice_train_full
```

### How to Add a Plugin

1. Create or reuse domain modules.
   - action generation: maps KK/KV pools to `ActionCandidate`
   - parser: maps tool output to new `(KK, value)` observations
   - reward observer: optional local-only feedback, after execution

2. Implement the plugin class.

```python
from apassr_tool.knowledge import KnowledgeStore, seed_knowledge
from apassr_tool.plugins import PluginMetadata, register_plugin


class MyLocalLabPlugin:
    name = "my-local-lab"
    metadata = PluginMetadata(
        name="my-local-lab",
        description="Controlled local lab adapter.",
        domain="local_lab",
        reward_observers=("none",),
        adapter_modules=("my_actions.py", "my_parser.py"),
        safety_notes=("loopback only", "no human-authored solution stages"),
    )

    def seed(self, base_url: str) -> KnowledgeStore:
        return seed_knowledge(base_url)

    def candidates(self, store: KnowledgeStore):
        return generate_my_candidates(store)

    def parse(self, result):
        return parse_my_result(result)

    def reward_observer(self, name: str, base_url: str):
        if name == "none":
            return None
        raise ValueError(f"unsupported reward observer: {name}")


register_plugin(MyLocalLabPlugin)
```

3. Add the plugin to the registry.
   - For v3, register it in `plugins.py` or import the module from there.
   - For v4, this should become file/package discovery.

4. Add tests.
   - plugin appears in `available_plugins()`
   - plugin exposes useful `PluginMetadata`
   - `seed()` creates only interface knowledge, not answers
   - `candidates()` works from KK/KV values
   - `parse()` extracts observations into KV values
   - DMP can run with `plugin="your-plugin"`

5. Run with the plugin:

```powershell
python -m apassr_tool.experiment --plugin my-local-lab --base-url http://127.0.0.1:PORT --condition APASSR
```

### Plugin Rules

Allowed:

- expose primitive actions
- parse observations into KV candidates
- define local-only reward observers
- seed basic interface knowledge such as base URL, host, port, and default entry paths

Not allowed:

- inject a challenge stage list
- inject known credentials, flags, or solution paths
- rank actions using a human-written writeup
- bypass the APASSR loop by directly solving the target

The line is simple: a plugin may describe the world's interaction grammar, but
the agent must learn how to use that grammar through observation, reward,
Prophecy, Imagination, and repeated execution.

Probe values are not seeded as human-provided hints. `KK_PROBE_VALUE` is
populated from observations such as query strings, JSON fields, discovered user
ids, and other short scalar values returned by the local target. Authenticated
paths are handled the same way: the agent only creates authenticated requests
for paths that have been observed and stored as `KK_AUTH_PATH`.

The first lab uses a small local sandbag web server that emits an observable
dependency chain. The sequence below documents what the target can reveal during
interaction; these values are not injected into the agent as initial knowledge.

```text
GET /
GET /robots.txt        -> discovers /debug
GET /debug             -> discovers user id 7 and /api/users?id=7
GET /api/users?id=7    -> discovers admin username and role
GET /static/app.js     -> discovers password rule
POST /login            -> obtains session cookie
GET observed auth path with cookie -> finds FLAG
```

## Current Tool Support

- `curl` template support when `curl` or `curl.exe` is available.
- Python HTTP fallback for smoke tests when curl is unavailable.
- Optional WSL Kali backend for running fixed `curl` and `nmap` templates through `wsl.exe`.
- `nmap` template support with allowlist checks. Nmap observations update `KK_PORT_STATE` and `KK_SERVICE`.
- Optional `whatweb` fingerprint template. It is marked unavailable when `whatweb` is not installed.

## Safety Rules

- Only loopback hosts are allowed by default: `127.0.0.1`, `localhost`, `::1`.
- Free-form shell commands are not accepted.
- Tool calls use fixed templates.
- Each call has a timeout.
- External URLs are rejected before execution.

## Run

Start the local sandbag server:

```powershell
cd X:\Dev\AASSR\v3
$env:PYTHONPATH='src'
python -m apassr_tool.sandbox_server --host 127.0.0.1 --port 8088
```

In another terminal:

```powershell
cd X:\Dev\AASSR\v3
$env:PYTHONPATH='src'
python -m apassr_tool.experiment --base-url http://127.0.0.1:8088 --condition APASSR
```

Use Kali Linux on WSL as the tool backend:

```powershell
cd X:\Dev\AASSR\v3
$env:PYTHONPATH='src'
python -m apassr_tool.experiment --base-url http://127.0.0.1:8088 --condition APASSR --backend wsl --wsl-distro kali-linux
```

The WSL backend still uses the same allowlist and fixed command templates. It does not enable arbitrary shell execution.

Optional Juice Shop target:

```powershell
cd X:\Dev\AASSR\v3
docker compose -f docker-compose.juice-shop.yml up -d
$env:PYTHONPATH='src'
python -m apassr_tool.experiment --base-url http://127.0.0.1:3000 --condition APASSR --backend wsl --wsl-distro kali-linux --step-limit 80
```

Run a learnable Juice Shop loop without a human task schedule:

```powershell
cd X:\Dev\AASSR\v3
$env:PYTHONPATH='src'
python -m apassr_tool.experiment --base-url http://127.0.0.1:3000 --condition APASSR --episodes 5 --step-limit 40 --reward-observer juice-shop --no-curl
```

Run the same loop with novelty-seeking behavior:

```powershell
python -m apassr_tool.experiment --base-url http://127.0.0.1:3000 --condition APASSR --episodes 5 --step-limit 40 --reward-observer juice-shop --objective novelty --no-curl
```

Run a continuous whole-Juice-Shop learning session:

```powershell
python -m apassr_tool.juice_train --plugin juice-shop-full --base-url http://127.0.0.1:3000 --episodes 50 --step-limit 80 --objective balanced --no-curl --output-dir runs\juice_train_balanced
```

Open the live Tkinter learning monitor:

```powershell
python -m apassr_tool.gui
```

The GUI shows the current action intent, Imagination top-k candidate branches,
action timeline, Knowledge Store, PolicyABC probabilities, Prophecy statistics,
plugin capabilities, and Juice Shop progress when the scoreboard observer is
enabled.

This does not inject a challenge order. Juice Shop's challenge API is used only
as a reward/progress observer after actions execute. The runner keeps PolicyABC,
`TableProphecyModel`, and optional novelty memory alive across episodes, prints
progress and ETA, and writes:

```text
runs\juice_train_balanced\juice_train_episodes.jsonl
runs\juice_train_balanced\juice_train_summary.json
runs\juice_train_balanced\checkpoint_latest.json
```

For trace-heavy runs:

```powershell
python -m apassr_tool.juice_train --base-url http://127.0.0.1:3000 --episodes 20 --step-limit 100 --objective weird --no-curl --include-records --output-dir runs\juice_train_weird
```

For a more aggressive, intentionally inefficient search:

```powershell
python -m apassr_tool.experiment --base-url http://127.0.0.1:3000 --condition APASSR --episodes 5 --step-limit 40 --reward-observer juice-shop --objective weird --no-curl
```

Save full step traces and extract unusual windows:

```powershell
python -m apassr_tool.experiment --base-url http://127.0.0.1:3000 --condition APASSR --episodes 10 --step-limit 80 --reward-observer juice-shop --objective weird --no-curl --include-records --json > weird-run.json
python -m apassr_tool.trace --input weird-run.json --output weird-trace-report.md --window 8
```

In this mode, knowledge still resets each episode, but PolicyABC and the
Prophecy Module persist. The agent learns from knowledge gain, HTTP execution
feedback, and newly solved Juice Shop challenges.

Juice Shop is a larger local target, not the default smoke test. Keep the
target on loopback and do not point the agent at external systems.

Run tests:

```powershell
cd X:\Dev\AASSR\v3
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## Next Upgrade

After this prototype is stable:

1. Keep v3 focused on the web pentesting adapter.
2. Add DVWA as another optional local target.
3. Add more primitive wrappers only when needed: `ffuf`/`gobuster`, `nikto`, `sqlmap`.
4. For v4, extract `actions.py`, `tools.py`, `parser.py`, and `reward.py` behind a plugin interface while keeping the APASSR core stable.
5. Keep allowlists and template-only execution for controlled local experiments.
