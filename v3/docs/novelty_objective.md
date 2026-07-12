# Novelty-Seeking APASSR Objective

The v3 local web prototype can run with objectives that intentionally favor
non-standard exploration. This is useful when the research goal is not to copy
known writeups, but to see whether APASSR can discover different action chains
from its own observations.

Novelty is an optional objective layered on top of the APASSR core. The main
framework remains KK/KV Knowledge Store, PolicyABC, Prophecy Module,
Imagination Cycle, and DMP. Novelty should not be described as the main method.

## Design Rule

The agent is not given a challenge stage list or canonical solution path.
Novelty is computed only from the agent's own experience:

- rare candidate signatures
- rare action-to-action chains
- rare response transitions

This keeps the objective different from a human-authored Juice Shop guide. It
does not say which challenge to solve next. It only rewards behavior that is
unusual relative to what APASSR has already tried.

## Objectives

`balanced`

- No novelty reward.
- Knowledge gain and solved challenge deltas drive learning.
- This tends to prefer high-yield `GET` exploration early.

`novelty`

- Adds moderate novelty reward and novelty candidate scoring.
- Reduces raw knowledge-gain reward so static/API scraping is less dominant.

`weird`

- Adds stronger novelty reward and candidate scoring.
- Strongly reduces raw knowledge-gain reward.
- Intended for inefficient but diverse exploration.
- Uses broad wild recombination: observed parameters and values are allowed to
  cross their original endpoint/source contexts. This intentionally preserves
  strange combinations such as using UI keys against API endpoints.

## Why Policy Reward Is Separated

Novelty bonus is included in total reward and stored in the Prophecy Module's
current tabular experience statistics, but
PolicyABC updates use `policy_reward`, which excludes novelty bonus. Without
this separation, a large number of one-off `GET` paths can receive novelty
bonus and push the WHAT table toward `HTTP_GET` again. Keeping novelty out of
PolicyABC prevents the policy table from collapsing into "new GET path" as the
main strategy.

## Latest Short Check

Command:

```powershell
python -m apassr_tool.experiment --base-url http://127.0.0.1:3000 --condition APASSR --episodes 5 --step-limit 40 --reward-observer juice-shop --objective weird --no-curl --json
```

Observed after separating novelty bonus from PolicyABC updates:

```text
HTTP_GET           0.199
HTTP_METADATA      0.178
QUERY_PROBE        0.139
PORT_SCAN          0.136
AUTHENTICATED_GET  0.131
FORM_POST          0.115
WEB_FINGERPRINT    0.102
```

This is much more diverse than the earlier collapsed run where `HTTP_GET`
dominated. The run did not solve a Juice Shop challenge yet, but it produced
172 distinct action signatures and 200 distinct action chains in 200 steps.

## Interpretation

This mode is not optimized for speed. It is meant to produce alternative traces
that can later be inspected for unusual paths, unexpected API behavior, or
non-canonical challenge solutions.

The remaining hard limits are safety and runtime limits, not semantic guidance:
the target must stay on the local allowlist, tool calls must use fixed
templates, and long runs may need to be split into smaller batches because wild
recombination can produce a very large candidate space.
