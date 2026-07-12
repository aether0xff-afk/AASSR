# Human-Hint Hygiene Notes

This note records the boundary between allowed interface knowledge and
human-provided solution hints in the v3 local tool prototype.

## Allowed Initial Knowledge

The initial seed is limited to the interaction interface:

- `KK_BASE_URL`
- `KK_HOST`
- `KK_PORT`
- root path `/`
- conventional metadata path `/robots.txt`

The seed does not include probe values, credentials, flags, endpoint-specific
parameters, authenticated paths, or challenge-specific solution values.

## Removed Human-Like Shortcuts

- `KK_PROBE_VALUE` no longer receives seeded values such as generic test inputs.
- Authenticated requests are not generated for a default `/admin` path.
- JavaScript parameter extraction no longer depends on a hand-picked list such
  as email, password, id, or search. It extracts bounded object keys observed in
  target responses.
- Unused candidate priority tiers were removed from the action generator.

## Current Binding Rule

Candidate actions are still generated from KK/KV binding, but parameter and
probe values are ranked by source dependency. A value observed from
`GET /api/example` is preferred when generating candidates for `/api/example`.
This is not a target-specific solution rule; it is the general APASSR dependency
assumption that knowledge produced by an action is often relevant to later
actions over the same object.

## Current Juice Shop Hygiene Check

A 40-step local Juice Shop run produced no seeded probe values:

```text
seed_probe_values = []
```

Example candidate after dependency-aware binding:

```text
POST_PROBE /api/Feedbacks UserId=1
PROBE /api/Feedbacks?UserId=1
```

Both `UserId` and `1` came from observed Juice Shop responses, not from initial
knowledge.

## Remaining Mismatch

The small sandbag server intentionally emits a simple observable dependency
chain so smoke tests can verify the closed loop. That target is for regression
testing only. Juice Shop and future local CTF targets should be used for
stronger experiments because they reduce the risk of overfitting to the sandbag
chain.
