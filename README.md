# AASSR v2 — 0.4.0

AASSR is a research architecture for sparse-reward autonomous agents that separates **Policy**, **Prophecy**, and **Imagination**, while keeping explicit transition knowledge and intervention traces.

Package/research milestone: **0.4.0**  
Architecture generation: **AASSR v2**

> 0.4.0 restores one canonical full-agent closed loop and incorporates the audited semantic ASEQ / response-causal observation work. It does **not** claim that high-level transfer is solved or that the newly integrated full AASSR already outperforms DQN.

## Canonical 0.4 architecture

```text
observable StateSnapshot
        |
        v
shared semantic-state contract
        |
        +----> empirical ASEQ (S,A,S') memory
        |
        v
Policy + learned Skill candidates
        |
        v
Prophecy + transition-effect composition + Imagination
        |
        v
real primitive transition(s)
        |
        +----> Knowledge Store with trace provenance
        +----> Prophecy/effect learning
        +----> feature memory
        +----> information-value learning
        +----> GOAL completion
        +----> repeated successful ASeq -> Skill promotion
        |
        v
terminal external return
        |
        v
delayed semantic-Policy credit
        |
        v
next real state / next episode
```

The integration is implemented by `IntegratedAASSRAgent`.

```python
from aassr_v2 import build_full_aassr_core

agent = build_full_aassr_core(prophecy, seed=7)
```

For the audited in-process pentest environment:

```python
from aassr_v2 import build_pentest_aassr_core

agent = build_pentest_aassr_core(prophecy, seed=7)
```

`build_pentest_aassr_core()` requires the current `response_causal_observation_v3` contract and rejects older privileged snapshots.

Detailed design: [`docs/aassr_v040_architecture.md`](docs/aassr_v040_architecture.md).

## What is integrated in 0.4.0

### Shared semantic state

Policy lookup, ASEQ self-loop memory, and Imagination repeated-state detection can use the **same state-equivalence function**.

For generic environments the default remains a backward-compatible vector + observable-facts key. The audited pentest factory instead uses the observation-derived `semantic_fingerprint()` so administrative request/audit/session noise does not create fake problem-solving states.

### ASEQ

ASEQ is the empirical transition tuple `(S, A, S')`.

The 0.4 guard is intentionally narrow:

- observe real transitions first,
- guard an action only after the same semantic `S -> A -> S` self-loop has occurred at least twice,
- never suppress a repeated state-changing `S -> A -> S'` merely because it repeats,
- if the same `(S,A)` has produced multiple semantic outcomes, do not classify it as an exact self-loop,
- if every available action would be guarded, restore the raw action set.

ASEQ therefore prevents the measured no-progress loop without turning repetition in general into a forbidden behavior.

### Policy

`SemanticContextualPolicy` learns state-conditioned action values using the shared semantic state contract. The integrated runtime owns delayed Policy credit so a real transition is not learned independently through multiple historical loops.

### Prophecy

The integrated agent accepts existing Prophecy implementations. It preserves:

- real-transition learning,
- optional recurrent prediction,
- transition-effect composition,
- holdout evaluation,
- optional Knowledge-aware `predict_with_context()` behavior.

The planner-facing Prophecy is bound to the live `KnowledgeStore`, so explicit learned knowledge is available during Imagination instead of being merely logged after execution.

### Imagination

AASSR retains counterfactual multi-step planning. Branches use Prophecy predictions, branch-local memory, confidence pruning, beam pruning, and configurable aggregation.

In 0.4 the planner's repeated-state identity is aligned with the same semantic contract used by real ASEQ and Policy lookup.

The default integrated scorer contains only the terminal external GOAL. A learned critic can be supplied explicitly; audited pentest does not reintroduce hidden progress or privileged resource-distance hints.

### Knowledge Store

Real added/removed facts are stored with trace provenance and unlocked-action signatures.

Generic AASSR can preserve this knowledge across episodes. The pentest constructor makes Knowledge episode-local because opaque route/profile/object identifiers are regenerated across scenario seeds; model parameters and learned general mechanisms still persist normally.

### Feature memory

`OnlineFeatureMemory` records observed information and identifier components and learns action-slot usefulness from delayed real outcome credit. It does not inject a correct value or delete candidates through a hand-written solution rule.

### GOAL

The default integrated GOAL is terminal success (`goal_progress == 1`). Callers may provide additional state-gap or knowledge goals without supplying an action demonstration.

GOAL score is an internal planning value, not an environment reward.

### Skill

When a real trajectory newly completes a GOAL, the existing `SkillLibrary` observes the real ASeq fragment. Repeated successful sequences may be promoted to a Skill.

A Skill appears as one candidate to Policy/Imagination, but reality still executes its primitive actions one by one. A failed primitive lowers Skill reliability and returns control to primitive behavior.

### Information value and delayed credit

`AdvancedTransitionEvaluator` separates:

- Knowledge-context effects,
- Prophecy parameter learning,
- holdout prediction gain,
- unlocked-action value,
- repetition/error signals,
- final delayed outcome credit.

The integrated runtime uses learned information value only as an **internal Policy-credit signal**. It never rewrites the environment's external reward.

## Learning ownership

A naive integration could train the same transition twice: once through the historical autonomous loop and once through `AdvancedTransitionEvaluator`.

0.4.0 explicitly prevents that.

```text
real primitive transition
        |
        v
AdvancedTransitionEvaluator
        |
        +--> Knowledge update
        +--> Prophecy/effect update
        +--> information-value features
        |
        v
episode terminal return
        |
        v
delayed credit
        |
        v
SemanticContextualPolicy
```

Inside `IntegratedAASSRAgent`, the older autonomous Policy/Prophecy update path is disabled. Regression tests verify that one real transition produces one base-Prophecy learning call.

## Audited pentest environment

AASSR includes a **safe in-process HTTP-shaped assessment environment**. It does not open real network sockets or execute shell commands.

The audited observation contract is `response_causal_observation_v3`.

It removes or masks:

- hidden curriculum level,
- hidden workflow depth,
- exact session countdown,
- hidden audit/lockout distance,
- hidden rate-limit distance,
- transient novelty hints,
- duplicate own/target role randomization,
- out-of-band next-step candidate unlocks.

Candidate routes/profiles/objects are exposed through simulated response evidence rather than through hidden world truth.

The transfer curriculum and the separately accepted HTTP benchmark are intentionally distinct:

```text
Transfer curriculum
= training / mechanism diagnosis / factor boundary study

Accepted HTTP benchmark
= evaluation environment
```

The historical transfer-stage name `full_benchmark` is retained for compatibility but is **not** byte-for-byte equivalence to the accepted HTTP benchmark.

## 0.4.0 ASEQ development evidence

The frozen predeclared 2x2 used three research seeds (`7`, `42`, `100`) and `10,000` real training transitions per cell.

| condition | success | stalled |
|---|---:|---:|
| `original` | 16/216 | 162/216 |
| `original_plus_aseq` | 45/216 | 9/216 |
| `learning_fix` | 7/216 | 166/216 |
| `learning_fix_plus_aseq` | 31/216 | 12/216 |

Across both learning backgrounds:

- without ASEQ: `23/432 = 5.3%` success, `328/432 = 75.9%` stalled,
- with ASEQ: `76/432 = 17.6%` success, `21/432 = 4.9%` stalled.

This supports the narrow semantic self-loop mechanism. It does **not** establish full-AASSR performance, and it does not justify suppressing arbitrary repeated transitions.

The development diagnostic remained at zero success from L4 upward, so high-level transfer is still an open research problem.

Release evidence and interpretation: [`docs/releases/v0.4.0.md`](docs/releases/v0.4.0.md).

## Environment limitations that remain explicit

The current transfer curriculum is useful for factor diagnosis, but not every level should be interpreted as a realistic dependency graph.

In particular, the current workflow-depth level repeats one workflow action while progress changes. It is a repetition/depth stressor, not yet a chain of distinct prerequisites such as `A -> B -> C -> D`.

That distinction is intentionally documented rather than hidden behind the word “complexity”.

## Research direction after 0.4.0

The next full-system work is **not another ASEQ tuning pass**. The main targets are:

1. retrain the integrated full AASSR against the audited observation contract,
2. revalidate Prophecy one-step and multi-step structural accuracy,
3. use learned/calibrated branch value rather than hidden handcrafted progress,
4. test whether Knowledge/effect reuse improves transfer across opaque scenarios,
5. add a true multi-prerequisite dependency environment,
6. compare integrated Policy-only vs Prophecy+Imagination from the same learned checkpoint,
7. only after methodology freeze, consume the separately blinded final evaluation seeds.

Environment Familiarization / Solve separation remains a research direction, not a 0.4 performance claim.

## Other environments and adapters

The repository also retains:

- `SandboxEnv` for generic observe/break/place/combine behavior,
- Grid/Escape research environments,
- dry-run Minecraft control interfaces,
- allowlisted authorized-assessment interfaces,
- counterexample worlds for randomness, irrelevant information, opaque names, and long dependencies.

Historical runners remain available for reproducing prior experiments. They are not all canonical 0.4 full-agent entry points.

## Reproducibility and paper runners

Paper-oriented P0–P5 reproduction remains documented separately:

- [`docs/paper_experiment_quickstart.md`](docs/paper_experiment_quickstart.md)
- [`docs/paper_protocol_implementation_status.md`](docs/paper_protocol_implementation_status.md)

The 0.4 integration does not rewrite frozen historical experiment evidence.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Compile and test:

```bash
python -m compileall -q src tests scripts
pytest -q
```

Focused 0.4 integration regressions:

```bash
pytest -q \
  tests/test_v040_integration.py \
  tests/test_roadmap_completion.py \
  tests/test_pentest_curriculum_causal.py \
  tests/test_pentest_training_mechanism_main.py
```

The dedicated workflow is:

```text
.github/workflows/aassr-v040-integration.yml
```

## Versioning

AASSR keeps architecture generation and package/research milestones separate.

- architecture generation: **AASSR v2**
- package/research milestone: **0.4.0**
- observation contract: **response_causal_observation_v3**
- ASEQ development experiment: **training-mechanism-2x2-causal-v1**

See [`docs/VERSIONING.md`](docs/VERSIONING.md).

The `v0.4.0` tag belongs on the final frozen merge commit, not on the earlier experiment-launch commit. Final blinded evaluation remains unconsumed until the predeclared methodology-freeze procedure is satisfied.
