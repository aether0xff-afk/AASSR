# AASSR 0.4 canonical integrated architecture

Date: 2026-08-08

Architecture generation: **AASSR v2**  
Package/research milestone: **0.4.0**

## Why this integration exists

During the GridPush and pentest diagnosis work, AASSR was intentionally decomposed into narrower experimental paths. The repository retained GOAL, Skill, Knowledge, Prophecy, Imagination, information-value, and autonomous-policy implementations, but no single current entry point exercised the whole closed loop. The pentest 2x2 narrowed the path further to DQN + curriculum + ASEQ so the self-loop mechanism could be isolated fairly.

AASSR 0.4 restores one canonical agent object without deleting the diagnostic runners or retroactively changing their evidence.

Canonical constructor:

```python
from aassr_v2 import build_full_aassr_core

agent = build_full_aassr_core(prophecy, seed=7)
```

Audited pentest constructor:

```python
from aassr_v2 import build_pentest_aassr_core

agent = build_pentest_aassr_core(prophecy, seed=7)
```

The pentest constructor refuses snapshots that are not produced under `response_causal_observation_v3`.

## Closed loop

```text
observable StateSnapshot
        |
        v
shared semantic-state contract
        |
        +----> ASEQ empirical (S,A,S') self-loop memory
        |
        v
SkillLibrary augments available actions
        |
        v
SemanticContextualPolicy + Prophecy + Imagination
        |
        v
one selected primitive action or learned Skill
        |
        v
real environment transition(s)
        |
        +----> KnowledgeStore with trace provenance
        +----> Prophecy + transition-effect learning
        +----> holdout/model-learning diagnostics
        +----> OnlineFeatureMemory
        +----> InformationValuePredictor
        +----> GOAL completion
        +----> repeated successful ASeq -> Skill promotion
        |
        v
terminal external return
        |
        v
delayed credit -> semantic Policy update
        |
        v
next episode / next real state
```

## One semantic identity contract

AASSR 0.4 does not let each module invent its own definition of state identity.

`SemanticContextualPolicy`, `SemanticSelfLoopASEQ`, and the active `ImaginationTree` instance receive the same semantic-key function.

For generic environments the backward-compatible raw key is:

```text
rounded state vector + sorted observable facts
```

For the audited pentest environment the canonical key is the existing observation-derived `semantic_fingerprint()`, which intentionally ignores administrative request/audit/session noise while retaining problem-solving semantics.

This means the real controller and imagined branches agree on what counts as returning to the same semantic state.

## ASEQ rule

ASEQ remains the empirical transition tuple `(S,A,S')`.

An action is guarded only when:

1. the same semantic `(S,A)` has been observed at least twice,
2. every observed outcome for that pair is the same semantic state `S`,
3. therefore the observed transition is a repeated `S -> A -> S` self-loop.

Repeated state-changing transitions are not suppressed. If every available action would be guarded, the unfiltered action set is restored.

ASEQ memory resets at episode boundaries; learned Policy/Prophecy/Knowledge/Skill state may persist according to the agent configuration.

## Learning ownership

Previous components could each learn from the same real transition if naively nested. AASSR 0.4 assigns ownership explicitly:

- `AdvancedTransitionEvaluator` owns Prophecy/effect learning for real primitive transitions.
- The historical `AutonomousLearningAgent.observe()` learning path is disabled inside the integrated agent.
- terminal credit is assigned once per real transition.
- `SemanticContextualPolicy` is updated by the integrated delayed-credit path.
- learned Skill macro-actions receive terminal outcome credit without a synthetic environment reward.

The regression suite checks that one real transition produces one base-Prophecy learning call.

## Reward versus internal value

The integration layer never rewrites the environment's external reward.

Pentest experiments can continue to use:

```text
+1 proof
-1 true lockout
 0 otherwise
```

Information value is an internal learned signal. `InformationValuePredictor` is trained from delayed real episode outcomes and may contribute to Policy credit through a separately logged weight. This is not an intermediate environment reward or a correct-action label.

GOAL scoring is also internal planning value. The default integrated agent contains only a terminal `goal_progress == 1` external goal unless the caller explicitly supplies additional goals.

## Knowledge and feature memory

Every real primitive transition records added/removed facts in `KnowledgeStore` with trace provenance and the signatures of actions unlocked by that observation.

New facts are also represented in `OnlineFeatureMemory`. The memory remains available to plugin/slot candidate resolution; the integrated core does not silently delete action candidates based on a hand-written information rule.

## GOAL and Skill

The default GOAL is terminal success. Domain code may add explicit state-gap or knowledge goals without supplying an action demonstration.

When a real trajectory newly completes a GOAL, `SkillLibrary` observes the real ASeq fragment. A repeated successful sequence is promoted according to the existing promotion rule. The learned Skill then appears as one candidate action; `SkillAwareProphecy` rolls its primitive transitions inside Imagination while reality still executes the primitives one by one.

## Pentest boundary

`build_pentest_aassr_core()` binds the integrated agent to:

- `response_causal_observation_v3`,
- the audited observation-derived `semantic_fingerprint()`,
- the existing safe in-process pentest worlds.

It intentionally does not re-enable the historical privileged consequence gate that depended on hidden audit/session-distance channels.

The transfer curriculum and the separately accepted HTTP benchmark remain different objects:

- transfer curriculum: development/training and factor-boundary diagnosis,
- accepted HTTP benchmark: evaluation environment.

The name `full_benchmark` in the historical transfer-stage tuple remains a compatibility name; it must not be interpreted as byte-for-byte equivalence to the accepted HTTP benchmark.

## What 0.4.0 does and does not claim

0.4.0 now means two things that are both frozen in the repository:

1. the audited pentest/ASEQ development evidence that established semantic self-loop suppression as a useful mechanism, and
2. restoration of one canonical full AASSR closed loop using the audited semantic/observation lessons.

It does **not** claim that the newly reintegrated full AASSR already beats DQN or solves L4+ transfer. Full-system performance must be measured in a new experiment after this integration passes its regression suite. The completed 2x2 evidence remains a mechanism-selection result and is not retroactively reinterpreted as a full-AASSR result.
