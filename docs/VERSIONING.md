# AASSR versioning

AASSR uses separate version namespaces so architecture generations, software releases, and experiment contracts do not get conflated.

## 1. Architecture generation

`AASSR v2` is the current architecture generation. It describes the broad research design and remains unchanged by ordinary software releases.

A change to `AASSR v3` is reserved for a deliberate architecture-generation break, not for an observation-contract revision, benchmark change, integration cleanup, or one mechanism improvement.

## 2. Research/software release

The Python package and repository milestones use semantic-style `0.x.y` versions.

- **minor** (`0.x.0`): a new reproducible research milestone, mechanism, benchmark contract, evaluation boundary, or canonical architecture integration.
- **patch** (`0.x.y`): implementation corrections that preserve the research claim and experimental contract.

While the project remains research software, `0.x` releases may still contain intentionally unresolved research limitations.

## 3. Experiment/protocol versions

Experiments and observation contracts keep their own explicit identifiers, for example:

- `training-mechanism-2x2-causal-v1`
- `response_causal_observation_v3`

These identifiers are not package versions and must not be used to infer an AASSR architecture generation.

## Release rule

A research milestone is prepared on its development branch by:

1. freezing the protocol and implementation,
2. running the predeclared development experiment when the release contains an empirical claim,
3. recording the evidence SHA/run and known limitations,
4. integrating the canonical runtime when the milestone changes which entry point represents AASSR,
5. bumping the package version and writing release notes.

The Git tag is created only on the final frozen merge commit. Final blinded evaluation seeds are derived only after the separately predeclared methodology-freeze rule is satisfied.

## Current milestone

`0.4.0` is the **audited semantic-ASEQ + integrated closed-loop milestone**.

Frozen empirical boundary:

- response-causal observation contract v3,
- repeated empirical semantic `S -> A -> S` suppression using ASEQ `(S,A,S')`,
- no suppression of state-changing repeats,
- all-guarded fallback preserving agent freedom,
- explicit corrected TD episode boundary in the corrected learning condition,
- predeclared 2x2 development comparison over research seeds 7, 42, and 100,
- no guided/oracle/correct-action learning,
- final blinded evaluation not consumed.

Canonical runtime boundary:

- `build_full_aassr_core()` is the generic integrated AASSR 0.4 constructor,
- `build_pentest_aassr_core()` binds the same closed loop to `response_causal_observation_v3`,
- Policy, ASEQ, and Imagination cycle detection can share one semantic-state identity function,
- Knowledge Store, Prophecy/effect learning, feature memory, information-value learning, GOAL completion, delayed Policy credit, and Skill promotion are routed through one real-transition loop,
- the integration prevents the same real transition from being learned twice through the old autonomous and evaluator paths.

The milestone does **not** claim that high-level transfer is solved or that the newly reintegrated full AASSR already outperforms DQN. The completed 2x2 shows that the semantic self-loop/stall bottleneck is substantially reduced; full-system transfer performance must be measured separately after integration.
