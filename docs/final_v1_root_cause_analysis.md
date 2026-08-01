# Final v1 root-cause analysis

This document freezes the diagnosis used to design Paper Protocol v2.  It does
not reinterpret or modify Final v1.  Evidence is labelled so an implementation
fact is not confused with an experimental or causal claim.

## Code-confirmed facts

### Evaluation worlds

- `autonomous_experiment._world_seed` maps training to
  `train_world_seeds`, `evaluation_seen` to `seen_world_seeds`, and zero-shot
  evaluation to `unseen_world_seeds`.
- `configs/paper_autonomy_final_v1.json` contains disjoint sets: train
  `51001..51008`, seen `61001..61004`, and unseen `71001..71008`.
- Consequently Final v1 has no frozen evaluation on the actual training-world
  seeds.  The name `evaluation_seen` means a separately generated world set,
  not a replay of worlds observed in training.

### Policy representation and tie-breaking

- `autonomous_agent.state_key` consists of the numeric state vector and opaque
  facts. `ContextualPolicy` indexes values by `(state_key, action.signature)`.
- `TabularQLearningAgent` uses the same identity key. `DQNAgent` consumes the
  state vector and a SHA-256-derived feature vector of the action signature.
- `ContextualPolicy`, Q-learning, DQN, and the imagination selector resolve
  equal values by lexicographically smallest action signature.  This is a
  deterministic tie-break, so an unseen state can repeatedly choose the same
  token even though its semantic effect is unknown.
- `OpaqueDependencyWorld.__init__` regenerates action signatures, observation
  labels, shuffled state slots, and the viable action independently for every
  world seed.  These identities therefore do not provide a stable causal key.

### Prophecy target and hidden failure

- `OpaqueDependencyWorld.corrupted` is private. `_vector` explicitly omits it,
  so correct and corrupting actions at a stage lead to observationally
  indistinguishable visible states until terminal reward.
- The v1 `ProphecyModel.learn` contract accepts only
  `(state, action, actual_next_state)`. `TabularProphecy` models the next
  `StateSnapshot`; it has no terminal-return or private-viability target.
- `AutonomousLearningAgent.observe` computes `prediction_score` as cosine-like
  similarity between predicted and actual visible next-state vectors.  A high
  value therefore means visible-state accuracy, not reward-relevant accuracy.

### Imagination and creativity

- `AutonomousLearningAgent.select_action` enables imagination from model
  coverage and interval checks, then chooses the highest imagined aggregate.
  It does not gate intervention on return calibration, OOD, or a same-unit
  comparison with policy expected return.
- Depth, branching, and aggregation are passed into `ImaginationTree`, but the
  v1 result alone cannot establish that these parameters changed decisions.
- `MultiSolutionDependencyWorld` defines five terminal operations in advance.
  `synthesize` is explicitly labelled `emergent_combination`; it is not an
  unenumerated composition discovered by the success predicate.
- `paper_runner._run_creativity` builds its reference pool from all non-AASSR
  strategies generated in the same run.  It is therefore capable of covering
  the closed environment's feasible graph set before AASSR novelty is scored.

## Experiment-supported observations

- Final v1 used 30 research seeds and completed all planned rows; see
  `paper_final_review/integrity_report.json`.
- Full AASSR frozen seen and unseen zero-shot success was zero for dependency
  lengths 4, 6, and 8, while training success was non-zero.
- `paper_final_review/anomaly_samples.csv` contains, for example, autonomy seed
  131/world 51001/episode 24 with prediction score `1.0`, success `0`, four real
  transitions, and no runtime error.  Across Final v1 the lightweight audit
  found 224,109 autonomy episodes with prediction score at least 0.8 and
  failure.
- Creativity produced 105,000 successful strategies but zero frozen-novelty
  passes; only 234 canonical causal graphs were observed and the baseline pool
  contained matching graphs.

## Supported diagnosis

The failure is not one isolated evaluation bug.  The phase name concealed the
absence of train-world replay, the environment removed cross-world reusable
causal structure, and the learned transition target omitted reward-relevant
hidden consequences.  These are confirmed design and measurement problems.
Whether a better-aligned environment will make Full AASSR outperform a fair
policy baseline remains an empirical hypothesis.

## Inferences requiring v2 experiments

- Deterministic tie-breaking may amplify repeated failure, but Final v1 does
  not isolate it from representation failure.
- Imagination may have degraded policy selection, but v1 lacks policy-only
  counterfactual logging and calibrated intervention records.
- Closed solution enumeration is a sufficient explanation for novelty zero;
  it does not prove that the agent lacks compositional creativity in an open
  environment.

## Unresolved questions

- Does a restored policy reproduce its training-tail performance on the exact
  training worlds?
- Can a learned relational encoder transfer across token remaps without using
  private causal labels?
- Does an exact return-calibrated transition model make planning useful?
- Can an independently frozen reference leave feasible, useful graph space for
  an agent to discover?
