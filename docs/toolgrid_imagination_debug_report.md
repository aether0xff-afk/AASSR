# ToolGrid Imagination v2 root-cause debugging

## Status

This report records a controlled diagnosis of the negative Imagination v2 result
from the ToolGrid factorial pilot. It does **not** replace the full factorial
experiment. The confirmed scope is:

- seed: `7`
- grid size: `3 x 3`
- action counts: `8` and `12` (`4` and `8` semantic tools)
- real transition budget: `5,000`
- unseen evaluation maps: `100` per cell
- no artificial episode step limit

Final diagnostic workflow:

- GitHub Actions run: `31072230333`
- commit: `d8a6e2efee0b72cebe7ea29b676077ffa3607b98`

## Controlled diagnostic design

The original factorial condition allowed Imagination to alter actions during
training. Its final policy therefore differed from the policy-only condition,
which confounded planner quality with a changed learning trajectory.

The debugging harness removes this confound:

1. Train one hybrid checkpoint while disabling Imagination interventions.
2. Continue learning Prophecy and the GRU branch critic from real transitions.
3. Evaluate the identical checkpoint twice on the same unseen map set:
   - learned policy only;
   - learned policy plus Imagination.
4. Clone the real environment only for post-hoc diagnostics.
5. For every candidate action, record:
   - whether the real successor remains solvable;
   - Prophecy error and predicted terminal class;
   - critic value on the predicted successor;
   - critic value on the real successor;
   - whether an intervention was beneficial or harmful.

The environment oracle is never exposed to the agent.

## Initial controlled result

| Tools | Policy only | Original Imagination | Delta | Harmful interventions | Tool-choice interventions |
|---:|---:|---:|---:|---:|---:|
| 4 | 52% | 41% | -11 pp | 25 | 0 |
| 8 | 29% | 19% | -10 pp | 15 | 0 |

Every action-changing intervention occurred during navigation. At the actual
semantic tool choice, Imagination never changed the policy action.

The critic was not the first-order failure. When scored on the **real** next
states, its best action was viable on approximately 96% of decisions in the
four-tool cell and 91% in the eight-tool cell. The failure occurred before or
around the critic: sparse calibration, world-model representation, and the
intervention gate.

## Root causes

### 1. Sparse-action calibration cache bug

The original cache key used the action-specific sample count divided by the
refresh stride. A zero calibration value could be cached before the action
reached `minimum_count=8`, then remain unchanged until 32 samples existed.
Sparse tool actions were therefore stuck at zero confidence after they had
become eligible for calibration.

Fix:

- never cache the pre-ready state;
- begin cache buckets at the minimum-count boundary.

### 2. Success and failure were collapsed into one terminal class

A correct ToolGrid tool and a wrong tool both terminate the episode. The
original calibration primarily checked whether the successor had available
actions, so terminal success and terminal failure appeared structurally equal.

Fix:

- calibrate three outcome classes separately:
  - nonterminal;
  - terminal success;
  - terminal failure.

### 3. Tool transitions were sparse in neural replay

Navigation transitions dominated the real stream. The one terminal tool choice
per episode was underrepresented, especially as the tool vocabulary grew.

Fix:

- repeat only train-split tool transitions in Prophecy replay;
- do not repeat or modify frozen holdout examples;
- keep the real-transition accounting unchanged.

This is a diagnostic balancing mechanism, not additional environment data.

### 4. Hashed action identity did not scale to the fixed tool vocabulary

The generic neural Prophecy encoded action signatures with signed hash
features. With eight semantic tools, the learned successor ranking remained
near random even after balancing replay.

Fix:

- use one-hot identity for the fixed action vocabulary.

This exposes only which action was selected. It does not encode what an action
does or which action is correct.

### 5. Required-tool identity was incorrectly represented as an ordinal scalar

The required tool was represented as `0, 1/7, ..., 1`. This imposes an artificial
ordinal geometry and forces the model to learn equality between an ordinal
state scalar and a categorical action identity.

Fix:

- use a categorical one-hot required-tool representation inside Prophecy;
- decode predictions back to the frozen environment observation schema;
- leave the policy and critic observations unchanged.

### 6. Imagination was invoked at the wrong decisions

The original global coverage gate allowed repeated navigation interventions but
blocked the semantic branch where actions had different terminal consequences.
After the representation fixes, enabling Imagination everywhere still produced
navigation harm.

Fix:

- use a model-derived terminal-choice gate;
- invoke Imagination only when every currently available action is predicted to
  terminate the episode;
- otherwise execute the learned policy directly.

The final gate does not inspect action names, ToolGrid phase, the correct tool,
or the environment oracle.

## Final same-checkpoint result

| Tools | Policy only | Terminal-choice Imagination | Delta | Improved maps | Worsened maps | Beneficial interventions | Harmful interventions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 46% | **95%** | **+49 pp** | 49 | 0 | 49 | 0 |
| 8 | 29% | **79%** | **+50 pp** | 50 | 0 | 50 | 0 |

All action-changing interventions occurred at terminal semantic choices. No
navigation action was overridden, and no intervention made a viable policy
choice fail.

## Final component diagnostics

| Metric | 4 tools | 8 tools |
|---|---:|---:|
| Prophecy vector MAE | 0.00397 | 0.00250 |
| Prophecy terminal accuracy | 99.59% | 99.60% |
| Predicted available-action accuracy | 98.69% | 98.20% |
| Predicted-state critic best-action viability | 95.35% | 91.29% |
| Real-state critic best-action viability | 95.35% | 92.20% |
| Predicted/real critic top-rank agreement | 91.59% | 95.10% |

## Interpretation

The negative pilot result did not show that imagination is intrinsically
useless. It showed a specific failure mode:

> Imagination was blocked where the world model could compare irreversible
> semantic outcomes, while being allowed to overwrite competent navigation
> decisions where deep rollout added little value.

After correcting calibration, categorical representation, replay balance, and
intervention placement, Imagination converted roughly half of the failed unseen
maps into successes without causing a single regression in these controlled
cells.

This supports a narrower claim than “AASSR now beats DQN generally”:

- a learned world model and critic can add substantial value at irreversible
  semantic branches;
- global always-on planning is unsafe and computationally wasteful;
- the decision to imagine must itself be learned or derived from predicted
  consequence structure.

## Remaining validation

Before changing the main conclusion, rerun the frozen factorial protocol with:

1. multiple seeds;
2. `3 x 3`, `5 x 5`, and `7 x 7` maps;
3. DQN, matched policy-only, and corrected Imagination;
4. the same real-transition budget and unseen maps;
5. explicit reporting of intervention benefit/harm and imagined-node cost;
6. a matched ablation for each fix:
   - original calibration;
   - hashed actions;
   - ordinal tool state;
   - unbalanced replay;
   - global planning gate.

The current result is strong root-cause evidence, but it is still a seed-7,
`3 x 3` controlled debugging result rather than the final factorial estimate.
