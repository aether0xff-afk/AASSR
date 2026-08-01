# GPT-2: Actionable Reward and Autonomous Academy

`GPT-2` is based on the updated `main` branch. It carries the GPT-1 design forward but reimplements it against the current hierarchical PolicyABC candidate-generation path.

## Implementation status

Implemented modules:

- `aassr.gpt2_reward`
  - consequence-based reward
  - executable-action unlock measurement
  - lifecycle progress
  - typed error penalties
  - semantic-repeat and two-cycle penalties
  - prediction-error curiosity forced off
  - one policy update per real action
- `aassr.gpt2_curriculum`
  - procedural capability bands
  - learning-progress scheduler
  - no stored solution trajectories
- `aassr.gpt2_experiment`
  - reward-only condition
  - autonomous academy pretraining
  - `model_only` and `full_prior` transfer
  - automatically matched no-academy baseline
  - creativity and policy-override diagnostics

Implemented tests:

- `tests/test_gpt2_reward.py`
- `tests/test_gpt2_curriculum.py`
- `tests/test_gpt2_experiment.py`

CI:

- `.github/workflows/gpt2-v2-tests.yml`

## Conditions

```text
GPT2_REWARD
GPT2_ACADEMY_MODEL
GPT2_ACADEMY_FULL
GPT2_ACADEMY_BASELINE
```

`GPT2_REWARD` uses the current C5 PolicyABC, Prophecy, and lightweight Imagination structure while replacing the learning reward.

`GPT2_ACADEMY_MODEL` retains the learned Prophecy/transition model but resets PolicyABC before target evaluation.

`GPT2_ACADEMY_FULL` retains both the learned model and the academy PolicyABC. It is an ablation for policy-prior interference.

`GPT2_ACADEMY_BASELINE` uses the same model architecture, initialization seed, evaluation policy seed, target worlds, and target-world seeds without academy pretraining.

## Consequence-based reward

The reward contains no key/door/flag-specific importance table.

```text
flag completion
+ log-scaled newly unlocked executable actions
+ resolved Knowledge lifecycle transitions
+ small capped semantic information signal
- typed execution errors
- repeated action with no semantic change
- A-B-A-B movement cycles
```

Candidate unlocks are deduplicated at the executable-action level. HOW labels and `KK_CURRENT_POS` do not create fake new actions.

Prediction-error curiosity is disabled during this phase. The base DMP policy update is intercepted so PolicyABC is not trained first with the legacy reward and then again with the new reward.

## Autonomous academy

The teacher chooses among four capability bands.

| Band | Environment family | Capacity |
| --- | --- | --- |
| foundation | `random_flag` | observation and movement |
| control | `random_wall_flag` | obstacle discovery and recovery |
| composition | `random_key_door` | knowledge-action composition |
| adversarial | `v2_complex` / `locked_bottleneck` | long dependency and transfer |

The first four academy tasks bootstrap one sample from each band. After that, selection uses:

```text
recent positive learning progress
+ proximity to the learnable mastery zone
+ under-sampled-band diversity bonus
+ small stochastic exploration
```

A curriculum task stores only:

```text
band
world kind
procedural seed
difficulty estimate
```

It does not store a solution, action sequence, or successful demonstration.

## Creativity guardrail

Every academy run also evaluates a matched no-academy baseline.

Reported metrics:

- success-rate delta
- successful-trajectory diversity delta
- trajectory-entropy delta
- novel-strategy rate relative to successful academy trajectories
- policy-override rate relative to the academy prior's top WHAT/WHERE axes

Default acceptance criteria:

```text
success delta >= 0.00
trajectory diversity drop <= 0.10
trajectory entropy drop <= 0.10
novel strategy rate >= 0.20
at least one successful target trajectory
```

A higher success rate alone is not enough. If academy transfer collapses strategy diversity, `guardrail.passed` is false.

Structural strategy signatures use action templates and collapse consecutive identical templates. Exact coordinates and HOW-only variants therefore do not create artificial novelty.

## Commands

Reward smoke run:

```powershell
cd v2
$env:PYTHONPATH='src'
python -m aassr.gpt2_experiment `
  --mode reward `
  --world v2_complex `
  --episodes 10 `
  --seeds 3 `
  --step-limit 120 `
  --output-dir runs/gpt2/reward_smoke
```

Creativity-preserving model-only academy:

```powershell
python -m aassr.gpt2_experiment `
  --mode academy `
  --world v2_complex `
  --pretrain-episodes 100 `
  --episodes 20 `
  --seeds 3 `
  --step-limit 120 `
  --prophecy-kind sequence `
  --transfer-mode model_only `
  --output-dir runs/gpt2/academy_model_smoke
```

Full-prior ablation:

```powershell
python -m aassr.gpt2_experiment `
  --mode academy `
  --world v2_complex `
  --pretrain-episodes 100 `
  --episodes 20 `
  --seeds 3 `
  --step-limit 120 `
  --prophecy-kind sequence `
  --transfer-mode full_prior `
  --output-dir runs/gpt2/academy_full_smoke
```

Outputs include standard step/episode/summary CSV files plus:

```text
gpt2_reward_diagnostics.csv
curriculum_history.csv
academy_report.json
baseline/
```

## Required evaluation order

```text
1. C5 vs GPT2_REWARD with identical seeds.
2. Reward-component ablation.
3. GPT2_ACADEMY_BASELINE vs GPT2_ACADEMY_MODEL.
4. GPT2_ACADEMY_MODEL vs GPT2_ACADEMY_FULL.
5. Accept academy transfer only when the creativity guardrail passes.
6. Add KK-update embedding only after reward and academy alignment are confirmed.
```
