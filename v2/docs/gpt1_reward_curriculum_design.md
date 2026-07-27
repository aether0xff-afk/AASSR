# GPT-1: Actionable Reward and Autonomous Academy

This branch keeps the published C0-C5/APASSR_FULL conditions unchanged. New work is isolated in three modules:

- `aassr.actionable`: consequence-based reward and `ActionableGridWorldDMP`
- `aassr.curriculum`: learning-progress scheduler with no stored solution trajectories
- `aassr.gpt1_experiment`: C6 reward and C7 academy runners

## Design goals

1. Minimize hand-authored task knowledge.
2. Reward consequences that are useful across tasks rather than specific KK names.
3. Teach transition, binding, lifecycle, and recovery fundamentals without teaching a fixed solution order.
4. Preserve creativity by supporting model-only transfer: academy Prophecy experience is retained while PolicyABC is reset before target evaluation.
5. Measure successful-trajectory diversity and novelty against academy trajectories.

## C6 reward

`C6_REWARD` starts from C5 and disables prediction-error curiosity. Its intrinsic signal is:

```text
unique executable actions unlocked
+ completed knowledge lifecycle transitions
+ small capped semantic information signal
- typed execution errors
- repeated action with no semantic change
- A-B-A-B movement cycles
```

No key/door/flag-specific reward weights are used. Candidate unlocks are deduplicated at the executable-action level, ignoring HOW labels and current-position bindings.

## C7 academy

The academy scheduler chooses among four procedural bands:

| Band | Environment family | Intended capacity |
| --- | --- | --- |
| foundation | random flag | observation and movement |
| control | random walls + flag | error discovery and recovery |
| composition | random key/door | knowledge-action composition |
| adversarial | v2 complex / locked bottleneck | transfer and long dependency |

The teacher uses recent learning progress, learnable-zone proximity, and exploration. It does not store or provide action sequences.

### Transfer modes

- `model_only`: retain learned Prophecy statistics, reset PolicyABC before evaluation. This is the primary creativity-preserving condition.
- `full_prior`: retain both Prophecy and PolicyABC. This is an ablation for measuring policy-prior interference.

## Creativity guardrails

Academy reports include:

- successful trajectory count
- unique successful trajectory count
- successful trajectory diversity
- novel strategy rate relative to successful academy sequences
- normalized trajectory entropy

The academy should only be accepted when target success improves without a material collapse in trajectory diversity or novel-strategy rate.

## Commands

Reward-only smoke test:

```powershell
cd v2
$env:PYTHONPATH='src'
python -m aassr.gpt1_experiment --mode reward --world v2_complex --episodes 10 --seeds 3 --step-limit 120 --output-dir runs/gpt1/c6_smoke
```

Creativity-preserving academy evaluation:

```powershell
python -m aassr.gpt1_experiment --mode academy --world v2_complex --pretrain-episodes 100 --episodes 20 --seeds 3 --step-limit 120 --transfer-mode model_only --output-dir runs/gpt1/c7_model_smoke
```

Policy-prior ablation:

```powershell
python -m aassr.gpt1_experiment --mode academy --world v2_complex --pretrain-episodes 100 --episodes 20 --seeds 3 --step-limit 120 --transfer-mode full_prior --output-dir runs/gpt1/c7_full_smoke
```

## Required comparison

Run the following on identical worlds and seeds:

```text
C5
C6_REWARD
C7_ACADEMY_MODEL
C7_ACADEMY_FULL
```

The next phase, KK-update embedding, must not begin until C6/C7 show that the reward and academy objectives align with actual success and do not suppress strategy diversity.
