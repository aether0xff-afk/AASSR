# APASSR_FULL_CAL Design Note

`APASSR_FULL_CAL` is a calibrated comparison condition for the predicted-state
APASSR rollout. It is intentionally separate from `APASSR_FULL` so the original
full-structure diagnostic condition remains reproducible.

## Motivation

The first `APASSR_FULL` diagnostic runs showed that the full structure was
active: it generated many virtual Knowledge Store transitions and many future
candidate actions. The main weakness was not inactivity, but over-crediting
future branches whose imagined candidates did not reliably become the next real
action.

`APASSR_FULL_CAL` therefore adds structural calibration without tuning the task:

- no reward change,
- no world-layout change,
- no rollout depth or branching change,
- no hidden-map access,
- no change to C0-C5 or baseline conditions.

## Implemented Changes

### Candidate Signature Deduplication

Future candidate identity uses a canonical signature:

```text
template, WHAT, HOW, WHERE, non-current-position bindings
```

Placeholder values are canonicalized per KK slot:

```text
KK_KEY_OBJECT=imagined-key#1
KK_KEY_OBJECT=imagined-key#2
-> placeholder:KK_KEY_OBJECT
```

Concrete values remain distinct. This prevents rollout value from being
inflated by multiple syntactic instances of the same imagined placeholder.

### Unique Future Metrics

The diagnostics now report both raw and unique expansion counts:

```text
raw_future_candidate_count
unique_future_candidate_count
duplicate_future_candidate_count
future_candidate_dedup_ratio
raw_newly_unlocked_action_count
unique_newly_unlocked_action_count
unique_unlock_ratio
```

These fields are propagated through step, episode, summary, and analysis CSVs.

### Confidence-Discounted Future Value

The real selected action remains executable and keeps its immediate value. Only
future rollout value is discounted:

```text
transition_confidence =
  mean(predicted positive KK probabilities, 1 - predicted_error_prob)

effective_confidence =
  clamp(transition_confidence * grounding_factor)

path_confidence_t =
  product_i<=t effective_confidence_i

future_step_value_t =
  gamma^t * path_confidence_t * immediate_value_t, for t > 0
```

When confidence is 1, `APASSR_FULL_CAL` matches the uncalibrated future value.
When confidence is 0, future value is removed while the first action's immediate
value remains.

### Placeholder Grounding Discount

Grounding factors distinguish concrete, placeholder-only, and mixed branches:

```text
concrete grounding: 1.00
mixed concrete/placeholder grounding: mixed_grounding_confidence_scale
placeholder-only grounding: placeholder_confidence_scale
```

The default comparison condition uses:

```text
placeholder_confidence_scale = 0.35
mixed_grounding_confidence_scale = 0.65
```

These constants are fixed implementation choices for the calibrated condition,
not per-environment tuning.

## Reporting

Run calibrated comparisons beside the original full condition:

```powershell
$env:PYTHONPATH='src'; python -m aassr.v2_compare --world v2_complex --episodes 30 --seeds 10 --step-limit 120 --workers 6 --include-apassr-full --include-apassr-full-cal --output-dir runs\apassr_full_cal_v2_complex_30x10
$env:PYTHONPATH='src'; python -m aassr.v2_compare --world locked_bottleneck --episodes 30 --seeds 10 --step-limit 120 --workers 6 --include-apassr-full --include-apassr-full-cal --output-dir runs\apassr_full_cal_locked_bottleneck_30x10
```

Report `APASSR_FULL_CAL` as a calibrated variant, not as the legacy paper
prototype and not as a replacement for C3/C5.
