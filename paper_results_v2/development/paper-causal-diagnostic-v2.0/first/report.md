# paper-causal-diagnostic-v2.0 — development_diagnostic

development_diagnostic results are not Final performance evidence.

## Engineering integrity
- frozen_checkpoint_immutable: True
- evaluation_learning_calls_zero: True
- private_state_leaks_zero: True
- gzip_trace_replay: True
- adaptation_branches_share_checkpoint: True
- frozen_representation_evaluation_immutable: True
- oracle_transition_accuracy_100_percent: True
- oracle_root_action_optimality_at_least_95_percent: False
- oracle_regret_below_policy: True
- oracle_dead_end_not_above_policy: True
- random_model_low_confidence_interventions_zero: True
- transfer_branches_share_checkpoint: True
- transfer_frozen_evaluation_immutable: True

## Benchmark adequacy
- world_certification: True
- contextual_training_above_random: True
- contextual_frozen_above_random: True
- contextual_replay_gap: True
- full_replay_gap: True
- open_creativity_world_adequate: True

## Diagnostic 1 metrics
- contextual_training_final_tail: 0.456667
- contextual_frozen_success: 0.500000
- random_success: 0.050000
- full_mean_absolute_replay_gap: 0.066667

Empirical hypotheses are reported, not used as progression gates.