# paper-causal-confirmation-v2.0 — locked_confirmation

locked_confirmation results are not Final performance evidence.

## Engineering integrity
- frozen_checkpoint_immutable: True
- evaluation_learning_calls_zero: True
- private_state_leaks_zero: True
- gzip_trace_replay: True
- adaptation_branches_share_checkpoint: True
- frozen_representation_evaluation_immutable: True
- oracle_transition_accuracy_100_percent: True
- oracle_root_action_optimality_at_least_95_percent: True
- oracle_regret_below_policy: True
- oracle_dead_end_not_above_policy: True
- random_model_low_confidence_interventions_zero: True
- transfer_branches_share_checkpoint: True
- transfer_frozen_evaluation_immutable: True

## Benchmark adequacy
- world_certification: True
- contextual_training_above_random: True
- contextual_frozen_above_random: True
- contextual_replay_gap: False
- full_replay_gap: False
- open_creativity_world_adequate: True

## Diagnostic 1 metrics
- contextual_training_final_tail: 0.873333
- contextual_frozen_success: 1.000000
- random_success: 0.046667
- full_mean_absolute_replay_gap: 0.163333

Empirical hypotheses are reported, not used as progression gates.