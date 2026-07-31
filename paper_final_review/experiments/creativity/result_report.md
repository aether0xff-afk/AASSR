# creativity Final review

- Config: `D:\AASSR\configs\paper_creativity_final_v1.json`
- Started: 2026-07-31T11:45:48.040+00:00
- Pipeline completed: 2026-07-31T11:49:14.397+00:00
- Planned/actual rows: 105,630 / 105,630
- Completed research seeds: 30 / 30
- Missing seeds: []
- Failed/retried runs: []
- Artifact validator: PASS
- Config/resolved hash matches manifest: True / True
- Final acceptance gate hash match: True
- Frozen creativity rule: applied and matched
- Pilot/Final research seed overlap: []
- Pilot/Final world seed overlap: []
- Final train/unseen world overlap: []
- Exact row duplicates: 0
- Grain duplicates: 0
- NaN/Inf: 0
- Invalid numeric: 0
- Abnormal domain values: 0
- Agent-visible private/oracle label leaks: 0
- Evaluation transitions with learning enabled: 0

## Largest episode CSV payload columns

| Column | Payload MiB | Share |
|---|---|---|
| strategy_id | 2.57 | 12.8% |
| environment | 2.52 | 12.6% |
| experiment | 2.52 | 12.6% |
| runtime_seconds | 2.14 | 10.7% |
| action_family | 1.91 | 9.5% |
| solution_family | 1.37 | 6.8% |
| condition | 1.14 | 5.7% |
| suite | 1.01 | 5.0% |
| phase | 0.82 | 4.1% |
| model | 0.60 | 3.0% |

Full per-condition/environment/phase counts and all integrity counters are in `integrity_report.json`.

## Primary seed-level results

| Condition | Environment | Phase | Metric | Seeds | Mean | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| aassr_no_imagination | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| aassr_no_imagination | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| aassr_no_novelty | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| aassr_no_novelty | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| dqn | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| dqn | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| full_aassr | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| full_aassr | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| novelty_search | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| novelty_search | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| q_learning | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| q_learning | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |
| random | multi_solution_dependency | evaluation_unseen_adaptation | success | 30 | 1.0000 | [1.0000, 1.0000] |
| random | multi_solution_dependency | training | success | 30 | 1.0000 | [1.0000, 1.0000] |

## Holm-significant paired comparisons

| Phase | Metric | Comparison | Paired seeds | Difference | 95% CI | Holm p |
|---|---|---|---|---|---|---|
| evaluation_unseen_adaptation | runtime_seconds | full_aassr vs aassr_no_imagination | 30 | 0.0001 | [0.0001, 0.0001] | 0.0003 |
| evaluation_unseen_adaptation | runtime_seconds | full_aassr vs dqn | 30 | -0.0020 | [-0.0021, -0.0018] | 0.0003 |
| evaluation_unseen_adaptation | runtime_seconds | full_aassr vs novelty_search | 30 | 0.0001 | [0.0001, 0.0001] | 0.0003 |
| evaluation_unseen_adaptation | runtime_seconds | full_aassr vs q_learning | 30 | 0.0002 | [0.0002, 0.0002] | 0.0003 |
| evaluation_unseen_adaptation | runtime_seconds | full_aassr vs random | 30 | 0.0002 | [0.0002, 0.0002] | 0.0003 |
| training | steps | full_aassr vs aassr_no_imagination | 30 | -0.0547 | [-0.0662, -0.0437] | 0.0003 |
| training | high_level_steps | full_aassr vs aassr_no_imagination | 30 | -0.0547 | [-0.0661, -0.0435] | 0.0003 |
| training | primitive_steps | full_aassr vs aassr_no_imagination | 30 | -0.0547 | [-0.0658, -0.0435] | 0.0003 |
| training | imagined_nodes | full_aassr vs aassr_no_imagination | 30 | 2.7279 | [2.6119, 2.8426] | 0.0003 |
| training | real_transitions | full_aassr vs aassr_no_imagination | 30 | -0.0547 | [-0.0661, -0.0435] | 0.0003 |
| training | imagined_transitions | full_aassr vs aassr_no_imagination | 30 | 2.7279 | [2.6159, 2.8403] | 0.0003 |
| training | action_proposals | full_aassr vs aassr_no_imagination | 30 | -0.0547 | [-0.0660, -0.0436] | 0.0003 |
| training | runtime_seconds | full_aassr vs aassr_no_novelty | 30 | 0.0004 | [0.0002, 0.0005] | 0.0003 |
| training | steps | full_aassr vs dqn | 30 | 0.0648 | [0.0485, 0.0815] | 0.0003 |
| training | high_level_steps | full_aassr vs dqn | 30 | 0.0648 | [0.0478, 0.0817] | 0.0003 |
| training | primitive_steps | full_aassr vs dqn | 30 | 0.0648 | [0.0483, 0.0815] | 0.0003 |
| training | imagined_nodes | full_aassr vs dqn | 30 | 2.7279 | [2.6149, 2.8397] | 0.0003 |
| training | runtime_seconds | full_aassr vs dqn | 30 | -0.0038 | [-0.0040, -0.0037] | 0.0003 |
| training | real_transitions | full_aassr vs dqn | 30 | 0.0648 | [0.0476, 0.0822] | 0.0003 |
| training | imagined_transitions | full_aassr vs dqn | 30 | 2.7279 | [2.6156, 2.8427] | 0.0003 |
| training | action_proposals | full_aassr vs dqn | 30 | 0.0648 | [0.0478, 0.0813] | 0.0003 |
| training | steps | full_aassr vs novelty_search | 30 | 0.0779 | [0.0630, 0.0913] | 0.0003 |
| training | high_level_steps | full_aassr vs novelty_search | 30 | 0.0779 | [0.0633, 0.0916] | 0.0003 |
| training | primitive_steps | full_aassr vs novelty_search | 30 | 0.0779 | [0.0628, 0.0919] | 0.0003 |
| training | imagined_nodes | full_aassr vs novelty_search | 30 | 2.7279 | [2.6168, 2.8440] | 0.0003 |
| training | runtime_seconds | full_aassr vs novelty_search | 30 | 0.0005 | [0.0003, 0.0007] | 0.0003 |
| training | real_transitions | full_aassr vs novelty_search | 30 | 0.0779 | [0.0633, 0.0917] | 0.0003 |
| training | imagined_transitions | full_aassr vs novelty_search | 30 | 2.7279 | [2.6151, 2.8414] | 0.0003 |
| training | action_proposals | full_aassr vs novelty_search | 30 | 0.0779 | [0.0637, 0.0915] | 0.0003 |
| training | steps | full_aassr vs q_learning | 30 | 0.2167 | [0.1981, 0.2342] | 0.0003 |
| training | high_level_steps | full_aassr vs q_learning | 30 | 0.2167 | [0.1977, 0.2343] | 0.0003 |
| training | primitive_steps | full_aassr vs q_learning | 30 | 0.2167 | [0.1977, 0.2348] | 0.0003 |
| training | imagined_nodes | full_aassr vs q_learning | 30 | 2.7279 | [2.6120, 2.8390] | 0.0003 |
| training | runtime_seconds | full_aassr vs q_learning | 30 | 0.0007 | [0.0004, 0.0008] | 0.0003 |
| training | real_transitions | full_aassr vs q_learning | 30 | 0.2167 | [0.1989, 0.2341] | 0.0003 |
| training | imagined_transitions | full_aassr vs q_learning | 30 | 2.7279 | [2.6112, 2.8437] | 0.0003 |
| training | action_proposals | full_aassr vs q_learning | 30 | 0.2167 | [0.1985, 0.2341] | 0.0003 |
| training | steps | full_aassr vs random | 30 | 0.0409 | [0.0175, 0.0653] | 0.0033 |
| training | high_level_steps | full_aassr vs random | 30 | 0.0409 | [0.0179, 0.0647] | 0.0040 |
| training | primitive_steps | full_aassr vs random | 30 | 0.0409 | [0.0178, 0.0659] | 0.0039 |
| training | imagined_nodes | full_aassr vs random | 30 | 2.7279 | [2.6111, 2.8421] | 0.0003 |
| training | runtime_seconds | full_aassr vs random | 30 | 0.0007 | [0.0006, 0.0008] | 0.0003 |
| training | real_transitions | full_aassr vs random | 30 | 0.0409 | [0.0183, 0.0648] | 0.0042 |
| training | imagined_transitions | full_aassr vs random | 30 | 2.7279 | [2.6114, 2.8418] | 0.0003 |
| training | action_proposals | full_aassr vs random | 30 | 0.0409 | [0.0175, 0.0651] | 0.0040 |
