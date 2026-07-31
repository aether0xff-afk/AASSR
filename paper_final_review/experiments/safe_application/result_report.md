# safe_application Final review

- Config: `D:\AASSR\configs\paper_safe_application_final_v1.json`
- Started: 2026-07-31T09:19:34.040+00:00
- Pipeline completed: 2026-07-31T09:19:35.485+00:00
- Planned/actual rows: 600 / 600
- Completed research seeds: 30 / 30
- Missing seeds: []
- Failed/retried runs: []
- Artifact validator: PASS
- Config/resolved hash matches manifest: True / True
- Final acceptance gate hash match: True
- Frozen creativity rule: not applicable
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
| experiment | 0.02 | 17.0% |
| phase | 0.02 | 14.8% |
| environment | 0.01 | 12.6% |
| solution_family | 0.01 | 11.3% |
| suite | 0.01 | 8.8% |
| condition | 0.01 | 8.2% |
| model | 0.01 | 7.1% |
| action_family | 0.01 | 5.5% |
| world_seed | 0.00 | 2.7% |
| research_seed | 0.00 | 1.9% |

Full per-condition/environment/phase counts and all integrity counters are in `integrity_report.json`.

## Primary seed-level results

| Condition | Environment | Phase | Metric | Seeds | Mean | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| safe_rule_agent | docker_local_assessment | evaluation_unseen_zero_shot | success | 30 | 1.0000 | [1.0000, 1.0000] |

## Holm-significant paired comparisons

No paired comparison remained significant after Holm correction.
