# Paper experiment quickstart

The paper workflow is intentionally fail-closed. Pilot suites can run
immediately; Final configs refuse to run until reviewed P0–P3 evidence and,
for creativity, reviewed novelty/utility rules have been frozen.

## Install

```powershell
python -m pip install -e ".[dev,paper]"
```

The core and tabular pilots remain usable without the `paper` extra. DQN and
PyTorch Prophecy conditions require it.

## Pilot sequence

```powershell
python scripts/run_paper_suite.py --config configs/paper_autonomy_pilot_v1.json --pilot
python scripts/run_paper_suite.py --config configs/paper_ablation_pilot_v1.json --pilot
python scripts/run_paper_suite.py --config configs/paper_transfer_pilot_v1.json --pilot
python scripts/run_paper_suite.py --config configs/paper_creativity_pilot_v1.json --pilot
python scripts/run_paper_suite.py --config configs/paper_safe_application_pilot_v1.json --pilot
```

Use `--dry-run` to validate a config and print its planned episode count.
Interrupted suites can be continued with `--resume`. Resume is allowed only
when the config SHA256 is unchanged.

Every run writes:

```text
paper_results/<protocol_version>/
├─ raw/
├─ seed_level/
├─ statistics/
├─ tables/
├─ figures/
├─ manifests/
└─ report.md
```

Validate a result independently:

```powershell
python scripts/validate_paper_artifacts.py --results paper_results/paper-autonomy-pilot-v1
```

## Freeze before Final

Review the creativity Pilot candidate before freezing it:

```powershell
python scripts/freeze_creativity_rules.py `
  --candidate paper_results/paper-creativity-pilot-v1/manifests/creativity_threshold_candidate.json `
  --output configs/frozen_creativity_rules_v1.json `
  --reviewer "reviewer-id"
```

Lock reviewed P0–P3 evidence:

```powershell
python scripts/lock_final_protocol.py `
  --p0-results paper_results/paper-autonomy-pilot-v1 `
  --p1-results paper_results/paper-ablation-pilot-v1 `
  --p2-results paper_results/paper-transfer-pilot-v1 `
  --p3-results paper_results/paper-creativity-pilot-v1 `
  --reviewer "reviewer-id" `
  --output configs/paper_acceptance_gates_v1.json
```

The lock records each source manifest SHA256. Do not regenerate either lock
after looking at Final results.

## Human study

The study UI binds to localhost only and does not request direct identifiers:

```powershell
python scripts/run_human_study.py `
  --config configs/paper_creativity_pilot_v1.json
```

Open `http://127.0.0.1:8765`. Path collection and blind ratings share an
anonymous participant ID. Exported data is written beside the database under
`export/`. In-progress paths are persisted in SQLite and resume after a browser
reload or server restart. Human data can be merged only when the exported
approval ID and dataset version match the config and at least two raters are
present. Institutional approval and participant consent remain the
researcher's responsibility.

## Isolated local application

Inspect the effective Compose configuration before starting it:

```powershell
python scripts/run_safe_application.py `
  --config configs/paper_safe_application_pilot_v1.json config
python scripts/run_safe_application.py `
  --config configs/paper_safe_application_pilot_v1.json smoke
```

The Compose network is internal, publishes no host ports, drops all
capabilities, runs read-only, and contains no real vulnerability or external
target. `smoke` builds and starts the container, verifies all three seeded
solution routes plus runtime isolation, and always tears the stack down.
