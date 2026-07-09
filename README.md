# AASSR Workspace

This root folder is only the version container.

## Folders

| Folder | Purpose |
| --- | --- |
| `v1/` | Frozen v1 snapshot: APASSR GridWorld, C0-C3, baselines, analysis, plots |
| `v2/` | Active workspace for the next, more complex version |
| `artifacts/` | Local experiment outputs and logs moved out of the root |

## Work In v2

```powershell
cd X:\Dev\AASSR\v2
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## v1 Status

v1 was snapshotted after:

```text
Ran 59 tests
OK
```

See:

- `v1/VERSION.md`
- `v2/V2_PLAN.md`
