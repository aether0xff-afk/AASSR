# GitHub Actions 안내

`main`에는 현재 개발과 검증에 필요한 workflow만 유지합니다.

## 유지하는 workflow

- `tests.yml` — 전체 portable test matrix
- `aassr-current-generation.yml` — current-generation 핵심 gate
- `dreamerv3-current-smoke.yml` — official DreamerV3 API/adapter smoke
- `current-status-rare-holdout.yml` — rare-status development diagnostic (manual)
- `aassr-current-base-diagnostic.yml` — current base 진단
- `aassr-repaired-smoke-diagnostic.yml` — repaired Imagination 진단

과거 v0.4, Imagination v2, GridPush/ToolGrid, stall/ablation one-off workflow는 `archive/pre-current-main-2026-08-12` 브랜치에 보존되어 있으며 main에서는 제거합니다.

긴 CUDA scaling 실험은 GitHub hosted CPU에서 실행하지 않습니다.
