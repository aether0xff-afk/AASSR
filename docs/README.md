# docs 안내

이 디렉터리는 AASSR의 문서 모음입니다. 파일 이름에 `current`가 들어간다고 해서 항상 현행 구현을 뜻하는 것은 아니므로, 아래 순서를 기준으로 보세요.

## 현행 문서

- `CURRENT_RUNTIME.md` — 현재 `main` runtime의 source-of-truth 탐색 문서
- `aassr_current_generation.md` — 현행 AASSR 기술 구조
- `dreamerv3_current_baseline.md` — 현재 DreamerV3 비교 조건과 adapter 계약

## 실험/진단 문서

`pentest_*`, `critic_*`, `prophecy_*`, `imagination_*` 문서는 특정 실험이나 감사 과정을 기록합니다. 현재 모델 정의보다 해당 실험의 재현과 해석을 위한 자료입니다.

## 과거 세대/재현 문서

- `aassr_v040_architecture.md`
- `releases/`
- GridPush / ToolGrid / Imagination v2 관련 문서
- paper protocol 및 과거 benchmark 결과

이들은 역사적 증거이며 현행 runtime 정의가 아닙니다. 병합 전 전체 저장소 상태는 `archive/pre-current-main-2026-08-12` 브랜치에 보존되어 있습니다.

## 현행 여부를 판단하는 최종 기준

문서보다 코드가 우선입니다.

1. `src/aassr_v2/current_manifest.py`
2. `src/aassr_v2/current_entrypoint.py`
3. `src/aassr_v2/pentest_current_generation_main.py`
4. `scripts/run_pentest_current_generation_main.py`
5. current-generation CI
