# scripts 안내

실행 스크립트 모음입니다. 새 실험은 아래 **현행 진입점**을 우선 사용하세요.

## 현행 실행

- `run_pentest_current_generation_main.py` — canonical AASSR current-generation 학습/평가
- `run_dreamerv3_current_baseline.py` — 현재 DreamerV3 비교 baseline
- `assemble_pentest_current_generation_suite.py` — current-generation 결과 조립
- `check_current_generation_hardware.py` — CUDA/하드웨어 경로 점검

## 현재 진단

- `run_repaired_imagination_final.py` — detailed Imagination decision/prediction trace
- `analyze_repaired_imagination_trace.py` — repaired trace 분석
- `run_current_status_rare_holdout.py` — rare public-status holdout 진단
- `run_pentest_current_generation_smoke.py` — 짧은 current runtime smoke

## 과거 재현

`v040`, `imagination_v2`, `toolgrid`, `gridpush`, `stall`, `paper`, `abandonment` 등의 이름이 들어간 스크립트는 과거 실험 재현용입니다. 새 current 실험의 진입점으로 사용하지 마세요.

병합 전 전체 파일 구조와 과거 workflow는 `archive/pre-current-main-2026-08-12` 브랜치에 보존되어 있습니다.
