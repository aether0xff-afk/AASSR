# configs 안내

이 디렉터리는 실험 설정과 과거 paper protocol 설정을 보관합니다.

현재 AASSR pentest main은 대부분의 핵심 설정을 CLI와 `current_*` protocol 코드에서 직접 고정/검증합니다. 따라서 오래된 JSON 파일을 보고 현행 runtime을 추론하지 마세요.

- `autonomous_*` — 과거 autonomous experiment 설정
- `paper_*` — paper reproduction / frozen protocol 설정
- `effect_*`, `frozen_*` — 특정 역사 실험 설정

현행 pentest 실행은 `scripts/run_pentest_current_generation_main.py`와 `src/aassr_v2/current_manifest.py`를 기준으로 합니다.
