# AASSR v2 실험 실행 안내

이 문서는 `aassr-v2` 브랜치의 배치 실험기 사용법을 설명한다. 파일럿은 실행 배선과 지표 방향을 확인하는 용도이며, 논문 성능 주장은 본 실험 설정과 충분한 seed를 사용한 뒤 내려야 한다.

## 1. 준비

PowerShell에서 저장소를 최신 상태로 맞춘다.

```powershell
git switch aassr-v2
git pull origin aassr-v2
python -m pip install -e ".[dev]"
python -m pip check
python -m compileall -q src tests scripts
pytest -q
```

모든 테스트가 통과한 뒤 실험을 실행한다.

## 2. 설정 검증만 하기

실험을 실행하지 않고 설정 문법과 예상 결과 행 수만 확인한다.

```powershell
python scripts/run_experiment.py --config configs/pilot.json --dry-run
```

정상적인 기본 파일럿은 `Planned result rows: 252`를 출력한다.

## 3. 파일럿 실행

```powershell
python scripts/run_experiment.py `
  --config configs/pilot.json `
  --output runs/pilot `
  --overwrite
```

한 줄 명령은 다음과 같다.

```powershell
python scripts/run_experiment.py --config configs/pilot.json --output runs/pilot --overwrite
```

파일럿에는 다음이 포함된다.

- 표 기반 Prophecy와 온라인 GRU 비교
- 학습 가능한 안정 전이와 순수 무작위 전이 비교
- 즉시 진행도 함정에서 깊이 1·2 Imagination 비교
- 길이 4·6 dependency 환경
- 두 번 성공한 ASeq의 Skill 승격
- 정보 노이즈 0·8 조건의 정보 가치 진단

## 4. 일부 실험만 실행

Imagination만 실행:

```powershell
python scripts/run_experiment.py --config configs/pilot.json --suite imagination --output runs/pilot_imagination --overwrite
```

Prophecy와 dependency만 실행:

```powershell
python scripts/run_experiment.py --config configs/pilot.json --suite prophecy --suite dependency --output runs/pilot_pd --overwrite
```

설정의 seed를 명령행에서 덮어쓰기:

```powershell
python scripts/run_experiment.py --config configs/pilot.json --seeds 7,13,21 --output runs/pilot_3seed --overwrite
```

## 5. 결과 파일

실험 폴더에는 다음 파일이 생성된다.

```text
runs/pilot/
├─ resolved_config.json
├─ episodes.csv
├─ seed_summary.csv
├─ summary.csv
├─ report.md
└─ traces/
   └─ information_seed*_noise*.jsonl
```

### episodes.csv

모든 평가 episode 또는 전이의 원자료다.

주요 열:

- `suite`, `condition`, `environment`, `model`
- `seed`, `episode`, `phase`
- `success`, `steps`, `errors`, `repeats`
- `prediction_score`, `holdout_score`, `holdout_gain`
- `imagined_nodes`, `imagination_depth`, `root_imagined_value`
- `actual_return`
- `skill_count`, `skill_uses`
- `noise_facts`, `novelty_score`, `intrinsic_value`
- `runtime_seconds`

### seed_summary.csv

각 condition에서 episode들을 먼저 seed 내부에서 평균 낸 결과다. 논문용 비교의 기본 단위로 사용한다.

### summary.csv

seed별 평균들을 다시 모아 다음을 계산한다.

- seed 평균
- seed 간 표준편차
- 95% 신뢰구간 근사치

개별 episode를 독립 표본처럼 세어 표본 수를 부풀리지 않는다.

### report.md

주요 결과를 빠르게 확인하는 표다. 세부 분석은 CSV를 기준으로 한다.

## 6. 파일럿에서 확인할 신호

### Imagination

`deceptive_choice` 환경은 다음 구조다.

```text
shortcut
→ 즉시 진행도 0.6
→ 막다른 상태

setup
→ 즉시 진행도 0.1
→ finish
→ 최종 목표 1.0
```

정상적인 핵심 신호:

- `policy_only`: 첫 행동 `shortcut`, 최종 실패
- `depth_1`: 첫 행동 `shortcut`, 최종 실패
- `depth_2`: 첫 행동 `setup`, 최종 성공

이 결과는 깊은 상상이 무조건 좋다는 증거가 아니라, 나무 탐색이 미래 상태를 실제로 역집계한다는 배선 검증이다.

### Skill

길이 4 dependency에서는 한 번 성공할 때 원시 행동 8개가 필요하다.

정상적인 핵심 신호:

- 첫 번째와 두 번째 성공: `high_level_steps = 8`
- 두 번 성공 후 Skill 승격
- 이후 성공: `high_level_steps = 1`
- `primitive_steps`는 계속 8

즉 Skill은 물리적 실행 비용을 마법처럼 없애는 것이 아니라, Policy와 Imagination이 다루는 고수준 계획 깊이를 압축한다.

### Prophecy

`prophecy` suite에서는 다음을 비교한다.

- `*_stable`: 규칙이 일정한 전이
- `*_random`: 순수 무작위 결과

안정 전이의 예측 점수가 무작위 전이보다 높아지는지 확인한다. 파일럿 결과만으로 GRU와 Tabular의 우열을 주장하지 않는다.

### 정보 가치

노이즈가 많아지면 `novelty_score`는 쉽게 커질 수 있다. 반면 `holdout_gain`과 `intrinsic_value`가 같은 비율로 커지지 않는지 확인한다. 이것이 단순 새 정보 개수와 예측 개선 기반 가치의 차이를 보는 첫 진단이다.

## 7. 본 실험 명령

### Prophecy 비교

```powershell
python scripts/run_experiment.py --config configs/prophecy.json --output runs/prophecy --overwrite
```

### Imagination 기능 제거

```powershell
python scripts/run_experiment.py --config configs/imagination.json --output runs/imagination --overwrite
```

### Dependency 길이 증가

```powershell
python scripts/run_experiment.py --config configs/dependency.json --output runs/dependency --overwrite
```

### GOAL·Skill 압축

```powershell
python scripts/run_experiment.py --config configs/goals_skills.json --output runs/goals_skills --overwrite
```

### 정보 가치와 무관 정보

```powershell
python scripts/run_experiment.py --config configs/information_value.json --output runs/information_value --overwrite
```

## 8. 결과 요약 다시 만들기

`episodes.csv`를 수정하거나 여러 분석을 거친 뒤 seed 단위 요약을 다시 만들 수 있다.

```powershell
python scripts/summarize_runs.py runs/pilot
```

이 명령은 다음을 다시 생성한다.

```text
seed_summary.csv
summary.csv
report.md
```

## 9. 출력 폴더 보호

`--overwrite`를 생략하면 같은 출력 폴더가 존재할 때 시간 문자열이 붙은 새 폴더를 만든다.

```powershell
python scripts/run_experiment.py --config configs/pilot.json --output runs/pilot
```

예:

```text
runs/pilot_20260728_213500/
```

기존 결과를 확실히 지우고 다시 실행할 때만 `--overwrite`를 사용한다.

## 10. GitHub Actions에서 파일럿 실행

GitHub 저장소에서:

```text
Actions
→ pilot-experiment
→ Run workflow
→ seeds 입력
→ Run workflow
```

워크플로는 다음을 수행한다.

```text
설치
→ 전체 컴파일
→ 전체 pytest
→ configs/pilot.json 실행
→ runs/pilot artifact 업로드
```

완료 후 workflow의 Artifacts에서 `aassr-v2-pilot-<commit>` 파일을 내려받으면 된다.

## 11. 연구 해석 기준

파일럿 통과 조건:

- 전체 pytest 통과
- 설정 dry-run 통과
- `episodes.csv`와 seed 단위 요약 생성
- depth 2가 즉시 진행도 함정을 회피
- 반복 성공 후 Skill 사용 발생
- JSONL trace 생성

본 실험의 최소 보고 항목:

- condition별 seed 수
- seed별 평가 평균
- 전체 seed 평균과 표준편차
- 95% 신뢰구간
- 성공률과 행동 수
- Prophecy 검증 점수
- 상상 노드 수와 실행 비용
- 실패 사례와 반례 환경 결과

파일럿을 통과해도 AASSR가 기존 방법보다 우수하다는 결론은 내리지 않는다. 본 실험과 기능 제거 실험에서 같은 seed끼리 짝지어 비교해야 한다.
