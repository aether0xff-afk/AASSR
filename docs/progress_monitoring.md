# 장시간 실험 진행 상황 확인

`autonomous_main` runner는 실행 중 전체 episode 기준 진행률, 처리 속도, 경과시간, ETA와 현재 작업 위치를 콘솔과 파일에 동시에 기록한다.

## 기본 실행

```powershell
python scripts/run_experiment.py --config configs/autonomous_main.json --output runs/autonomous_main --overwrite
```

출력 예시:

```text
[AASSR:progress]  37.42% 246,972/660,000 |   84.31 ep/s | elapsed 00:48:49 | ETA 01:21:41 | job=113/300 | seed=211 | environment=opaque_dependency_l6 | condition=full_aassr | phase=training | episode=972/2000 | recent_success=0.830
```

- `%`와 `완료/전체`: 전체 seed × 환경 × 조건 × episode 진행률
- `ep/s`: 실행 시작 이후 평균 episode 처리 속도
- `elapsed`: 누적 경과시간
- `ETA`: 현재 평균 처리 속도를 기준으로 계산한 남은 시간
- `job`: 현재 seed·환경·조건 조합 번호
- `recent_success`: training에서는 최근 최대 100 episode 성공률, evaluation에서는 현재까지 평가 성공률

초기 몇 episode 동안 ETA는 표본이 적어 크게 흔들릴 수 있다. 실행이 진행될수록 평균 속도 기반 ETA가 안정된다.

## 출력 주기 조절

설정 파일:

```json
"progress": {
  "every_episodes": 100,
  "every_seconds": 10
}
```

둘 중 하나의 조건이 먼저 충족되면 로그를 기록한다. CLI 옵션이 설정 파일보다 우선한다.

```powershell
python scripts/run_experiment.py `
  --config configs/autonomous_main.json `
  --output runs/autonomous_main `
  --overwrite `
  --progress-every 50 `
  --progress-seconds 5
```

콘솔 출력을 숨기되 파일 기록은 유지하려면:

```powershell
python scripts/run_experiment.py `
  --config configs/autonomous_main.json `
  --output runs/autonomous_main `
  --overwrite `
  --quiet-progress
```

## 생성되는 진행 파일

```text
runs/autonomous_main/
├─ progress.log
├─ progress.jsonl
├─ progress.json
└─ episodes.csv
```

### `progress.log`

사람이 읽는 전체 진행 로그다. PowerShell에서 실시간으로 볼 수 있다.

```powershell
Get-Content runs/autonomous_main/progress.log -Wait
```

### `progress.json`

항상 마지막 상태 하나만 원자적으로 갱신한다. 실행이 중단되어도 마지막 완료량, ETA, 현재 seed·환경·조건이 남는다.

```powershell
Get-Content runs/autonomous_main/progress.json
```

주기적으로 한 줄 요약만 보려면:

```powershell
while ($true) {
  $p = Get-Content runs/autonomous_main/progress.json -Raw | ConvertFrom-Json
  "{0:N2}%  {1:N0}/{2:N0}  ETA {3}  {4}/{5}/{6}" -f `
    $p.percent, $p.completed, $p.total, $p.eta_at, `
    $p.context.seed, $p.context.environment, $p.context.condition
  Start-Sleep 5
}
```

### `progress.jsonl`

모든 `start`, `job_start`, `progress`, `training_complete`, `job_complete`, `finish`, `failed` 이벤트를 JSON 한 줄씩 저장한다. 나중에 속도 변화나 중단 위치를 분석할 때 사용한다.

### `episodes.csv`

이제 실험 종료 후 한꺼번에 쓰지 않고 episode마다 스트리밍한다. 일정 episode마다 flush하므로 장시간 실행 중에도 중간 결과를 확인하고, 비정상 종료 시 이미 기록된 결과를 보존할 수 있다.

## 정상 종료와 오류

정상 종료 시 `progress.json`의 값은 다음과 같다.

```json
{
  "event": "finish",
  "completed": 660000,
  "total": 660000,
  "percent": 100.0,
  "eta_seconds": 0.0
}
```

예외가 발생하면 `event`가 `failed`가 되고 마지막 context와 오류 종류가 기록된다. 오류가 발생한 seed·환경·조건·phase·episode를 `progress.log`와 `progress.json`에서 바로 확인할 수 있다.

## 후처리

전체 episode 실행이 끝난 뒤 seed 통계 파일을 만드는 동안에는 CLI가 다음 로그를 출력한다.

```text
[AASSR:postprocess] episode execution complete in 02:10:31; generating seed summaries...
[AASSR:postprocess] summaries complete in 00:00:08
```

진행률 100% 이후에도 프로세스가 잠시 종료되지 않는 경우 이 후처리 단계가 진행 중인 것이다.
