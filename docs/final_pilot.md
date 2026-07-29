# AASSR v2 최종 파일럿

이 설정은 이전 파일럿에서 발견한 두 문제를 수정한 뒤 본 실험 직전에 돌리는 최종 검증이다.

- 현재 샘플이 검증셋을 바꾸지 않도록 고정 holdout 전후 비교
- holdout 최소 4개 이전에는 holdout gain을 보상에 사용하지 않음
- Skill 결과에 실제 seed 기록
- 즉시 진행도가 높지만 복구 불가능한 dependency 함정
- 유용한 정보 1개와 무관한 정보 8개 중 에이전트가 직접 선택
- raw novelty, sparse reward, 검증된 information value 비교
- 5 seeds에서 seed 내부 평균 후 seed 간 통계

## 한 줄 실행

```powershell
python scripts/run_experiment.py --config configs/pilot.json --output runs/final_pilot --overwrite
```

실행 전 규모만 확인:

```powershell
python scripts/run_experiment.py --config configs/pilot.json --dry-run
```

정상 계획 행 수는 `1980`이다.

## 정상 신호

1. `imagination/depth_2`가 `setup`을 골라 성공하고 `depth_1`은 `shortcut`에서 실패한다.
2. `dependency/full_chain_imagination`이 `advance_safe`를 고르고, `policy_greedy`와 `shallow_imagination`은 `advance_greedy` 함정에 빠진다.
3. `skills/skill_after_2`는 각 실제 seed에서 두 번 성공 후 `high_level_steps=1`이 된다.
4. `information_value/raw_novelty`는 `noise_probe`를 선호하고, `validated_information_value`는 `useful_probe`를 선호해야 한다.
5. holdout trace에서 검증셋이 충분하지 않은 초기 구간의 `holdout_gain`은 0이며, 전후 점수는 같은 고정 표본으로 계산된다.

## 결과 파일

- `episodes.csv`: 원자료
- `seed_summary.csv`: seed 내부 episode 평균
- `summary.csv`: seed 평균들의 평균·표준편차·95% 구간
- `report.md`: 빠른 결과표
- `traces/`: 고정 holdout 및 information value 세부 로그

이 파일럿이 통과해야 더 큰 episode 수와 환경 다양성을 가진 본 실험으로 넘어간다.
