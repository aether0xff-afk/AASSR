# AASSR v2 최종 파일럿

이 설정은 이전 파일럿에서 발견한 측정·환경 설계 문제를 수정한 뒤 본 실험 직전에 돌리는 최종 검증이다.

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
4. `information_value/validated_information_value`는 `useful_probe`를 안정적으로 선택해야 하며, `raw_novelty`는 무관한 사실 수에 흔들려 더 낮고 불안정한 성공률을 보여야 한다.
5. holdout trace에서 검증셋이 충분하지 않은 초기 구간의 `holdout_gain`은 0이며, 전후 점수는 같은 고정 표본으로 계산된다.

## 2026-07-29 GitHub 검증 결과

GitHub Actions에서 전체 테스트와 5-seed 최종 파일럿을 실행했고 `1980`개 결과 행과 artifact 생성을 확인했다.

- Prophecy: Tabular stable `1.000`, random `0.421`; GRU stable `0.879`, random `0.370`
- Imagination: policy/depth 1은 `shortcut` 선택 후 성공률 `0`; depth 2는 `setup` 선택 후 성공률 `1`
- 함정 dependency: policy와 depth 2는 `advance_greedy` 선택 후 성공률 `0`; 전체 사슬 상상은 길이 4·6 모두 `advance_safe` 선택 후 성공률 `1`
- Skill: 다섯 실제 seed가 각각 기록됐고, 모든 seed에서 두 번째 성공 후 고수준 계획 단계가 `8 → 1`로 감소했다. 원시 실행 단계는 계속 `8`이다.
- 정보 선택: sparse reward와 validated information value는 5/5 seed에서 `useful_probe`를 선택하고 성공했다. raw novelty는 3/5 seed에서 `noise_probe`를 선택해 실패했고 2/5 seed에서만 유용한 정보를 선택했다.
- 고정 holdout: 현재 샘플 자기채점은 제거됐다. 이 결정론적 Tabular 환경에서는 전후 일반화 점수 차이가 없어 실제 `holdout_gain`도 모두 `0`이었다.

마지막 항목은 측정 오류가 수정됐다는 증거이지, holdout 기반 일반화 향상이 이미 증명됐다는 뜻은 아니다. 양의 holdout gain을 검증하려면 GRU와 관측 노이즈, 미관측 상태 조합을 포함한 별도 일반화 환경이 필요하다.

## 결과 파일

- `episodes.csv`: 원자료
- `seed_summary.csv`: seed 내부 episode 평균
- `summary.csv`: seed 평균들의 평균·표준편차·95% 구간
- `report.md`: 빠른 결과표
- `traces/`: 고정 holdout 및 information value 세부 로그

이 파일럿은 실행 배선과 인과 방향을 검증한다. 논문용 성능 주장은 더 큰 episode 수, 환경 다양성, 독립 seed를 사용한 본 실험 뒤에 내려야 한다.
