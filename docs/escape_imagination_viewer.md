# Escape GridWorld Imagination Viewer

## 언제 상상하는가

기본 설정은 `imagination_interval=1`이지만 모든 환경 tick에서 무조건 Imagination을 실행하지는 않는다.

한 tick에서 Imagination이 실행되려면 다음 조건을 모두 만족해야 한다.

1. `use_imagination=True`
2. epsilon 탐색으로 뽑힌 무작위 행동이 아님
3. 해당 tick이 `imagination_interval`에 해당함
4. 현재 사용 가능한 행동에 대한 Prophecy model coverage가 `imagination_minimum_coverage` 이상임

Escape GUI 기본값은 다음과 같다.

```text
imagination_interval = 1
imagination_minimum_coverage = 0.75
```

따라서 정확한 의미는 **Prophecy를 충분히 학습한 뒤, 조건을 만족하는 모든 비무작위 선택 tick에서 상상한다**는 것이다. 초기 학습에서는 epsilon이 높고 model coverage가 낮아 상상 횟수가 적으며, 학습이 진행될수록 상상 비율이 증가한다.

## GUI

```bash
python scripts/run_escape_gridworld.py --gui
```

기존 GridWorld 창과 함께 `AASSR Imagination Viewer` 창이 자동으로 열린다.

상상 창에는 다음 정보가 표시된다.

- 상상 순번과 현실 root tick
- root 시점의 실제 위치
- 최종 선택된 첫 행동
- 생성 노드 수, 확장 노드 수, 최대 깊이
- 깊이별 전체 Imagination tree
- 선택된 최선 경로 강조
- 각 노드의 누적 가치와 누적 신뢰도
- 각 노드의 종료 원인
  - goal
  - low_confidence
  - no_actions
  - repeated_state
  - depth_limit
- 루트 행동별 aggregate value, leaf 값, 최선 경로
- 모든 노드의 예측 상태, facts, available actions, policy memory, prophecy memory
- 이전/다음 상상 탐색
- 최신 상상 자동 추적

### 실시간 모드

상상이 실행될 때마다 각 트리가 순서대로 창에 전달된다. 다음 상상이 실행되기 전까지 현재 트리가 유지되므로 분기와 선택 경로를 직접 볼 수 있다.

### 최대 속도 모드

모든 상상 트리는 빠짐없이 파일에 저장하지만, GUI 이벤트가 학습을 늦추거나 메모리를 채우지 않도록 창은 가장 최근 트리로 합쳐서 갱신한다. 따라서 화면 중간 갱신 일부를 건너뛸 수 있지만 원본 데이터는 손실되지 않는다.

## 저장 파일

기존 결과 폴더에 다음 파일이 추가된다.

```text
imaginations.jsonl
imagination_summary.json
```

`imaginations.jsonl`은 Imagination 실행 직후 한 줄씩 즉시 flush된다. 각 줄에는 다음이 들어간다.

- 상상 순번과 UTC 시각
- 현실 root tick과 위치
- 선택된 첫 행동
- root action별 평가값과 최선 leaf
- 전체 노드와 부모 관계
- 각 노드의 전체 예측 StateSnapshot
- action path와 state path
- 누적 가치
- step/cumulative confidence
- policy memory와 prophecy memory
- terminal reason

강제 종료되더라도 이미 실행된 상상 기록은 남는다.

`imagination_summary.json`에는 전체 상상 횟수, 총 노드 수, 평균 노드 수, 최대 깊이가 저장된다.
