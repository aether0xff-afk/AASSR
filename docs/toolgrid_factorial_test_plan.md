# ToolGrid map-size × action-branching factorial pilot

## 목적

기존 final complexity scaling은 주로 최단 성공 경로 길이를 증가시켰고, DQN의 실제 난이도는 단조 증가하지 않았다. 이번 파일럿은 복잡도를 하나의 Level로 합치지 않고 다음 두 축을 직접 조작한다.

1. **공간/경로 복잡도**: map size `3×3`, `5×5`, `7×7`
2. **의미 있는 행동 분기**: action count `8`, `12`

주 질문은 각 축이 커질수록 `Imagination v2 − DQN` unseen 성공률 차이가 증가하는지다.

## 환경

ToolGrid는 네 개 이동 행동과 여러 tool 행동을 동시에 노출한다.

- `move_north`, `move_south`, `move_west`, `move_east`
- action count 8: tool 4개
- action count 12: tool 8개

맵마다 네 개 station이 생성되고 각 station은 하나의 tool을 요구한다. 에이전트는 station까지 이동한 뒤 맞는 tool을 사용해야 다음 station으로 진행한다. 잘못된 이동, 이미 사용한 칸 재방문, 현재 station에 맞지 않는 tool 사용은 비가역 실패다.

추가 tool은 영구적으로 쓸모없는 함정 버튼이 아니다. map seed에 따라 모든 tool이 어느 station에서는 정답으로 등장하며, 충분한 map pool에서 각 tool 사용이 균형 있게 나타난다.

## 동결 조건

- 외부 보상: 최종 성공 `1`, 그 외 `0`
- 인공 step/tick 제한 없음
- 성공 또는 환경 자체의 비가역 실패로 종료
- station 수: 4
- 같은 seed·factorial cell에서 모든 비교 조건은 같은 train/unseen map을 사용
- train/unseen map seed 범위 분리

## 비교 모델

1. DQN
2. Neural Policy-only
3. Imagination v2

Imagination v2는 ToolGrid용 고정 차원 codec, Neural Delta Prophecy, holdout calibration, branch-local GRU critic을 사용한다. Neural Policy-only는 같은 DQN Policy와 Neural Prophecy를 사용하지만 Imagination을 끈 matched control이다.

## 파일럿 규모

- 독립 seed: `7, 21, 42`
- map size: `3, 5, 7`
- action count: `8, 12`
- 조건: 3개
- 총 cell: `3 × 3 × 2 × 3 = 54`
- seed·cell당 real transition budget: `5,000`
- training maps: 48
- unseen maps: 100
- checkpoints: `0, 2,500, 5,000`

## 분석

각 seed와 factorial cell에서 unseen 성공률을 먼저 계산한다. 그 후

```text
Delta = success(Imagination v2) - success(DQN)
```

에 대해 seed별로 다음 모형을 적합한다.

```text
Delta = b0 + b_size·map_size + b_branch·action_count
        + b_interaction·map_size·action_count
```

- `b_size > 0`: 맵이 커질수록 Imagination의 상대 가치 증가
- `b_branch > 0`: 의미 있는 행동 가지 수가 늘수록 상대 가치 증가
- `b_interaction > 0`: 두 복잡도가 함께 증가할 때 추가 상승

3 seeds는 최종 확증이 아니라 방향성을 보는 파일럿이다. 유망한 축만 20 seeds 이상의 확증 실험으로 확대한다.
