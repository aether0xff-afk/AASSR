# ToolGrid map-size × action-branching factorial pilot

## 목적

기존 final complexity scaling은 주로 최단 성공 경로 길이를 증가시켰고, DQN의 실제 난이도는 단조 증가하지 않았다. 이번 파일럿은 복잡도를 하나의 Level로 합치지 않고 다음 두 축을 직접 조작한다.

1. **공간/경로 복잡도**: map size `3×3`, `5×5`, `7×7`
2. **의미 있는 도구 분기**: tool choices `4`, `8` (`action_count` 표기는 이동 행동을 포함해 `8`, `12`)

주 질문은 각 축이 커질수록 `Imagination v2 − DQN` unseen 성공률 차이가 증가하는지다.

## 환경

ToolGrid에는 네 개 이동 행동과 여러 tool 행동이 있다.

- `move_north`, `move_south`, `move_west`, `move_east`
- action count 8: 이동 4개 + tool 4개
- action count 12: 이동 4개 + tool 8개

두 행동 종류를 매 순간 함께 노출하지는 않는다.

- 이동 중: 격자 안에 있고 아직 방문하지 않은 이동 행동만 노출
- station 도달 후: 이동 행동을 숨기고 모든 tool 후보만 노출

따라서 map size는 공간 탐색 부담을, tool 수는 station에서의 의미 있는 선택 분기를 독립적으로 조작한다. 맵마다 station 하나와 그 station에 필요한 tool 하나가 생성된다. 에이전트는 station까지 이동한 뒤 올바른 tool을 사용하면 성공한다. 이미 방문한 칸은 다시 밟을 수 없고, 이동 가능한 칸을 모두 소진하거나 잘못된 tool을 쓰면 비가역 실패한다.

추가 tool은 영구적으로 쓸모없는 함정 버튼이 아니다. map seed에 따라 모든 tool이 다른 맵에서는 정답으로 등장하며, 충분한 map pool에서 모든 tool이 실제로 요구되는지 테스트한다.

station 수를 1로 고정한 이유는 이번 파일럿에서 **맵 크기와 tool branching만 격리**하기 위해서다. 초기 2-stage smoke에서는 어려운 셀의 최종 성공이 너무 적어 GRU critic이 성공·실패 양쪽 사례를 보지 못했고 Imagination 사용률이 0%가 되는 바닥 효과가 확인됐다. dependency depth는 이번 파일럿 결과 뒤 별도 제3요인으로 추가한다.

## 동결 조건

- 외부 보상: 최종 성공 `1`, 그 외 `0`
- 인공 step/tick 제한 없음
- 성공 또는 환경 자체의 비가역 실패로 종료
- station 수: 1
- 같은 seed·factorial cell에서 모든 비교 조건은 같은 train/unseen map 사용
- train/unseen map seed 범위 분리
- 공간 조작과 tool branching 조작 외 규칙은 셀 사이에서 동일

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

## 조작 점검

최종 성능을 보기 전에 다음이 만족되는지 확인한다.

1. 평균 oracle shortest steps가 `3×3 < 5×5 < 7×7`
2. station에서 실제 available tool count가 `4 < 8`
3. 모든 tool이 map pool 어딘가에서 정답으로 등장
4. DQN 성공률이 모든 셀에서 0 또는 1로 포화되지 않음
5. Imagination v2의 critic이 준비되고 최종 평가에서 Imagination 사용률이 0보다 큼

5번을 만족하지 못한 셀은 상상 효과 검정에서 제외하고, 왜 critic이 준비되지 않았는지 별도로 보고한다.

## 분석

각 seed와 factorial cell에서 unseen 성공률을 먼저 계산한다. 그 후

```text
Delta = success(Imagination v2) - success(DQN)
```

에 대해 seed별로 다음 모형을 적합한다.

```text
Delta = b0 + b_size·map_size + b_branch·tool_count
        + b_interaction·map_size·tool_count
```

- `b_size > 0`: 맵이 커질수록 Imagination의 상대 가치 증가
- `b_branch > 0`: 의미 있는 tool 가지 수가 늘수록 상대 가치 증가
- `b_interaction > 0`: 두 복잡도가 함께 증가할 때 추가 상승

3 seeds는 최종 확증이 아니라 방향성을 보는 파일럿이다. 유망한 축만 20 seeds 이상의 확증 실험으로 확대한다.
