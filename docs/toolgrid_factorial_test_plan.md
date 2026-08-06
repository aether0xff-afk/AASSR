# ToolGrid map-size × semantic-branching factorial pilot

## 목적

기존 final complexity scaling은 주로 최단 성공 경로 길이를 증가시켰고, DQN의 실제 난이도는 단조 증가하지 않았다. 이번 파일럿은 복잡도를 하나의 Level로 합치지 않고 다음 두 축을 직접 조작한다.

1. **공간/경로 복잡도**: map size `3×3`, `5×5`, `7×7`
2. **의미 있는 도구 분기**: tool choices `4`, `8` (`action_count`는 이동 행동을 포함해 `8`, `12`)

주 질문은 각 축이 커질수록 `Imagination v2 − DQN` unseen 성공률 차이가 증가하는지다.

## 환경

ToolGrid에는 네 개 이동 행동과 여러 tool 행동이 있다.

- `move_north`, `move_south`, `move_west`, `move_east`
- action count 8: 이동 4개 + tool 4개
- action count 12: 이동 4개 + tool 8개

두 행동 종류를 매 순간 함께 노출하지 않는다.

- 이동 중: 격자 안에 있고 아직 방문하지 않은 이동 행동만 노출
- station 도달 후: 이동 행동을 숨기고 모든 tool 후보만 노출

따라서 map size는 공간 탐색 부담을, tool 수는 station에서의 의미 있는 선택 분기를 독립적으로 조작한다. 맵마다 station 하나와 그 station에 필요한 tool 하나가 생성된다. 에이전트는 station까지 이동한 뒤 올바른 tool을 사용하면 성공한다. 이미 방문한 칸은 다시 밟을 수 없고, 이동 가능한 칸을 모두 소진하거나 잘못된 tool을 쓰면 비가역 실패한다.

추가 tool은 영구적으로 쓸모없는 함정 버튼이 아니다. map seed에 따라 모든 tool이 다른 맵에서는 정답으로 등장한다. 집계기는 train/unseen pool에서 실제 등장한 required-tool ID를 파싱하며, 하나라도 누락되면 결과 생성을 중단한다.

station 수를 1로 고정한 이유는 이번 파일럿에서 **맵 크기와 tool branching만 격리**하기 위해서다. 초기 2-stage smoke에서는 어려운 셀의 최종 성공이 너무 적어 GRU critic이 성공·실패 양쪽 사례를 보지 못했고 Imagination 사용률이 0%가 되는 바닥 효과가 확인됐다. dependency depth는 이번 파일럿 결과 뒤 별도 제3요인으로 추가한다.

## 수정된 동결 조건

- 외부 보상: 최종 성공 `1`, 그 외 `0`
- 환경 자체에는 인공 step/tick 제한 없음
- 성공 또는 환경 자체의 비가역 실패로 종료
- station 수: 1
- 같은 seed·factorial cell에서 모든 비교 조건은 같은 train/unseen map 사용
- train/unseen map seed 범위 분리
- real transition budget: 정확히 `5,000`
- 체크포인트: `0`, `5,000`
- `2,500` 중간 체크포인트는 제거한다. 중간에서 진행 중인 episode를 자르면 후반 학습 궤적에 인공 개입이 되기 때문이다.
- 최종 budget 경계에서 끝나지 않은 episode는 성공/실패로 기록하지 않고 `budget_checkpoint`로 표시하며 episodic return/critic buffer를 폐기한다.
- `training_wall_seconds`에는 평가 시간이 포함되지 않는다.

## 비교 모델과 matched-control 규칙

1. DQN
2. Neural Policy-only
3. Imagination v2

Neural Policy-only와 Imagination v2는 다음 학습 구성요소를 모두 동일하게 가진다.

- 동일한 DQN policy
- 동일한 Neural Delta Prophecy
- 동일한 holdout calibration
- 동일한 GRU branch critic
- 동일한 real transitions 및 random seed

차이는 최종 평가에서 terminal-choice Imagination을 사용할지뿐이다. **학습 중 Imagination 개입은 두 조건 모두 0회**여야 하며, 집계기는 두 조건의 training map, action 결과, step 수, termination 및 누적 transition이 행 단위로 완전히 같은지 검사한다. 하나라도 다르면 factorial 결과를 생성하지 않는다.

## 실험 결과로 수정된 모델 동작

### Prophecy 표현

- 고정 action vocabulary는 signed hash 대신 collision-free one-hot identity로 표현
- required-tool ID는 ordinal scalar 대신 categorical one-hot으로 표현
- categorical 표현은 Prophecy 내부에서만 사용하고 DQN/critic의 frozen raw observation은 유지

### Prophecy 학습과 calibration

- real transition은 replay에 정확히 한 번만 저장
- minibatch는 `(action identity, nonterminal/success/failure)` strata를 균등 샘플링
- pre-ready confidence `0`을 cache하지 않음
- calibration은 nonterminal, terminal success, terminal failure를 구분
- predicted available-action set까지 구조 검증에 포함

### DQN

- Bellman target의 `max Q(s', a')`는 다음 상태에서 실제 available action만 대상으로 계산
- terminal state의 bootstrap value는 정확히 0

### Imagination

- 학습 중에는 항상 꺼서 matched policy trajectory를 보장
- 평가 시 critic이 준비되고, Prophecy가 현재 모든 후보 action을 terminal successor로 예측한 결정에서만 실행
- action 이름, ToolGrid phase, 정답 tool, 환경 oracle은 gate에 사용하지 않음
- 내부 coverage 및 intervention-margin 검사를 추가로 통과해야 실제 action을 변경

## 파일럿 규모

- 독립 seed: `7, 21, 42`
- map size: `3, 5, 7`
- action count: `8, 12`
- 조건: 3개
- 총 cell: `3 × 3 × 2 × 3 = 54`
- seed·cell당 real transition budget: 정확히 `5,000`
- training maps: 48
- unseen maps: 100
- checkpoints: `0, 5,000`

## 조작 및 프로토콜 점검

최종 성능을 보기 전에 다음이 만족되어야 한다.

1. 평균 oracle shortest steps가 `3×3 < 5×5 < 7×7`
2. station의 semantic branching factor가 `4 < 8`
3. 모든 tool ID가 각 unseen map pool에서 실제 정답으로 등장
4. 모든 cell의 actual training transitions가 정확히 `5,000`
5. Imagination v2 training rows의 `imagination_runs` 합이 `0`
6. Neural Policy-only와 Imagination v2의 training trajectory가 완전히 동일
7. DQN 성공률이 모든 셀에서 0 또는 1로 포화되지 않음
8. 최종 평가에서 Imagination 사용률, imagined-node 비용, action-change 비율을 함께 보고

1–6 중 하나라도 실패하면 aggregate job은 오류를 내고 통계 결과를 생성하지 않는다.

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
