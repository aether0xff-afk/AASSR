# ToolGrid map-size × semantic-branching factorial pilot

## 목적

기존 final complexity scaling은 사실상 최단 성공 경로 길이 중심이었고 DQN의 실제 난이도는 단조 증가하지 않았다. 이번 파일럿은 복잡도를 하나의 Level로 합치지 않고 다음 두 축을 직접 조작한다.

1. **공간/경로 복잡도**: map size `3×3`, `5×5`, `7×7`
2. **의미적 선택 분기**: tool choices `4`, `8` (`action_count`는 이동 4개를 포함해 `8`, `12`)

주 질문은 각 축이 커질수록 `Imagination v2 − DQN` unseen 성공률 차이가 증가하는지다.

## 환경

ToolGrid에는 네 이동 행동과 여러 tool 행동이 있다.

- 이동 중에는 격자 안의 미방문 이동 행동만 노출한다.
- station 도달 후에는 이동을 숨기고 모든 tool 후보만 노출한다.
- 올바른 tool을 사용하면 성공하고, 잘못된 tool 사용 또는 이동 경로 소진은 비가역 실패다.
- 추가 tool은 영구적인 가짜 버튼이 아니라 다른 map에서는 실제 정답으로 등장한다.
- 외부 보상은 최종 성공 `1`, 그 외 `0`이다.
- 환경 자체에는 인공 step/tick limit가 없다.

station 수는 1로 고정한다. 이번 파일럿에서는 map size와 semantic branching만 격리하고 dependency depth는 후속 제3요인으로 남긴다.

## 숨은 비교 버그와 프로토콜 교정

초기 수정판은 `neural_policy_only`와 `imagination_v2`를 같은 seed로 별도 GitHub runner에서 각각 다시 학습했다. 학습 중 Imagination 실행은 0회였지만, seed 42·7×7·8-tool cell에서 두 training stream은 1,060 transitions까지 같다가 이후 갈라졌다.

- Policy-only training rows: 287
- Imagination training rows: 281
- 최초 불일치: episode 56, map seed `420712008`

이는 긴 신경망 학습에서 runner별 수치 차이가 행동 선택으로 증폭된 것이다. 따라서 **같은 seed와 같은 구성은 같은 checkpoint를 보장하지 않는다.** 해당 run의 aggregate validator가 실패한 것은 올바른 동작이며, 그 결과는 최종 통계로 사용하지 않는다.

교정된 프로토콜은 다음과 같다.

1. 각 seed × map size × action count마다 hybrid agent를 정확히 한 번만 학습한다.
2. 학습 중 Imagination intervention은 항상 0회다.
3. 학습이 끝난 동일 객체와 동일 model state를 두 번 평가한다.
   - `use_imagination=False`: Neural Policy-only
   - `use_imagination=True`: Imagination v2
4. 두 평가 사이에는 재초기화, checkpoint reload, 추가 학습, 별도 runner가 없다.
5. training CSV의 두 condition 행은 하나의 실제 training stream을 가리키는 읽기 전용 paired view다.
6. aggregate validator는 두 view의 map, episode, success, steps, termination, 누적 transition이 행 단위로 완전히 같은지 확인한다.

이제 Policy-only와 Imagination의 차이는 **동일 checkpoint에서 planner를 켰는지 여부 하나뿐**이다.

## 수정된 production 동작

### Prophecy

- action identity: signed hash → collision-free one-hot
- required-tool identity: ordinal scalar → categorical one-hot
- real transition은 replay에 한 번만 저장
- minibatch는 `(action identity, nonterminal/success/failure)` strata에서 균형 샘플링
- minimum-count 이전 confidence 0은 cache하지 않음
- calibration은 nonterminal, terminal success, terminal failure와 available-action set을 구분

### DQN

- Bellman target은 다음 상태에서 실행 가능한 action만 대상으로 `max Q` 계산
- terminal bootstrap은 정확히 0

### Imagination

- 학습 중 intervention 0회
- 평가 시 critic이 준비되고 Prophecy가 모든 현재 후보를 terminal successor로 예측할 때만 실행
- action 이름, ToolGrid phase, 정답 tool, oracle은 gate에 사용하지 않음
- coverage와 intervention-margin을 통과해야 실제 정책 행동을 변경

## 동결 조건

- real transition budget: 정확히 `5,000`
- checkpoints: `0`, `5,000`
- 최종 budget 경계의 미완료 episode는 `budget_checkpoint`로 기록하고 episodic return/critic buffer를 실패 사례로 학습하지 않음
- `training_wall_seconds`는 training segment 시간만 포함
- 같은 cell의 DQN과 shared hybrid는 동일 train/unseen map seed 사용
- train/unseen map seed 범위 분리

## 파일럿 규모

- 독립 seed: `7, 21, 42`
- map size: `3, 5, 7`
- action count: `8, 12`
- 실제 학습 job:
  - DQN: `3 × 3 × 2 = 18`
  - shared hybrid: `3 × 3 × 2 = 18`
  - 합계: **36 training jobs**
- 최종 평가 condition cell은 DQN / Policy-only / Imagination v2의 **54개**
- training maps: 48
- unseen maps: 100

## 조작 및 프로토콜 점검

통계를 생성하기 전에 다음을 검사한다.

1. 평균 oracle shortest steps가 `3×3 < 5×5 < 7×7`
2. station semantic branching factor가 `4 < 8`
3. 모든 tool ID가 unseen pool에서 실제 정답으로 등장
4. 모든 final checkpoint의 actual transitions가 정확히 `5,000`
5. shared hybrid의 training-time `imagination_runs` 합이 0
6. Policy-only와 Imagination training views가 행 단위로 완전히 동일
7. 두 평가가 같은 checkpoint, 같은 map, 같은 episode index를 사용
8. 최종 평가의 Imagination 사용률과 imagined-node 비용을 함께 보고

1–7 중 하나라도 실패하면 aggregate는 통계 결과를 만들지 않는다.

## 분석

각 seed와 factorial cell에서 unseen 성공률을 계산한 뒤

```text
Delta = success(Imagination v2) - success(DQN)
```

에 대해 seed별 모형을 적합한다.

```text
Delta = b0 + b_size·map_size + b_branch·tool_count
        + b_interaction·map_size·tool_count
```

- `b_size > 0`: 맵이 커질수록 Imagination 상대 가치 증가
- `b_branch > 0`: semantic tool 가지 수가 늘수록 상대 가치 증가
- `b_interaction > 0`: 두 요인이 함께 커질 때 추가 상승

3 seeds는 방향성을 보는 파일럿이다. 유망한 축만 20 seeds 이상의 확증 실험으로 확대한다.
