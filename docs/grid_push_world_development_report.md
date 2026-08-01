# AASSR Grid-Push World Development Diagnostic 보고서

이 문서는 `paper-grid-push-development-v2.0/fourth` 실행을 권위 있는 Development
Diagnostic으로 삼는다. 이 실행은 코드 구현 커밋
`eaa2de89e12dede0afde6906de19b5396a1f7d56`에서 수행했다. 결과는 개발 증거일
뿐이며 Locked Confirmation, Pilot, Final 연구 증거가 아니다. 이전 Development 실행
`first`, `second`, `third`도 실패를 덮어쓰지 않고 별도 디렉터리에 보존했다.

## 1. 브랜치와 기준 커밋

- 새 브랜치: `codex/aassr-v2-grid-push-world`
- 기준 브랜치: `codex/aassr-v2-pr7-transfer-creativity`
- 기준 커밋: `fa7a63f`
- 구현 커밋: `eaa2de89e12dede0afde6906de19b5396a1f7d56`
- 권위 있는 Development 실행: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth`

`git merge-base codex/aassr-v2-grid-push-world fa7a63f`와 ancestry 검사로 새 브랜치가
지정 커밋에서 이어졌음을 확인했다. 원격 push는 수행하지 않았다.

## 2. Minecraft 격리

`rg --files | rg -i minecraft`는 이 브랜치에서 결과를 반환하지 않는다. 따라서
Minecraft world, diagnostic, 전용 script/config/lock/result/test/backend 문서는 새
브랜치에 없다. 과거 `codex/aassr-v2-pr8-minecraft` 브랜치는 삭제하거나 변경하지
않았다.

## 3. 실제 생성 격자 5개

범례는 `#` 벽, `.` 바닥, `A` 플레이어, `B` 블록, `O` 구덩이, `P` 압력판,
`D` 닫힌 문, `G` 목표다. 아래 내용은
`fourth/world_examples.json`에서 그대로 옮겼다.

### world 76001

```text
########
#ABD...#
#P.#...#
#.B##.G#
#..#...#
#..O...#
########
```

최소 13행동, 해 1개, 구조 해 1개, bounded dead end 8개, Random 성공 추정 0.00.

### world 76002

```text
#######
#..D..#
#PB####
#..O.G#
#B.#..#
#A.#..#
#######
```

최소 14행동, 해 5개, 구조 해 5개, bounded dead end 85개, Random 성공 추정 0.00.

### world 76003

```text
#######
#...O.#
#.B.D.#
#...#G#
#AB.#.#
#.P##.#
#######
```

최소 9행동, 해 25개, 구조 해 18개, bounded dead end 308개, Random 성공 추정 0.00.

### world 76004

```text
#######
#...O.#
#.A.D.#
#PB.#.#
#.B.#G#
#...#.#
#######
```

최소 9행동, 해 27개, 구조 해 16개, bounded dead end 395개, Random 성공 추정 0.00.

### world 76005

```text
#########
#..#..DG#
#.B#..#.#
#ABO.P#.#
#..#..#.#
#..#..#.#
#########
```

최소 16행동, 해 1개, 구조 해 1개, bounded dead end 80개, Random 성공 추정 0.00.

## 4. 20개 격자의 검사기 최소 경로 길이

월드 seed 76001부터 76020까지의 최소 행동 수는 순서대로 다음과 같다.

```text
13, 14, 9, 9, 16, 9, 4, 14, 12, 17,
9, 17, 15, 7, 17, 13, 16, 8, 9, 15
```

20/20이 인증을 통과했고 solver 탐색 truncation은 0/20이었다. 모두 블록 push가
필요하고 bounded dead end가 8개 이상이며, Random 성공 추정의 최댓값은 0.10이다.
12/20 월드에는 구조적으로 다른 해가 2개 이상 있었다. 구덩이와 문 메커니즘을
모두 요구한다고 검사기가 판정한 월드는 9/20이었다. 각 동결 기준은
`fourth/manifests/solver_references/world_*.json`에 있다.

## 5. 검사기 경로가 agent 입력에 들어가지 않는 코드 근거

`GridPushWorld.observe()`는 `RawCausalObservation`만 만든다. 이 값에는 inventory,
observable facts, 네 개의 일반 이동 action과 affordance, resource/health/damage,
관측된 cell, 직전 행동 성공, terminal reward와 terminal 여부만 있다. private
`plate_links`, solver path, minimum actions, 역할, goal distance/progress는 직렬화하지
않는다. 검사기와 인증은 `solve_grid_world()`와 `certify_grid_world()`의 별도 분석
경로에 있고 agent runner는 `spec`으로 world를 다시 생성해 `world.observe()`만
`choose`에 전달한다.

실행 중 private leak 검사는 0건이었고 agent checkpoint 문자열 검사에도
`plate_links`, `solver_reference`, `minimum_actions`, `correct_path`, `goal_distance`,
`goal_progress`, `viability`가 없었다. Solver reference 20개는 agent 시작 시각보다
먼저 동결됐고 각 hash가 재검증됐다.

## 6. 에이전트가 실제 받은 관측 예시

다음은 gzip trace의 실제 `full_aassr` agent episode 첫 관측을 축약한 것이다.

```json
{
  "inventory": {},
  "observable_facts": [],
  "available_actions": [
    "MOVE_NORTH", "MOVE_SOUTH", "MOVE_WEST", "MOVE_EAST"
  ],
  "action_affordances": {
    "MOVE_NORTH": ["move", "north"],
    "MOVE_SOUTH": ["move", "south"],
    "MOVE_WEST": ["move", "west"],
    "MOVE_EAST": ["move", "east"]
  },
  "spatial_observations": {
    "width": 8,
    "height": 7,
    "observation_mode": "full_map",
    "cell:1,1": "floor+player",
    "cell:2,1": "floor+block",
    "cell:3,1": "door_closed",
    "cell:1,2": "pressure_plate",
    "cell:3,5": "pit",
    "cell:6,3": "goal"
  },
  "last_action_succeeded": null,
  "terminal_reward": 0.0,
  "terminal": false
}
```

원본 전체 관측은 `fourth/raw/trace.jsonl.gz`의 `record_type=agent_episode`,
`condition=full_aassr` 레코드에 있다. action은 네 방향 이동뿐이고 push는 블록을
향해 이동할 때 공통 물리법칙으로 발생한다.

## 7. AASSR 실제 호출 경로

실행 경로는 다음과 같다.

```text
scripts/run_grid_push_development.py
  -> run_grid_push_development()
  -> run_small_grid_diagnostic()
  -> CausalAASSRAgent(GridRelationalEffectEncoder)
  -> CausalImaginationPlanner(LearnedReturnModel(EmpiricalCausalProphecy))
  -> planner.decide(observation, policy)
  -> GridPushWorld.step(final_selected_action)
  -> grid_observable_transition()
  -> CausalAASSRAgent.observe_transition()       # training only
  -> CausalAASSRAgent.finish_episode()
```

학습 종료 후 checkpoint를 fresh clone에 복원해 frozen 평가했다. 세 연구 seed 모두
Full checkpoint의 평가 전후 SHA-256이 동일했고 평가 학습 호출은 0회였다. 실제
decision 레코드는 gzip trace에 기록됐다. 이번 소형 실행에서는 gate가 한 번도
개입을 허용하지 않아 policy-only action이 최종 action으로 유지됐다.

## 8. 블록 밀기 전후 trace

`fourth/physics_traces.json`의 block-push probe에서 seed 76001의 solver 행동 중
step 2와 3에 `block_moved`가 발생한다. 공통 `step()`은 이동 앞 칸에 블록이 있으면
그 뒤가 통과 가능할 때만 블록을 한 칸 옮기고 플레이어를 블록의 이전 위치로
옮긴다. 벽, 닫힌 문, 다른 블록 뒤에서는 상태를 바꾸지 않는다. pull과 두 블록
동시 push용 action/API는 없다.

## 9. 압력판과 문 상태 변화 trace

명시적 physics probe 행동은 `MOVE_EAST, MOVE_EAST, MOVE_SOUTH`다.

```text
초기             첫 push 후        둘째 push 후       player가 plate 이탈
######           ######            ######              ######
#...D#           #...d#            #...d#              #...D#
#ABP.#           #.Ab.#            #..AB#              #..PB#
#...G#           #...G#            #...G#              #..AG#
######           ######            ######              ######
```

여기서 `b`는 plate 위 block, `d`는 열린 문이다. 첫 step event는
`block_moved, player_moved, plate_pressed, door_opened`; 둘째 step에서는 block이
plate를 떠나도 player가 plate 위라 문이 열린 채다. 셋째 step에서
`player_moved, plate_released, door_closed`가 기록됐다. 숨은 plate-door link는
관측에 없고 문 cell의 변화만 보인다.

## 10. 구덩이 메움 trace

명시적 pit probe에서 `MOVE_EAST` 한 번의 전후 상태다.

```text
before           after
######           ######
#....#           #....#
#ABO.#           #.A=.#
#...G#           #...G#
######           ######
```

event는 `block_moved, pit_filled, player_moved`다. `=`는 메워져 통과 가능한 pit이고
원래 block은 사라진다. 역변환 API가 없어 이 변화는 불가역이다.

## 11. Relational key 불연속 수정 전후

먼저 `test_unknown_action_value_survives_learned_effect_key_transition`을 추가했다.
수정 전에는 첫 action을 `unknown:<token hash>`에 기록한 뒤 effect를 학습하면
`effect:<profile>`로 조회해 다음 assertion이 실제 실패했다.

```text
assert agent.q_value(transition.before, transition.action) == 1.0
E assert 0.0 == 1.0
```

수정은 private effect label이 아니라 관측 transition으로 얻은 old/new key를
`RepresentedReturnAgent._migrate_key()`에서 병합하고 episode credit key도 바꾼다.
Relational state key에서도 동적으로 변하는 learned profile을 제거했다. 수정 후 같은
테스트가 통과했고 최종 Full 실행에서 seed별 30, 36, 36회의 실제 key migration이
기록됐다. 즉 첫 기록의 가치가 learned-effect 표현으로 이어졌다.

## 12. Random, Contextual, Full AASSR 소형 결과

research seed는 9101, 9103, 9109 세 개다. 각 학습 조건은 150 episode, 각 조건의
frozen 평가는 seed당 20 episode, 최대 35 step이며 첫 5개 인증 world를 같은
schedule로 사용했다.

| 조건 | training final-tail success | frozen success | 평균 step | 실패 action 비율 | 평균 block move | imagination 개입 비율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.0000 | 0.0000 | 35.0000 | 0.3890 | 1.5500 | 0.0000 |
| Contextual Policy | 0.1133 | 0.1333 | 31.8000 | 0.9204 | 0.5333 | 0.0000 |
| Full AASSR | 0.1000 | 0.1333 | 31.9333 | 0.9165 | 0.6000 | 0.0000 |

Full의 평균 Prophecy terminal-return Brier score는 0.00822, observable-effect error는
0.20709였다. 그러나 성공률은 Contextual과 동일하고, Imagination은 0회 개입했으므로
Full 또는 Imagination이 더 좋다는 증거가 아니다. 실패 action 비율도 약 0.92로 매우
높아 아직 적절한 sparse-reward 학습 성능이라고 볼 수 없다.

## 13. 검사기와 다른 행동열의 창의성 계산 예시

world 76002의 사후 solver alternative는 첫 frozen reference의 action list와 실제로
달랐다. 그러나 정규화된 인과 graph가 reference와 동일했다.

- source: `posthoc_solver_alternative_example_not_agent_evidence`
- success: true
- action list differs: true
- effect sequence: `door_becomes_passable -> block_enters_pit -> pit_becomes_passable -> traverse_filled_pit -> goal_reached`
- graph-edit, motif-Jaccard, prerequisite-edge, family, sequence novelty: 모두 0.0
- novelty: 0.0
- utility: 0.79167
- reproducibility: 0.0
- creativity: 0.0
- 탈락: `novelty_not_above_frozen_threshold`, `utility_below_threshold`, `not_reproduced`

즉 단지 action 문자열/길이가 다른 해결법을 창의적으로 세지 않았다. 성공한 agent
episode 16개를 사후 채점했으나 최종 창의 후보는 0개였다. 이 결과는 미화하지 않으며
현재 AASSR가 새로운 유효 전략을 찾았다는 증거가 없다.

## 14. 전체 테스트와 artifact 무결성

최종 소스에서 실행한 전체 테스트 결과는 다음과 같다.

```text
149 passed, 3 skipped in 56.18s
```

Docker Desktop 경로를 PATH에 포함해 기존 Docker 안전 테스트까지 실행했다. 추가로
최종 실행은 180 episode row와 11,212 gzip trace row를 기록했고 gzip 전체 재생,
CSV/trace/world/physics artifact SHA-256, config hash, seed commitment, causal-law hash,
20개 solver reference hash를 통과했다. 11개 engineering gate가 모두 true다.

## 15. 아직 작동하지 않거나 입증되지 않은 부분

- Locked Confirmation, Pilot, Final은 만들거나 실행하지 않았다.
- Full AASSR는 이 진단에서 Contextual보다 낫지 않았고 Imagination 개입은 0회였다.
- agent가 성공한 16개 전략 중 frozen reference 밖의 최종 창의 후보는 0개였다.
- 평균 실패 action 비율이 Contextual 0.9204, Full 0.9165로 높다.
- 제한 시야 관측 모드는 인터페이스로 구현했지만 이번 진단은 full-map만 실행했다.
- generator는 randomized dual-route와 combined multiroom grammar를 사용한다. 20개
  layout은 달랐지만 임의 room topology 전체를 생성하는 범용 dungeon generator는
  아직 아니다.
- 20개 중 8개는 인증 해 집합에서 구조적으로 다른 해가 2개 미만이었다. 전체 suite가
  모든 개별 월드에 복수 구조 해를 보장하지는 않는다.
- solver는 bounded 탐색이다. 이번 20개는 truncation 0건이지만 더 큰 grid에 대한
  완전성은 입증하지 않았다.
- 실제 인간 비교와 인간 창의성 평가는 없다.
- 이 Development 결과는 성능 가설의 연구 증거로 사용할 수 없다.

## 실행 artifact 색인

- 요약 manifest: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/manifests/protocol_manifest.json`
- 실제 관측/decision/episode trace: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/raw/trace.jsonl.gz`
- episode CSV: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/raw/episodes.csv`
- 물리 trace: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/physics_traces.json`
- 5개 world 예: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/world_examples.json`
- 조건별 수치: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/condition_summary.csv`
- 창의성 계산: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/creativity_evaluation.json`
- 동결 solver reference: `paper_results_v2/development/paper-grid-push-development-v2.0/fourth/manifests/solver_references/`
