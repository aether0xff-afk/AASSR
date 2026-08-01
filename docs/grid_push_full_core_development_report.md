# GridPush `AASSRCore` 통합 Development 보고서

## 판정

요청한 공통 코어 통합과 GridPush plugin 연결은 완료됐다. 전체 모듈 호출·학습
갱신 engineering probe, 20개 월드 사전 인증, fresh-clone frozen 평가, checkpoint 및
trace 누출 검사를 통과했다.

그러나 생성 월드 소형 Development 성능은 training final-tail과 frozen 평가 모두
0%였다. 이번 결과는 통합 무결성만 입증하며 AASSR 성능 가설은 지지하지 않는다.
Locked Confirmation, Pilot, Final은 실행하지 않았다.

권위 있는 완주 실행은 다음이다.

```text
paper_results_v2/development/paper-grid-push-core-development-v2.1/third
```

구현 커밋은 `ac42f37`, trace 직렬화 수정 커밋은 `3981b41`이다.

## 공통 실행 구조

`AASSRCore`가 다음 기존 모듈 인스턴스를 한 lifecycle에서 직접 소유하고 연결한다.

- `KnowledgeStore`
- `OnlineFeatureMemory`
- `GoalSet`과 terminal `GOAL`
- `WeightedPolicy`
- `TabularProphecy`와 `SkillAwareProphecy`
- `ImaginationTree`
- `AdvancedTransitionEvaluator`
- `ReplayBuffer`와 `PredictionValidator` holdout
- `InformationValuePredictor`
- `DelayedCreditAssigner`
- `SkillLibrary`

각 primitive transition은 다음 하나의 환경 중립 경로를 지난다.

```text
EnvironmentPlugin raw observation
  -> CoreObservationEncoder
  -> Knowledge / FeatureMemory / GOAL query
  -> Policy + ImaginationTree
  -> plugin.execute(primitive action)
  -> AdvancedTransitionEvaluator
  -> Knowledge / Prophecy / Replay-Holdout update
  -> OnlineFeatureMemory update
  -> terminal GOAL / SkillLibrary
  -> DelayedCreditAssigner
  -> InformationValuePredictor / Policy update
```

환경별 학습 loop는 없다. `AASSRCore.run_episode()`가 training과 frozen evaluation의
공통 lifecycle을 소유한다.

## EnvironmentPlugin과 GridPush 경계

공통 `EnvironmentPlugin`은 reset, raw observation, `ActionSchema`, primitive action
execution, observable transition, terminal/final sparse reward, rendering만 정의한다.
`ObservableEnvironmentTransition`은 non-terminal reward가 0이 아니면 생성 단계에서
거부한다.

`GridPushEnvironmentPlugin`의 action schema는 다음 네 개뿐이다.

```text
MOVE_NORTH
MOVE_SOUTH
MOVE_WEST
MOVE_EAST
```

별도 PUSH, solver, optimal action, goal distance, progress, block role, solution family,
plate-door link API는 없다. Push와 plate/door/pit 변화는 `GridPushWorld`의 공통 물리로
발생하고 plugin은 agent-visible before/after observation만 반환한다.

`CoreObservationEncoder`는 raw observation의 visible token만 hash encoding하고 모든
상태에서 `StateSnapshot.goal_progress = 0.0`으로 둔다. 성공은 terminal 이후
`terminal_reward=1`과 `terminal_success` fact로만 관측한다.

## Solver/runtime 분리

- Runtime 물리: `grid_push_world.py`
- Analysis-only solver/generator/certifier: `grid_push_solver.py`
- Runtime adapter: `grid_push_plugin.py`

Plugin source는 solver를 import하지 않는다. fresh Python process에서 plugin을 import한
직후 `aassr_v2.grid_push_solver in sys.modules`는 `False`였다.

완주 실행에서는 solver가 20개 월드를 먼저 인증하고 reference를 배타적으로 동결했다.
그 뒤 solver result 객체를 폐기하고 core와 plugin을 생성했다. Core checkpoint와 raw
trace에서 다음 private/analysis 문자열 출현 수는 모두 0이었다.

```text
plate_links, solver_reference, minimum_actions, optimal_action,
correct_path, goal_distance, goal_progress_delta, block_role,
solution_family, viability
```

## 전체 모듈 호출 probe

Probe는 성공이 보장되는 한 칸짜리 GridPush engineering fixture를 사용한다. 이는 각
모듈의 정상 호출과 update 경로를 검사하기 위한 것이며 성능 증거가 아니다. Solver
행동이나 정답 action은 core에 전달하지 않았다. 6개 training episode 모두 terminal
성공했고 holdout 1개와 skill 1개가 실제 생성됐다.

| 모듈 | 실제 호출 | persistent 학습 갱신 | 비학습 work unit |
| --- | ---: | ---: | ---: |
| KnowledgeStore | 12 | 6 | 0 |
| OnlineFeatureMemory | 29 | 12 | 0 |
| GOAL | 18 | 0 | 6 achievement events |
| Policy | 12 | 6 | 0 |
| Prophecy | 24 | 5 | 0 |
| ImaginationTree | 6 | 0 | 12 imagined nodes |
| AdvancedTransitionEvaluator | 6 | 6 | 0 |
| Replay | 6 | 6 | 0 |
| Holdout | 12 | 1 | 0 |
| InformationValuePredictor | 12 | 6 | 0 |
| DelayedCreditAssigner | 6 | 0 | 6 credit assignments |
| SkillLibrary | 12 | 6 | 0 |

GOAL, ImaginationTree, DelayedCreditAssigner는 학습 파라미터를 가진 모듈로 가장하지
않고 achievement, imagined node, credit assignment를 별도 work unit으로 기록한다.
나머지 trainable 모듈은 모두 persistent update가 0보다 컸다.

## Frozen clone 결과

Probe checkpoint fingerprint는 평가 전후 모두 다음 값으로 동일했다.

```text
05ec554b1ca9f30027b914c6d009dfa0d2bc02ae60f0395db95c0a9321576e84
```

Frozen probe에서는 evaluator, Knowledge, FeatureMemory, GOAL, Policy, Prophecy,
ImaginationTree, Replay/Holdout, InformationValuePredictor, SkillLibrary의 inference
호출이 실제 발생했다. 모든 모듈의 `learning_updates`는 정확히 0이었다.
DelayedCreditAssigner는 frozen 평가에서 학습 종료 credit을 만들지 않으므로 호출하지
않았다.

생성 월드의 세 research seed에서도 전체 frozen 평가 전후 checkpoint hash가 각각
일치했고 모든 learning update가 0이었다.

## 생성 월드 Development 성능

연구 seed `9201, 9203, 9209`마다 인증된 train world 3개에서 40 episode를 학습하고
fresh checkpoint clone으로 6 episode를 frozen 평가했다. episode당 최대 30 primitive
step이다.

| Research seed | Training final-tail | Frozen success | Training/Frozen 평균 step | Training imagination 사용률 | Frozen imagination 사용률 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9201 | 0.0000 | 0.0000 | 30 / 30 | 0.5608 | 1.0000 |
| 9203 | 0.0000 | 0.0000 | 30 / 30 | 0.5825 | 1.0000 |
| 9209 | 0.0000 | 0.0000 | 30 / 30 | 0.5567 | 1.0000 |
| 평균 | 0.0000 | 0.0000 | 30 / 30 | 0.5667 | 1.0000 |

생성 월드 training은 seed당 실제 transition 1,200개였다. 각 seed에서 Prophecy 960회,
holdout 240회, Policy 1,200회, InformationValuePredictor 1,200회가 갱신됐고
ImaginationTree는 seed별 2,672, 2,702, 2,772개의 node를 생성했다. 그럼에도 성공은
한 번도 없어서 GOAL achievement와 SkillLibrary 갱신은 생성 월드에서는 0이었다.

따라서 "전체 모듈이 호출됐다"와 "전체 AASSR가 이 환경을 학습했다"는 별개다.
현재 결과는 전자는 만족하지만 후자의 성능 증거는 실패다. Frozen imagination 사용률
100% 역시 유효한 계획을 의미하지 않으며 성공률 0%와 함께 해석해야 한다.

## 조건 명칭

새 plugin/core 경로의 episode만 `full_aassr`로 기록했다. 완주 trace의 condition
집합은 정확히 `{full_aassr}`다.

기존 GridPush 전용 `CausalAASSRAgent` 경로는 삭제하지 않고
`reduced_causal_agent`로 명칭을 변경했다. 과거 `paper-grid-push-development-v2.0`
artifact의 당시 문자열은 불변 보존했으며 사후 수정하지 않았다.

## 실행 실패 보존

- `first`: shell의 10초 외부 timeout으로 인증 단계 후 중단됐다.
- `second`: module probe trace 직렬화 중 immutable `mappingproxy` deepcopy 오류가
  발생했다.
- `third`: 직렬화 수정 커밋 `3981b41`에서 완주했다.

`first`와 `second`의 claim과 이미 생성된 20개 solver reference는 덮어쓰지 않았다.
각 디렉터리에 별도 `failure.json`을 추가했다.

## Artifact 무결성

- 인증 월드: 20/20
- episode CSV: 138행
- gzip trace: 145행
- gzip 완전 재생: 통과
- config hash: 일치
- CSV/trace/audit/seed-summary SHA-256: 모두 일치
- frozen solver reference: 20개, 내부 hash 모두 일치
- raw trace private term: 0건
- non-terminal environment reward violation: 0건
- engineering gate: 전부 통과
- Locked Confirmation/Pilot/Final: 모두 미실행

## 테스트

구현 커밋에서 전체 저장소 테스트 결과는 다음과 같다.

```text
156 passed, 3 skipped
```

Docker Desktop 실행 경로를 포함해 기존 Docker 안전 테스트도 함께 실행했다. 이후
발견된 trace 직렬화 결함에는 실제 `AASSREpisodeRecord.to_dict()` regression assertion을
추가했다.

## 남은 한계

- 생성 월드 성공률은 training과 frozen 모두 0%다.
- 생성 월드에서는 성공이 없어 GOAL achievement와 Skill 생성이 일어나지 않았다.
- 현재 concrete `EnvironmentPlugin` 구현은 우선 요청된 GridPush 하나다. 기존 legacy
  environment runner는 호환성 보존을 위해 삭제하거나 새 코어로 강제 전환하지 않았다.
- Prophecy는 현재 전체 코어의 기존 tabular implementation이다. 신경 모델 성능은
  이번 범위에서 검증하지 않았다.
- Checkpoint는 저장소 내부에서 생성한 trusted local pickle만 복원하도록 설계했다.
  외부의 신뢰할 수 없는 checkpoint를 로드하면 안 된다.
- 이 결과는 Development engineering evidence이며 연구 성능 주장에 사용할 수 없다.

## 산출물

- `paper_results_v2/development/paper-grid-push-core-development-v2.1/third/manifests/protocol_manifest.json`
- `paper_results_v2/development/paper-grid-push-core-development-v2.1/third/module_call_audit.json`
- `paper_results_v2/development/paper-grid-push-core-development-v2.1/third/research_seed_summary.json`
- `paper_results_v2/development/paper-grid-push-core-development-v2.1/third/raw/episodes.csv`
- `paper_results_v2/development/paper-grid-push-core-development-v2.1/third/raw/trace.jsonl.gz`
- `paper_results_v2/development/paper-grid-push-core-development-v2.1/third/manifests/solver_references/`
