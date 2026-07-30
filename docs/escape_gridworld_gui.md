# Colored-key Escape GridWorld GUI

## 목적

기존 불투명 이진 의존 사슬보다 공간 탐색과 물체 의존성을 가진 작은 환경에서 AASSR의 온라인 학습 과정을 관찰한다.

환경은 중립 상자, 빨강·파랑·초록 열쇠, 같은 색 열쇠로 여는 문, 출구, 상하좌우 이동과 `interact`만 사용한다. 상자 내용은 열기 전에는 노출되지 않고 문 색은 관측할 수 있다.

## 생성 규칙

`generate_escape_grid()`는 색 수에 맞춰 다음 의존 구조를 만든다.

```text
시작 구역
  └─ 빨간 열쇠 상자
      └─ 빨간 문
          └─ 파란 열쇠 상자
              └─ 파란 문
                  └─ 초록 열쇠 상자
                      └─ 초록 문
                          └─ 출구
```

각 구역의 문과 상자 위치는 seed로 결정되며 빈 미끼 상자를 추가할 수 있다. 생성된 세계는 `oracle_plan()`의 BFS로 해결 가능성과 최단 경로 길이를 검사하지만 oracle 경로는 에이전트에게 제공하지 않는다.

## 에피소드 종료와 점수

에피소드에는 tick 제한이 없다.

```text
출구 도달 전: 계속 진행
출구 도달: 성공 및 에피소드 종료
사용자 중지: 현재 에피소드를 중단 기록으로 남기고 전체 세션 종료
```

상자 열기, 열쇠 획득, 문 개방에는 외부 성공 점수를 주지 않는다. 출구에 도달한 경우에만 다음 성공 점수를 최종 return으로 사용한다.

```text
성공 점수 배수 = 1 + oracle 최단 tick / 실제 성공 tick
```

기본 설정에서 최단 경로 성공은 `2.0x`이고, 실제 경로가 길어질수록 `1.0x`에 가까워진다.

## 실제 학습기

GUI는 기존 코어의 다음 구성을 사용한다.

```text
AutonomousLearningAgent
+ ContextualPolicy
+ TabularProphecy
+ ImaginationTree
+ holdout 기반 validated information gain
+ 반복·오류 감점
```

에이전트는 시범 없이 실제 상호작용으로 전이를 학습한다. 출구 도달 후 계산된 성공 점수가 전체 행동열에 할인 역전파된다.

## GUI 실행

```bash
python -m pip install -e ".[dev]"
python scripts/run_escape_gridworld.py --gui
```

### 하나의 세션에서 속도 전환

두 버튼은 새 세션을 따로 시작하는 버튼이 아니라 실행 중 현재 세션의 표시 모드를 바꾼다.

- `실시간으로 보기`: 모든 primitive step을 렌더링하고 짧은 지연을 둔다.
- `안 보고 최대 속도`: step 렌더링과 인위적 sleep을 제거한다.
- 실행 중 어느 방향으로든 즉시 전환할 수 있다.
- 현재 맵, episode, tick, Policy, Prophecy, Imagination, holdout, RNG 상태는 유지된다.
- 두 모드 모두 모든 step 기록을 디스크에 저장한다.

## 모든 실행 정보 저장

GUI와 headless 실행 모두 기본적으로 다음 경로를 자동 생성한다.

```text
runs/escape_gridworld/<timestamp>_seed<seed>_<session-id>/
```

각 step과 episode 기록은 실행 도중 즉시 flush된다. 따라서 프로세스가 비정상 종료되어도 이미 기록된 줄은 남는다.

```text
session.json                 # 세션 ID, 설정, 상태, 파일 목록
world.json                   # 맵 크기, 벽, 상자, 열쇠, 문, 출구, seed
steps.jsonl                  # 모든 tick의 전체 전후 상태와 학습 지표
episodes.csv                 # 에피소드별 분석용 표
episodes.jsonl               # 에피소드별 전체 구조화 기록
mode_switches.jsonl          # 실시간/최대 속도 전환 시각과 위치
session.log                  # 사람이 읽는 실행 로그
summary.json                 # 최종 세션 요약과 기술통계
summary.txt                  # 간단한 텍스트 요약
statistics.json              # 분포, 사분위수, 상관계수, 행동/이벤트 집계
checkpoints/
  episode_000001.json.gz     # 각 episode 종료 후 전체 에이전트 상태
  ...
  latest.json.gz             # 가장 최근 checkpoint
  final.json.gz              # 종료 시 최종 checkpoint
charts/
  episode_steps.svg
  episode_scores.svg
  episode_duration.svg
  prediction_and_holdout.svg
  intrinsic_value.svg
  imagination_usage.svg
  errors_and_repeats.svg
  action_distribution.svg
  event_distribution.svg
```

### `steps.jsonl`에 저장되는 항목

- UTC 기록 시각
- session 경과시간, episode 경과시간
- tick 전체 소요시간과 계산 소요시간
- episode 번호와 step 번호
- 실행 모드와 epsilon
- 행동의 verb, target, tool, destination, parameters, signature
- 행동 전후 전체 state vector
- 행동 전후 전체 facts
- 행동 전후 available actions
- 상태 metadata와 goal progress
- event, error, 외부 보상, goal 도달 여부
- 추가·삭제된 facts와 새로 열린 actions
- Imagination 사용 여부, node 수, 깊이, root imagined value
- Prediction score
- holdout before, after, gain
- intrinsic value
- 반복 행동 여부

### `episodes.csv/jsonl`에 저장되는 항목

- 시작·종료 UTC 시각
- episode 소요시간과 session 누적시간
- 전체 step 수와 oracle 최단 step 수
- 효율, 점수, 최근 100회 평균 점수
- epsilon
- 이동·상호작용 횟수
- 오류, 반복, blocked 횟수
- 발견한 열쇠, 연 문, 빈 상자 수
- Imagination 호출 수, node 수, 최대 깊이
- Prediction, holdout, intrinsic value 통계
- 실시간·최대 속도에서 소비한 시간
- Policy entry, Prophecy entry, holdout 크기
- 행동별·이벤트별 횟수
- 성공, 중지 여부

### Checkpoint에 저장되는 항목

- Contextual Policy의 state-action value, count, state visits
- 전역 action value
- Tabular Prophecy의 exact/global 전이 횟수와 상태 snapshot
- holdout transition 전체
- agent와 holdout RNG 상태
- transition/decision index
- 학습 설정

각 episode checkpoint는 저장 공간을 절약하기 위해 gzip JSON으로 기록한다. 저장량이 너무 커지는 실험에서는 headless 옵션 `--no-episode-checkpoints`로 episode별 checkpoint만 끌 수 있다. 최종 checkpoint와 step/episode 기록은 계속 저장된다.

## 종료 후 통계 창

세션이 완료되거나 사용자가 중지하면 별도 통계 창이 자동으로 열린다. 메인 GUI의 `통계 창` 버튼으로 다시 열 수 있다.

통계 창은 다음 탭을 제공한다.

- 요약 통계와 결과 폴더 열기
- **에피소드별 step 수와 100회 이동평균**
- 성공 점수와 이동평균
- oracle 대비 경로 효율
- 에피소드 소요시간
- Prediction score
- holdout before/after/gain
- intrinsic value
- Imagination node와 호출 수
- 오류와 반복 행동
- 실시간/최대 속도별 소요시간
- 행동 분포
- 이벤트 분포
- 모든 episode의 상세 표

같은 그래프는 종료 시 `charts/*.svg`에도 저장된다.

## Headless 실행

최대 속도 Full AASSR:

```bash
python scripts/run_escape_gridworld.py --episodes 2000 --colors 2 --seed 7 --mode fast
```

출력 폴더 지정:

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --colors 2 \
  --seed 7 \
  --mode fast \
  --output runs/escape_gridworld/my_run
```

Contextual Policy 중심 ablation:

```bash
python scripts/run_escape_gridworld.py --episodes 2000 --colors 2 --seed 7 --mode fast --no-imagination
```

## 연구 해석상의 제한

현재 GUI 실행기는 하나의 procedural seed로 만든 맵을 반복 학습한다. 높은 점수가 처음 보는 맵에 대한 일반화를 의미하지는 않는다.

본 실험에서는 다음을 별도로 구성해야 한다.

```text
training map seeds != validation map seeds != test map seeds
```

## 검증

```bash
python -m compileall -q src tests scripts
pytest -q tests/test_escape_gridworld.py
```

테스트는 다음을 확인한다.

- 여러 seed의 생성 맵이 해결 가능함
- 상자에서 열쇠가 획득되고 같은 색 문만 열림
- 출구 도달 전 외부 성공 보상이 0임
- 500 tick을 넘어도 자동 timeout되지 않음
- 실시간·최대 속도 모드가 같은 학습 결과를 냄
- 실행 중 runtime mode를 변경할 수 있음
- 짧은 성공 경로가 더 높은 점수 배수를 받음
- 모든 step·episode 파일, checkpoint, summary, SVG chart가 생성됨
