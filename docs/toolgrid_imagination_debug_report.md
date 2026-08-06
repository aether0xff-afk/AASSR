# ToolGrid Imagination v2 숨은 동작 버그 진단 및 수정 보고서

## 범위

이 문서는 ToolGrid factorial pilot에서 Imagination v2가 DQN과 Policy-only보다 낮게 나온 원인을 실행 오류가 아니라 **의도와 다른 실제 동작**의 관점에서 추적한 기록이다.

초기 root-cause 진단 범위는 다음과 같다.

- seed: `7`
- grid size: `3×3`
- semantic tools: `4`, `8` (`action_count` 8, 12)
- real transition budget: `5,000`
- unseen maps: cell당 100
- 환경 자체의 인공 step limit 없음

참조 진단 run:

- GitHub Actions: `31072230333`
- commit: `d8a6e2efee0b72cebe7ea29b676077ffa3607b98`

이 run은 원인을 확인한 디버그 기준점이다. 이후 수정 사항은 debug monkey patch가 아니라 실제 `toolgrid_factorial_masked.py` production 경로에 통합했다.

## 통제 진단 설계

원래 factorial 비교에서는 Imagination이 학습 중 행동을 변경했다. 따라서 최종 `neural_policy_only`와 `imagination_v2`는 같은 정책에 planner만 켜고 끈 비교가 아니라 서로 다른 학습 궤적을 가진 모델이었다.

이를 제거하기 위해 다음 paired 진단을 사용했다.

1. Imagination intervention을 끈 상태에서 hybrid checkpoint 하나를 학습한다.
2. Prophecy와 GRU critic은 실제 transition으로 계속 학습한다.
3. 같은 checkpoint와 같은 unseen map을 두 번 평가한다.
   - greedy learned policy only
   - 같은 learned policy + Imagination
4. 환경 clone과 oracle은 사후 개입 평가에만 사용한다.
5. 각 후보 action에 대해 실제 생존 가능성, Prophecy 예측, predicted/real successor의 critic value, 개입 손익을 기록한다.

환경 oracle은 agent 입력, gate, planner score에 노출하지 않았다.

## 초기 통제 결과

| tools | policy only | original Imagination | delta | harmful interventions | tool-choice interventions |
|---:|---:|---:|---:|---:|---:|
| 4 | 52% | 41% | -11 pp | 25 | 0 |
| 8 | 29% | 19% | -10 pp | 15 | 0 |

행동을 실제로 바꾼 Imagination intervention은 전부 navigation 중에 발생했고, 정작 비가역적인 tool choice에서는 한 번도 정책 행동을 바꾸지 못했다.

critic 자체가 첫 번째 병목은 아니었다. 실제 next state를 critic에 넣었을 때 best action의 viable rate는 4-tool 약 96%, 8-tool 약 91%였다. 실패는 critic 앞단의 representation, Prophecy calibration, replay distribution, intervention gate에 집중돼 있었다.

## 확인된 숨은 버그와 production 수정

### 1. calibration pre-ready cache 버그

기존 cache key는 action별 holdout sample 수를 refresh stride로 나눈 값이었다. `minimum_count=8` 이전에 confidence `0`이 cache되면 sample이 32개가 될 때까지 갱신되지 않았다. 희소 tool action은 학습 가능해진 뒤에도 confidence 0에 묶였다.

수정:

- minimum count 이전 값은 cache하지 않는다.
- ready 경계부터 cache bucket을 시작한다.
- calibration은 모델 gradient revision에 따라서도 갱신한다.

### 2. terminal success와 terminal failure 혼동

정답 tool과 오답 tool은 모두 episode를 종료한다. 기존 calibration은 주로 `available_actions`의 존재 여부만 비교해 성공 종료와 실패 종료를 같은 구조로 취급했다.

수정:

- `nonterminal`, `terminal success`, `terminal failure`를 별도 class로 검증한다.
- predicted available-action set 일치도 함께 검사한다.
- success/failure class가 다르면 calibration score를 0으로 만든다.

주의: 초기 debug CSV의 `prophecy_terminal_match`는 ended/not-ended 일치만 측정했다. 따라서 이전 보고서의 “terminal accuracy”는 정확히는 **termination-status accuracy**다. production validation summary에서는 이 이름을 바로잡고 3-class accuracy로 과장하지 않는다.

### 3. semantic transition이 navigation replay에 묻힘

episode마다 navigation transition은 여러 개지만 terminal tool transition은 한 개뿐이다. action 수가 늘수록 중요한 분기 표본이 minibatch에서 거의 선택되지 않았다.

초기 진단에서는 tool transition을 반복 삽입했지만, production에서는 real-data count를 왜곡하지 않도록 방식 자체를 수정했다.

수정:

- 모든 real transition은 replay에 정확히 한 번만 저장한다.
- minibatch를 `(action identity, outcome class)` strata에서 균등 샘플링한다.
- frozen holdout은 재샘플링하거나 복제하지 않는다.

### 4. fixed action vocabulary를 hash feature로 표현

generic NeuralDeltaProphecy는 opaque action signature를 signed hash bucket으로 표현했다. 8-tool 조건에서는 action identity 충돌과 불필요한 geometry 때문에 successor ranking이 거의 random에 머물렀다.

수정:

- ToolGrid의 고정 action vocabulary를 collision-free one-hot identity로 표현한다.
- action의 의미나 정답 여부는 넣지 않고 선택된 action ID만 제공한다.

### 5. required-tool ID를 ordinal scalar로 표현

required tool을 `0, 1/7, …, 1`로 넣으면 tool ID 사이에 존재하지 않는 순서와 거리 관계가 생긴다. 모델은 ordinal state scalar와 categorical action identity의 equality를 억지로 학습해야 했다.

수정:

- Prophecy codec 내부에서 required-tool identity를 categorical one-hot으로 표현한다.
- prediction은 frozen raw environment schema로 다시 decode한다.
- DQN과 GRU critic의 observation schema는 변경하지 않는다.

### 6. Imagination 실행 위치가 반대

기존 global coverage gate는 navigation에서 반복적으로 planner를 실행하면서 semantic terminal choice에서는 coverage 부족으로 막았다.

수정:

- 학습 중 Imagination intervention을 항상 끈다.
- 평가 시 critic이 준비된 뒤, Prophecy가 **현재 available action 전부를 terminal successor로 예측한 결정**에서만 planner를 실행한다.
- action 이름, `tool_` prefix, ToolGrid phase, 정답 tool, oracle을 사용하지 않는다.
- 내부 coverage 및 intervention advantage gate는 그대로 적용한다.

### 7. Policy-only와 Imagination의 학습 궤적 불일치

원래 Imagination condition은 학습 중 action을 바꿔 policy replay와 Prophecy data까지 달라졌다. 이는 planner의 평가가 아니었다.

수정:

- Policy-only와 Imagination 모두 DQN, Prophecy, holdout calibrator, GRU critic을 동일하게 가진다.
- 두 조건 모두 training-time Imagination run은 0이어야 한다.
- aggregate 단계에서 map, segment, success, steps, termination, cumulative transition을 행 단위로 비교한다.
- 하나라도 다르면 통계 결과 생성을 중단한다.

### 8. DQN Bellman target이 금지된 action을 포함

환경은 context에 따라 navigation action 또는 tool action만 노출하지만 기존 target은 network의 모든 output에 대해 `max Q(s',a')`를 계산했다. 다음 상태에서 실행할 수 없는 action의 큰 Q값이 valid action target을 오염시켰다.

수정:

- replay에 next-state available-action mask를 저장한다.
- bootstrap max는 valid action만 대상으로 계산한다.
- terminal next value는 정확히 0으로 설정한다.

### 9. transition budget 초과

기존 runner는 목표 transition을 넘더라도 episode가 끝날 때까지 계속 실행했다. 조건별 episode 길이가 다르므로 같은 “5,000 budget”이라도 실제 data 수가 달랐다.

수정:

- final real-transition budget을 정확히 5,000으로 맞춘다.
- budget 경계에서 끝나지 않은 segment는 환경 실패로 기록하지 않고 `budget_checkpoint`로 표시한다.
- partial episodic return/critic buffer는 실패 사례로 학습시키지 않고 폐기한다.
- aggregate 단계에서 actual transition이 target과 다르면 실패한다.

### 10. 중간 checkpoint가 학습에 개입

2,500 transition에서 exact budget을 맞추기 위해 진행 중 episode를 자르면, checkpoint가 단순 측정점이 아니라 후반 학습 궤적을 바꾸는 intervention이 된다.

수정:

- factorial checkpoint를 `0`, `5,000`만 사용한다.
- 0에서 5,000까지 연속 학습한다.
- 최종 경계에서만 정확히 정지한다.

### 11. training wall time에 evaluation 시간 포함

기존 누적 timer는 중간 seen/unseen evaluation 시간까지 포함했다.

수정:

- training segment 시간만 별도 누적한다.
- evaluation 시간은 training wall time에서 제외한다.

### 12. model size에서 Prophecy bytes 누락

`model_units`에는 Prophecy가 포함되지만 `model_bytes`에는 DQN과 critic만 포함돼 두 지표의 범위가 달랐다.

수정:

- Prophecy ensemble state bytes를 model bytes에 포함한다.

### 13. branching 조작 및 tool coverage 집계 오류

기존 manifest의 `effective_branching_factor`는 context와 무관한 global `action_count`였고, aggregate의 `unique_tools`는 실제 observed tool을 세지 않고 `tool_count` 평균을 복사했다.

수정:

- global action vocabulary와 station semantic branching factor를 분리해 기록한다.
- `required_tools` JSON을 실제 파싱해 observed tool ID를 계산한다.
- unseen pool에서 tool coverage가 불완전하면 aggregate를 중단한다.

### 14. production validation이 wrapper diagnostics를 잘라냄

ToolGrid wrapper는 internal planner decision의 상세 필드를 `action`, `imagined_nodes`, `used_imagination` 세 값으로 축약했다. production gate를 직접 검증하는 harness가 intervention reason과 value를 읽을 수 없었다.

수정:

- validation script가 production gate 조건을 그대로 적용하면서 internal `ActionDecision`의 전체 diagnostics를 보존한다.
- debug-only patch class는 제거했다.

## root-cause 기준 결과

아래 수치는 원인을 확인한 seed-7, 3×3 debug 기준점이다.

| tools | policy only | terminal-choice Imagination | delta | improved maps | worsened maps | beneficial | harmful |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 46% | **95%** | **+49 pp** | 49 | 0 | 49 | 0 |
| 8 | 29% | **79%** | **+50 pp** | 50 | 0 | 50 | 0 |

모든 action-changing intervention은 terminal semantic choice에서 발생했고 navigation override와 harmful intervention은 0회였다.

이 수치는 production 통합 전 root-cause 증거이며, production 경로는 별도의 same-checkpoint workflow에서 다시 검증한다. 표현·replay·budget 구현이 더 엄격해졌으므로 production 수치가 이 표와 완전히 같다고 미리 가정하지 않는다.

## 해석

초기 음수 결과는 “Imagination이 본질적으로 무가치하다”는 증거가 아니었다. 실제 실패 모드는 다음과 같다.

> 세계 모델이 비가역적 의미 결과를 비교할 수 있는 곳에서는 Imagination이 막혔고, 깊은 rollout의 가치가 낮은 navigation에서는 이미 학습된 정책을 덮어썼다.

확인된 교훈은 더 제한적이고 실용적이다.

- learned world model과 critic은 irreversible semantic branch에서 큰 가치를 줄 수 있다.
- global always-on planning은 안전하지 않고 계산을 낭비한다.
- “언제 상상할지”도 predicted consequence structure에서 결정해야 한다.
- 비교 조건의 학습 trajectory와 실제 transition count를 먼저 고정하지 않으면 planner 효과를 해석할 수 없다.

## 최종 검증 조건

production 수정판의 결론은 다음 검증이 끝난 뒤 확정한다.

1. production same-checkpoint seed-7, 3×3, tools 4/8
2. multi-seed `3×3`, `5×5`, `7×7`
3. DQN, matched Policy-only, corrected Imagination
4. cell당 정확히 5,000 real transitions
5. training-time Imagination 0회
6. Policy-only/Imagination training trajectory 완전 일치
7. intervention benefit/harm 및 imagined-node cost 보고
8. aggregate protocol validator 통과

root-cause 결과는 강한 원인 증거지만 전체 factorial 우위를 대신하지 않는다.
