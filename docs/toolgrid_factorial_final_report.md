# ToolGrid shared-checkpoint factorial 최종 보고서

## 1. 실험 목적

ToolGrid factorial은 AASSR Imagination v2가 다음 두 질문에서 어떤 성질을 보이는지 확인하기 위해 수행했다.

1. 동일한 학습 checkpoint에서 Imagination을 켰을 때 DQN 및 Policy-only보다 나은가?
2. 맵 크기와 의미 선택지가 증가할수록 DQN 대비 우위가 커지는가?

초기 구현은 같은 seed라도 Policy-only와 Imagination을 별도 runner에서 다시 학습해 서로 다른 checkpoint를 비교하는 문제가 있었다. 최종 실험에서는 hybrid agent를 각 cell에서 정확히 한 번만 학습하고, 같은 객체와 같은 model state에서 `use_imagination=False/True`만 바꾸어 평가했다.

## 2. 최종 프로토콜

- seeds: `7`, `21`, `42`
- grid sizes: `3×3`, `5×5`, `7×7`
- semantic tools: `4`, `8`
- conditions: DQN / Neural Policy-only / Imagination v2
- cell당 real training transitions: 정확히 `5,000`
- training maps: `48`
- unseen evaluation maps: `100`
- training-time Imagination: `0회`
- Policy-only와 Imagination: 동일한 training rows, checkpoint, evaluation maps
- aggregate 전제 검증 실패 시 통계 생성 중단

참조:

- final commit: `04cb93a93f0d6fdffd3b336418aa93a580df1270`
- GitHub Actions run: `31081140848`
- pull request: `#23`

## 3. unseen success 결과

| map | tools | DQN | Policy-only | Imagination v2 |
|---:|---:|---:|---:|---:|
| 3×3 | 4 | 75.0% | 72.3% | **100.0%** |
| 3×3 | 8 | 27.7% | 29.0% | **42.0%** |
| 5×5 | 4 | 33.0% | 36.0% | **44.0%** |
| 5×5 | 8 | 14.3% | 17.0% | **17.0%** |
| 7×7 | 4 | 15.0% | **23.3%** | **23.3%** |
| 7×7 | 8 | 5.3% | **9.0%** | **9.0%** |

18개 seed×environment cell에서:

- Imagination v2 > Policy-only: `5`
- Imagination v2 = Policy-only: `13`
- Imagination v2 < Policy-only: `0`
- Imagination v2 > DQN: `18 / 18`

따라서 이번 실험 범위에서는 Imagination v2가 DQN보다 일관되게 높았고, 동일 checkpoint의 Policy-only 성능을 악화시키지 않았다.

## 4. complexity hypothesis 결과

가설은 다음과 같았다.

> 맵 크기와 semantic branching이 증가할수록 Imagination v2의 DQN 대비 우위가 커질 것이다.

Imagination-DQN 차이에 대한 평균 계수:

- map-size effect: `-0.0683`
- semantic-branch effect: `-0.0394`
- interaction: `+0.0150`

map-size와 branch effect는 세 seed 모두 음수였다. 따라서 해당 가설은 지지되지 않았으며 기각한다.

현재 데이터가 지지하는 결론은 더 제한적이다.

> Imagination v2는 이번 ToolGrid 범위에서 DQN보다 우수했지만, 환경 크기와 도구 수가 증가할수록 그 우위가 확대되지는 않았다.

## 5. Imagination 사용률 해석

Imagination 사용률은 대략 다음과 같이 감소했다.

- `3×3 / 4 tools`: 약 `99.7%`
- `7×7 / 8 tools`: 약 `9.3%`

이 감소를 곧바로 “복잡한 환경에서 올바르게 더 신중해졌다”고 해석할 수는 없다.

ToolGrid 한 에피소드에는 station이 항상 4개이므로 성공 경로의 semantic tool choice도 정확히 4번이다. 반면 맵 크기가 커질수록 station 사이 이동 step만 증가한다. 따라서 전체 step을 분모로 한 Imagination use rate는 자연스럽게 감소한다.

동시에 감소 폭이 매우 크므로, terminal-choice gate가 실제 중요 선택에서도 닫히는 과도한 보수성이 일부 포함됐을 가능성이 있다. 이를 구분하려면 다음 지표가 필요하다.

- episode당 Imagination 호출 횟수
- 실제 irreversible decision point에서의 gate recall
- gate-open 시 성공률 개선
- missed opportunity rate
- action-changing intervention 수와 손익

## 6. ToolGrid 환경이 실제로 조작한 것

ToolGrid 맵은 다음으로 구성된다.

- 빈 정사각 격자
- 시작점 1개
- 순서가 고정된 station 4개
- 각 station의 정답 tool 1개
- 상하좌우 이동
- 전역적으로 사용 가능한 tool actions

없는 요소:

- 벽 및 장애물
- 문, 열쇠, 권한 상태
- 관측에 따라 열리는 새로운 행동
- 자원 획득과 소비
- 여러 목표 사이의 선택
- 되돌릴 수 있는 오류 복구
- 서로 다른 정보 수집 경로
- 장기적인 credential/session/state dependency

따라서 `map size`는 주로 의미적 복잡도가 아니라 공간적 horizon과 navigation transition 수를 증가시킨다. `tool count`는 terminal semantic branching을 증가시키지만, semantic decision point의 수는 4개로 고정된다.

이 한계 때문에 ToolGrid 결과를 Minecraft 또는 penetration testing 성능의 직접 증거로 사용해서는 안 된다. ToolGrid는 다음 성질을 통제해서 확인한 추상 benchmark다.

- 동일 checkpoint에서의 planner causal effect
- 비가역 semantic choice에서의 selective correction
- 학습 중 개입을 제거한 안전한 paired evaluation
- DQN 대비 성능 비교

## 7. 최종 결론

1. **DQN 대비 우위:** 이번 18개 seed×environment cell 모두에서 확인됐다.
2. **Policy 보호:** 5개 개선, 13개 동률, 악화 0개였다.
3. **복잡도 증가 우위 가설:** 지지되지 않았다.
4. **현재 병목:** 상상 깊이 자체보다 언제 Imagination을 호출할지 탐지하는 gate에 가깝다.
5. **다음 환경 요구:** 단순 이동 길이가 아니라 정보 수집, 권한, 세션, 의존관계, 비가역 행동, 복구 비용을 독립적으로 조작해야 한다.

## 8. 다음 단계

다음 benchmark는 실제 외부 시스템을 공격하지 않는 in-process penetration-testing simulation으로 구성한다.

- target은 deterministic local mock state machine
- raw shell과 임의 URL 금지
- 행동은 고정 allowlisted request/inspection template
- 정보 수집 → 상태 변화 → 권한 획득 → 목표 확인의 dependency graph
- Policy-only와 Imagination의 동일-checkpoint paired evaluation 유지
- 실제 localhost 도구 transport는 별도 단계로 분리

이 전환의 목적은 “맵이 커진다”가 아니라, 실제 문제 해결에서 중요한 **관측되지 않은 상태, 장기 의존성, 의미 분기, 비가역 비용**을 benchmark에 넣는 것이다.
