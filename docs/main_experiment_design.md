# AASSR v2 본 실험 주장과 설계

## 연구 질문

1. **자율 발견**: 정답 행동 시범, 정답 경로 사전학습, 의미 있는 행동 이름 없이도 최종 희소 보상만으로 성공 경로를 발견할 수 있는가?
2. **장기 계획**: 같은 온라인 경험 조건에서 Prophecy와 Imagination이 상태 조건부 Policy보다 긴 의존 관계의 성공률 또는 표본 효율을 개선하는가?
3. **세계 모델 학습**: 에이전트가 직접 수집한 경험만으로 실제 다음 상태 예측이 개선되는가?
4. **이름 비의존성**: 행동과 상태 이름을 seed마다 불투명하게 바꿔도 동일한 학습 알고리즘이 작동하는가?
5. **정보 가치**: 검증된 예측 개선 보상을 제거했을 때 학습 안정성 또는 표본 효율이 악화되는가?
6. **규모 증가**: 의존 길이가 4, 6, 8로 증가할 때 효과와 계산 비용은 어떻게 변하는가?

이 질문들은 사전에 참이라고 가정하지 않는다. 조건 간 차이가 없으면 해당 가설은 기각하거나 제한적으로 해석한다.

## 허용할 주장

결과가 기준을 충족할 때만 다음을 주장한다.

> AASSR v2는 정답 경로가 제공되지 않은 불투명 희소 보상 환경에서 온라인 경험을 통해 성공 경로를 발견했다.

> 동일한 탐색 예산에서 Prophecy와 Imagination을 포함한 조건은 상태 조건부 Policy 기준선보다 긴 의존 환경에서 더 높은 성공률 또는 더 빠른 학습을 보였다.

> 결과는 행동 이름의 의미가 아니라 seed별 불투명 행동에 대한 실제 전이 경험에서 형성되었다.

현재 설계만으로 모든 현실 환경 일반화, 인간의 환경 설계가 전혀 없음, 처음 보는 행동에 대한 zero-shot 정답은 주장하지 않는다.

## 누수 방지 규칙

- 본 실험 runner는 `_pretrain_choice`, `_pretrain_trap` 같은 oracle 경로 생성기를 호출하지 않는다.
- 에이전트는 환경 내부의 viable action 목록에 접근하지 않는다.
- 행동명은 `op_<무작위 48비트 값>`이며 `safe`, `trap`, `greedy`, `finish` 같은 의미 단어를 포함하지 않는다.
- 상태 fact도 불투명한 `obs_<무작위 값>`이다.
- 중간 `goal_progress`는 항상 0이고, 손상 없이 마지막 단계에 도달했을 때만 reward 1과 goal progress 1을 준다.
- Policy 초기값은 모든 상태·행동에서 동일하다.
- 각 seed는 독립된 행동 매핑과 상태 표현 순열을 가진다.
- 평가 구간에서는 탐색과 학습을 끄고 agent를 고정한다.
- 실행 결과에 `protocol_manifest.json`을 저장한다.

## 일반형 에이전트 구조

```text
StateSnapshot + opaque available actions
        ↓
ContextualPolicy
- state-action별 가치
- epsilon + UCB 탐색
- 행동 문자열 의미 미사용
        ↓
전이 coverage가 부족하면 직접 탐색
충분하면 Prophecy 기반 ImaginationTree 계획
        ↓
현실에서 첫 행동만 실행
        ↓
온라인 Prophecy 학습
- transition을 train 또는 frozen holdout으로 분리
- 현재 표본으로 같은 갱신을 자기채점하지 않음
        ↓
최종 희소 보상을 episode 전체에 할인 역전파
+ 선택적 검증 holdout gain
- 반복 감점
- 오류 감점
```

기존 `WeightedPolicy`의 전역 action weight 대신 `ContextualPolicy`를 사용한다. 같은 행동도 상태에 따라 다른 가치를 가질 수 있고, 코어는 행동 이름을 해석하지 않는다.

## 환경

`OpaqueDependencyWorld(length=L)`은 L단계 비가역 의존 환경이다.

- 매 단계 불투명 행동 2개가 주어진다.
- 하나는 성공 가능성을 보존하고 하나는 irreversible corruption을 만든다.
- 두 행동 모두 다음 단계로 이동하므로 즉시 보상으로 구분되지 않는다.
- corruption은 관측되지만 의미 없는 벡터와 fact로 표현된다.
- 마지막 단계까지 corruption 없이 도달한 경우에만 reward 1을 준다.

## 비교 조건

| 조건 | 상태별 Policy | Prophecy | Imagination | 검증 정보 가치 |
|---|---:|---:|---:|---:|
| random | 아니요 | 아니요 | 아니요 | 아니요 |
| contextual_policy | 예 | 아니요 | 아니요 | 아니요 |
| prophecy_no_imagination | 예 | 예 | 아니요 | 예 |
| imagination_no_validated_value | 예 | 예 | 예 | 아니요 |
| full_aassr | 예 | 예 | 예 | 예 |

모든 조건은 동일 episode 수와 환경 seed 집합을 사용하며 독립 agent를 처음부터 학습한다.

## 본 실험 규모

`configs/autonomous_main.json`

- 독립 seed 20개
- 의존 길이 4, 6, 8
- 조건 5개
- train 2,000 episode
- 고정 평가 200 episode
- 총 결과 행 `20 × 3 × 5 × 2,200 = 660,000`

배선 확인은 `configs/autonomous_smoke.json`으로 수행한다.

## 지표와 통계

주요 지표는 평가 성공률, 학습 곡선 AUC, 최초 성공 episode, 최근 구간 성공률, prediction similarity, Imagination 사용량과 실행 시간이다. 보조 지표는 holdout score/gain, 반복, 오류, intrinsic value다.

Episode를 독립 표본으로 세지 않는다. 각 seed 내부에서 먼저 요약하고 seed 간 평균·표준편차·95% 구간을 계산한다. training과 evaluation phase는 분리한다.

## 성공 판정

- `full_aassr`가 random보다 높고 여러 seed에서 반복되어야 자율 발견을 주장한다.
- Imagination 조건이 `prophecy_no_imagination`보다 평가 성공률, AUC, 최초 성공 시점 중 하나 이상에서 개선되어야 계획 효과를 주장한다.
- 차이가 없으면 이 환경에서 Imagination 추가 이득은 검증되지 않았다고 결론낸다.
- `full_aassr`와 `imagination_no_validated_value` 차이가 없고 holdout gain이 0이면 정보 가치 효과를 주장하지 않는다.

## 기존 최종 파일럿의 위치

`configs/pilot.json`과 `final_pilot.py`의 oracle pretraining 실험은 삭제하지 않고 **모듈 진단**으로만 남긴다.

> 전이 모델이 필요한 경로를 이미 학습한 조건에서 Imagination 깊이가 근시안적 선택을 교정할 수 있는가?

이 결과를 자율 발견이나 인간 개입 최소화의 근거로 사용하지 않는다.
