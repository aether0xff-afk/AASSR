# Imagination v2 5-seed 비교 결과

## 실험 목적

이 실험은 이전 진단에서 가장 유망했던 두 요소를 실제 Imagination에 결합했다.

- Neural Delta Prophecy: 현재 상태에서 행동으로 바뀌는 부분을 예측한다.
- GRU Branch Critic: 상상 가지가 지나온 상태와 행동 순서를 기억하고, 최종 성공 가능성을 예측한다.

Imagination v2의 구조는 다음과 같다.

```text
Policy가 행동 후보 생성
→ Neural Delta Prophecy가 행동별 다음 상태 예측
→ 각 미래를 독립된 가지로 확장
→ 가지마다 별도의 GRU Critic 기억 유지
→ Critic 점수로 가지 확장 순서와 최종 행동 개입 판단
```

Critic 신경망의 가중치는 모든 가지가 공유한다. 가지가 갈라질 때 복사되는 것은 그 가지가 지금까지 본 순서를 요약한 작은 기억값뿐이다.

## 실험 조건

- seed: `7, 13, 21, 42, 100`
- 학습: seed마다 1,000 episodes
- 학습 맵: 64개
- 최종 평가: 학습한 맵 100 episodes, 처음 보는 맵 100 episodes
- 외부 보상: 최종 성공 시 1, 그 외 0
- 모든 조건은 동일한 실제 환경 상호작용 예산을 사용했다.

## 비교 조건

- `legacy_aassr`: 기존 AASSR Policy, legacy Prophecy, 사람이 만든 가지 점수식
- `controlled_legacy`: Imagination v2와 같은 DQN Policy, legacy Prophecy, 사람이 만든 점수식
- `neural_manual`: Neural Delta Prophecy, 사람이 만든 점수식
- `legacy_gru_critic`: legacy Prophecy, GRU Critic
- `imagination_v2`: Neural Delta Prophecy, GRU Critic
- `neural_policy_only`: Imagination v2와 같은 Policy와 Neural Prophecy를 사용하지만 상상은 완전히 끔
- `dqn`: 외부 기준선

`neural_policy_only`는 Imagination v2가 높은 성능을 낸 원인이 실제 상상 개입인지, 기본 Policy 학습인지 구분하기 위해 추가했다.

## 최종 결과

평균은 5개 seed의 평균이다.

| 조건 | 학습한 맵 성공률 | 처음 보는 맵 성공률 | 행동 선택 계산 시간 |
|---|---:|---:|---:|
| 기존 AASSR | 1.0% | 0.0% | 48.8초 |
| 같은 Policy의 controlled legacy | 0.4% | 0.0% | 37.6초 |
| Neural Prophecy + 수동 점수식 | 7.0% | 4.0% | 70.5초 |
| Legacy Prophecy + GRU Critic | 6.6% | 6.6% | 55.1초 |
| DQN | 16.0% | 7.4% | 0.39초 |
| Neural Policy만 사용 | 28.6% | 11.2% | 11.0초 |
| **Imagination v2** | **28.8%** | **12.8%** | **95.2초** |

## Legacy와의 비교

Imagination v2는 기존 AASSR보다 확실히 높았다.

```text
학습한 맵: 1.0% → 28.8%
처음 보는 맵: 0.0% → 12.8%
```

같은 DQN Policy를 사용한 controlled legacy도 처음 보는 맵 성공률이 0%였다. 따라서 단순히 Policy를 DQN으로 바꾼 것만으로는 설명되지 않는다. Legacy Prophecy와 수동 점수식을 통해 상상이 행동을 바꾸면 Policy 학습을 크게 방해했다.

Neural Prophecy만 넣고 기존 수동 점수식을 유지한 조건도 처음 보는 맵에서 4.0%에 그쳤다. 수동 점수식은 상상 실행 중 약 50.6%에서 Policy 행동을 바꿨고, 부정확한 예측에 너무 자주 개입했다.

## GRU Critic이 실제로 도움을 주었는가

Imagination v2는 Neural Policy만 사용한 조건보다 다음만큼 높았다.

```text
학습한 맵: 28.6% → 28.8%, +0.2%p
처음 보는 맵: 11.2% → 12.8%, +1.6%p
```

하지만 seed별 처음 보는 맵 차이는 다음과 같았다.

```text
seed 7:   -2%p
seed 13:   0%p
seed 21:  +9%p
seed 42:  +1%p
seed 100:  0%p
```

차이가 seed 21 하나에 크게 의존한다. 5개 seed만으로 GRU Critic이 Policy-only보다 일반적으로 우수하다고 결론 내릴 수 없다.

계산 비용은 오히려 크게 증가했다.

```text
Neural Policy-only: 11.0초
Imagination v2:     95.2초
```

행동 선택 계산 시간이 약 8.7배가 되었지만 처음 보는 맵 성공률 증가는 1.6%p였다.

## Critic이 행동을 얼마나 바꾸었는가

Imagination v2는 상상을 수천 번 실행했지만 Policy가 고른 행동을 실제로 바꾼 비율은 평균 약 0.33%였다. 다섯 seed 중 세 seed에서는 최종 기록상 행동을 한 번도 바꾸지 않았다.

실제 환경 정답 시뮬레이터로 행동 변경의 결과를 사후 검사한 값은 다음과 같다. 이 정답 정보는 학습에는 사용하지 않았다.

```text
학습 중 행동 변경: 26회
- 더 좋은 행동으로 수정: 0회
- 더 나쁜 행동으로 변경: 22회
- 나머지: 같은 결과 또는 판별 불가

평가 중 행동 변경: 56회
- 더 좋은 행동으로 수정: 2회
- 더 나쁜 행동으로 변경: 41회
- 나머지: 같은 결과 또는 판별 불가
```

따라서 현재 GRU Critic이 성공 경로를 잘 찾아 행동을 개선했다고 말할 수 없다. 현재의 주요 효과는 낮은 점수 차이에서는 행동을 바꾸지 않도록 억제하여, 기존 수동 점수식의 과도한 개입을 막은 것이다.

## 두 요소를 합친 결과의 정확한 해석

이번 실험에서 확인된 사실은 다음과 같다.

1. Neural Delta Prophecy와 GRU Critic을 결합한 시스템은 legacy AASSR보다 훨씬 높았다.
2. GRU Critic은 기존 수동 점수식보다 훨씬 보수적으로 행동했다.
3. 부정확한 상상으로 Policy 행동을 자주 바꾸는 것보다, 확실하지 않으면 Policy를 유지하는 편이 성능이 높았다.
4. 그러나 현재 GRU Critic의 실제 행동 변경은 대부분 해로웠다.
5. Imagination v2와 완전히 상상을 끈 Policy-only의 차이는 작고 seed에 따라 불안정했다.

따라서 이번 결과를 다음처럼 표현해야 한다.

> Imagination v2는 legacy의 과도하고 잘못된 상상 개입을 제거해 시스템 성능을 크게 회복했다. 하지만 GRU Critic이 유용한 미래를 적극적으로 골라 성능을 높였다는 증거는 아직 없다.

## 인간 개입

Critic 학습에는 다음 정보를 사용하지 않았다.

- 목표까지 남은 거리
- 정답 행동 순서
- 열쇠, 문, 상자 중 무엇이 중요한지에 대한 표시
- 사람이 판단한 좋은 가지 점수
- Oracle이 계산한 가지 가치

Critic의 학습 정답은 실제 에피소드가 최종 성공했는지 실패했는지뿐이다.

다만 인간 개입이 0인 것은 아니다.

- 상태를 25개 숫자로 표현하는 방법을 사람이 정했다.
- 상상 깊이 5, beam 16을 사람이 정했다.
- Critic을 켜기 전에 최소 64 episodes, 성공 4회, 실패 4회를 요구하도록 정했다.
- Prophecy 신뢰도와 행동 개입 최소 차이에 임계값을 사용했다.

이 값들은 정답 경로를 알려주지는 않지만 학습과 탐색 방법에 대한 인간 설계다. 특히 Neural Delta의 상태 표현은 어떤 환경 정보를 보존할지 사람이 선택한 것이므로, 모델이 필요한 표현까지 스스로 발견했다고 주장하면 안 된다.

## 현재 결정

- Imagination v2는 legacy보다 우수한 후속 구조로 보존한다.
- 기존 수동 점수식은 최종 구조에서 제거할 근거가 생겼다. 너무 자주 잘못 개입한다.
- 현재 GRU Critic에는 강한 pruning 또는 행동 변경 권한을 주지 않는다.
- 다음 실험에서는 Critic을 먼저 관찰 전용으로 학습하고, 실제 검증 정확도가 충분할 때만 행동 변경 권한을 단계적으로 부여해야 한다.
- 상상용 난수와 Policy 탐색용 난수를 분리해야 한다. 현재는 상상을 실행하기만 해도 난수 사용 순서가 바뀌어, 직접 행동을 바꾸지 않은 경우에도 학습 경로가 달라질 수 있다.
- Imagination v2가 Policy-only보다 유의미하게 우수한지는 더 많은 seed와 분리된 난수 흐름으로 다시 검증해야 한다.
