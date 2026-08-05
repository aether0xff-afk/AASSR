# AASSR 자발적 포기 smoke 결과

## 결론

현재 Imagination v2의 GRU Branch Critic 출력값을 그대로 포기 판단에 사용하면 안 된다.

가장 보수적인 임계값인 `0.05`에서도 대부분의 에피소드를 2스텝 직후 포기했으며, 포기를 선언한 모든 상태에 Oracle 기준 성공 경로가 남아 있었다.

> 현재 Critic은 `이 상태가 회복 불가능한가`를 판단하지 않는다. 현재 Policy와 지금까지의 궤적이 최종 성공으로 이어질 빈도를 예측하므로, 성공 자체가 드문 환경에서는 해결 가능한 상태도 거의 모두 낮게 평가한다.

따라서 자발적 포기 아이디어는 보존하지만, 현재 Critic을 포기 모듈로 연결하는 구현은 채택하지 않는다.

## 실험 구조

- 환경: 최종 strict GridPush
- 모델: Imagination v2
- seed: `7`
- 최종 학습 조건: `1,000 episodes`, `64 training maps`
- 평가: seen 30 episodes + unseen 30 episodes
- 포기 판단은 학습 완료 후 모델을 동결한 상태에서만 사용
- 포기한 에피소드는 Critic 학습에 다시 넣지 않음
- Oracle은 포기 상태가 실제로 해결 가능한지 사후 판정하는 용도로만 사용
- 고정 에피소드 스텝 제한 없음
- 비종료 방지 안전 상한 128은 별도 `safety_stop`으로 기록

포기 규칙:

```text
Critic 성공확률 <= threshold
조건이 연속 2회 유지
최소 2 real transitions 관찰
→ 포기 선언
```

검사한 threshold:

```text
0.05, 0.15, 0.30
```

## 사전 300-episode smoke

첫 소형 실행은 다음 결과를 냈다.

```text
training success = 2 / 300 = 0.67%
critic_ready = false
포기 선언 = 0회
```

최종 안전장치는 Critic이 성공 에피소드와 실패 에피소드를 각각 최소 4회 본 뒤에만 판단 권한을 허용한다. 300회 학습에서는 성공 사례가 2개뿐이어서 Critic이 포기를 선언하지 않은 것이 정상이다.

안전장치를 약화해 억지로 포기시키지 않고, 최종 비교와 같은 1,000-episode 학습량으로 다시 실행했다.

## 1,000-episode 학습 상태

```text
training success rate = 5.0%
real environment transitions = 5,301
critic episodes = 1,000
critic gradient updates = 1,970
critic_ready = true
```

Critic은 작동 가능한 상태였지만 성공 사례가 전체의 약 5%로 매우 적었다.

## 가장 보수적인 threshold 0.05

### Seen maps

```text
포기 선언: 25 / 30 episodes = 83.3%
평균 선언 시점: 2.0 steps
포기 상태가 실제 해결 불가능: 0 / 25
포기 상태가 실제 해결 가능: 25 / 25
포기 정밀도: 0%
```

그림자 모드에서는 포기를 무시하고 계속 진행했을 때 11/30이 성공했다.
실제 포기 모드에서는 3/30만 성공했다.

```text
shadow success = 36.7%
active-abandon success = 10.0%
포기가 막은 성공 = 8 episodes
```

### Unseen maps

```text
포기 선언: 30 / 30 episodes = 100%
평균 선언 시점: 2.27 steps
포기 상태가 실제 해결 불가능: 0 / 30
포기 상태가 실제 해결 가능: 30 / 30
포기 정밀도: 0%
```

그림자 모드에서는 4/30이 성공했지만 실제 포기 모드에서는 전부 포기하여 성공이 0이었다.

```text
shadow success = 13.3%
active-abandon success = 0.0%
포기가 막은 성공 = 4 episodes
```

따라서 threshold 0.05 하나만 보더라도 총 12개의 실제 성공 가능 에피소드를 포기가 차단했다.

## threshold 0.15와 0.30

두 기준에서는 seen과 unseen의 모든 에피소드가 거의 정확히 2스텝 후 포기됐다.

```text
seen: 30 / 30 포기, active success 0%
unseen: 30 / 30 포기, active success 0%
```

각 threshold에서 그림자 모드의 성공 15개를 모두 차단했다.

`0.15`와 `0.30` 결과가 동일한 이유는 2스텝 시점 Critic 출력이 대부분 0.002~0.114 범위에 있었기 때문이다.

## 포기 위치

threshold 0.05의 그림자 모드 포기 선언 55회 중:

```text
step 2: 52회
step 4: 1회
step 5: 2회
```

상태 phase 분포:

```text
phase 0: 32회
phase 1: 22회
phase 2: 1회
```

즉 대부분 문제의 극초반, 상자에 도달하기 전 또는 상자를 움직이기 시작한 직후에 포기했다.

실제로 성공한 궤적에서도 다음과 같은 낮은 값을 출력했다.

```text
0.0295 → 이후 성공
0.0157 → 이후 성공
0.0218 → 이후 성공
0.0401 → 이후 성공
0.0039 → 이후 성공
```

따라서 낮은 Critic 값은 `성공 경로가 사라졌다`는 신호가 아니다.

## Dead-end 탐지

그림자 모드로 끝까지 진행했을 때 비종료 전 실제 성공 불가능 상태를 거친 에피소드가 있었다.

```text
seen: 2 episodes
unseen: 6 episodes
```

그러나 Critic은 그 상태까지 기다리지 않고 모든 경우에 앞선 해결 가능한 상태에서 먼저 포기를 선언했다.

```text
적절한 dead-end 포기 = 0회
dead-end detection rate = 0%
```

## 계산 자원 절약과 손실

세 threshold 전체에서 실제 포기는 그림자 실패 경로 대비 총 997 환경 스텝을 절약했다.

하지만 동시에 threshold별 반복 평가를 합쳐 성공 42개를 차단했다.

이 값은 포기가 계산량을 줄일 수 있다는 사실만 보여주며, 현재 판단이 적절하다는 증거가 아니다. 성공을 대량으로 포기하면서 얻은 절약이므로 사용할 수 없다.

안전 상한 도달은 한 번도 없었다.

## 왜 실패했는가

현재 GRU Critic의 학습 target은 실제 에피소드의 최종 성공 여부다.

```text
성공 에피소드의 모든 prefix → 1
실패 에피소드의 모든 prefix → 0
```

이 target이 답하는 질문은 다음에 가깝다.

> 현재 Policy가 이 궤적에서 계속 행동할 때 최종적으로 성공할 가능성이 얼마나 되는가?

우리가 포기 판단에 필요한 질문은 다르다.

> 현재 상태에서 가능한 어떤 합리적 continuation을 사용해도 성공할 수 없는가?

현재 Policy가 실패할 가능성이 높은 것과 환경 상태가 회복 불가능한 것은 같지 않다.

또한 학습 성공률이 5%이므로 Critic은 대부분의 prefix에 실패 label을 받는다. 그 결과 해결 가능한 상태와 실제 성공 궤적에도 매우 낮은 값을 주는 강한 비관 편향이 생겼다.

## 현재 결정

1. 최종 AASSR에 현재 방식의 active abandonment를 넣지 않는다.
2. GRU Critic의 절대 출력값을 성공 가능성으로 해석하지 않는다.
3. threshold 조정만으로 해결하려 하지 않는다. `0.05`에서도 이미 과도하게 포기했다.
4. minimum steps나 patience를 늘리는 것은 증상을 늦출 뿐 target 불일치를 해결하지 않는다.
5. 자발적 포기 아이디어는 폐기하지 않고 별도 연구 아이디어로 보존한다.

## 다음에 필요한 포기 target

포기 모델은 `현재 Policy의 성공 빈도`가 아니라 `상태의 회복 가능성`을 배워야 한다.

인간이 정답 경로나 Oracle label을 넣지 않는 후보는 다음과 같다.

```text
저장된 실제 상태에서 여러 개의 독립 continuation을 실행
→ 하나라도 성공하면 recoverable
→ 충분한 다양한 continuation이 모두 실패하면 abandonment evidence 증가
```

이 방식은 실제 환경 결과만 사용하면서 현재 Policy의 단일 실패 궤적과 상태 자체의 불가능성을 분리할 가능성이 있다.

다만 continuation 수가 부족하면 해결 가능한 상태를 실패로 오인할 수 있으므로, 이 또한 바로 최종 구조로 넣지 않고 별도 진단이 필요하다.

## 최종 한 문장

> 현재 GRU Critic은 포기할 때를 아는 모델이 아니라, 성공이 드문 환경에서 거의 모든 초기 상태를 비관적으로 보는 모델이었다. 포기 기능은 작동했지만 판단은 전부 조기 포기였으므로 최종 모델에는 활성화하지 않는다.
