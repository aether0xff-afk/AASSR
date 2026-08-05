# Q-learning·DQN·DreamerV3·AASSR 성능 및 효율 비교

## 결론

이번 고정-action 희소보상 GridPush 비교에서는 **DQN이 성능과 계산비용을 함께 고려했을 때 가장 우수했다.**

- DQN: 학습한 맵 `16.4%`, 처음 보는 맵 `7.2%`
- AASSR full: 학습한 맵 `0.8%`, 처음 보는 맵 `0.0%`
- 공식 DreamerV3 `size1m`: 학습한 맵 `0.0%`, 처음 보는 맵 `0.0%`
- Q-learning·AASSR Policy-only·Random: 모두 `0.0%`

따라서 현재 결과로 **AASSR이 Q-learning·DQN·DreamerV3보다 일반적으로 우수하다고 주장할 수 없다.** 오히려 이 benchmark에서는 DQN이 AASSR보다 성공률이 높고 학습 계산도 훨씬 적게 들었다.

다만 AASSR이 성공한 두 episode는 모두 oracle 최단 경로와 일치했다. 즉 현재 AASSR은 성공 빈도는 매우 낮지만, 올바른 미래를 찾았을 때의 경로 품질은 높았다.

## 공정한 비교 조건

모든 알고리즘에 다음 조건을 동일하게 적용했다.

- 25차원 수치 관측
- 항상 표시되는 4개의 불투명 행동 `choice_0..choice_3`
- 최종 성공 시에만 외부 보상 `1`
- 중간 보상 없음
- 인위적인 tick·energy 제한 없음
- 지나간 칸이 무너지는 비가역적 경로
- BFS oracle로 실제 완주 가능한 procedural map만 선별
- seed `7, 13, 21, 42, 100`
- 학습 맵과 처음 보는 평가 맵 분리

Native 조건은 1,000 training episode를 수행했다. episode 길이가 알고리즘의 행동에 따라 달라지므로 실제 환경 transition 수는 조건마다 다르며 별도로 기록했다.

공식 DreamerV3는 upstream commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe`, 공식 `size1m` 설정, JAX CPU로 실행했다. `run.steps=5000`이지만 Dreamer driver의 step 집계와 환경 reset 때문에 실제 환경 transition은 seed당 평균 약 `3,470`회였다.

## 최종 5-seed 평균

| 조건 | 학습한 맵 성공률 | 처음 보는 맵 성공률 | 실제 학습 transition | 학습 벽시계 | peak RSS | 저장 모델/테이블 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.0% | 0.0% | 2,247.8 | 0.127 s | 33.0 MB | 0 MB |
| Q-learning | 0.0% | 0.0% | 2,228.2 | 0.206 s | 34.6 MB | 0.290 MB |
| AASSR Policy-only | 0.0% | 0.0% | 2,503.0 | 6.471 s | 62.9 MB | 2.842 MB |
| AASSR full | 0.8% | 0.0% | 3,677.2 | 76.661 s | 66.2 MB | 3.216 MB |
| DQN | **16.4%** | **7.2%** | 5,032.8 | **7.346 s** | 306.8 MB | 0.084 MB¹ |
| DreamerV3 official `size1m` | 0.0% | 0.0% | 3,470.0 | 170.546 s | 1,317.5 MB | 7.966 MB |

¹ DQN의 0.084 MB는 network checkpoint만 센 값이다. replay buffer와 PyTorch runtime은 포함하지 않으므로 실제 실행 메모리는 peak RSS를 함께 봐야 한다.

## seed별 성공률

### DQN

| seed | 학습한 맵 | 처음 보는 맵 |
|---:|---:|---:|
| 7 | 48% | 22% |
| 13 | 10% | 8% |
| 21 | 10% | 2% |
| 42 | 8% | 4% |
| 100 | 6% | 0% |

### AASSR full

| seed | 학습한 맵 | 처음 보는 맵 |
|---:|---:|---:|
| 7 | 0% | 0% |
| 13 | 0% | 0% |
| 21 | 2% | 0% |
| 42 | 0% | 0% |
| 100 | 2% | 0% |

### DreamerV3 official `size1m`

5개 seed 모두 학습·seen·unseen에서 성공 episode가 없었다.

## 같은 평가 맵에서 DQN과 AASSR 비교

최종 checkpoint의 동일 seed·동일 map·동일 episode를 대응시켰다.

| 평가 | DQN만 성공 | AASSR만 성공 | 둘 다 성공 | 둘 다 실패 | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| 학습한 맵 | 40 | 1 | 1 | 208 | `3.82e-11` |
| 처음 보는 맵 | 18 | 0 | 0 | 232 | `7.63e-06` |

이 benchmark에서는 DQN의 우위가 단순 평균 하나나 특정 seed 하나에서 생긴 결과가 아니다.

## 샘플 효율

Native 조건의 checkpoint별 5-seed 평균은 다음과 같다.

| training episode | DQN transition | DQN seen/unseen | AASSR transition | AASSR seen/unseen |
|---:|---:|---:|---:|---:|
| 50 | 113.8 | 0.0% / 0.0% | 115.8 | 0.0% / 0.0% |
| 100 | 224.2 | 0.0% / 0.0% | 229.0 | 0.0% / 0.0% |
| 250 | 619.4 | 0.8% / 2.8% | 601.2 | 0.0% / 0.0% |
| 500 | 1,444.2 | 4.0% / 4.4% | 1,427.2 | 0.0% / 0.0% |
| 1,000 | 5,032.8 | 16.4% / 7.2% | 3,677.2 | 0.8% / 0.0% |

DQN은 AASSR과 거의 같은 약 1,400 transition 구간에서도 이미 seen `4.0%`, unseen `4.4%`를 보였다. 따라서 DQN의 우위가 단순히 최종적으로 더 많은 transition을 사용했기 때문만은 아니다.

DreamerV3는 평균 약 3,470 실제 transition을 경험했지만 학습 중에도 양의 외부 보상을 한 번도 얻지 못했다. 이 결과는 작은 예산에서의 희소보상 발견 능력을 보여주지만, 훨씬 큰 data budget에서의 DreamerV3 최종 성능을 판정하는 실험은 아니다.

## 계산 효율

Native agent 내부에서 직접 측정한 행동 선택+학습 업데이트 시간은 다음과 같다.

| 조건 | 선택+업데이트 시간 |
|---|---:|
| Random | 0.003 s |
| Q-learning | 0.049 s |
| AASSR Policy-only | 4.948 s |
| DQN | 6.689 s |
| AASSR full | 57.155 s |

AASSR full은 DQN보다 약 `8.5배` 많은 선택·업데이트 계산을 사용했다.

최종 평가의 행동 선택 지연은 다음과 같다.

| 조건 | 행동당 평균 선택 시간 |
|---|---:|
| Random | 1.1 μs |
| Q-learning | 14.8 μs |
| DQN | 63.7 μs |
| AASSR Policy-only | 약 0.51 ms |
| AASSR full | 약 17.7 ms |

AASSR full의 평가 행동 선택은 DQN보다 약 `278배` 느렸다. AASSR은 평가 episode당 평균 약 427개의 imagined node를 만들었으며, 실패는 Imagination이 비활성이라서 발생한 것이 아니다.

DreamerV3의 학습 벽시계에는 JAX compilation이 포함되어 있다. seed 평균 학습 시간은 약 `170.5초`, peak RSS는 약 `1.32GB`였다. 독립 평가 프로세스도 JAX compilation과 checkpoint 복원을 포함해 seen·unseen 각각 평균 약 `25초`가 걸렸다.

## 성공했을 때의 경로 품질

| 조건 | 평가 | 성공 수 | 성공 시 평균 step | oracle 평균 | 경로 효율 |
|---|---|---:|---:|---:|---:|
| AASSR full | 학습한 맵 | 2 | 9.50 | 9.50 | **1.000** |
| DQN | 학습한 맵 | 41 | 14.32 | 11.49 | 0.830 |
| DQN | 처음 보는 맵 | 18 | 18.28 | 12.06 | 0.693 |

AASSR은 거의 성공하지 못했지만 성공한 두 경우에는 최단 경로를 선택했다. 반대로 DQN은 훨씬 자주 성공했지만 성공 경로에는 우회가 있었다.

따라서 성능을 하나의 숫자로만 보면 안 된다.

- **신뢰성:** DQN 우위
- **성공 시 경로 품질:** 관측된 소수 사례에서는 AASSR 우위
- **샘플 효율:** DQN 우위
- **학습 계산량:** DQN 우위
- **추론 속도:** DQN 우위
- **메모리:** AASSR이 DQN·DreamerV3보다 작지만 Q-learning보다는 큼

## 기존 GridPush 결과와 달라진 이유

기존 GridPush에서는 현재 가능한 행동만 동적으로 노출했다. 예를 들어 벽 밖 이동이나 이미 무너진 칸으로 가는 행동은 후보에서 빠졌고, 이동·밀기·줍기·사용처럼 행동 의미도 드러났다.

이번 benchmark에서는 모든 상태에서 항상 동일한 네 개의 불투명 선택지를 제공했다. 따라서 에이전트가 다음을 직접 학습해야 했다.

- 현재 위치와 경계에 따라 어떤 선택이 실패하는지
- 무너진 칸에 따라 같은 선택의 결과가 어떻게 바뀌는지
- procedural map의 좌표 관계를 다른 맵에 어떻게 일반화하는지

DQN의 neural function approximation은 이 좌표 패턴을 일부 일반화했지만, 현재 AASSR의 tabular Policy와 effect-context Prophecy는 이를 충분히 일반화하지 못한 것으로 해석된다. 또한 부정확한 transition model 위에서 큰 Imagination tree를 만들면서 계산량은 증가했지만 성공률로 이어지지 않았다.

이 해석은 결과에 기반한 원인 추론이며, Prophecy의 위치별 예측 오차를 별도로 측정하는 추가 진단으로 확인해야 한다.

## DreamerV3 결과의 올바른 해석

이번 실행은 공식 DreamerV3 코드와 `size1m` 설정을 사용했지만 작은 벡터 환경에서 약 3,470 실제 transition만 제공한 **초기 샘플 효율 실험**이다.

따라서 다음은 말할 수 있다.

> 이 작은 최종보상-only 예산에서는 공식 DreamerV3가 한 번도 성공 경험을 발견하지 못했고, DQN보다 훨씬 많은 시간과 메모리를 사용했다.

반면 다음은 말할 수 없다.

> DreamerV3가 충분한 학습 예산에서도 AASSR 또는 DQN보다 항상 나쁘다.

DreamerV3의 강점은 학습된 world model과 latent imagination을 큰 규모의 다양한 문제에 적용하는 데 있다. 정확한 장기 비교를 위해서는 `50k`, `200k` 이상의 transition budget과 GPU 실행을 별도 실험으로 추가해야 한다.

## 현재 연구 주장에 미치는 영향

이번 결과는 다음 주장을 기각한다.

> AASSR은 희소보상 환경에서 Q-learning·DQN·DreamerV3보다 일반적으로 우수하다.

현재 안전하게 주장할 수 있는 범위는 다음과 같다.

> AASSR의 multi-step Imagination은 행동 affordance와 재사용 가능한 transition 구조가 제공되는 특정 환경에서는 Policy-only보다 성능을 높였다. 그러나 동일한 고정 action space와 강한 procedural 일반화를 요구한 환경에서는 DQN이 AASSR보다 높은 성공률과 계산 효율을 보였다.

## 다음 개선 우선순위

1. **Prophecy 일반화 개선**
   - 전체 상태 exact match가 아니라 국소 좌표·경계·무너진 칸 관계를 행동 결과 특징으로 학습
   - action-conditioned effect encoder 도입
   - 위치가 달라도 같은 국소 인과를 재사용하는지 별도 평가

2. **Imagination 계산 제어**
   - 예측 오차가 큰 상태에서는 깊이를 자동 축소
   - node budget 고정 비교
   - DQN과 동일한 추론 시간 예산에서의 성능 측정

3. **동일 transition budget 재실험**
   - `1k / 3.5k / 5k / 20k` transition checkpoint
   - 모든 조건의 실제 transition 수를 정확히 맞춤

4. **DreamerV3 확대 실험**
   - GPU에서 `50k / 200k` transition
   - compilation을 제외한 steady-state throughput도 별도 측정

5. **AASSR 실패 진단**
   - 좌표·경계·collapsed-cell 유형별 Prophecy 오차
   - Imagination이 선택한 행동의 correction/harm 비율
   - imagined node 수와 실제 성공 기여의 상관관계
