# AASSR 최종 테스트: frozen strict GridPush 복잡도 스케일링

## 1. 결정 상태

이 문서는 최종 실험 시작 전 고정한 프로토콜이다.

- 포기 기능: **제외**
- 고정 tick/step 제한: **사용하지 않음**
- 에피소드 종료: 성공 또는 환경 자체의 비가역 실패
- 외부 보상: 최종 성공 시 `1`, 그 외 `0`
- 환경 규칙: 기존 strict `BenchmarkGridPushWorld`를 수정하지 않음
- 주 평가 split: 처음 보는 unseen 맵
- 독립 반복 단위: episode가 아니라 **seed**

포기 smoke에서 현재 Critic의 낮은 출력은 상태의 회복 불가능성을 의미하지 않았고, 해결 가능한 상태를 대량으로 조기 포기했다. 따라서 포기 임계값을 조정해 최종 실험에 넣지 않는다.

## 2. 주가설

> frozen strict GridPush 안에서 맵의 구조적 복잡도가 증가할수록 DQN의 성공률이 더 빠르게 감소하여 `Imagination v2 − DQN`의 상대 성공률 차이가 AASSR에 유리한 방향으로 증가한다.

절대 성공률이 복잡도와 함께 증가한다는 가설은 사용하지 않는다.

```text
Delta(seed, level)
= success_rate(Imagination v2, seed, level)
- success_rate(DQN, seed, level)
```

주 검정량은 각 seed에서 Level 1~5에 대한 `Delta`의 선형 기울기다.

## 3. 보조가설

> 복잡도가 증가할수록 `Imagination v2 − Neural Policy-only`의 상대 성공률 차이도 증가한다.

이 검정은 같은 neural Policy와 Neural Delta Prophecy를 쓰면서 Imagination의 추가 효과가 복잡도에 따라 달라지는지 본다.

## 4. 환경을 새로 만들지 않는 이유

복잡도 가설을 확인하기 위해 최종 결과 직전에 AASSR에 유리한 새 규칙을 추가하면 사후 환경 설계라는 비판을 피하기 어렵다.

따라서 환경의 행동 공간, 관측 표현, 보상, 단계 구조, 비가역 실패 규칙은 동결한다. 복잡도 Level은 동일한 procedural generator가 만든 맵의 실제 상태 그래프만 분석해 나눈다. 에이전트 성공률은 Level 결정에 사용하지 않는다.

이 선택은 KPDE의 모든 요소를 독립적으로 조작하는 실험보다 범위가 좁지만, 현재 최종 환경에서 복잡도와 상대 성능의 관계를 가장 덜 편향되게 검정한다.

## 5. 맵 복잡도 원시 지표

각 solvable map의 전체 도달 가능 상태 그래프를 BFS로 열거하고 다음 값을 저장한다.

1. `oracle_shortest_steps`
   - 정확한 최단 성공 행동 수 `L*`
2. `irreversible_failure_ratio`
   - 도달 가능한 비종료 상태에서 선택 가능한 모든 행동 중 즉시 실패 terminal로 가는 비율
3. `reachable_nonterminal_states`
   - 도달 가능한 고유 비종료 상태 수
4. `max_graph_depth`
   - 초기 상태에서 도달 가능한 상태 그래프의 최대 깊이
5. `mean_nonfailure_actions`
   - 상태당 실패하지 않는 평균 행동 수
6. `success_edge_ratio`
   - 전체 검사 행동 중 성공 terminal로 직접 이어지는 비율

Level 정렬은 다음 사전 고정된 lexicographic 순서를 쓴다.

```text
oracle_shortest_steps
→ irreversible_failure_ratio
→ reachable_nonterminal_states
→ max_graph_depth
```

가중합 점수를 주 결과에 사용하지 않는다.

## 6. Level 1~5 구성

각 research seed마다 학습용과 unseen 평가용으로 완전히 분리된 map-seed 범위를 사용한다.

1. 필요한 맵 수의 3배를 solvable map 후보로 수집한다.
2. 위 구조적 순서로 정렬한다.
3. 정렬된 후보를 동일 크기의 5개 분위로 분할한다.
4. 각 분위에서 고정 난수로 필요한 수만 선택한다.

따라서 Level 1은 해당 seed 후보군의 가장 단순한 20%, Level 5는 가장 복잡한 20%다. 모든 비교 조건은 동일 seed에서 정확히 같은 맵 manifest를 사용한다.

Level별 원시 지표 평균을 결과에 함께 기록해 실제로 난이도 지표가 증가했는지 확인한다.

## 7. 비교 조건

최종 비교는 네 조건으로 고정한다.

1. `DQN`
2. `Legacy AASSR`
3. `Neural Policy-only`
4. `Imagination v2`

Oracle은 다음 용도로만 사용한다.

- solvable map 선별
- 최단 성공 길이 계산
- 맵 복잡도 기록
- 성공 경로 효율 계산

Oracle 정보는 에이전트의 학습 입력, 보상, 행동 선택, Imagination scorer에 들어가지 않는다.

## 8. 학습 및 평가 규모

### 독립 seed

총 20개다.

```text
7, 13, 21, 42, 100,
131, 173, 211, 257, 307,
353, 401, 457, 503, 557,
601, 653, 701, 751, 809
```

### 학습

- 조건당 seed별 실제 환경 transition budget: `20,000`
- Level별 training maps: `32`
- 전체 training maps: `160`
- 각 Level이 균형 있게 반복되도록 training sequence 구성
- checkpoint transition: `0, 2,500, 5,000, 10,000, 20,000`

환경 transition budget 도달 도중 episode를 인위적으로 끊지 않는다. 현재 episode가 환경 자체로 종료될 때까지 진행하므로 마지막 checkpoint에는 짧은 overshoot가 생길 수 있고 이를 별도 기록한다.

### 탐색률 공정성

기존 러너는 epsilon을 episode 수에 따라 줄였지만, 알고리즘마다 episode 길이가 달라 실제 상호작용량이 달라졌다.

최종 러너는 정책에 전달되는 학습 진행도를 완료된 real transition 수로 바꾼다. 따라서 DQN과 AASSR 계열 모두 같은 transition 진행률에서 같은 exploration schedule 위치를 갖는다.

### 평가

각 checkpoint와 Level에서:

- seen: `100 episodes`
- unseen: `100 unique maps`

최종 checkpoint 기준 조건당 총 unseen 평가 episode는 seed별 `500`, 전체 20 seeds에서 `10,000`이다.

## 9. 주 통계 검정

주 평가 split은 unseen이다.

1. 각 seed·Level·조건의 성공률을 먼저 계산한다.
2. 각 seed·Level에서 `Imagination v2 − DQN` 차이를 계산한다.
3. 각 seed의 Level 1~5 차이에 선형 기울기를 적합한다.
4. 20개 seed 기울기의 평균과 seed bootstrap 95% CI를 계산한다.
5. 기울기가 0보다 큰지 one-sided Wilcoxon signed-rank test를 수행한다.

가설 지지 기준은 다음 두 조건을 모두 만족하는 경우다.

```text
seed-bootstrap 95% CI lower bound > 0
AND
one-sided Wilcoxon p < 0.05
```

episode 수천 개를 독립 표본으로 취급하지 않는다.

## 10. Level별 보조 분석

- 조건별 seed 평균 성공률과 seed-bootstrap 95% CI
- Level별 `Imagination v2 − DQN` paired difference
- Level별 `Imagination v2 − Neural Policy-only` paired difference
- 동일 seed·동일 unseen map의 McNemar exact test
- 5개 Level McNemar p-value에 Holm 보정
- 평균 상대 차이가 처음 0 이상이 되는 crossover Level
- CI가 0보다 완전히 큰 Level
- seen 결과와 학습곡선은 보조 결과로 분리

## 11. 기록 지표

### episode 단위

- condition, seed, split, Level, map seed
- success, reward, steps, `L*`, path efficiency
- real transition 누계
- action selection/update 시간
- imagined node 수, Imagination 실행 여부
- 맵 원시 복잡도 지표
- 종료 원인

### checkpoint 단위

- Level별 seen/unseen 성공률
- 성공 시 평균 step과 path efficiency
- Imagination 사용률과 평균 imagined node
- 학습 episode 수와 실제 transition 수
- 계산 시간, 모델 크기, gradient update 수

### 최종 출력

- `seed_level_rates.csv`
- `unseen_success_by_level.csv`
- `primary_relative_effect.csv`
- `primary_seed_slopes.csv`
- `mcnemar_by_level.csv`
- `complexity_by_level.csv`
- `hypothesis_test.json`
- 성공률/상대 우위/seed 기울기 그래프
- 한국어 최종 결과 보고서

## 12. 해석 범위

이 실험이 지지되면 다음을 말할 수 있다.

> 현재 frozen strict GridPush generator 안에서 구조적으로 복잡한 맵으로 갈수록 Imagination v2의 DQN 대비 상대 성능이 개선됐다.

다음까지 자동으로 증명되지는 않는다.

- 모든 KPDE에서 AASSR이 DQN보다 우수함
- 부분관측이나 지식-행동 결합이 그 원인임
- 복잡도가 증가하면 AASSR 절대 성공률이 증가함
- 계산 효율에서도 AASSR이 우수함

가설이 지지되지 않으면 현재 AASSR 구조의 최종 성능 주장에 그대로 반영하고, 결과를 본 뒤 같은 실험을 덮어쓰는 모델 수정은 하지 않는다.
