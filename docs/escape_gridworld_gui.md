# Colored-key Escape GridWorld GUI

## 목적

기존 불투명 이진 의존 사슬보다 공간적 탐색과 물체 의존성을 가진 작은 환경에서 AASSR의 온라인 학습 과정을 관찰한다.

환경은 다음 요소만 사용한다.

- 중립 상자
- 빨강·파랑·초록 열쇠
- 같은 색 열쇠로 여는 색 문
- 출구
- 상하좌우 이동과 `interact`

상자 내용은 열기 전에는 노출되지 않는다. 문 색은 관측할 수 있다. 환경이 주는 외부 보상은 출구에 도달했을 때의 `+1`뿐이며, 상자 열기·열쇠 획득·문 개방에는 중간 외부 보상을 주지 않는다.

## 생성 규칙

`generate_escape_grid()`는 색 수에 맞춰 구역을 순서대로 나눈다.

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

각 구역의 문 위치와 상자 위치는 seed로 결정된다. 빈 미끼 상자를 추가할 수 있다. 생성된 세계는 `oracle_plan()`의 BFS로 해결 가능성을 검증하지만, oracle 경로는 에이전트 학습에 제공되지 않는다.

## 실제 학습기

GUI는 별도 데모용 정답 정책이 아니라 기존 코어의 다음 구성을 사용한다.

```text
AutonomousLearningAgent
+ ContextualPolicy
+ TabularProphecy
+ ImaginationTree
+ holdout 기반 validated information gain
+ 반복·오류 감점
```

에이전트는 시범 없이 실제 상호작용으로 전이를 학습한다. episode 종료 시 최종 탈출 보상이 전체 행동열에 할인 역전파된다.

## GUI 실행

먼저 개발 의존성을 설치한다.

```bash
python -m pip install -e ".[dev]"
```

GUI를 실행한다.

```bash
python scripts/run_escape_gridworld.py --gui
```

시작 화면에는 두 실행 버튼이 있다.

### 실시간으로 보기

- 모든 primitive step을 GridWorld에 그린다.
- step 사이에 짧은 지연을 둔다.
- 에이전트 위치, 열린 상자, 보유 열쇠, 열린 문을 즉시 표시한다.
- Prediction score, holdout gain, Imagination 사용 여부와 imagined node 수를 로그로 확인할 수 있다.

### 안 보고 최대 속도

- step 렌더링을 하지 않는다.
- 인위적인 sleep을 하지 않는다.
- 지정된 episode 간격으로 진행률과 rolling success만 갱신한다.
- 학습 순서, RNG seed, Policy·Prophecy·Imagination 업데이트는 실시간 모드와 동일하다.

따라서 두 모드의 차이는 표시 비용뿐이며 학습 알고리즘은 바뀌지 않는다.

## Headless 실행

최대 속도 Full AASSR:

```bash
python scripts/run_escape_gridworld.py --episodes 2000 --colors 2 --seed 7 --mode fast
```

실시간 콘솔 출력:

```bash
python scripts/run_escape_gridworld.py --episodes 100 --colors 1 --seed 7 --mode live
```

Contextual Policy 중심 ablation:

```bash
python scripts/run_escape_gridworld.py --episodes 2000 --colors 2 --seed 7 --mode fast --no-imagination
```

## GUI에 표시되는 항목

- 전체 episode 진행률
- 누적 성공률
- 최근 100 episode 성공률
- epsilon
- 현재 inventory
- 상자·열쇠·문 상호작용 이벤트
- Prophecy prediction score
- holdout gain
- intrinsic value
- Imagination 사용 여부
- 현재 imagined node 수
- 전체 Imagination 호출 수와 누적 node 수
- oracle 최단 경로 길이

## 연구 해석상의 제한

현재 GUI 실행기는 하나의 procedural seed로 만든 맵을 반복 학습한다. 목적은 환경과 학습 루프를 디버깅하고, 행동이 어떻게 바뀌는지 눈으로 확인하는 것이다.

따라서 GUI에서 높은 성공률이 나와도 처음 보는 맵에 일반화했다는 뜻은 아니다. 본 실험에서는 다음을 별도로 구현해야 한다.

```text
training map seeds != validation map seeds != test map seeds
```

권장 본 실험은 여러 train map에서 학습한 뒤, 보지 않은 test map마다 1회 또는 소수의 적응 기회만 주고 평가하는 방식이다. GUI 코어와 환경은 그 실험의 시각화 및 단위 검증에 재사용한다.

## 검증

```bash
python -m compileall -q src tests scripts
pytest -q tests/test_escape_gridworld.py
```

테스트는 다음을 확인한다.

- 여러 seed의 생성 맵이 실제로 해결 가능함
- 상자에서 열쇠가 획득됨
- 같은 색 열쇠가 있어야 문이 열림
- 최종 탈출 전에는 외부 보상이 0임
- 실시간·최대 속도 모드가 같은 학습 결과를 냄
- epsilon이 단조 감소함
