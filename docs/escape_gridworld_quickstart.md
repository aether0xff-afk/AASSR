# Escape GridWorld GUI 빠른 시작

이 문서는 AASSR v2의 색 열쇠·색 문 Escape GridWorld GUI를 실행하고 기록·상상·모델 기능을 사용하는 방법을 정리한다.

## 실행

```bash
python -m pip install -e ".[dev]"
python scripts/run_escape_gridworld.py --gui
```

GUI는 기본 GridWorld 창과 `AASSR Imagination Viewer` 창을 함께 연다.

## 학습 속도 전환

하나의 세션을 유지한 채 다음 버튼으로 언제든 전환한다.

- `실시간으로 보기`: primitive step을 원래 속도로 렌더링한다.
- `안 보고 최대 속도`: 렌더링과 인위적 대기를 중단하고 같은 상태에서 최대 속도로 계속 학습한다.

속도 전환은 맵, 현재 episode, Policy, Prophecy, Imagination, holdout, RNG 상태를 초기화하지 않는다.

## 에피소드와 점수

에피소드는 출구에 도달할 때까지 계속된다. 고정 tick 제한은 없다. 출구까지 도달한 경우만 성공이며 성공 점수는 다음과 같다.

```text
성공 점수 = 1 + oracle 최단 tick / 실제 성공 tick
```

최단 경로 성공은 `2.0x`, 더 오래 걸릴수록 `1.0x`에 가까워진다.

## 전체 기록과 통계

모든 step과 episode는 실행 중 즉시 디스크에 기록된다.

```text
runs/escape_gridworld/<session>/
├─ session.json
├─ world.json
├─ steps.jsonl
├─ episodes.csv
├─ episodes.jsonl
├─ mode_switches.jsonl
├─ imaginations.jsonl
├─ imagination_summary.json
├─ summary.json
├─ summary.txt
├─ statistics.json
├─ session.log
├─ checkpoints/
├─ models/
└─ charts/
```

학습 완료 또는 중지 후 별도 통계 창이 자동으로 열린다. 필수 그래프인 episode별 step 수와 이동평균 외에도 점수, 효율, 시간, Prediction, holdout, intrinsic value, Imagination, 오류·반복, 행동·이벤트 분포를 볼 수 있다. 그래프는 SVG 파일로도 저장된다.

## Imagination Viewer

현재 설정의 `imagination_interval=1`은 모든 환경 tick에서 무조건 상상한다는 뜻이 아니다. 다음 조건을 모두 만족하는 비무작위 step마다 Imagination을 실행한다.

- Imagination 기능이 켜져 있음
- epsilon random exploration이 아님
- interval 조건 충족
- Prophecy coverage가 기본 임계값 이상

상상 창은 실제 트리의 전체 노드, 부모 관계, 깊이, 누적 가치, 신뢰도, 종료 이유, 루트 행동별 평가, 선택된 첫 행동과 최선 경로를 표시한다. 최대속도 모드에서는 화면은 최신 트리로 갱신하지만 모든 상상 트리는 `imaginations.jsonl`에 빠짐없이 저장된다.

## 모델 저장과 불러오기

세션 종료 시 다음 파일이 자동 생성된다.

```text
runs/escape_gridworld/<session>/models/final.aassr-model.gz
```

GUI 버튼:

- `모델 불러오기`: 저장 모델을 다음 세션의 초기 학습 상태로 사용한다.
- `현재 모델 저장`: 학습 중 또는 종료 후 원하는 경로에 저장한다.
- `새 모델로 시작`: 선택한 모델을 해제한다.

모델에는 Policy, Prophecy, holdout, RNG, transition/decision index와 누적 episode가 저장된다. 불러오면 epsilon 감쇠도 저장 지점 다음부터 이어진다. 현재 episode 중간 위치와 임시 행동열은 모델에 넣지 않으므로 새 episode부터 시작한다.

CLI 예시:

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --colors 2 \
  --distractors 2 \
  --load-model models/my_agent.aassr-model.gz \
  --save-model models/continued_agent.aassr-model.gz
```

자동 최종 모델 저장을 끄려면 `--no-auto-save-model`을 사용한다.

## Headless 실행

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --colors 2 \
  --seed 7 \
  --mode fast
```

Imagination 제거 비교:

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --colors 2 \
  --seed 7 \
  --mode fast \
  --no-imagination
```

현재 GUI 실험은 한 procedural seed의 맵을 반복해 학습 과정을 관찰하는 도구다. 같은 맵에서의 개선만으로 처음 보는 맵 일반화를 주장하지 않는다. 일반화 검증은 train/test map seed를 분리한 별도 실험으로 수행해야 한다.
