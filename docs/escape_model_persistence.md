# Escape GridWorld 모델 저장과 불러오기

## 모델 파일

정식 모델 파일 확장자는 다음과 같다.

```text
.aassr-model.gz
```

모델에는 다음 학습 상태가 포함된다.

- Contextual Policy의 상태별 행동 가치, 전역 가치, 방문 횟수
- Tabular Prophecy의 정확 전이, 행동 계열 전이, 상태 스냅샷
- holdout 전이와 holdout 난수 상태
- 에이전트 난수 상태
- transition/decision index
- 누적 완료 episode 수
- 저장 당시 학습 설정과 호환성 정보

현재 진행 중인 episode의 임시 행동열과 GridWorld 위치는 모델에 포함하지 않는다. 모델을 불러오면 새 episode부터 학습을 이어간다. 전체 세션을 정확히 복구하려면 기존 `checkpoints/*.json.gz`와 세션 기록을 사용한다.

## GUI

```bash
python scripts/run_escape_gridworld.py --gui
```

- `모델 불러오기`: 학습 시작 전에 모델 파일을 선택한다.
- `현재 모델 저장`: 학습 중 또는 학습 종료 후 일관된 모델 스냅샷을 저장한다.
- `새 모델로 시작`: 선택한 불러오기 모델을 해제한다.

학습 중 저장은 에이전트 잠금을 사용한다. 현재 primitive step이 끝난 뒤 Policy, Prophecy, holdout을 한 시점의 상태로 저장하므로 딕셔너리 일부만 저장되는 경쟁 조건이 없다.

세션 종료 시 다음 파일이 자동 생성된다.

```text
<session output>/models/final.aassr-model.gz
```

## CLI

저장된 모델에서 이어 학습:

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --colors 2 \
  --distractors 2 \
  --load-model models/my_agent.aassr-model.gz
```

최종 모델을 지정 경로에도 저장:

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --save-model models/continued_agent.aassr-model.gz
```

자동 최종 모델 저장을 끄려면:

```bash
python scripts/run_escape_gridworld.py --no-auto-save-model
```

## 이어 학습 규칙

모델에 누적 완료 episode 수가 저장된다. 불러온 뒤 epsilon 감쇠는 0부터 다시 시작하지 않고 저장된 episode 다음 지점부터 이어진다.

다음 환경 구조는 모델과 현재 실행이 같아야 한다.

- 색 수 `color_count`
- 미끼 상자 수 `distractor_boxes`

seed는 달라도 불러올 수 있다. 따라서 같은 크기의 새로운 배치에서 전이 학습과 정책 전이를 시험할 수 있다.

## 체크포인트와 차이

```text
checkpoints/*.json.gz   연구 실행 복구와 episode별 내부 기록
*.aassr-model.gz        사용자가 저장·불러오기 위한 이식 가능한 학습 모델
```

모델 파일은 새 세션에서 이어 학습하거나 평가할 때 사용하고, 체크포인트는 원래 세션 분석과 복구에 사용한다.
