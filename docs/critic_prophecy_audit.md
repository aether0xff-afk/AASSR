# Branch Critic and one-step Prophecy audit

이 실험은 서로 다른 두 질문을 한 결과로 섞지 않는다.

## 1. 상상 가지 pruning Critic

현재 상상 트리는 가지마다 별도의 상태, 행동 경로, 임시 Policy 기억을 가진다. 이번 실험에서는 가지를 제거하는 학습 모델을 두 종류만 구현한다.

- `ParentTransitionCritic`: 직전 부모 상태, 행동, 예측된 자식 상태와 Prophecy 신뢰도만 본다.
- `GRUBranchCritic`: 뿌리부터 현재 가지까지의 전이 순서를 읽는다. 가지가 갈라질 때 신경망 전체가 아니라 GRU의 작은 기억 벡터만 복사한다.

기존 수동 `StateDeltaScorer`는 비교 기준으로만 유지한다. 다른 Critic 아이디어는 구현하지 않는다.

Critic의 학습 정답은 실제 에피소드가 최종적으로 성공했는지 여부뿐이다. 목표까지 남은 거리, 열쇠나 문의 중요도, 정답 행동 순서, 사람이 지정한 창의성 점수는 학습에 사용하지 않는다. 실행하지 않은 상상 가지에는 성공·실패 라벨을 만들지 않는다.

pruning 평가는 정확한 Prophecy를 사용해 Prophecy 오차를 제거한 상태에서 진행한다. 정확한 환경 모델과 최단거리 계산은 Critic 학습에는 들어가지 않으며, 가지를 잘못 제거했는지 확인하는 평가에만 사용한다.

## 2. Prophecy one-step 정확도

이 실험에서는 Policy와 Imagination을 사용하지 않는다. 모든 모델은 동일한 실제 전이만 학습한다.

```text
현재 상태 + 실제 행동 -> 실제 다음 상태
```

비교 모델은 다음 세 종류다.

- `legacy_gru`: 기존 Online GRU. 숫자 다음 상태를 예측한 뒤 과거에 실제로 본 상태 중 가장 비슷한 상태를 반환한다.
- `neural_delta`: 현재 Neural Delta Prophecy. 상태 변화량과 종료 상태를 학습하고 replay와 3개 모델 평균을 사용한다.
- `transition_prefix`: `[현재 상태, 행동, 다음 상태]`를 하나의 전이 단위로 보고, `[현재 상태, 행동]`만 입력해 다음 상태를 맞힌다. 첫 구현은 작은 causal Transformer지만 핵심은 Transformer 자체가 아니라 전이 prefix 사전학습 방식이다.

주 결과는 전체 다음 상태 완전 일치율이다. 숫자 평균 오차, 사실 집합, 가능한 행동, 성공·실패도 따로 기록한다. 일반 이동과 상태 단계가 바뀌는 전이, 성공, 실패도 분리한다.

## 3. Neural Prophecy 상승 원인 분리

다음 조건을 함께 실행한다.

- `ablation_direct_target`: 같은 MLP와 replay를 사용하되 변화량 대신 다음 상태를 직접 예측한다.
- `ablation_single_model`: 3개 모델 평균을 하나의 모델로 줄인다.
- `ablation_no_replay`: 과거 전이를 다시 뽑지 않고 직전 전이만 한 번 학습한다.
- `ablation_16_value_state`: 사용된 칸 9개를 명시적으로 넣지 않고 기존 16개 상태 값만 사용한다.

이 비교를 통해 개선이 변화량 예측, replay, ensemble, 더 완전한 상태 표현 중 어디에서 왔는지 분리한다.

## 실행

```bash
python scripts/run_critic_prophecy_audit.py \
  --output runs/critic_prophecy_audit \
  --seed 7
```

결과는 `summary.json`에 저장된다. Critic 결과와 Prophecy 결과는 서로 다른 절에 기록된다.
