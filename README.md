# AASSR v2

논문용 P0–P5 재현 실험은
[`docs/paper_experiment_quickstart.md`](docs/paper_experiment_quickstart.md)를
따른다. Pilot/Final seed 분리, 평가 동결, 구조 전이, 창의성 전략 거리,
익명 블라인드 평가, 내부망 전용 Docker 검증이 독립된 논문 runner에
구현되어 있다. 요구사항별 구현·실행 증거는
[`docs/paper_protocol_implementation_status.md`](docs/paper_protocol_implementation_status.md)에
정리되어 있다.

AASSR v2는 기존 구현을 복사하지 않고 처음부터 다시 설계한 연구용 코드베이스다.

목표는 에이전트에게 사물별 정답 규칙이나 행동 순서를 직접 가르치는 것이 아니다. 플러그인은 명령 문법과 기본 조작만 제공하고, 어떤 정보와 행동 조합이 목표에 유용한지는 실제 경험·Prophecy·Imagination을 통해 학습한다.

현재 연구와 우위 검증의 범위는 **Policy·Prophecy·Imagination**까지다. GOAL 생성·수행 분리와 자율 Skill은 핵심 구조의 우위가 먼저 확정된 뒤 추가할 후속 확장 기능으로 분리한다.

## 현재 연구 브랜치와 진행 상황

기준일: **2026-08-05**

- 활성 연구 브랜치: `codex/aassr-v2-bottleneck-sota-audit`
- 비교 실험 기반 브랜치: `codex/aassr-v2-baseline-efficiency-benchmark`
- Pull Request: **#18**, draft·미병합
- 현재 목적: AASSR의 병목을 Policy, Prophecy, imagined-state scoring, Imagination 탐색 비용으로 분리하고, 각 병목이 AASSR의 정체성에 필요한지 판별한다.
- GOAL·Skill은 본 우위 검증에서 제외하고 후속 기능으로 유지한다.

### 동일 환경 baseline과 확대 pilot

고정된 네 개의 불투명 행동, procedural map, 마지막 성공에만 외부 보상을 주는 strict GridPush에서 같은 관측·행동·seed 조건으로 비교했다.

2,000 training episodes, 128 train maps, 5 seeds, seed별 seen/unseen 각 100회의 확대 pilot 결과:

| 조건 | 학습한 맵 | 처음 보는 맵 |
|---|---:|---:|
| 현재 AASSR | 1.2% | 0.0% |
| Neural Prophecy + Imagination | 29.4% | 25.6% |
| Neural Prophecy + 단순 adaptive depth | 24.8% | 22.2% |
| Neural Prophecy + holdout calibration | 31.6% | 29.0% |
| Neural Prophecy + 보수적 calibration | **40.4%** | **37.6%** |
| DQN | **69.0%** | **50.4%** |

따라서 현재 AASSR이 DQN보다 우수하다고 주장할 수 없다. strict GridPush에서는 DQN이 성공률과 계산 효율 모두 앞선다.

반면 원인 분리용으로 완전한 Prophecy를 넣고 현재 scorer와 multi-step Imagination을 유지한 조건은 seen/unseen 평균 약 **98% / 98%**를 기록했다. Oracle Prophecy는 최종 모델 후보가 아니라 상한과 병목 위치를 확인하기 위한 진단 조건이다.

### 현재까지 확인된 병목

1. **가장 큰 병목은 Prophecy의 일반화다.**
   - exact/tabular 상태 키는 좌표나 맵이 달라진 동일한 전이 규칙을 재사용하지 못한다.
   - Neural Delta Prophecy로 바꾸자 기존 AASSR보다 seen `+28.2%p`, unseen `+25.6%p` 개선됐다.

2. **두 번째 병목은 Prophecy 신뢰도의 calibration이다.**
   - ensemble끼리 같은 예측을 한다는 사실은 그 예측이 실제로 맞다는 뜻이 아니다.
   - 학습에 쓰지 않은 실제 전이에서 vector, facts, available actions, goal progress, terminal 정확도를 검증하고 개입을 제한하자 unseen 성능이 추가로 약 `+12.0%p` 개선됐다.

3. **현재 one-step 정확도만으로 multi-step 개입 안전성을 보장할 수 없다.**
   - 일부 seed의 intervention audit에서는 vector 예측이 높아 보여도 Imagination 개입의 harm이 correction보다 많았다.
   - ensemble 평균으로 만든 하나의 가상 상태와 depth가 늘어날수록 누적되는 오차가 다음 병목으로 남아 있다.

4. **Imagination 자체는 제거 대상이 아니다.**
   - 완전한 모델을 주면 현재 Imagination이 과제를 거의 해결한다.
   - 다만 매 step마다 같은 depth·beam을 여는 고정 계산 예산은 정체성이 아니라 구현 선택이므로 개선해야 한다.

5. **Policy와 scorer는 1차 병목이 아니지만 현재 구현을 유지할 이유도 없다.**
   - DQN Policy에 현재 Prophecy를 연결해도 성능이 살아나지 않았다.
   - 완전한 Prophecy에서는 현재 scorer도 작동했다.
   - 따라서 우선순위는 Prophecy지만 exact Policy, replay 없는 episodic credit, 고정 scorer는 후속 교체 대상이다.

### AASSR 정체성 기준

유지해야 하는 요소:

- Policy·Prophecy·Imagination의 역할 분리
- 실제 경험으로 학습되는 Prophecy
- 여러 미래를 비교하는 counterfactual multi-step Imagination
- KK/KV, fact, transition effect의 명시적 저장과 재사용
- 왜 행동을 바꿨는지 추적할 수 있는 개입 기록
- 최종 외부 보상 중심의 문제 해결

교체하거나 제거해도 되는 구현상 병목:

- 완전한 상태를 그대로 키로 쓰는 exact/tabular Policy와 Prophecy
- ensemble 예측을 평균 내 하나의 확정 상태로 만드는 방식
- 매 step 고정 depth·beam tree search
- 고정된 hand-written StateDeltaScorer 가중치
- replay 없는 episodic Monte-Carlo식 credit assignment
- 실제 정확도가 아닌 ensemble 합의만으로 계산한 confidence

현재 의견은 다음과 같다.

> AASSR의 경쟁력은 DQN이나 Dreamer를 그대로 복제하는 데 있지 않다. SOTA 계열의 공유 표현, replay, world-model 일반화, calibration은 받아들이되, Policy·Prophecy·Imagination의 분리와 명시적 지식·효과·개입 근거는 유지해야 한다.

현재 strict benchmark에서 SOTA보다 낫다고 말할 수 있는 부분은 없다. 다만 명시적인 전이 지식과 행동 개입 감사 가능성, 지식이 행동 파라미터로 재사용되는 KPDE 환경에서의 잠재적 장점은 별도 검증 가치가 있다. 이 장점은 아직 공개 benchmark 우위로 입증되지 않았다.

## 다음 Prophecy 연구 방향: Environment Familiarization과 Solve 분리

이 절은 **다음 구현 제안이며 현재 완료된 기능이 아니다.**

현재처럼 문제 해결과 동시에 미완성 Prophecy를 바로 Imagination에 투입하면 다음 악순환이 생길 수 있다.

```text
부정확한 Prophecy
→ 잘못된 미래 상상
→ 나쁜 행동 선택
→ 편향된 상태만 방문
→ 편향된 데이터로 다시 학습
```

이를 줄이기 위해 Prophecy 학습을 세 단계로 분리한다.

### 1. Environment Familiarization

- 보상, 목표, 성공 경로, 정답 행동을 사용하지 않는다.
- 랜덤 행동으로 시작한 뒤 낮은 방문 횟수, 새로운 effect, 새로운 fact/action 조합, 높은 모델 불확실성을 우선하는 탐색으로 전환한다.
- 실제로 도달한 상태만 사용해 `(상태, 행동, 다음 상태)` 전이를 수집한다.
- Prophecy는 좋은 행동이 아니라 환경의 world dynamics만 학습한다.

```text
입력
- 현재 상태 또는 관측·행동 history
- 다음 후보 행동

정답
- state delta
- 추가·제거된 facts
- 열린·닫힌 available actions
- active / success / failure terminal class
- 각 출력의 불확실성
```

### 2. Solve

- Policy가 행동 후보를 제안한다.
- 충분히 검증된 Planning Prophecy가 후보별 미래 분포를 예측한다.
- Imagination은 여러 경로를 비교하되, Prophecy confidence와 holdout calibration이 낮으면 Policy를 덮어쓰지 않는다.
- 현실에서는 첫 행동만 실행하고 실제 결과를 본 뒤 다시 계획한다.

### 3. Online Adaptation

- Solve 중 새로 관측한 전이는 Online Prophecy에 계속 추가한다.
- 새 업데이트가 frozen holdout의 one-step·multi-step 정확도를 높일 때만 Planning Prophecy에 반영한다.
- Planning Prophecy는 검증된 snapshot 또는 EMA 방식으로 천천히 갱신해 세계 모델이 갑자기 흔들리는 것을 막는다.

### Transformer 기반 Prophecy에 대한 판단

Transformer는 무조건적인 정답이 아니다.

- 현재 상태가 완전한 Markov state라면 작은 MLP·GRU를 강한 baseline으로 유지한다.
- 현재 관측만으로 숨겨진 과거 정보와 환경 상태를 복원할 수 없는 부분관측 환경에서는 Transformer가 자연스럽다.

부분관측 Prophecy의 권장 입력은 다음과 같다.

```text
(o0, a0, o1, a1, ..., ot, 후보 행동 at)
→ 다음 상태 변화 분포
```

history에는 상태 vector뿐 아니라 facts, available actions, KK/KV, 관측 mask, WHAT/HOW/WHERE 행동 표현을 포함한다.

권장 학습 목표:

```text
L =
  state-delta loss
+ fact-change loss
+ action-availability loss
+ terminal-class loss
+ multi-step rollout loss
+ confidence calibration loss
```

one-step만 학습하지 않고 같은 rollout에서 2·3·5-step 예측 쌍을 함께 만든다. 일부 학습 구간에는 실제 다음 상태가 아니라 Prophecy가 예측한 상태를 다시 입력해, 실제 Imagination에서 자기 오차가 누적될 때의 안정성도 학습한다.

### 평균 미래 대신 여러 미래 유지

현재 neural ensemble의 평균을 하나의 상태로 decode하면 실제로 존재하지 않는 중간 상태가 만들어질 수 있다. 다음 Imagination은 ensemble member 또는 확률적 outcome을 별도 branch로 유지해야 한다.

```text
같은 상태·행동
├─ model/outcome 1
├─ model/outcome 2
└─ model/outcome 3
```

행동 개입은 평균적으로 좋아 보이는 경로가 아니라, 여러 plausible model에서 충분히 안전하거나 risk-adjusted value가 명확히 높은 경우에만 허용한다.

### Familiarization 종료 기준

고정 episode 수만으로 Solve로 넘어가지 않고 world-model 자체의 준비도를 본다.

- frozen holdout one-step 정확도
- 3·5-step rollout 정확도
- fact/action/terminal 정확도
- 행동별 최소 coverage
- 최근 데이터 추가에 따른 validation 개선량
- OOD 또는 ensemble disagreement

정확한 임계값은 pilot으로 결정하며, 성공률이나 목표 보상을 Familiarization 종료 기준으로 사용하지 않는다.

### 비교해야 할 조건

같은 real-transition budget으로 다음을 비교한다.

1. 기존 joint online learning
2. Familiarization 후 Prophecy 동결
3. Familiarization + online adaptation
4. 기존 Neural Delta Prophecy warm-up control
5. Transformer one-step
6. Transformer multi-step
7. ensemble branching + calibrated intervention
8. Oracle Prophecy 상한

공정성을 위해 두 표를 별도로 낸다.

- **총 transition matched:** Familiarization과 Solve를 합친 실제 환경 상호작용 수를 동일하게 맞춤
- **Solve transition matched:** 동일한 Solve 경험에 pretrained Prophecy 제공 효과를 따로 측정

### 확대 실험 순서

구조 후보가 pilot에서 살아남은 뒤에만 큰 실험으로 확대한다.

1. strict 좌표 일반화 환경
   - 20 seeds, 5,000 training episodes, 256 train maps, seen/unseen 각 200
   - transition·wall-clock·decision-latency·memory matched 결과를 분리
2. KPDE 전용 환경
   - 획득한 KK/KV가 이후 행동의 HOW/WHERE 파라미터로 재사용됨
   - 중간 외부 보상 없이 지식 획득과 긴 의존 관계가 필요함
   - 새로운 환경에서 기존 effect를 재조합해야 함
3. 공개 world-model benchmark
   - DreamerV3 등은 충분한 GPU와 50k/200k real-transition budget으로 비교
   - 작은 CPU pilot 결과를 알고리즘 전체의 최종 성능으로 일반화하지 않음

최종 판단은 성공률 하나가 아니라 다음을 함께 본다.

- 성공률과 return
- 최단 경로 대비 효율
- 실제 환경 transition 수
- 학습·추론 벽시계 시간
- 행동당 Imagination node와 latency
- peak memory와 모델 크기
- Prophecy one-step·multi-step 정확도
- Imagination correction/harm 비율
- 처음 보는 환경에서의 재사용성

## Escape GridWorld 연구 GUI

현재 가장 쉽게 AASSR의 학습 과정을 관찰할 수 있는 실행 환경은 색 열쇠·색 문 Escape GridWorld다.

- 상자 안의 색 열쇠를 찾아 같은 색 문을 연 뒤 출구까지 이동
- 출구 도달만 외부 성공으로 판정
- 고정 episode tick 제한 없음
- 더 짧게 성공할수록 높은 성공 점수
- 하나의 세션에서 실시간 렌더링과 최대속도 학습 즉시 전환
- 모든 step·episode·상상 트리·모델 상태 영구 저장
- 종료 후 통계 창과 전체 그래프 자동 표시
- 별도 Imagination Viewer에서 실제 상상 트리 실시간 관찰
- 학습 모델 저장·불러오기 및 이어 학습

### 실행

```bash
python -m pip install -e ".[dev]"
python scripts/run_escape_gridworld.py --gui
```

GUI는 현재 GridWorld 창과 `AASSR Imagination Viewer` 창을 함께 연다.

### 같은 세션에서 속도 전환

학습 도중 다음 버튼을 언제든 누를 수 있다.

- `실시간으로 보기`: primitive step을 적당한 속도로 렌더링한다.
- `안 보고 최대 속도`: 렌더링과 인위적 대기를 중단하고 같은 학습 상태에서 최대 속도로 계속한다.

속도 전환은 현재 맵, episode, tick, Policy, Prophecy, Imagination, holdout, RNG 상태를 초기화하지 않는다.

### 에피소드와 성공 점수

에피소드는 출구에 도달할 때까지 계속된다. 출구까지 간 경우만 성공이며 점수는 다음과 같다.

```text
성공 점수 = 1 + oracle 최단 tick / 실제 성공 tick
```

최단 경로 성공은 `2.0x`, 오래 걸릴수록 `1.0x`에 가까워진다. 상자 열기나 문 열기 자체에는 외부 성공 보상을 주지 않는다.

### 전체 기록과 통계

모든 step과 episode는 실행 중 즉시 파일에 기록된다.

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

학습 완료 또는 중지 후 별도 통계 창이 자동으로 열린다. episode별 step 수와 최근 이동평균을 필수로 표시하며, 점수, 경로 효율, 소요시간, Prediction, holdout, intrinsic value, Imagination, 오류·반복, 행동·이벤트 분포도 함께 제공한다. 그래프는 SVG 파일로도 저장된다.

### Imagination Viewer

`imagination_interval=1`은 모든 환경 tick에서 무조건 상상한다는 뜻이 아니다. 다음 조건을 만족하는 비무작위 step마다 Imagination을 실행한다.

- Imagination 기능이 켜져 있음
- epsilon random exploration이 아님
- interval 조건 충족
- Prophecy model coverage가 기본 임계값 이상

상상 창은 실제 트리의 전체 노드, 부모 관계, 깊이, 누적 가치, 신뢰도, 종료 이유, 루트 행동별 평가, 선택된 첫 행동과 최선 경로를 표시한다. 최대속도 모드에서는 화면은 최신 트리로 갱신하지만 모든 상상 트리는 `imaginations.jsonl`에 빠짐없이 저장된다.

### 모델 저장과 불러오기

세션 종료 시 정식 모델이 자동 저장된다.

```text
runs/escape_gridworld/<session>/models/final.aassr-model.gz
```

GUI 버튼:

- `모델 불러오기`: 저장 모델을 다음 세션의 초기 학습 상태로 사용
- `현재 모델 저장`: 학습 중 또는 종료 후 원하는 경로에 저장
- `새 모델로 시작`: 선택한 모델을 해제하고 처음부터 학습

모델에는 Policy, Prophecy, holdout, RNG, transition/decision index와 누적 episode가 들어간다. 불러오면 epsilon 감쇠도 저장 지점 다음부터 이어진다. 현재 episode 중간 위치와 임시 행동열은 포함하지 않으므로 새 episode부터 시작한다.

CLI 예시:

```bash
python scripts/run_escape_gridworld.py \
  --episodes 2000 \
  --colors 2 \
  --distractors 2 \
  --load-model models/my_agent.aassr-model.gz \
  --save-model models/continued_agent.aassr-model.gz
```

최대속도 headless 실행:

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

상세 사용법은 [`docs/escape_gridworld_quickstart.md`](docs/escape_gridworld_quickstart.md)를 참고한다.

> 현재 GUI는 한 procedural seed의 맵을 반복해 학습 과정을 관찰하는 도구다. 같은 맵에서의 개선만으로 처음 보는 맵 일반화를 주장하지 않는다. 일반화 검증은 train/test map seed를 분리한 별도 실험으로 수행해야 한다.

## 핵심 폐루프

```text
관측
→ 원본 경험과 Knowledge Store 기록
→ 정보 특징 생성 및 온라인 비지도 군집화
→ Policy 상위 행동 후보 생성
→ Prophecy 기반 평행우주 나무 생성
→ 가장 높은 가치의 미래로 이어지는 첫 행동 선택
→ 현실에서 선택된 첫 행동 실행
→ 실제 다음 상태로 Prophecy 학습
→ 검증 전이에서 예측 개선 측정
→ 정보 가치와 지연 기여를 Policy에 반영
→ 다음 현실 상태에서 다시 계획
```

## 0.2.0에서 구현된 구조

### 범용 행동 플러그인

코어는 `MOVE`, `SCAN`, `BREAK` 같은 단어의 의미를 가정하지 않는다. 플러인은 `ActionSchema`와 `ParameterSpec`으로 행동 문법·필수/선택 파라미터·기본값만 선언한다.

- 임의 문자열 행동 verb 지원
- 동적 파라미터와 안정적인 action signature
- 플러그인 등록 및 실행 분리
- 행동별 슬롯 후보의 2단계 선택
- 커리큘럼용 파라미터 문법 수업 자동 생성

### 정보 특징과 의미 형성

`OnlineFeatureMemory`는 정보를 이름으로 고정 분류하지 않고 관측 특징과 `행동-슬롯-결과` 경험으로 표현한다.

- 해시 특징 기본선
- 온라인 비지도 군집화와 재배치
- 군집 역할 점수와 개별 정보 점수
- 군집 선택 후 구체 값 선택
- 대규모 관측에서만 임베딩을 쓰는 선택적 라우터
- 경험 특징·임베딩·혼합 표현 기능 제거 설정

### 후속 확장 기능: GOAL과 자율 Skill

GOAL과 자율 Skill은 현재 AASSR v2의 필수 코어와 우위 주장에 포함하지 않는다. 관련 구현과 진단 코드는 연구 중인 후보로만 유지하며, Policy·Prophecy·Imagination의 성능과 일반성이 먼저 독립적으로 검증된 뒤 정식 기능으로 추가한다.

후속 설계 방향은 다음과 같다.

- GOAL Maker는 Imagination으로 도달할 중간 상태를 생성
- GOAL Executor는 전달받은 상태에 가까워지는 행동을 수행
- Maker와 Executor를 분리해 GOAL 생성과 실행을 독립적으로 평가
- 같은 GOAL을 반복해서 해결한 ASeq를 이후 Skill 후보로 확장
- GOAL 남발, 잘못된 GOAL 고착, 계산량 증가를 별도 실험으로 검증

현재 README의 핵심 폐루프와 본 실험 결과를 해석할 때 GOAL·Skill 성능을 AASSR의 입증된 장점으로 사용하지 않는다.

### 순환형 Prophecy와 평행우주 Imagination

- 순수 Python 온라인 GRU Prophecy
- 실제 전이에 대한 one-step truncated backpropagation
- 우주별 독립 GRU 은닉 상태
- 관측된 다음 상태 템플릿과 예측 벡터의 근접 검색
- 기본 `n=2` 분기, Beam 가지치기, 신뢰도 기반 가변 깊이
- `max`, `mean`, `top_mean`, `risk_adjusted` 집계
- 현실에서는 첫 행동만 실행한 뒤 다시 계획

### 정보 가치 학습

최근 행동을 바로 외운 성능을 보상하지 않도록 학습과 검증을 분리한다.

- 학습 전이와 holdout 전이 분리
- KK 문맥 갱신 효과와 Prophecy 파라미터 갱신 효과 분리
- 검증 전이의 예측 개선량 측정
- 새로 열린 행동의 지연된 실제 가치 추정
- 반복·오류 감점
- 최종 결과를 ASeq에 할인 역배분
- 정보 가치 예측기와 Policy 강화 연결
- 모든 상태·행동·예측·지표 JSONL 직렬화

### 커리큘럼과 반례 환경

자동 Teacher는 성공률 창을 보고 기본기 단계를 올리거나 내린다. 첫 단계 외에는 정답 행동 시범을 제공하지 않는다.

- 기본 조작과 관찰
- 장애물 우회
- 물체 획득
- 상태 변화
- 긴 의존 관계
- 속성 관계
- 처음 보는 복합 환경
- 플러그인 필수·선택 파라미터 문법

반례 환경에는 무관한 대량 정보, 학습 가능한 인과와 순수 무작위성, 불투명 이름, 무작위 배치, 긴 의존 사슬이 포함된다.

### 확장 검증용 플러그인

- `SandboxEnv`: 관찰, 부수기, 설치, 조합. 숨겨진 recipe는 플러그인 문법에 노출되지 않는다.
- `MinecraftControlPlugin`: 이동, 시점 변경, 버튼 입력, 상호작용의 dry-run 연결 규약.
- `AuthorizedAssessmentPlugin`: 승인된 대상만 허용하는 추상적 scan/connect/read 규약. exploit·shell command는 생성하지 않으며 실제 도구 연결은 외부 transport가 담당한다.

Minecraft와 모의 침투 테스트 항목은 **코어 호환성과 안전한 연결 규약까지 구현**된 상태이며 실제 게임 클라이언트나 네트워크 도구를 이 저장소에서 직접 실행하지 않는다.

## 기능 제거 실험

`ablations.py`는 다음 비교 설정을 자동 생성한다.

- 분기 수 `1/2/3`
- 깊이 `1/2/3` 및 적응형 깊이
- 우주 집계 `max/mean/risk_adjusted`
- 정보 가치 제거
- 특징 없음, 특징만, 군집, 2단계 선택, 온라인 재군집, 행동-슬롯 문맥
- 경험 특징, 임베딩 특징, 혼합 특징

GOAL·Skill 관련 제거 설정은 후속 확장 진단용으로만 남기며, 현재 본 실험의 우위 판정에서는 제외한다.

## 시범 없는 자율 본 실험

기존 `final_pilot.py`는 전이 경로를 미리 학습시킨 뒤 Imagination 모듈 자체를 확인하는 **진단 실험**이다. 자율 발견의 근거로 사용하지 않는다.

`autonomous_main` runner는 다음을 강제한다.

- oracle pretraining 없음
- 정답 행동이나 경로 시범 없음
- `safe`, `trap`, `finish` 같은 의미 있는 행동명 없음
- 중간 보상과 중간 goal progress 없음
- 모든 조건의 Policy 초기값 동일
- seed별 행동 매핑과 상태 표현 무작위화
- 학습 후 평가에서는 탐색과 갱신 중지

에이전트는 전역 action weight 대신 상태별 `ContextualPolicy`를 사용한다. 전이 coverage가 부족할 때는 epsilon/UCB로 탐색하고, 충분한 실제 경험이 쌓인 뒤에만 Prophecy와 Imagination을 사용한다.

축소 검증:

```bash
python scripts/run_experiment.py --config configs/autonomous_smoke.json --dry-run
python scripts/run_experiment.py --config configs/autonomous_smoke.json --output runs/autonomous_smoke --overwrite
```

본 실험:

```bash
python scripts/run_experiment.py --config configs/autonomous_main.json --dry-run
python scripts/run_experiment.py --config configs/autonomous_main.json --output runs/autonomous_main --overwrite
```

본 실험은 20 seeds, 의존 길이 4/6/8, 5개 조건, train 2,000 + evaluation 200 episode로 총 `660,000`개 결과 행을 계획한다. 주장, 누수 방지 규칙, 비교 조건과 성공 판정은 [`docs/main_experiment_design.md`](docs/main_experiment_design.md)에 정리되어 있다.

### 진행률·ETA·로그

`autonomous_main`은 전체 episode 기준 진행률, 처리 속도, 경과시간, ETA, 현재 seed·환경·조건·phase를 콘솔에 출력한다.

```text
[AASSR:progress]  37.42% 246,972/660,000 |   84.31 ep/s | elapsed 00:48:49 | ETA 01:21:41 | job=113/300 | seed=211 | environment=opaque_dependency_l6 | condition=full_aassr | phase=training | episode=972/2000 | recent_success=0.830
```

기본 출력 주기는 설정 파일의 `progress.every_episodes`와 `progress.every_seconds`로 정하고 CLI에서 덮어쓸 수 있다.

```bash
python scripts/run_experiment.py \
  --config configs/autonomous_main.json \
  --output runs/autonomous_main \
  --overwrite \
  --progress-every 50 \
  --progress-seconds 5
```

실행 중 다음 파일이 계속 갱신된다.

```text
progress.log
progress.jsonl
progress.json
episodes.csv
```

PowerShell에서 로그를 실시간 확인하려면:

```powershell
Get-Content runs/autonomous_main/progress.log -Wait
```

상세 사용법은 [`docs/progress_monitoring.md`](docs/progress_monitoring.md)를 참고한다.

## 기존 진단·배치 실험 실행

설정 문법과 실행 규모만 확인:

```bash
python scripts/run_experiment.py --config configs/pilot.json --dry-run
```

최종 진단 파일럿 실행:

```bash
python scripts/run_experiment.py --config configs/pilot.json --output runs/final_pilot --overwrite
```

결과는 다음으로 저장된다.

```text
runs/<experiment>/
├─ resolved_config.json
├─ protocol_manifest.json
├─ progress.log
├─ progress.jsonl
├─ progress.json
├─ episodes.csv
├─ seed_summary.csv
├─ summary.csv
└─ report.md
```

기존 실험 설정:

- `configs/prophecy.json`
- `configs/imagination.json`
- `configs/dependency.json`
- `configs/goals_skills.json` — 후속 GOAL·Skill 확장 진단용
- `configs/information_value.json`

자세한 기존 명령은 [`docs/experiments.md`](docs/experiments.md)를 참고한다.

## 검증

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src tests scripts
pytest -q
```

주요 테스트:

- `tests/test_autonomous_experiment.py`: 자율 학습 구조와 누수 방지
- `tests/test_progress_reporting.py`: 진행률·ETA·영속 로그
- `tests/test_escape_gridworld.py`: Escape 환경·무제한 episode·점수
- `tests/test_escape_imagination_capture.py`: 전체 상상 트리 저장
- `tests/test_escape_model_io.py`: 모델 저장·복원·호환성
- `tests/test_experiment_runner.py`: 배치 실험기·진단 파일럿·seed 통계

## 버전

- 연구 세대: **AASSR v2**
- 코드 패키지: **0.2.0**
- 안정 개발 브랜치: **aassr-v2**
- 활성 연구 브랜치: **codex/aassr-v2-bottleneck-sota-audit**
- 비교 실험 기반 브랜치: **codex/aassr-v2-baseline-efficiency-benchmark**
- 활성 연구 PR: **#18**, draft·미병합
- 기존 `main` 및 이전 AASSR 구현은 수정하지 않는다.
