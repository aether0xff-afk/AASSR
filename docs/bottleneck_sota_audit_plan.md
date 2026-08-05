# AASSR 병목·정체성·SOTA 비교 및 확대 실험 계획

## 1. 목적

이번 단계의 질문은 단순히 "AASSR이 DQN보다 낮다"가 아니다.

1. 성능과 계산 병목이 어느 모듈에서 발생하는가?
2. 그 병목은 AASSR의 정체성 때문에 반드시 감수해야 하는가?
3. 정체성을 유지하면서 교체하거나 줄일 수 있는 구현 선택은 무엇인가?
4. 현재 AASSR이 최신 강한 방법보다 실제로 나은 부분과 나쁜 부분은 무엇인가?
5. 작은 5-seed 진단에서 확인한 원인이 확대 실험에서도 유지되는가?

## 2. AASSR의 정체성 기준

다음은 유지해야 한다.

- **Policy / Prophecy / Imagination 분리**: 무엇을 할지, 세계가 어떻게 변할지, 여러 미래를 비교하는 기능이 분리되어야 한다.
- **학습된 Prophecy**: 환경 규칙을 사람이 직접 넣는 것이 아니라 경험으로 배워야 한다.
- **counterfactual multi-step Imagination**: 실제로 실행하기 전에 여러 행동과 결과를 비교해야 한다.
- **지식·효과의 명시적 재사용**: 완전한 상태 암기보다 전이 효과와 지식을 새로운 상태에 조합할 수 있어야 한다.
- **최종 외부 보상 중심**: 환경별 중간 보상과 정답 경로를 사람이 설계해서는 안 된다.

다음은 정체성이 아니라 현재 구현 선택이다. 성능을 막으면 교체해야 한다.

- 완전한 상태를 키로 쓰는 tabular Policy
- 완전한 상태 또는 전체 vector를 거의 그대로 쓰는 Prophecy lookup
- 모든 exploit step마다 동일한 깊이와 beam으로 상상하기
- 사람이 정한 고정 `StateDeltaScorer` 가중치
- 실패 transition을 거의 한 번만 사용하는 episodic Monte-Carlo credit
- 고정 depth, branching, beam, outcome sample
- 상상 결과를 다음 결정에 amortize하거나 cache하지 않는 구조

## 3. 현재 코드에서 보이는 1차 병목 가설

### 3.1 Policy 표현 일반화

`ContextualPolicy`는 `(전체 vector, 정렬된 facts, action)`을 사실상 exact key로 사용한다. 다른 procedural map에서 좌표와 used-cell 조합이 달라지면 경험을 거의 재사용하지 못한다.

DQN은 동일 25차원 입력을 신경망으로 공유하므로 좌표·경계·used-cell 패턴을 여러 맵에 걸쳐 일반화한다. 현재 strict benchmark에서 DQN이 AASSR보다 강한 가장 직접적인 구조 차이다.

**판별 실험:** tabular Policy만 DQN value network로 교체하고 Prophecy·Imagination은 그대로 둔 `hybrid_current_model`.

### 3.2 Prophecy의 전이 일반화와 calibration

`TabularProphecy`는 exact state-action이 없으면 같은 opaque action의 모든 결과를 하나의 global family에 섞는다. `EffectComposedProphecy`도 exact context가 전체 vector를 포함하므로 다른 좌표에서 exact effect를 재사용하기 어렵다. family fallback은 북/남/동/서 선택의 결과를 서로 다른 위치와 phase에서 섞어 multimodal하고 부정확해질 수 있다.

부정확한 모델에서 깊은 상상을 하면 model error가 누적된다. 계산량은 늘지만 좋은 행동을 고를 근거는 약해진다.

**판별 실험:** DQN Policy를 유지하면서 Prophecy만 완전한 oracle로 바꾼 `hybrid_oracle_model`.

### 3.3 imagined-state 평가 함수

현재 scorer는 `goal_progress`, 새 fact 수, 새 action 수, step cost를 사용한다. strict benchmark는 최종 성공 직전까지 `goal_progress=0`이고 action set도 항상 네 개다. 따라서 실제 목표에 가까워지는 이동과 멀어지는 이동이 비슷한 값을 받을 수 있다.

**판별 실험:** 학습된 Prophecy는 유지하고, oracle 최단거리 변화로만 imagined transition을 평가하는 `hybrid_current_model_oracle_score`.

이 scorer는 원인 진단용이며 최종 모델에는 넣지 않는다. 정답 경로를 사람이 제공하므로 AASSR 정체성에 어긋난다.

### 3.4 계획 탐색과 계산량

현재 full AASSR은 행동당 약 17.7ms, DQN은 약 63.7μs였다. multi-step planning 자체는 정체성에 필요하지만, 매 step 동일한 tree를 처음부터 만드는 것은 필요하지 않다.

**판별 실험:** 완전한 모델과 진단 scorer에서 깊은 tree와 `depth=4, interval=4` budgeted tree를 비교한다. 성능이 같다면 현재 계산 병목 대부분은 불필요한 구현 비용이다.

### 3.5 탐험과 credit assignment

AASSR Policy는 성공한 episode의 final return을 역방향으로 한 번 반영한다. DQN은 replay buffer와 TD bootstrap으로 같은 transition을 여러 차례 재사용한다. 희소보상에서 최초 성공이 드물 때 이 차이가 매우 크다.

후속 개선 후보:

- Policy replay
- n-step TD / eligibility trace
- Prophecy uncertainty와 연결된 optimistic exploration
- 실제 transition과 imagined transition의 분리된 replay

## 4. 진단 조건

| condition | 바꾸는 것 | 질문 | 최종 모델 후보인가? |
|---|---|---|---|
| `dqn` | 외부 baseline | 신경망 value와 replay만으로 얼마나 일반화하는가? | 비교 기준 |
| `aassr_current` | 없음 | 현재 전체 구조의 성능은? | 현재 모델 |
| `hybrid_current_model` | Policy만 DQN network로 교체 | tabular Policy가 1차 병목인가? | 예 |
| `hybrid_oracle_model` | Policy=DQN, Prophecy=완전 모델 | 모델 오차를 없애면 현재 scorer/search가 작동하는가? | 아니오 |
| `hybrid_current_model_oracle_score` | scorer만 정답 거리 사용 | Prophecy가 완벽한 평가 아래에서도 막는가? | 아니오 |
| `hybrid_oracle_model_oracle_score` | 모델과 scorer 모두 완전 | 현재 tree search의 상한은? | 아니오 |
| `hybrid_oracle_budgeted` | 완전 모델/scorer, 얕고 드문 planning | planning 비용 중 불필요한 부분은? | 구조만 후보 |
| `oracle_bfs` | 환경 완전 탐색 | 환경 상한과 최단경로 검산 | 아니오 |

## 5. 최신 강한 계열과의 구조 비교

이 custom GridPush에 공인 SOTA leaderboard는 없다. 따라서 "SOTA보다 높다"는 표현은 직접 같은 benchmark에서 구현해 비교한 경우에만 사용한다. 아래는 구조적 비교다.

### DreamerV3

DreamerV3는 recurrent latent world model 안에서 actor와 critic을 imagined trajectories로 학습한다. 여러 domain에서 한 설정을 사용하도록 normalization과 loss 설계를 안정화했다.

- AASSR보다 강한 점: 공유 latent representation, replay 기반 world-model 학습, imagined actor-critic의 amortized policy improvement, 큰 data/model scale.
- AASSR이 잠재적으로 강한 점: 명시적인 effect·fact trace, 작은 메모리, 행동 개입 이유를 root별로 검사 가능, symbolic knowledge 결합.
- 이번 측정: 작은 약 3,470 real-transition 예산에서는 DreamerV3 0%, AASSR 0.8% seen이었으나 둘 다 거의 실패했으므로 우위 주장에 쓰지 않는다.

참고: Hafner et al., *Mastering Diverse Domains through World Models*, arXiv:2301.04104.

### EfficientZero V2

EfficientZero V2는 learned representation, dynamics, reward/value prediction과 planning을 제한된 데이터에서 함께 최적화하며 discrete/continuous, visual/state input을 다룬다. 논문은 66개 평가 중 50개에서 DreamerV3보다 높은 결과를 보고한다.

- AASSR보다 강한 점: representation learning, value prefix/return modeling, replay와 reanalysis, 학습된 latent planning score.
- AASSR이 유지할 차별점: 사람이 읽을 수 있는 KK/effect, explicit Goal/Prophecy/Imagination 분리, 새로운 symbolic action template 조합 가능성.

참고: Wang et al., *EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data*, arXiv:2403.00564.

### TD-MPC2

TD-MPC2는 decoder-free latent world model과 local trajectory optimization을 결합해 많은 continuous-control task에서 한 설정으로 강한 성능을 보인다.

- AASSR보다 강한 점: latent model/value의 공동 학습, decision-time planning이 연속 optimization으로 효율적, 규모 확장 검증.
- 직접 비교 제한: 현재 GridPush는 discrete opaque action이고 TD-MPC2의 대표 영역은 continuous control이다.

참고: Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control*, arXiv:2310.16828.

### 2025~2026 효율 개선 방향

최근 world-model 연구는 긴 sequence의 계산을 줄이는 SSM/Mamba, real+imagined data를 함께 쓰는 Dyna warmup, 모델 불확실성에 기반한 optimistic exploration을 강화하고 있다. AASSR의 다음 버전은 단순히 tree를 키우기보다 이 세 방향을 받아들여야 한다.

- Drama: Mamba 기반 linear sequence complexity와 작은 world model.
- Optimistic World Models: sparse reward에서 optimistic dynamics learning을 DreamerV3/STORM에 결합.

이 방법들은 아직 AASSR custom benchmark에서 직접 실행하지 않았으므로 구조적 참고일 뿐 성능 비교 결과가 아니다.

## 6. 현재 AASSR이 나은 부분

현재 실험으로 지지되는 범위만 적는다.

- AASSR full peak RSS 약 66MB로 DQN 약 307MB, DreamerV3 약 1.32GB보다 작았다.
- 성공한 두 seen episode는 모두 oracle 최단경로였다. 단 표본이 2개이므로 일반적 우위로 확정할 수 없다.
- 기존 dynamic-affordance GridPush에서 full Imagination은 Policy-only보다 seen/unseen 성공률을 높였다.
- Imagination의 root 평가, intervention, model coverage와 effect source를 직접 기록할 수 있어 실패 원인을 모듈별로 추적하기 쉽다.

## 7. 현재 AASSR이 나쁜 부분

- strict procedural benchmark의 성공률과 일반화: DQN 우위.
- 같은 transition 수 구간의 sample efficiency: DQN 우위.
- 행동 선택 latency와 전체 계산량: DQN 우위.
- replay와 TD credit assignment 부재.
- Prophecy의 out-of-distribution state calibration과 multi-step error 누적.
- hand-written imagined-state score 의존.
- fixed tree budget으로 인한 불필요한 계산.
- end-to-end representation learning 부재.

## 8. 해결 순서

진단 결과에 따라 다음 순서로 고친다.

1. **Policy 병목 확인**: neural Policy hybrid가 DQN 수준을 회복하는지 확인.
2. **Prophecy 병목 확인**: oracle model이 hybrid를 얼마나 올리는지 확인.
3. **scorer 병목 확인**: oracle score가 learned model에서 얼마나 작동하는지 확인.
4. **불필요한 planning 비용 제거**: uncertainty/value-gap 기반 adaptive depth, cache, interval 적용.
5. **일반 Prophecy 교체**: action-conditioned local/relational encoder와 delta prediction 도입. 환경 이름이나 key/door 규칙은 넣지 않는다.
6. **credit 개선**: replay+n-step TD를 Policy에 추가하되 Policy/Prophecy/Imagination 분리는 유지.
7. **학습된 imagined value**: 고정 scorer 대신 outcome/value head를 Prophecy와 함께 학습.
8. **GOAL 재도입**: downstream success/value를 실제로 올리는 상태만 GOAL로 유지.

## 9. 확대 실험

### 단계 A: 병목 pilot

- seeds: 7, 13, 21, 42, 100
- learned condition: 1,000 training episodes
- 64 train maps
- checkpoint별 seen 50 + unseen 50
- oracle upper bound 별도

이 단계는 원인을 고르는 용도다.

### 단계 B: identity-preserving 개선 비교

pilot에서 살아남은 실제 후보만 사용한다.

- seeds: 20개
- 5,000 training episodes
- 256 train maps
- checkpoint: 0, 100, 250, 500, 1k, 2k, 5k
- seen/unseen 각 200
- transition-matched, wall-clock-matched, decision-latency-matched 세 표를 별도로 작성

### 단계 C: DreamerV3 확대

- 공식 구현과 고정 commit
- 50k와 200k real transition
- GPU 실행
- 최소 10 seeds
- DQN/AASSR도 동일 real-transition budget으로 재평가

### 통계

- seed를 독립 표본으로 bootstrap confidence interval 계산
- 동일 map에 대한 paired success 비교
- success뿐 아니라 return, transition, wall-clock, action latency, RSS, model size, path efficiency 보고
- Imagination correction/harm, Prophecy one-step/multi-step error, calibration을 함께 보고

## 10. 중단 기준

다음 중 하나면 해당 개선은 채택하지 않는다.

- 성공률이 오르지만 oracle 정보나 task-specific 규칙을 사용함
- AASSR의 Policy/Prophecy/Imagination 분리를 없애 사실상 DQN/Dreamer 복제품이 됨
- seen만 개선하고 unseen이 악화됨
- 같은 transition 또는 같은 wall-clock에서 DQN보다 나빠짐
- Imagination intervention이 늘지만 correction보다 harm이 많음
- 평균만 좋아지고 seed 절반 이상에서 악화됨
