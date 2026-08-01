# AASSR Paper Protocol v2 구현 및 실행 보고서

이 보고서는 2026-08-01의 `aassr-v2` 기준 로컬 구현과 실행 결과를
기록한다. Development Diagnostic은 연구 증거가 아니며, Locked
Confirmation과 Pilot을 분리한다. 기존 Final v1 결과와 config는 수정하거나
삭제하지 않았다. Protocol v2 Final 및 실제 Minecraft 런타임은 실행하지
않았다.

## 1. Final v1 실패 원인 중 코드로 확정된 것

- `evaluation_seen`은 train world replay가 아니라 별도 disjoint world seed를
  사용했다. 따라서 이름과 달리 실제 train-world frozen 평가가 없었다.
- Contextual Policy와 tabular Q-learning은 `(state identity, action identity)`를,
  DQN은 opaque state/action feature를 사용했다. world마다 action signature,
  fact, vector slot이 바뀌므로 안정적인 cross-world causal key가 없었다.
- 동점은 action signature 사전순으로 결정되어 unseen state에서 같은 opaque
  token을 반복 선택할 수 있었다.
- private `corrupted` 상태가 observation에서 빠졌고 Prophecy는 observable
  next state만 학습했다. 따라서 높은 prediction score가 terminal success
  예측을 뜻하지 않았다.
- v1 Imagination에는 return calibration, OOD, 같은 단위의 policy/model Q
  margin gate가 없었다.
- v1 creativity 환경은 terminal operation을 미리 열거했고 같은 run의
  baseline pool이 닫힌 graph space를 포괄할 수 있었다.

근거와 v1 trace 사례는 `docs/final_v1_root_cause_analysis.md`에
`코드 확정/실험 근거`로 분리했다. 결론은 하나의 버그가 아니라 phase 의미,
환경 설계, representation, Prophecy target, creativity 해 공간이 결합된
문제다.

## 2. 추정만 가능한 것

- deterministic tie-break가 실패 반복을 얼마나 증폭했는지는 v1만으로
  분리할 수 없다.
- v1 Imagination이 policy-only보다 실제로 악화시켰는지는 counterfactual
  action log가 없어 확정할 수 없다.
- novelty 0은 닫힌 reference space로 설명 가능하지만 agent가 원천적으로
  compositional creativity가 없다는 증거는 아니다.
- v2에서 relational transfer, learned Prophecy, gated Imagination이 일반적으로
  우수하다는 가설은 이번 소규모 Confirmation으로 확정되지 않았다.

## 3. 변경한 파일 목록

PR-ready 전달 단위는 아래 순차 branch/commit이다.

1. `codex/aassr-v2-pr1-audit` — `fd30551`
2. `codex/aassr-v2-pr2-protocol` — `cda5d64`
3. `codex/aassr-v2-pr3-causal-world` — `e95351f`
4. `codex/aassr-v2-pr4-relational` — `347e27d`
5. `codex/aassr-v2-pr5-prophecy` — `79af6ad`
6. `codex/aassr-v2-pr6-imagination` — `c83e9bc`
7. `codex/aassr-v2-pr7-transfer-creativity` — `fa7a63f`
8. `codex/aassr-v2-pr8-minecraft` — Minecraft code 기준 `e0e070e`

주요 변경은 다음과 같다.

- 감사/보호: `docs/final_v1_root_cause_analysis.md`,
  `configs/final_v1_preservation_manifest.json`, `v2_immutability.py`
- Protocol: `paper_v2_types.py`, `paper_v2_protocol.py`,
  `paper_v2_runner.py`, v2 run/lock/freeze CLI
- 환경/표현: `causal_dependency_world.py`, `causal_representation.py`,
  `representation_diagnostic.py`
- 모델/계획: `causal_prophecy.py`, `causal_agent_v2.py`,
  `causal_imagination.py`, `imagination_diagnostic_v2.py`
- 전이/창의성: `transfer_diagnostic_v2.py`, `open_creativity_v2.py`
- Minecraft-like: `minecraft_causal_world.py`,
  `minecraft_diagnostic_v2.py`, Semantic/Opaque별 config 및 실행 CLI
- 테스트: Protocol v2 전용 8개 test module

기존 `paper_final_review/`, `scripts/build_paper_final_review.py`,
`paper_results/`, v1 config/API는 변경하지 않았다.

## 4. 새 환경의 인과법칙

`CausalDependencyWorldV2`는 world 사이에 유지되는 primitive law와 world별
token/layout/resource/composition 변형을 분리한다. 공통 effect는 정보 획득,
parameter binding, 자원 획득, 도구 형성, 장애물 제거, 위험 감소, 경로 생성,
목표 달성이다. 기본 `strict_sparse`는 terminal reward만 제공한다.

모든 world는 실행 전에 bounded exact solver로 검사한다. solvable, 최소 경로
3~12, valid solution 2개 이상, causal family 3개 이상, reachable dead end와
irreversible decision, random success 0.10 이하, shortcut와 private leak 여부를
manifest에 기록한다. Confirmation의 모든 world certification이 통과했다.

Open creativity world는 695개 feasible graph와 11개 family를 열거했다.
동결 baseline 14개 밖의 graph는 681개였고 reference coverage는 0.02014였다.

## 5. agent-visible 정보와 private 정보

Agent-visible `RawCausalObservation`에는 inventory, opaque observable facts,
available actions/선택적 affordance, 실제 resource cost, health/damage,
observable spatial change, action success/failure, terminal reward만 있다.
`strict_sparse`에는 explicit/normalized goal progress가 없다.

`PrivateWorldState`에는 true causal graph/family/viability, latent risk,
optimal plan, oracle transition과 solver용 effect history가 있다. 이 객체는
agent observation으로 직렬화하지 않는다. true prerequisite/effect/family를
supervised agent target으로 사용하지 않는다. Development와 Confirmation의
private leak gate는 모두 0건으로 통과했다.

## 6. 상태·효과 표현 구조

하나의 raw-observation pipeline 뒤에서 `IdentityEncoder`와
`RelationalEffectEncoder`가 분기한다. observation, action space, reward,
transition/update budget, exploration schedule과 capacity를 같게 검증한다.

Relational encoder는 visible transition delta로 `LearnedEffectMemory`를
갱신한다. prerequisite, effect, risk, return probability는 실행 history에서
귀납하며 private graph를 읽지 않는다. true effect를 받는 조건은
`privileged_effect_upper_bound`로만 허용하고 본 비교에서 제외한다.

Diagnostic 2A는 action schema를 유지하고 state/layout token만 바꾸며,
2B는 action token도 바꾸고 `[1,4,16]` probing budget을 제공한다.

## 7. Prophecy target 변경

v2.0 empirical Prophecy는 next observable state, observable effect delta,
unlock probability, discounted terminal-return probability를 예측한다.
Success head target은 private viability가 아니라 episode 종료 후 관측된
`gamma^(T-t) * terminal_success`뿐이다.

v2.1 interface는 resource cost, damage/risk proxy, empirical visit-count와
frozen holdout error 기반 uncertainty, OOD와 reliability-bin calibration을
추가한다. state/effect/unlock/return calibration metric을 분리한다. GRU는
tabular interface 진단 뒤에만 허용하며 이번 실행에는 사용하지 않았다.

## 8. Imagination gating 구조

`policy_q`와 `model_q`는 모두 expected discounted terminal return `[0,1]`이다.
policy-only action을 먼저 기록하고 다음 조건을 모두 만족할 때만 learned/random
model이 action을 바꾼다.

```text
calibration confidence >= threshold
AND uncertainty <= threshold
AND OOD <= threshold
AND model_q(best) - policy_q(policy_action) >= minimum margin
```

각 decision은 policy-only/final action, root별 두 Q, uncertainty/OOD,
calibration, margin, intervention reason, imagined nodes/depth, 실제 결과를
gzip JSONL에 기록한다. Random model은 low-confidence/OOD gate 때문에 개입
0이었다. Oracle은 본 비교에서 제외되는 ungated exact upper bound로 두어
transition accuracy 100%, root optimality 100%와 regret 검증에 사용했다.

## 9. 평가 phase 의미

- `training`: 학습 활성화
- `evaluation_train_world_frozen`: 같은 train world seed와 generator spec을
  새 checkpoint clone에서 epsilon 0, 학습 0으로 평가
- `evaluation_isomorphic_world_zero_shot`: causal law/composition은 같고
  token/layout/resource 위치만 변경, 학습 0
- `evaluation_unseen_composition_zero_shot`: primitive law hash는 같고
  composition-template hash는 다름, 학습 0
- `adaptation`: 독립 branch별 `[0,1,4,16,64]` budget만 학습
- `evaluation_unseen_after_adaptation`: adaptation 후 완전 frozen 평가

Fingerprint에는 policy, Prophecy, holdout, RNG, planner cache, counters,
replay, normalization/calibration buffer와 learned relational memory가 포함된다.
각 adaptation branch는 동일 serialized checkpoint에서 fresh clone으로 시작한다.

## 10. 테스트 결과

최종 전체 회귀 결과는 `135 passed, 3 skipped`였다. skip 3개는 환경상 선택
의존 테스트이며 Docker 안전 테스트는 Docker Desktop 경로를 명시해 통과했다.

추가 검증은 seed/phase 분리, v1 path 보호, one-shot confirmation claim,
concurrent resume lease, gzip corruption/replay, full checkpoint clone,
evaluation mutation 0, world certification 경계, raw/private 분리,
identity/relational 공정성, terminal-return target, Q 단위, low-confidence gate,
oracle transition, depth/branching tree 변화, transfer branch fingerprint,
token-invariant graph canonicalization, creativity reference hash,
Minecraft mock contract/track 분리를 포함한다.

첫 Development 실행 중 동일 run의 concurrent resume이 가능한 결함을
발견했다. active PID lease를 추가해 두 번째 writer를 차단하고, stale lock과
partial raw를 삭제하지 않고 `failed_attempts/`로 보존하도록 수정했다.

## 11. diagnostic 실행 명령

```powershell
python scripts/run_paper_suite_v2.py --config configs/paper_causal_diagnostic_v2.json --run-id second
python scripts/freeze_creativity_reference_v2.py --config configs/paper_causal_confirmation_v2.json --output configs/locks/open_creativity_reference_v2.json
python scripts/lock_paper_v2_protocol.py --config configs/paper_causal_confirmation_v2.json --output configs/locks/paper_causal_confirmation_v2.lock.json
python scripts/run_paper_suite_v2.py --config configs/paper_causal_confirmation_v2.json --run-id first

python scripts/run_minecraft_causal_suite.py --config configs/paper_minecraft_semantic_diagnostic_v2.json --run-id first
python scripts/run_minecraft_causal_suite.py --config configs/paper_minecraft_opaque_diagnostic_v2.json --run-id first
python scripts/run_minecraft_causal_suite.py --config configs/paper_minecraft_semantic_confirmation_v2.json --run-id first
python scripts/run_minecraft_causal_suite.py --config configs/paper_minecraft_opaque_confirmation_v2.json --run-id first
python scripts/run_minecraft_causal_suite.py --config configs/paper_minecraft_semantic_pilot_v2.json --run-id first
python scripts/run_minecraft_causal_suite.py --config configs/paper_minecraft_opaque_pilot_v2.json --run-id first
```

Causal/creativity Pilot 명령은 Locked Confirmation adequacy failure 때문에
실행하지 않았다. Final 명령/config는 만들거나 실행하지 않았다.

## 12. diagnostic 실제 결과

### Development Diagnostic

권위 있는 수정 후 run은
`paper_results_v2/development/paper-causal-diagnostic-v2.0/second/`이다.
72 episode rows와 3,672 replayable trace records를 기록했고 engineering
13/13, adequacy 6/6을 통과했다.

- Contextual train tail `0.4567`, frozen `0.5000`, Random `0.0500`
- Full mean absolute tail/frozen gap `0.0667`
- 2A state-token remap: Identity `0.0`, Relational `1.0`
- 2B action-token remap budget 16: Identity `0.3333`, Relational `0.0`
- transfer AUC: Relational `1.0`, from-scratch `1.0`, gain `0.0`
- policy-only success `0.6667`, learned gated `1.0`, Oracle `1.0`
- creativity: successful/novel/final candidate `0/0/0`

이 결과는 코드 수정에 사용한 Development 결과이며 연구 증거가 아니다.

### Locked Confirmation

Causal Confirmation은
`paper_results_v2/locked_confirmation/paper-causal-confirmation-v2.0/first/`에
81 rows로 불변 보존했다. Engineering은 13/13 통과했지만 adequacy는 4/6이다.

- Contextual train tail `0.8733`, frozen `1.0000`, Random `0.0467`:
  replay gap `0.1267 > 0.10`
- Full tail은 seed별 `0.81, 0.81, 0.89`, frozen은 모두 `1.0`:
  mean absolute gap `0.1633 > 0.10`
- 2A: Identity `0.0`, Relational `1.0`
- 2B budget 16: Identity `0.6667`, Relational `0.0`
- transfer는 모든 budget에서 Relational/From-scratch 모두 `1.0`; gain `0.0`
- Imagination은 모든 condition success `1.0`; Oracle regret `0`, root optimality
  `1.0`, learned/random은 policy-only 대비 success 차이 `0`
- creativity는 successful unique 9, frozen-reference novelty 7이지만 모든
  utility/reproduction 조건을 만족한 final creative candidate는 `0`

따라서 Confirmation 결과를 본 뒤 threshold나 환경을 고치지 않았고 causal 및
creativity Pilot을 차단했다.

Minecraft Semantic/Opaque Confirmation은 각각 engineering 7/7, adequacy
3/3을 통과했다. 별도 mock Pilot은 각각 6 rows를 생성하고 같은 gate를 모두
통과했다. 이는 환경/adapter 계약 검증이며 agent 성능 또는 실제 Minecraft
증거가 아니다.

### 질문별 판정

- 기존 실패는 평가 버그, 환경 설계, 모델 문제 중 무엇인가: **복합**이다.
  `evaluation_seen` 의미/phase 설계 문제, 재사용 불가능한 world identity 설계,
  reward-irrelevant Prophecy target이 모두 코드로 확인됐다.
- 같은 학습 world에서 학습 정책이 재현되는가: **Partial**. frozen success는
  Contextual/Full 모두 `1.0`이었으나 tail과의 gap `0.1267/0.1633`이 사전 기준
  `0.10`을 넘었다. 성능 붕괴는 아니지만 benchmark replay gate는 실패다.
- token만 바뀐 동형 world에서 effect representation이 유리한가: **2A Yes,
  2B No**. state-token zero-shot은 `1.0 vs 0.0`; action-token few-shot budget
  16은 `0.0 vs 0.6667`이었다.
- 정확한 Oracle model에서 Imagination이 유리한가: **Partial**. root
  optimality `100%`, transition accuracy `100%`, regret `0`이었지만
  Confirmation success는 Oracle과 policy-only 모두 `1.0`이라 success gain은
  없었다.
- learned Prophecy가 reward-relevant 미래를 예측하는가: **Partial/미해결**.
  target과 calibration interface는 terminal outcome 기반으로 검증됐지만
  Confirmation에서 별도 cross-seed prediction-calibration 통계를 내지 않았고
  learned Imagination success도 policy-only와 모두 `1.0`이었다.
- 열린 조합 환경에 reference 밖 재현 가능 전략이 발생하는가: **환경 Yes,
  agent 결과 No**. feasible 695개 중 681개가 reference 밖이었고 agent가 novel
  graph 7개를 만들었지만 최종 재현·utility 조건을 모두 통과한 후보는 0개다.

## 13. 아직 해결되지 않은 문제

- Confirmation의 train-tail/frozen gap gate가 실패해 causal/creativity Pilot이
  없다. 수정하려면 `v2.0` artifact를 건드리지 않고 새 protocol version과
  새 confirmation seed를 사용해야 한다.
- Diagnostic 3은 budget 0부터 transfer/from-scratch가 모두 100%여서 ceiling
  effect가 있고 transfer 가설을 식별하지 못했다. 이 결과 때문에 현 protocol
  환경을 사후 수정하지 않았다.
- 2B에서 relational action-effect induction이 identity보다 나빴다.
- Prophecy의 reward-relevant calibration을 seed 간 통계로 보고하는 별도
  artifact가 아직 없다.
- Confirmation novel graph 7개는 재현성과 전체 utility 조건을 만족하지 못해
  creative candidate가 아니다.
- 실제 인간 reference/evaluation과 실제 Minecraft backend 데이터는 없다.
- Protocol v2 Final config와 Final 결과는 의도적으로 없다.

## 14. 실제 Minecraft adapter로 넘어가기 위한 조건

현재 구현은 high-level `MinecraftAdapter`와 deterministic mock뿐이다.
Semantic/Opaque track의 config, 결과와 claim은 분리했다. 실제 backend로
넘어가려면 다음을 새 protocol에서 충족해야 한다.

- MineStudio/MineRL/Malmo 중 하나를 공식 문서 기준으로 선택하고 dependency,
  asset provenance, license, Python/Java/OS/display/network 요구를 동결
- deterministic reset, seed replay, structured observation 및 private-state leak
  검증
- Semantic skill 의미와 Opaque probing budget을 backend action에 명시적으로
  매핑
- mock과 실제 backend의 causal-law/adapter conformance test
- 실제 backend 전용 Confirmation을 신규 미사용 seed로 최초 1회 실행
- backend와 track이 다른 결과를 같은 표나 claim에 병합하지 않음

후보 비교는 `docs/minecraft_backend_comparison.md`에 기록했다. 이번 범위에는
실제 서버, RGB, keyboard/mouse control, MineStudio/MineRL/Malmo dependency가
포함되지 않는다.
