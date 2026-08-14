# AASSR Core Architecture v1

상태: **구조 동결(frozen)**  
버전: `aassr-core-architecture-v1`

이 문서는 새 AASSR Core의 **책임 경계와 데이터 흐름**을 고정한다. 이후 DQN, GRU, ensemble, feature 크기, 학습률 같은 구현은 바뀔 수 있지만, 아래 구성요소의 소유권과 방향을 바꾸려면 아키텍처 버전을 올려야 한다.

## 1. 최상위 원칙

> **Plugin은 세계의 문법과 공개 I/O만 제공하고, 문제 해결의 의미·기억·판단·예측·계획은 Core가 소유한다.**

Plugin은 후보의 가치, 정답/오답, 진전, 역할, world model, shaping reward를 제공하지 않는다.

## 2. 확정된 Core 구성요소와 실행 순서

```text
REAL WORLD
   │ public I/O
   ▼
Plugin
   │ PluginStepResult
   ▼
PluginContract / Transition
   │
   ▼
Knowledge
   │
   ├──────────────┐
   ▼              │
ActionSurface     │
   │              │
   ▼              │
Representation ◀─┘
   │
   ▼
Skills augment action surface
   │
   ▼
ASEQ exact S→A→S guard
   │
   ▼
Policy base decision
   │
   ▼
Imagination gate
   ├─ not ready ─────────────────────────────┐
   │                                         │
   └─ ready → Prophecy → Critic → compare ──┤
                                             ▼
                                         final action
                                             │
                                             ▼
                                           Plugin
                                             │
                                             ▼
                                      real Transition
                                             │
                 ┌───────────────────────────┼───────────────────────────┐
                 ▼                           ▼                           ▼
             Knowledge                  Policy/Prophecy              Critic/Skills
                 │                           │                           │
                 └───────────────────────────┴──────────────┬────────────┘
                                                            ▼
                                                   ASEQ records (S,A,S')

Runtime wraps the whole lifecycle and owns orchestration only.
```

이 순서는 고정한다. **ASEQ는 Policy/Imagination이 판단하기 전에 후보 surface를 guard한다.** Imagination은 ASEQ를 통과한 후보들에 대해 Policy의 기본 결정을 유지하거나 대체할 수 있다.

## 3. 각 구성요소의 책임

### PluginContract / Transition

환경과 Core 사이의 유일한 경계다. 행동 문법, 공개 관찰 자료형, 외부 reward, terminated/truncated/error만 받는다. Core의 학습 객체는 Plugin 구현을 직접 알지 않는다.

### Knowledge

AASSR의 canonical memory다.

- 공개 관찰 증거 기억
- 실행을 위해 필요한 concrete public value 기억
- 실제 action/outcome 경험 기억
- episode-local / persistent lifetime 구분

Knowledge는 행동을 평가하거나 선택하지 않는다.

### ActionSurface

`PluginSchema + current public observation + Knowledge`에서 **기계적으로 실행 가능한 후보**를 만든다.

- 후보 생성만 담당
- ranking 금지
- 정답/진전 기반 필터 금지
- concrete 이름 사전순 우선순위 금지

Plugin이 후보 목록을 제공하지 않는 이유를 이 구성요소가 명확히 담당한다.

### Representation

Core 내부의 공통 언어다.

- state vector
- action feature / structural action representation
- semantic state identity
- StateSnapshot 구성

Policy, Prophecy, Critic, ASEQ, Skills가 서로 다른 환경별 표현을 만들지 않고 이 표현을 공유한다.

### Policy

현재 상태에서 행동의 **기본 가치와 기본 선택**을 학습한다.

- 외부 sparse reward 학습
- 내부 information value는 외부 reward와 분리된 채널로 유지
- 환경별 수동 rule 금지

DQN은 현재 구현일 뿐 아키텍처의 필수 조건이 아니다.

### Prophecy

`S, A -> S'`를 학습하는 Core-owned world model이다.

- 실제 transition으로만 학습
- Calibration은 Prophecy의 하위 책임
- Plugin이 별도 환경 모델을 설치할 수 없음

GRU/ensemble/neural-delta는 구현 선택이다.

### Critic

현재/예상 trajectory의 sparse-return 가치를 평가한다.

- Policy와 별도
- Prophecy와 별도
- 실제 episode return으로 학습
- Imagination branch scoring에 사용

### Imagination

Prophecy와 Critic을 사용해 실제 행동 전에 미래를 비교한다.

- 스스로 세계의 사실을 생성해 Knowledge에 기록하지 않음
- imagined transition을 real transition처럼 학습 데이터로 취급하지 않음
- readiness/coverage/calibration 조건이 만족될 때만 개입
- 실제 planner run이 0이면 ON/OFF 성능 비교를 유효한 treatment로 보지 않음

### ASEQ

사용자가 고정한 의미를 그대로 유지한다.

> 반복된 정확한 semantic `S -> A -> S`만 억제한다.

- `S -> A -> S'`, `S' != S`는 허용
- volatile counter만 달라졌다고 다른 semantic state로 보지 않음
- 모든 후보가 guard되면 원래 action set으로 fallback
- 일반적인 anti-repeat penalty가 아님

### Skills

성공한 행동 구조를 재사용 가능한 macro/sequence로 학습한다.

- concrete ID가 아닌 structural action template 저장
- 현재 concrete 세계에 grounding할 때 Core Policy를 사용
- domain role/정답 label을 저장하지 않음

### Runtime

Runtime은 **조립과 lifecycle만** 담당한다.

- begin/reset
- select -> execute -> observe
- episode finish
- 각 learner의 update 순서
- diagnostics

Runtime 안에 환경 의미, feature 설계, 가치 함수, world model 규칙을 넣지 않는다.

## 4. 확정된 실제 데이터 흐름

```text
Plugin observation
    ↓
Knowledge.observe(public evidence)
    ↓
ActionSurface.generate(schema, observation, knowledge)
    ↓
Representation.build_state(...)
    ↓
Skills augment reusable macro actions
    ↓
ASEQ exact-self-loop guard
    ↓
Policy base decision
    ↓
[Imagination gate ready?]
    ├─ no  → Policy action
    └─ yes → Prophecy + Critic planning → keep/replace Policy action
    ↓
Plugin real execution
    ↓
real Transition
    ├─ Knowledge update
    ├─ Policy update
    ├─ Prophecy + Calibration update
    ├─ Critic episodic update
    ├─ Skills update
    └─ ASEQ records (S, A, S')
```

## 5. 구조와 구현을 분리한다

다음은 **구조가 아니다**. 이후 한 변수씩 교체·실험할 수 있다.

- DQN vs 다른 Policy learner
- GRU vs Transformer Prophecy
- ensemble 크기
- hidden units
- feature vector 차원
- Critic network 종류
- imagination depth/beam/threshold
- optimizer / batch size / replay 방식

반대로 아래 변경은 아키텍처 변경으로 취급한다.

- Plugin이 representation 또는 ranking을 소유하게 함
- Knowledge를 Plugin으로 이동
- Policy/Prophecy/Critic이 서로 다른 환경 전용 state 표현을 사용
- imagined transition을 real knowledge로 기록
- ASEQ를 일반 반복 패널티로 변경
- Runtime에 task-specific 의미 규칙을 삽입
- ActionSurface의 전략적 후보 필터를 Plugin 또는 환경 코드에 맡김

## 6. 현재 코드와의 정렬 작업

아키텍처는 이 문서로 먼저 확정한다. 현재 구현은 아직 파일 단위로 이 구조와 1:1 정렬되지 않은 부분이 있다.

특히 다음은 **구조 변경이 아니라 코드 정리 대상**이다.

1. `public_memory.py` 안에 Knowledge와 ActionSurface 책임이 같이 있음
2. `representation.py` / `schema_representation.py`의 역할이 겹침
3. Imagination/ASEQ/일부 generic type이 historical top-level module에 남아 있음
4. Runtime이 여러 legacy generic implementation을 직접 import함

다음 단계는 새로운 알고리즘을 추가하는 것이 아니라, 위 확정 구조에 맞게 파일/클래스 소유권을 정렬하는 것이다. 이 정렬이 끝나기 전에는 localhost 난도 상승이나 성능 실험을 진행하지 않는다.

## 7. 연구 증거 경계

기존 10k pentest current-generation 경로는 historical reproduction으로 유지한다. 새 Core v1 아키텍처와 과거 checkpoint 결과를 같은 구현 세대로 섞어 해석하지 않는다.
