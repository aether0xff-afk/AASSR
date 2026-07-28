# AASSR v2

AASSR v2는 기존 구현을 복사하지 않고 처음부터 다시 설계한 연구용 코드베이스다.

목표는 에이전트에게 사물별 정답 규칙이나 행동 순서를 직접 가르치는 것이 아니다. 플러그인은 명령 문법과 기본 조작만 제공하고, 어떤 정보와 행동 조합이 목표에 유용한지는 실제 경험·Prophecy·Imagination을 통해 학습한다.

## 핵심 폐루프

```text
관측
→ 원본 경험과 Knowledge Store 기록
→ 정보 특징 생성 및 온라인 비지도 군집화
→ GOAL 상태 차이 계산
→ Policy 상위 행동과 학습된 Skill 후보 생성
→ Prophecy 기반 평행우주 나무 생성
→ 가장 높은 가치의 미래로 이어지는 첫 행동 선택
→ 현실에서 첫 행동 또는 Skill의 원시 행동 실행
→ 실제 다음 상태로 Prophecy 학습
→ 검증 전이에서 예측 개선 측정
→ 정보 가치와 지연 기여를 Policy에 반영
→ 다음 현실 상태에서 다시 계획
```

## 0.2.0에서 구현된 구조

### 범용 행동 플러그인

코어는 `MOVE`, `SCAN`, `BREAK` 같은 단어의 의미를 가정하지 않는다. 플러그인은 `ActionSchema`와 `ParameterSpec`으로 행동 문법·필수/선택 파라미터·기본값만 선언한다.

- 임의의 문자열 행동 verb 지원
- 동적 파라미터와 안정적인 action signature
- 플러그인 등록 및 실행 분리
- 행동별 슬롯 후보의 2단계 선택
- 커리큘럼용 파라미터 문법 수업 자동 생성

### 정보 특징과 의미 형성

`OnlineFeatureMemory`는 정보를 이름으로 고정 분류하지 않고, 관측 특징과 `행동-슬롯-결과` 경험으로 표현한다.

- 해시 특징 기본선
- 온라인 비지도 군집화와 재배치
- 군집 역할 점수와 개별 정보 점수
- 군집 선택 후 구체 값 선택
- 대규모 관측에서만 임베딩을 쓰는 선택적 라우터
- 경험 특징/임베딩/혼합 표현 기능 제거 설정

### GOAL과 자율 Skill

GOAL은 행동 명령이 아니라 현재 상태와 원하는 상태의 차이다.

- 사실 보유·부재, 행동 가능성, 최종 진행도, 지식 필요, 벡터 목표
- 막힌 행동이나 원하는 상태로부터 내부 GOAL 생성
- GOAL 진전을 직접 평가하는 Imagination scorer
- 같은 GOAL을 반복해서 해결한 ASeq만 Skill 후보로 승격
- Skill을 하나의 행동처럼 Policy와 Imagination 후보에 포함
- Skill 실패 시 신뢰도 하락 및 원시 행동 수준으로 복귀

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
- GOAL, Skill, 정보 가치 제거
- 특징 없음, 특징만, 군집, 2단계 선택, 온라인 재군집, 행동-슬롯 문맥
- 경험 특징, 임베딩 특징, 혼합 특징

## 실험 실행

설정 문법과 실행 규모만 확인:

```bash
python scripts/run_experiment.py --config configs/pilot.json --dry-run
```

전체 파일럿 실행:

```bash
python scripts/run_experiment.py --config configs/pilot.json --output runs/pilot --overwrite
```

결과는 다음으로 저장된다.

```text
runs/pilot/
├─ resolved_config.json
├─ episodes.csv
├─ seed_summary.csv
├─ summary.csv
├─ report.md
└─ traces/
```

본 실험 설정:

- `configs/prophecy.json`
- `configs/imagination.json`
- `configs/dependency.json`
- `configs/goals_skills.json`
- `configs/information_value.json`

자세한 명령과 파일럿 해석 기준은 [`docs/experiments.md`](docs/experiments.md)를 참고한다.

## 검증

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src tests scripts
pytest -q
```

새 로드맵 기능은 `tests/test_roadmap_completion.py`에서 검증하고, 배치 실험기·파일럿·seed 단위 통계는 `tests/test_experiment_runner.py`에서 검증한다.

## 버전

- 연구 세대: **AASSR v2**
- 코드 패키지: **0.2.0**
- 개발 브랜치: **aassr-v2**
- 기존 `main` 및 이전 AASSR 구현은 수정하지 않는다.
